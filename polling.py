###############################################################################
# FPGA–VPU MATRIX MULTIPLICATION SIMULATOR (FULL DUPLEX, ONE PACKET PER MATRIX)
#
# Goal:
#   - Model a system where each VPU computes C = A × B and returns the **entire**
#     C matrix as **one** packet (frame) over a VPU→FPGA link.
#   - Links are full-duplex and rate-limited per direction (e.g., 200 Mb/s each).
#   - Data motion and compute occur in discrete time steps (ticks).
#   - We keep strict byte-accurate accounting: headers and payload consume wire
#     bandwidth; polling/control is logic-only (no wire cost).
#
# Key mechanics:
#   - FPGA -> VPU (ingress): pushes B-matrix bytes into a VPU FIFO (rate-limited).
#   - VPU moves a whole B matrix from FIFO to SPM (when space allows).
#   - VPU starts compute when SPM holds both A and B (A preloaded once).
#   - On completion, VPU enqueues **one frame** whose payload is the entire C.
#   - VPU -> FPGA (egress): link drains bytes from the current frame (header first,
#     then payload). When a frame finishes on the wire, the FPGA assembler
#     "receives" the complete matrix (single-frame = instant reassembly).
#
# Outputs:
#   - Aggregate stats (bits transferred, utilization, payload efficiency, etc.)
#   - Per-VPU stats (bytes in/out, frames, header/payload split, stalls, etc.)
#   - JSON dump of the results and a CSV in ~/Downloads with per-VPU metrics.
###############################################################################

# ------------------------------ Standard libs --------------------------------
import math       # arithmetic helpers (ceil, etc.)
import csv        # write per-VPU CSV
import os         # create output folders and resolve paths
import json       # dump results to a JSON file
import tempfile   # make a temp directory for artifacts
import statistics # compute simple averages for sampled occupancies
from collections import deque   # queue for frames on the VPU egress
from dataclasses import dataclass  # compact, typed "struct"-like classes

# ------------------------------ Parameters -----------------------------------
# All tunable knobs live here. You can edit and re-run to sweep behaviors.
params = {
    "num_vpus": 16,              # How many independent VPUs we simulate.
    "matrix_dim": 16,            # N in an N×N matrix.
    "elem_bytes": 4,             # Bytes per matrix element (e.g., int32 => 4).
    "vpu_spm_kb": 512,           # Scratchpad capacity per VPU (in kilobytes).
    "fifo_bytes": 64 * 1024,     # Ingress FIFO depth per VPU (in bytes).

    # Per-direction link bandwidths (FULL DUPLEX) in megabits per second.
    "link_in_Mbps": 200.0,       # FPGA → VPU (feeds FIFO with B).
    "link_out_Mbps": 200.0,      # VPU → FPGA (sends one frame per C).

    "tick_s": 1e-6,              # Simulation tick duration (seconds). Here: 1 µs.
    "sim_time_s": 0.5,           # Total simulated time (seconds). Here: 0.5 s.
    "clock_MHz": 200.0,          # VPU internal clock (in MHz). Used for compute time.
    "macs_per_cycle": 1.0,       # Parallel MACs / cycle (VPU compute width).
    "unconditional_push": True,  # If True, FPGA always tries to push at link rate.

    # Frame header size (bytes). Payload is entire C. Header+payload consume wire.
    "PKT_HEADER_BYTES": 24,
}

# ------------------------- Derived constants/quantities -----------------------
# For readability, pull a couple into locals.
N = params["matrix_dim"]                 # N in N×N
elem_bytes = params["elem_bytes"]        # bytes per element

# Size of a single N×N matrix in BYTES and BITS (this is the C payload size).
size_matrix_bytes = N * N * elem_bytes   # e.g., 16*16*4 = 1024 bytes
size_matrix_bits = size_matrix_bytes * 8 # e.g., 8192 bits

# Store these back into params (handy to see in the printed config).
params["size_matrix_bytes"] = size_matrix_bytes
params["size_matrix_bits"]  = size_matrix_bits

# Convert link rates (Mb/s) into BYTES PER TICK for each direction.
#   link_in_Mbps  : FPGA → VPU
#   link_out_Mbps : VPU → FPGA
in_bps   = params["link_in_Mbps"]  * 1e6              # bits/sec
out_bps  = params["link_out_Mbps"] * 1e6              # bits/sec
in_bytes_per_tick  = (in_bps  / 8.0) * params["tick_s"]  # bytes/tick
out_bytes_per_tick = (out_bps / 8.0) * params["tick_s"]  # bytes/tick

# Compute time for ONE matrix multiply on a VPU:
#   Total MACs for N×N × N×N → N^3
#   VPU can do macs_per_cycle per cycle. So cycles = N^3 / macs_per_cycle.
#   With a clock of clock_MHz, we get cycles/sec = clock_MHz*1e6.
#   So time (sec) = cycles / (clock_Hz).
compute_cycles_per_matrix = (N**3) / params["macs_per_cycle"]
compute_time_s = compute_cycles_per_matrix / (params["clock_MHz"] * 1e6)
params["compute_time_s"] = compute_time_s  # stash for reference

# A convenient reference: effective GMAC/s (giga-MACs per second) per VPU.
effective_gmacps = (params["macs_per_cycle"] * params["clock_MHz"]) / 1000.0

# ----------------------- Print the full configuration ------------------------
print("Simulation parameters:")
for k, v in params.items():
    print(f"  {k}: {v}")
print("Derived:")
print(f"  matrix_bytes={size_matrix_bytes}")
print(f"  in_bytes_per_tick={in_bytes_per_tick:.3f}  out_bytes_per_tick={out_bytes_per_tick:.3f}")
print(f"  compute_time_s={compute_time_s:.6e}  (~{compute_time_s*1e6:.2f} us)")
print(f"  VPU effective throughput ≈ {effective_gmacps:.2f} GMAC/s")

# ----------------------- Frame (packet) representation -----------------------
@dataclass
class Frame:
    """
    Represents one **wire packet** emitted by a VPU toward the FPGA.

    This model is "one-packet-per-matrix":
      - payload_bytes = size of full C matrix (N*N*elem_bytes)
      - header_bytes  = fixed protocol tag size (PKT_HEADER_BYTES)
      - tx_offset     = how many bytes of this frame have already gone out
      - rx_time_s     = time when the last byte of this frame hit the FPGA
    """
    vpu_id: int                 # Which VPU produced/sent this frame.
    txn_id: int                 # Transaction ID assigned at compute start.
    tile_id: int                # 0 in this model; placeholder for tiling.
    seq_idx: int                # Always 0 in this model (single frame).
    seq_count: int              # Always 1 in this model (single frame).
    header_bytes: int           # Fixed tag/header size (counts on wire).
    payload_bytes: int          # Size of full C matrix (counts on wire).
    tx_offset: int = 0          # How many bytes sent so far (0..wire_len).
    rx_time_s: float = None     # Timestamp when frame completed on the wire.

    @property
    def wire_len(self) -> int:
        """Total bytes on the wire for this frame: header + payload."""
        return self.header_bytes + self.payload_bytes

# --------------------------- FPGA-side assembler -----------------------------
class FPGAAssembler:
    """
    Collects frames "arriving" from VPUs.
    In this single-frame model, a frame == a complete matrix, so reassembly
    is trivial: we count arrivals and collect stats.

    We still track header vs. payload bits to measure payload efficiency.
    """
    def __init__(self):
        self.completed_latencies_us = []  # List of "reassembly latencies" (here always 0).
        self.total_frames = 0             # How many frames arrived (across all VPUs).
        self.header_bits = 0              # Sum of header bits observed on the wire.
        self.payload_bits = 0             # Sum of payload bits observed on the wire.

    def note_wire_bytes(self, hdr_bytes: int, pay_bytes: int) -> None:
        """
        Called each tick by the link layer to attribute header/payload bytes
        to the FPGA-side accounting. We keep *bits* to match link capacity units.
        """
        self.header_bits  += hdr_bytes * 8
        self.payload_bits += pay_bytes * 8

    def on_frame_rx(self, frame: Frame, t_now_s: float) -> None:
        """
        Called exactly when a full frame finishes on the wire this tick.
        Because it's one-frame-per-matrix, the "reassembly latency" is 0.
        (In a multi-frame model, you'd store first/last arrival times here.)
        """
        self.total_frames += 1
        self.completed_latencies_us.append(0.0)  # one-frame => immediate completion

# ------------------------------- VPU model -----------------------------------
class VPUState:
    """
    Captures the per-VPU internal state and behavior:
      - Ingress FIFO that accepts bytes from FPGA (rate-limited).
      - SPM storage for matrix tiles (we preload A once at sim start).
      - Compute engine that consumes "B" and produces "C".
      - Egress frame queue; each completed compute enqueues one frame.
      - Statistics (I/O bytes, frames, stall/active ticks, utilizations).
    """
    def __init__(self, id: int, params: dict):
        self.id = id

        # ----- Ingress FIFO (FPGA → VPU) ---------------------------------
        self.fifo_capacity = params["fifo_bytes"]   # Max FIFO depth in bytes.
        self.fifo_bytes = 0.0                       # Current FIFO occupancy.

        # ----- Scratchpad (SPM) ------------------------------------------
        self.spm_capacity = params["vpu_spm_kb"] * 1024  # Capacity in bytes.
        self.spm_bytes = size_matrix_bytes               # Preload A once.

        # ----- Compute engine --------------------------------------------
        self.compute_busy = False        # True while compute is in flight.
        self.compute_end_time = 0.0      # Sim time when compute will finish.
        self._current_txn_id = None      # txn_id for the compute in progress.
        self.next_txn_id = 1             # Monotonic txn counter for this VPU.

        # ----- Egress (VPU → FPGA) ---------------------------------------
        self.egress_frames = deque()     # Queue of frames waiting to be sent.

        # ----- Counters for I/O and link accounting ----------------------
        self.total_in_bytes = 0.0        # Sum of bytes successfully accepted (FPGA→VPU).
        self.total_out_bytes = 0.0       # Sum of bytes sent on wire (VPU→FPGA, header+payload).
        self.dropped_in_bytes = 0.0      # Bytes the FPGA "tried" to send but FIFO was full.
        self.link_bytes_transferred = 0.0# Total bytes "on the wire" in both directions.

        # Fine-grained VPU→FPGA split (to compute payload efficiency).
        self.header_bytes_sent = 0       # Header bytes actually transmitted.
        self.payload_bytes_sent = 0      # Payload bytes actually transmitted.
        self.frames_sent = 0             # Count of fully transmitted frames.

        # Activity tracking (useful for stall vs active fractions).
        self.ticks_stalled = 0           # Ticks with no compute and no ready data.
        self.ticks_active = 0            # Ticks with compute active or data available.

        # Time-series samples to compute average FIFO/SPM occupancy.
        self.fifo_occupancy_samples = []
        self.spm_occupancy_samples = []

    # -------------------- Small helpers for clarity ---------------------------
    def sample(self) -> None:
        """Sample FIFO and SPM occupancy once per tick for averaging."""
        self.fifo_occupancy_samples.append(self.fifo_bytes)
        self.spm_occupancy_samples.append(self.spm_bytes)

    def begin_compute(self, t_now: float, compute_time_s: float) -> None:
        """
        Start a new compute:
          - Mark busy
          - Set end time
          - Allocate a new transaction ID
        """
        self.compute_busy = True
        self.compute_end_time = t_now + compute_time_s
        self._current_txn_id = self.next_txn_id
        self.next_txn_id += 1

    def enqueue_result_frame(self, header_bytes: int, payload_bytes: int) -> None:
        """
        Push exactly ONE frame (header + full C payload) into the egress queue.
        The frame carries the txn_id of the compute that just finished.
        """
        fr = Frame(
            vpu_id=self.id,
            txn_id=self._current_txn_id,
            tile_id=0,            # No tiling in this model.
            seq_idx=0,            # Single-frame message => index 0
            seq_count=1,          # ...and count 1.
            header_bytes=header_bytes,
            payload_bytes=payload_bytes,
        )
        self.egress_frames.append(fr)
        self._current_txn_id = None  # Clear; the frame now holds the identity.

# ------------------------------- Simulator -----------------------------------
def run_sim(params: dict) -> dict:
    """
    Main time-step simulation.

    Returns:
      results: dict with 'params', 'aggregate', and 'per_vpu' stats that we
               also dump to JSON/CSV after the run.
    """
    # Pull out a few locals for speed/readability.
    num_vpus = params["num_vpus"]
    tick = params["tick_s"]
    total_ticks = int(math.ceil(params["sim_time_s"] / tick))  # how many steps

    # Instantiate VPUs and the single FPGA assembler (shared sink).
    vpus = [VPUState(i, params) for i in range(num_vpus)]
    assembler = FPGAAssembler()

    # Running simulation time (seconds).
    t = 0.0

    # Aggregate link usage across all VPUs and both directions (in bits).
    aggregate_bits_transferred = 0.0

    # Constant header size for frames (in bytes).
    PKT_HDR = params["PKT_HEADER_BYTES"]

    # ------------------------------ Main loop --------------------------------
    for _ in range(total_ticks):

        # ========================= PHASE 1: Links ============================
        # For each VPU, apply the per-direction link budgets for this tick.
        # INCOMING: FPGA -> VPU (fills FIFO)
        # OUTGOING: VPU -> FPGA (drains frames)
        for v in vpus:
            # Budget (in bytes) available this tick on each direction.
            in_budget  = in_bytes_per_tick
            out_budget = out_bytes_per_tick

            # ---- FPGA → VPU (Ingress) -----------------------------------
            # If unconditional_push, FPGA "tries" to send at the full budget.
            # Otherwise you'd clamp to remaining FIFO space here.
            if params["unconditional_push"]:
                incoming_attempt = in_budget
            else:
                incoming_attempt = min(in_budget, v.fifo_capacity - v.fifo_bytes)

            # How much free space do we actually have in the FIFO right now?
            free_fifo = v.fifo_capacity - v.fifo_bytes

            # Actual bytes accepted is the minimum of attempt and space.
            incoming_accepted = min(incoming_attempt, free_fifo)

            # Excess bytes "attempted" beyond FIFO capacity are "dropped"
            # (modeling a sender that pushes optimistically). Drops do NOT
            # count as consumed wire bandwidth in this model.
            dropped = max(0.0, incoming_attempt - incoming_accepted)

            # Update the FIFO occupancy and cumulative accepted/dropped counters.
            v.fifo_bytes       += incoming_accepted
            v.total_in_bytes   += incoming_accepted
            v.dropped_in_bytes += dropped

            # Account wire usage for accepted ingress bytes.
            v.link_bytes_transferred += incoming_accepted
            aggregate_bits_transferred += incoming_accepted * 8.0

            # ---- VPU → FPGA (Egress) ------------------------------------
            # We now drain bytes from the front frame in the egress queue.
            # We split transmitted bytes into header vs payload for efficiency.
            bytes_sent_this_tick   = 0  # total VPU→FPGA bytes we managed to send
            header_sent_this_tick  = 0  # header subset of that
            payload_sent_this_tick = 0  # payload subset of that

            # While we still have budget and frames to send...
            while out_budget > 0 and v.egress_frames:
                # Look at the current head-of-queue frame.
                fr = v.egress_frames[0]

                # How many bytes of this frame remain to send this tick?
                remaining = fr.wire_len - fr.tx_offset
                if remaining <= 0:
                    # Defensive: if somehow already complete, pop it.
                    v.egress_frames.popleft()
                    continue

                # Send at most 'remaining' or 'out_budget', whichever is smaller.
                to_send = int(min(out_budget, remaining))
                if to_send <= 0:
                    break  # no more budget

                # 'before' is how many bytes were already sent.
                # 'after'  is how many will be sent after this step.
                before = fr.tx_offset
                after  = fr.tx_offset + to_send

                # Split the 'to_send' region into header and payload portions.
                # Header occupies byte indices [0, header_bytes).
                # Payload occupies [header_bytes, wire_len).
                # Bytes that overlap [before, after) with header count as header.
                hdr_sent = max(0, min(fr.header_bytes - before, to_send))
                pay_sent = to_send - hdr_sent

                # Advance frame progress and reduce remaining budget.
                fr.tx_offset = after
                out_budget  -= to_send

                # Track this tick's contribution.
                bytes_sent_this_tick   += to_send
                header_sent_this_tick  += hdr_sent
                payload_sent_this_tick += pay_sent

                # If we've finished this frame this tick, mark it "received" by FPGA.
                if fr.tx_offset >= fr.wire_len:
                    fr.rx_time_s = t                       # arrival timestamp
                    v.egress_frames.popleft()              # remove from queue
                    assembler.on_frame_rx(fr, t)           # FPGA sees it now
                    v.frames_sent += 1                     # frame count++

            # End of VPU→FPGA draining loop; update cumulative counters.
            v.total_out_bytes     += bytes_sent_this_tick
            v.header_bytes_sent   += header_sent_this_tick
            v.payload_bytes_sent  += payload_sent_this_tick

            # All outgoing bytes are wire usage; add to the link sums.
            v.link_bytes_transferred += bytes_sent_this_tick
            aggregate_bits_transferred += bytes_sent_this_tick * 8.0

            # Let the FPGA assembler accumulate header/payload bits for efficiency.
            assembler.note_wire_bytes(header_sent_this_tick, payload_sent_this_tick)

        # ====================== PHASE 2: FIFO → SPM ==========================
        # Move whole B matrices from FIFO into SPM whenever space allows.
        # We only move **whole** matrices to keep semantics clean.
        for v in vpus:
            # How much free space remains in SPM?
            spm_free = v.spm_capacity - v.spm_bytes

            # How many whole matrices are sitting in the FIFO?
            avail_fifo_mats = int(v.fifo_bytes // size_matrix_bytes)

            # How many whole matrices would fit into the SPM right now?
            fit_in_spm_mats = int(spm_free // size_matrix_bytes)

            # We can move at most the min of those two counts.
            move_count = min(avail_fifo_mats, fit_in_spm_mats)
            if move_count > 0:
                # Convert count to bytes and do the transfer.
                move_bytes = move_count * size_matrix_bytes
                v.fifo_bytes -= move_bytes
                v.spm_bytes  += move_bytes

        # ======================== PHASE 3: COMPUTE ===========================
        # If we have A preloaded and at least one B in SPM, start compute.
        # (We model A being present as SPM >= 2 * matrix_bytes: one A + one B.)
        for v in vpus:
            # Start a compute if not busy and there is A+B resident in SPM.
            if not v.compute_busy and (v.spm_bytes >= 2 * size_matrix_bytes):
                # "Consume" the B tile from SPM for the compute (A remains).
                v.spm_bytes -= size_matrix_bytes
                # Begin compute; record txn_id and completion time.
                v.begin_compute(t_now=t, compute_time_s=params["compute_time_s"])

            # Track activity vs stall: if we aren't busy and don't yet have B
            # (either in SPM or enough bytes in FIFO to form 1 matrix), count
            # this tick as "stalled"; otherwise as "active".
            has_B_available = (v.spm_bytes >= 2 * size_matrix_bytes)
            B_in_fifo       = (v.fifo_bytes >= size_matrix_bytes)
            if (not v.compute_busy) and (not has_B_available) and (not B_in_fifo):
                v.ticks_stalled += 1
            else:
                v.ticks_active  += 1

        # ======================== PHASE 4: COMPLETE ==========================
        # Any compute that has reached its end time emits exactly one frame
        # (header + full C payload) into the VPU egress queue.
        for v in vpus:
            if v.compute_busy and t >= v.compute_end_time:
                v.compute_busy = False
                v.compute_end_time = 0.0
                v.enqueue_result_frame(
                    header_bytes=params["PKT_HEADER_BYTES"],
                    payload_bytes=size_matrix_bytes
                )

        # ======================== PHASE 5: SAMPLE ============================
        # Record FIFO/SPM occupancies for averages.
        for v in vpus:
            v.sample()

        # Advance simulated time by 1 tick.
        t += tick

    # ------------------------- End of main loop -------------------------------

    # =========================== Post-processing ==============================
    results = {
        "params": dict(params),   # echo params back for provenance
        "aggregate": {},          # system-wide metrics
        "per_vpu": []             # per-VPU metrics array
    }

    # Total bits on the wire across all VPUs and both directions.
    results["aggregate"]["total_bits_transferred"] = aggregate_bits_transferred

    # Theoretical capacity over the whole system and window:
    #   num_vpus * (in+out) * (Mb/s→b/s) * sim_time
    total_link_bps = (params["link_in_Mbps"] + params["link_out_Mbps"]) * 1e6 * params["num_vpus"]
    results["aggregate"]["aggregate_utilization"] = (
        aggregate_bits_transferred / (params["sim_time_s"] * total_link_bps)
    )

    # Payload-vs-header efficiency for VPU→FPGA direction (assembler's view).
    denom_bits = assembler.header_bits + assembler.payload_bits
    payload_eff = (assembler.payload_bits / denom_bits) if denom_bits > 0 else 0.0
    results["aggregate"]["header_bits_transferred"]  = assembler.header_bits
    results["aggregate"]["payload_bits_transferred"] = assembler.payload_bits
    results["aggregate"]["payload_efficiency"]       = payload_eff
    results["aggregate"]["total_frames_rx"]          = assembler.total_frames

    # Matrix "reassembly" latency stats (always 0 because single-frame).
    lats = assembler.completed_latencies_us
    results["aggregate"]["matrix_latency_us"] = (
        {"count": len(lats),
         "min": float(min(lats)) if lats else None,
         "avg": float(sum(lats)/len(lats)) if lats else None,
         "max": float(max(lats)) if lats else None}
    )

    # Per-VPU metrics:
    # Normalize link_util_fraction per VPU against that VPU's full-duplex capacity:
    #   per_vpu_full_duplex_bps = (in+out) * 1e6
    per_vpu_full_duplex_bps = (params["link_in_Mbps"] + params["link_out_Mbps"]) * 1e6

    for v in vpus:
        # Averages over sampled occupancy histories.
        avg_fifo = statistics.mean(v.fifo_occupancy_samples) if v.fifo_occupancy_samples else 0.0
        avg_spm  = statistics.mean(v.spm_occupancy_samples)  if v.spm_occupancy_samples  else 0.0

        # Stall fraction := stalled_ticks / (stalled + active) (avoid div-by-zero).
        total_ticks_counted = (v.ticks_stalled + v.ticks_active)
        stall_fraction = (v.ticks_stalled / total_ticks_counted) if total_ticks_counted > 0 else 0.0

        # Link utilization for this VPU across both directions, normalized by that
        # VPU's full-duplex capacity over the whole sim window.
        link_util_fraction = (
            (v.link_bytes_transferred * 8.0) / (params["sim_time_s"] * per_vpu_full_duplex_bps)
            if per_vpu_full_duplex_bps > 0 else 0.0
        )

        # Per-VPU payload efficiency (header vs payload actually sent).
        per_eff = (
            v.payload_bytes_sent / (v.header_bytes_sent + v.payload_bytes_sent)
            if (v.header_bytes_sent + v.payload_bytes_sent) > 0 else 0.0
        )

        # Compose the per-VPU record.
        results["per_vpu"].append({
            "id": v.id,
            "total_in_bytes": v.total_in_bytes,         # bytes accepted from FPGA
            "total_out_bytes": v.total_out_bytes,       # header + payload bytes sent
            "dropped_in_bytes": v.dropped_in_bytes,     # push attempts beyond FIFO cap
            "avg_fifo_bytes": avg_fifo,                 # avg FIFO occupancy
            "avg_spm_bytes": avg_spm,                   # avg SPM occupancy
            "stall_fraction": stall_fraction,           # time spent idle (no work ready)
            "link_util_fraction": link_util_fraction,   # normalized to this VPU's cap

            # Egress framing stats (useful for payload efficiency insight)
            "frames_sent": v.frames_sent,
            "header_bytes_sent": v.header_bytes_sent,
            "payload_bytes_sent": v.payload_bytes_sent,
            "payload_efficiency_vpu": per_eff,

            # Handy references so you don't have to recompute later
            "effective_gmacps": effective_gmacps,       # derived from params
            "compute_time_us": params["compute_time_s"] * 1e6,
        })

    # Hand back everything to caller.
    return results

# ------------------------------- Run sim -------------------------------------
results = run_sim(params)

# ---------------------------- Persist results --------------------------------
# Create a temp directory for artifacts so you can inspect JSON later.
out_dir = tempfile.mkdtemp(prefix="vpu_sim_onepkt_")
out_json = os.path.join(out_dir, "sim_results.json")

# Write a pretty JSON dump with all metrics.
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to", out_json)

# ------------------------------ Print summary --------------------------------
print("\nAggregate summary:")
agg = results["aggregate"]

# For readability, compute the system's total full-duplex Mbps.
total_fd_Mbps = (params["link_in_Mbps"] + params["link_out_Mbps"]) * params["num_vpus"]
print(f"  Total bits transferred: {agg['total_bits_transferred']:.0f} bits")
print(f"  Aggregate utilization (vs full-duplex {total_fd_Mbps:.0f} Mbps): {agg['aggregate_utilization']:.3f}")
print(f"  Frames received: {agg['total_frames_rx']}")
print(f"  Payload efficiency (VPU→FPGA): {agg['payload_efficiency']:.3f} "
      f"[payload_bits={agg['payload_bits_transferred']}, header_bits={agg['header_bits_transferred']}]")

lat = agg["matrix_latency_us"]
print(f"  Matrix reassembly latency (us): count={lat['count']} min={lat['min']} avg={lat['avg']} max={lat['max']}")

print("\nPer-VPU summary (first 6):")
for p in results["per_vpu"][:6]:
    print(f"  VPU{p['id']}: in={p['total_in_bytes']:.0f}B out={p['total_out_bytes']:.0f}B "
          f"dropped={p['dropped_in_bytes']:.0f}B avg_fifo={p['avg_fifo_bytes']:.1f}B "
          f"avg_spm={p['avg_spm_bytes']:.1f}B stall_frac={p['stall_fraction']:.3f} "
          f"link_util={p['link_util_fraction']:.3f} frames={p['frames_sent']} "
          f"eff={p['payload_efficiency_vpu']:.3f} GMAC/s≈{p['effective_gmacps']:.2f} "
          f"t_compute≈{p['compute_time_us']:.2f} µs")

# ----------------------------- Export per-VPU CSV ----------------------------
# We also export per-VPU stats to ~/Downloads for quick spreadsheet inspection.
downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
os.makedirs(downloads_dir, exist_ok=True)
csv_file = os.path.join(downloads_dir, "per_vpu_stats.csv")

with open(csv_file, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "id","total_in_bytes","total_out_bytes","dropped_in_bytes",
        "avg_fifo_bytes","avg_spm_bytes","stall_fraction","link_util_fraction",
        "frames_sent","header_bytes_sent","payload_bytes_sent","payload_efficiency_vpu",
        "effective_gmacps","compute_time_us"
    ])
    for p in results["per_vpu"]:
        w.writerow([
            p["id"], p["total_in_bytes"], p["total_out_bytes"],
            p["dropped_in_bytes"], p["avg_fifo_bytes"], p["avg_spm_bytes"],
            p["stall_fraction"], p["link_util_fraction"],
            p["frames_sent"], p["header_bytes_sent"], p["payload_bytes_sent"],
            p["payload_efficiency_vpu"], p["effective_gmacps"], p["compute_time_us"]
        ])

print("CSV saved to", csv_file)
print(f"Temp dir: {out_dir}")
