###############################################################################
# FPGA–VPU MATRIX MULTIPLICATION SIMULATOR (CYCLE-ACCURATE)
# Each tick = 1 clock cycle at 200 MHz (5ns)
# 16 compute VPUs + 1 accumulator VPU (id=16)
###############################################################################

import math, csv, os, json, tempfile, statistics
from collections import deque
from dataclasses import dataclass

# ------------------------------ Parameters -----------------------------------
params = {
    # -------------------- System topology --------------------
    "num_vpus": 16,              # 16 compute VPUs; the 17th (id=16) is the accumulator
    "matrix_dim": 16,            # Each VPU computes a 16×16 matrix multiply (C = A×B)
    "elem_bytes": 4,             # Each element (float/int32) = 4 bytes

    # -------------------- Local memory capacities --------------------
    "vpu_spm_kb": 512,           # Each compute VPU scratchpad memory (KB)
    "fifo_bytes": 64 * 1024,     # FIFO buffer between FPGA link and SPM (bytes)

    # -------------------- Link characteristics --------------------
    # 200 Mb/s full-duplex = 25 MB/s = 25 bytes per microsecond
    "link_in_Mbps": 200.0,       # FPGA → Compute VPU
    "link_out_Mbps": 200.0,      # Compute VPU → FPGA
    "acc_link_in_Mbps": 200.0,   # FPGA → Accumulate VPU
    "acc_link_out_Mbps": 200.0,  # Accumulate VPU → FPGA

    # -------------------- Timing model --------------------
    "clock_MHz": 200.0,          # System clock = 200 MHz → 5ns per cycle
    "sim_time_us": 5000,         # Run simulation for 5000 microseconds (5ms)
    "macs_per_cycle": 64.0,      # Each VPU can perform 64 MACs per clock cycle

    # -------------------- Protocol framing --------------------
    "PKT_HEADER_BYTES": 24,      # Fixed 24-byte frame header
}

# ------------------------- Derived constants ---------------------------------
N = params["matrix_dim"]
elem_bytes = params["elem_bytes"]

# Matrix size
size_matrix_bytes = N * N * elem_bytes           # 16×16×4 = 1024 bytes
size_matrix_bits  = size_matrix_bytes * 8        # 8192 bits
params["size_matrix_bytes"] = size_matrix_bytes
params["size_matrix_bits"]  = size_matrix_bits

# Cycle time in seconds
cycle_time_s = 1.0 / (params["clock_MHz"] * 1e6)  # 5e-9 seconds = 5ns
params["cycle_time_s"] = cycle_time_s

# Total simulation cycles
total_cycles = int(params["sim_time_us"] * 1e-6 / cycle_time_s)
params["total_cycles"] = total_cycles

# Link throughput: bytes per cycle
link_bytes_per_cycle = (params["link_in_Mbps"] * 1e6 / 8.0) / (params["clock_MHz"] * 1e6)
params["link_bytes_per_cycle"] = link_bytes_per_cycle

# Compute latency: cycles per matrix multiply
compute_cycles = (N**3) / params["macs_per_cycle"]  # 4096 / 64 = 64 cycles
params["compute_cycles"] = compute_cycles

# Accumulator latency: sum N×N elements from 16 VPUs = 16*256 = 4096 operations
accum_ops = params["num_vpus"] * N * N
accum_cycles = accum_ops / params["macs_per_cycle"]  # 4096 / 64 = 64 cycles
params["accum_cycles"] = accum_cycles

print("=== Simulation Parameters ===")
for k, v in params.items():
    print(f"  {k}: {v}")
print(f"\nDerived:")
print(f"  Cycle time: {cycle_time_s*1e9:.2f} ns")
print(f"  Total cycles: {total_cycles:,}")
print(f"  Link throughput: {link_bytes_per_cycle:.4f} bytes/cycle")
print(f"  Compute latency: {compute_cycles:.0f} cycles ({compute_cycles*cycle_time_s*1e6:.2f} µs)")
print(f"  Accumulator latency: {accum_cycles:.0f} cycles ({accum_cycles*cycle_time_s*1e6:.2f} µs)")
print(f"  VPU throughput: {params['macs_per_cycle']*params['clock_MHz']/1000:.1f} GMAC/s\n")

# ------------------------------ Data Structures ------------------------------
@dataclass
class Frame:
    """Represents one frame/packet."""
    vpu_id: int
    txn_id: int
    header_bytes: int
    payload_bytes: int
    tx_offset: int = 0      # bytes already transmitted
    
    @property
    def wire_len(self) -> int:
        return self.header_bytes + self.payload_bytes
    
    @property
    def complete(self) -> bool:
        return self.tx_offset >= self.wire_len


# ------------------------------ Link Model -----------------------------------
class Link:
    """Models a unidirectional link with fractional byte accumulation."""
    def __init__(self, name: str, bytes_per_cycle: float):
        self.name = name
        self.bytes_per_cycle = bytes_per_cycle
        self.credit = 0.0  # fractional byte accumulator
        self.total_bytes = 0.0
        
    def tick(self):
        """Add bandwidth for this cycle."""
        self.credit += self.bytes_per_cycle
        
    def get_budget(self) -> int:
        """Get integer bytes available this cycle."""
        budget = int(self.credit)
        self.credit -= budget
        return budget
    
    def consume(self, bytes_used: int):
        """Track bytes transmitted."""
        self.total_bytes += bytes_used


# ------------------------------ VPU State ------------------------------------
class VPU:
    """Models a compute VPU."""
    def __init__(self, vpu_id: int):
        self.id = vpu_id
        
        # Buffers
        self.fifo_capacity = params["fifo_bytes"]
        self.fifo_bytes = 0
        self.spm_capacity = params["vpu_spm_kb"] * 1024
        self.spm_bytes = size_matrix_bytes  # Pre-load matrix A
        
        # Compute state
        self.compute_busy = False
        self.compute_end_cycle = 0
        self.current_txn_id = None
        self.next_txn_id = 1
        
        # Egress queue
        self.egress_queue = deque()
        
        # Statistics
        self.bytes_received = 0
        self.bytes_sent = 0
        self.bytes_dropped = 0
        self.frames_sent = 0
        self.cycles_stalled = 0
        self.cycles_computing = 0
        self.fifo_samples = []
        self.spm_samples = []
        
    def can_start_compute(self) -> bool:
        """Check if VPU can start next computation (NO BARRIER)."""
        return (not self.compute_busy and 
                self.spm_bytes >= 2 * size_matrix_bytes)
    
    def start_compute(self, cycle: int):
        """Begin matrix multiplication."""
        self.compute_busy = True
        self.compute_end_cycle = cycle + compute_cycles
        self.current_txn_id = self.next_txn_id
        self.next_txn_id += 1
        self.spm_bytes -= size_matrix_bytes  # Consume matrix B
        
    def finish_compute(self):
        """Complete computation and enqueue result frame."""
        self.compute_busy = False
        frame = Frame(
            vpu_id=self.id,
            txn_id=self.current_txn_id,
            header_bytes=params["PKT_HEADER_BYTES"],
            payload_bytes=size_matrix_bytes
        )
        self.egress_queue.append(frame)
        self.current_txn_id = None
        
    def sample_state(self):
        """Record current buffer occupancy."""
        self.fifo_samples.append(self.fifo_bytes)
        self.spm_samples.append(self.spm_bytes)


# ------------------------------ Accumulator VPU ------------------------------
class AccumulatorVPU:
    """Models the reduction/accumulation VPU."""
    def __init__(self):
        self.id = params["num_vpus"]
        self.num_vpus = params["num_vpus"]
        
        # Received frames tracking
        self.pending = {}  # txn_id -> set of vpu_ids
        
        # Compute queue - can process multiple transactions in pipeline
        self.compute_queue = deque()  # List of (txn_id, start_cycle, end_cycle)
        
        # Egress queue
        self.egress_queue = deque()
        
        # Statistics
        self.frames_received = 0
        self.frames_sent = 0
        self.bytes_sent = 0
        self.cycles_busy = 0
        
    def receive_frame(self, frame: Frame, cycle: int):
        """Process incoming frame from compute VPU."""
        self.frames_received += 1
        txn = frame.txn_id
        
        # Track which VPUs have reported
        if txn not in self.pending:
            self.pending[txn] = set()
        self.pending[txn].add(frame.vpu_id)
        
        # When all VPUs have sent their frame, queue accumulation
        if len(self.pending[txn]) == self.num_vpus:
            self.queue_accumulation(txn, cycle)
    
    def queue_accumulation(self, txn: int, cycle: int):
        """Queue a new accumulation task."""
        # Find when we can start (after previous accumulation finishes)
        if self.compute_queue:
            last_end = self.compute_queue[-1][2]
            start_cycle = max(cycle, last_end)
        else:
            start_cycle = cycle
        
        end_cycle = start_cycle + accum_cycles
        self.compute_queue.append((txn, start_cycle, end_cycle))
    
    def process_compute_queue(self, cycle: int):
        """Check if any accumulations have finished."""
        while self.compute_queue and self.compute_queue[0][2] <= cycle:
            txn, start, end = self.compute_queue.popleft()
            self.emit_result(txn)
            self.cycles_busy += (end - start)
    
    def emit_result(self, txn: int):
        """Create result frame for completed accumulation."""
        frame = Frame(
            vpu_id=self.id,
            txn_id=txn,
            header_bytes=params["PKT_HEADER_BYTES"],
            payload_bytes=size_matrix_bytes
        )
        self.egress_queue.append(frame)
        self.frames_sent += 1
        
        # Clean up
        if txn in self.pending:
            del self.pending[txn]


# ------------------------------ FPGA Model -----------------------------------
class FPGA:
    """Models the FPGA coordinator."""
    def __init__(self):
        self.frames_from_compute = 0
        self.frames_from_acc = 0
        self.forward_queue = deque()  # Frames to forward to accumulator
        
    def receive_from_compute(self, frame: Frame):
        """Receive completed frame from compute VPU."""
        self.frames_from_compute += 1
        # Reset tx_offset so we can forward to accumulator
        frame.tx_offset = 0
        self.forward_queue.append(frame)
        
    def receive_from_acc(self, frame: Frame):
        """Receive reduced result from accumulator."""
        self.frames_from_acc += 1


# ------------------------------ Simulator ------------------------------------
def run_simulation():
    """Main cycle-accurate simulation loop."""
    
    # Initialize components
    vpus = [VPU(i) for i in range(params["num_vpus"])]
    acc = AccumulatorVPU()
    fpga = FPGA()
    
    # Initialize links (4 per compute VPU + 2 for accumulator)
    vpu_in_bpc = (params["link_in_Mbps"] * 1e6 / 8.0) / (params["clock_MHz"] * 1e6)
    vpu_out_bpc = (params["link_out_Mbps"] * 1e6 / 8.0) / (params["clock_MHz"] * 1e6)
    acc_in_bpc = (params["acc_link_in_Mbps"] * 1e6 / 8.0) / (params["clock_MHz"] * 1e6)
    acc_out_bpc = (params["acc_link_out_Mbps"] * 1e6 / 8.0) / (params["clock_MHz"] * 1e6)
    
    # Per-VPU links
    vpu_in_links = [Link(f"VPU{i}_IN", vpu_in_bpc) for i in range(params["num_vpus"])]
    vpu_out_links = [Link(f"VPU{i}_OUT", vpu_out_bpc) for i in range(params["num_vpus"])]
    
    # Accumulator links
    acc_in_link = Link("ACC_IN", acc_in_bpc)
    acc_out_link = Link("ACC_OUT", acc_out_bpc)
    
    # Statistics
    total_bits_transferred = 0
    
    # Main simulation loop
    for cycle in range(total_cycles):
        
        # ====================================================================
        # PHASE 1: LINK TICKS (accumulate bandwidth)
        # ====================================================================
        for link in vpu_in_links + vpu_out_links + [acc_in_link, acc_out_link]:
            link.tick()
        
        # ====================================================================
        # PHASE 2: FPGA → COMPUTE VPU (ingress)
        # ====================================================================
        for i, vpu in enumerate(vpus):
            budget = vpu_in_links[i].get_budget()
            
            # FPGA always tries to push at link rate
            free_space = vpu.fifo_capacity - vpu.fifo_bytes
            bytes_to_send = min(budget, free_space)
            bytes_dropped = budget - bytes_to_send
            
            vpu.fifo_bytes += bytes_to_send
            vpu.bytes_received += bytes_to_send
            vpu.bytes_dropped += bytes_dropped
            
            if bytes_to_send > 0:
                vpu_in_links[i].consume(bytes_to_send)
                total_bits_transferred += bytes_to_send * 8
        
        # ====================================================================
        # PHASE 3: COMPUTE VPU → FPGA (egress)
        # ====================================================================
        for i, vpu in enumerate(vpus):
            budget = vpu_out_links[i].get_budget()
            bytes_sent_this_cycle = 0
            
            while budget > 0 and vpu.egress_queue:
                frame = vpu.egress_queue[0]
                remaining = frame.wire_len - frame.tx_offset
                
                if remaining <= 0:
                    vpu.egress_queue.popleft()
                    continue
                
                to_send = min(budget, remaining)
                frame.tx_offset += to_send
                budget -= to_send
                bytes_sent_this_cycle += to_send
                
                if frame.complete:
                    vpu.egress_queue.popleft()
                    vpu.frames_sent += 1
                    fpga.receive_from_compute(frame)
            
            if bytes_sent_this_cycle > 0:
                vpu.bytes_sent += bytes_sent_this_cycle
                vpu_out_links[i].consume(bytes_sent_this_cycle)
                total_bits_transferred += bytes_sent_this_cycle * 8
        
        # ====================================================================
        # PHASE 4: FPGA → ACCUMULATOR (forward)
        # ====================================================================
        budget = acc_in_link.get_budget()
        
        while budget > 0 and fpga.forward_queue:
            frame = fpga.forward_queue[0]
            remaining = frame.wire_len - frame.tx_offset
            
            if remaining <= 0:
                # Frame already complete, shouldn't happen
                fpga.forward_queue.popleft()
                continue
            
            to_send = min(budget, remaining)
            frame.tx_offset += to_send
            budget -= to_send
            
            # Track bytes on link
            acc_in_link.consume(to_send)
            total_bits_transferred += to_send * 8
            
            # Only deliver when fully forwarded
            if frame.tx_offset >= frame.wire_len:
                fpga.forward_queue.popleft()
                acc.receive_frame(frame, cycle)
        
        # ====================================================================
        # PHASE 5: ACCUMULATOR → FPGA (egress)
        # ====================================================================
        budget = acc_out_link.get_budget()
        bytes_sent_this_cycle = 0
        
        while budget > 0 and acc.egress_queue:
            frame = acc.egress_queue[0]
            remaining = frame.wire_len - frame.tx_offset
            
            if remaining <= 0:
                acc.egress_queue.popleft()
                continue
            
            to_send = min(budget, remaining)
            frame.tx_offset += to_send
            budget -= to_send
            bytes_sent_this_cycle += to_send
            
            if frame.complete:
                acc.egress_queue.popleft()
                fpga.receive_from_acc(frame)
        
        if bytes_sent_this_cycle > 0:
            acc.bytes_sent += bytes_sent_this_cycle
            acc_out_link.consume(bytes_sent_this_cycle)
            total_bits_transferred += bytes_sent_this_cycle * 8
        
        # ====================================================================
        # PHASE 6: FIFO → SPM TRANSFER
        # ====================================================================
        for vpu in vpus:
            available_matrices = int(vpu.fifo_bytes // size_matrix_bytes)
            spm_free = vpu.spm_capacity - vpu.spm_bytes
            can_fit = int(spm_free // size_matrix_bytes)
            
            to_move = min(available_matrices, can_fit)
            if to_move > 0:
                move_bytes = to_move * size_matrix_bytes
                vpu.fifo_bytes -= move_bytes
                vpu.spm_bytes += move_bytes
        
        # ====================================================================
        # PHASE 7: COMPUTE START (NO BARRIER - VPUs run independently)
        # ====================================================================
        for vpu in vpus:
            if vpu.can_start_compute():
                vpu.start_compute(cycle)
        
        # ====================================================================
        # PHASE 8: COMPUTE FINISH
        # ====================================================================
        for vpu in vpus:
            if vpu.compute_busy:
                if cycle >= vpu.compute_end_cycle:
                    vpu.finish_compute()
                vpu.cycles_computing += 1
            else:
                # Check if stalled (no data to compute)
                has_data = vpu.spm_bytes >= 2 * size_matrix_bytes
                if not has_data:
                    vpu.cycles_stalled += 1
        
        # Accumulator compute processing
        acc.process_compute_queue(cycle)
        
        # ====================================================================
        # PHASE 9: SAMPLE STATE
        # ====================================================================
        if cycle % 1000 == 0:  # Sample every 1000 cycles
            for vpu in vpus:
                vpu.sample_state()
    
    # ====================================================================
    # POST-PROCESSING
    # ====================================================================
    results = {
        "params": dict(params),
        "aggregate": {},
        "per_vpu": [],
        "accumulator": {}
    }
    
    # Aggregate statistics
    total_link_capacity = (
        params["num_vpus"] * (params["link_in_Mbps"] + params["link_out_Mbps"]) +
        (params["acc_link_in_Mbps"] + params["acc_link_out_Mbps"])
    ) * 1e6
    
    sim_time_s = total_cycles * cycle_time_s
    
    results["aggregate"]["total_bits_transferred"] = total_bits_transferred
    results["aggregate"]["aggregate_utilization"] = total_bits_transferred / (sim_time_s * total_link_capacity)
    results["aggregate"]["frames_from_compute"] = fpga.frames_from_compute
    results["aggregate"]["frames_from_acc"] = fpga.frames_from_acc
    results["aggregate"]["total_frames"] = fpga.frames_from_compute + fpga.frames_from_acc
    
    # Per-VPU statistics
    for vpu in vpus:
        avg_fifo = statistics.mean(vpu.fifo_samples) if vpu.fifo_samples else 0
        avg_spm = statistics.mean(vpu.spm_samples) if vpu.spm_samples else 0
        
        total_active = vpu.cycles_computing + vpu.cycles_stalled
        stall_frac = vpu.cycles_stalled / total_active if total_active > 0 else 0
        compute_util = vpu.cycles_computing / total_cycles
        
        results["per_vpu"].append({
            "id": vpu.id,
            "bytes_received": vpu.bytes_received,
            "bytes_sent": vpu.bytes_sent,
            "bytes_dropped": vpu.bytes_dropped,
            "frames_sent": vpu.frames_sent,
            "avg_fifo_bytes": avg_fifo,
            "avg_spm_bytes": avg_spm,
            "stall_fraction": stall_frac,
            "compute_utilization": compute_util,
        })
    
    # Accumulator statistics
    acc_util = acc.cycles_busy / total_cycles if total_cycles > 0 else 0
    results["accumulator"] = {
        "id": acc.id,
        "frames_received": acc.frames_received,
        "frames_sent": acc.frames_sent,
        "bytes_sent": acc.bytes_sent,
        "compute_utilization": acc_util,
        "pending_count": len(acc.pending),
        "queue_depth": len(acc.compute_queue),
    }
    
    # Link statistics
    results["links"] = {
        "vpu_ingress": [{"id": i, "bytes": link.total_bytes} for i, link in enumerate(vpu_in_links)],
        "vpu_egress": [{"id": i, "bytes": link.total_bytes} for i, link in enumerate(vpu_out_links)],
        "acc_ingress_bytes": acc_in_link.total_bytes,
        "acc_egress_bytes": acc_out_link.total_bytes,
    }
    
    return results


# ------------------------------ Run Simulation -------------------------------
print("Running simulation...")
results = run_simulation()

# ------------------------------ Print Results --------------------------------
print("\n" + "="*80)
print("KERNEL TIMING")
print("="*80)
print(f"Cycles per kernel: {compute_cycles:.0f} cycles")
print(f"Time per kernel: {compute_cycles * cycle_time_s * 1e6:.3f} µs")
print(f"Compute throughput per VPU: {params['macs_per_cycle'] * params['clock_MHz'] / 1000:.1f} GMAC/s")

print("\n" + "="*80)
print("AGGREGATE RESULTS")
print("="*80)
agg = results["aggregate"]
print(f"Total bits transferred: {agg['total_bits_transferred']/1e6:.2f} Mbits")
print(f"Aggregate link utilization: {agg['aggregate_utilization']*100:.2f}%")
print(f"Frames from compute VPUs: {agg['frames_from_compute']}")
print(f"Frames from accumulator: {agg['frames_from_acc']}")
print(f"Total frames: {agg['total_frames']}")

print("\n" + "="*80)
print("PER-VPU LINK UTILIZATION")
print("="*80)
sim_time_s = params["sim_time_us"] * 1e-6
link_capacity_bits = params["link_in_Mbps"] * 1e6 * sim_time_s
for i in range(params["num_vpus"]):
    in_bytes = results["links"]["vpu_ingress"][i]["bytes"]
    out_bytes = results["links"]["vpu_egress"][i]["bytes"]
    in_util = (in_bytes * 8) / link_capacity_bits * 100
    out_util = (out_bytes * 8) / link_capacity_bits * 100
    print(f"VPU {i:2d}: "
          f"INGRESS={in_util:.2f}% ({in_bytes/1e6:.3f} MB)  "
          f"EGRESS={out_util:.2f}% ({out_bytes/1e6:.3f} MB)")

print("\n" + "="*80)
print("ACCUMULATOR LINK UTILIZATION")
print("="*80)
acc_in_bytes = results["links"]["acc_ingress_bytes"]
acc_out_bytes = results["links"]["acc_egress_bytes"]
acc_in_capacity_bits = params["acc_link_in_Mbps"] * 1e6 * sim_time_s
acc_out_capacity_bits = params["acc_link_out_Mbps"] * 1e6 * sim_time_s
acc_in_util = (acc_in_bytes * 8) / acc_in_capacity_bits * 100
acc_out_util = (acc_out_bytes * 8) / acc_out_capacity_bits * 100
acc_stat = results["accumulator"]
print(f"ACC VPU {acc_stat['id']}: "
      f"INGRESS={acc_in_util:.2f}% ({acc_in_bytes/1e6:.3f} MB)  "
      f"EGRESS={acc_out_util:.2f}% ({acc_out_bytes/1e6:.3f} MB)")

print("\n" + "="*80)
print("AGGREGATE LINK DATA TRANSFER")
print("="*80)
total_ingress_compute = sum(link["bytes"] for link in results["links"]["vpu_ingress"])
total_egress_compute = sum(link["bytes"] for link in results["links"]["vpu_egress"])
total_ingress_acc = results["links"]["acc_ingress_bytes"]
total_egress_acc = results["links"]["acc_egress_bytes"]
total_all_links = total_ingress_compute + total_egress_compute + total_ingress_acc + total_egress_acc

print(f"Compute VPU ingress (FPGA→VPU): {total_ingress_compute/1e6:.3f} MB")
print(f"Compute VPU egress (VPU→FPGA):  {total_egress_compute/1e6:.3f} MB")
print(f"Accumulator ingress (FPGA→ACC): {total_ingress_acc/1e6:.3f} MB")
print(f"Accumulator egress (ACC→FPGA):  {total_egress_acc/1e6:.3f} MB")
print(f"Total all links:                 {total_all_links/1e6:.3f} MB")

print("\n" + "="*80)
print("PER-VPU DETAILED STATS (first 4)")
print("="*80)
for vpu_stat in results["per_vpu"][:4]:
    print(f"VPU {vpu_stat['id']:2d}: "
          f"frames={vpu_stat['frames_sent']} "
          f"dropped={vpu_stat['bytes_dropped']/1e3:.1f}KB "
          f"stall={vpu_stat['stall_fraction']*100:.1f}% "
          f"compute_util={vpu_stat['compute_utilization']*100:.1f}% "
          f"avg_fifo={vpu_stat['avg_fifo_bytes']/1e3:.1f}KB "
          f"avg_spm={vpu_stat['avg_spm_bytes']/1e3:.1f}KB")

print("\n" + "="*80)
print("ACCUMULATOR DETAILED STATS")
print("="*80)
print(f"Frames received: {acc_stat['frames_received']}")
print(f"Frames sent: {acc_stat['frames_sent']}")
print(f"Compute utilization: {acc_stat['compute_utilization']*100:.2f}%")
print(f"Pending transactions at end: {acc_stat['pending_count']}")
print(f"Queue depth at end: {acc_stat['queue_depth']}")

# ------------------------------ Save Results ---------------------------------
out_dir = tempfile.mkdtemp(prefix="vpu_sim_cycle_")
json_file = os.path.join(out_dir, "results.json")
with open(json_file, "w") as f:
    json.dump(results, f, indent=2)

csv_file = os.path.join(os.path.expanduser("~"), "Downloads", "vpu_cycle_sim.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results["per_vpu"][0].keys())
    writer.writeheader()
    writer.writerows(results["per_vpu"])

print(f"\nResults saved to:")
print(f"  JSON: {json_file}")
print(f"  CSV: {csv_file}")
