import math, csv, os, json, tempfile, statistics, matplotlib.pyplot as plt

# ============================================================================
# VPU MATRIX MULTIPLICATION SIMULATOR
# ============================================================================
# This simulator models a multi-VPU (Vector Processing Unit) system that
# performs matrix multiplication operations (C = A * B). It simulates:
# - Data transfer over QSPI links from FPGA to VPUs
# - FIFO buffering at each VPU
# - Scratchpad Memory (SPM) storage
# - Matrix multiplication computation
# - Results being sent back
# ============================================================================

# ---------------- Simulation Parameters -----------------
params = {
    "num_vpus": 16,              # Number of VPUs in the system
    "matrix_dim": 16,            # Dimension of square matrices (16x16)
    "elem_bytes": 4,             # Bytes per matrix element (int32 = 4 bytes)
    "vpu_spm_kb": 512,           # Scratchpad memory size per VPU in KB
    "fifo_bytes": 64 * 1024,     # FIFO buffer size in bytes (64 KB)
    "link_Mbps": 100.0,          # Data rate per QSPI link in Megabits/sec
    "tick_s": 1e-6,              # Simulation time step (1 microsecond)
    "sim_time_s": 0.5,           # Total simulation duration (0.5 seconds)
    "clock_MHz": 200.0,          # VPU clock frequency in MHz
    "macs_per_cycle": 1.0,       # Multiply-accumulate ops per clock cycle - more means more parallelism within the VPU
    "unconditional_push": True   # FPGA always pushes data (worst-case scenario)
}

# ---------------- Derived Parameters -----------------
# Calculate how big one matrix is in memory
N = params["matrix_dim"]                      # 16
elem_bytes = params["elem_bytes"]             # 4 bytes
size_matrix_bytes = N * N * elem_bytes        # 16*16*4 = 1024 bytes per matrix
size_matrix_bits = size_matrix_bytes * 8.0    # 8192 bits per matrix
params["size_matrix_bytes"] = size_matrix_bytes
params["size_matrix_bits"] = size_matrix_bits

# Calculate link bandwidth per simulation tick
link_bps = params["link_Mbps"] * 1e6          # Convert Mbps to bits/sec
link_bytes_per_tick = (link_bps / 8.0) * params["tick_s"]  # Bytes transferable per tick

# Calculate how long matrix multiplication takes
# For N×N matrices: N^3 multiply-accumulate operations needed
compute_cycles_per_matrix = (N**3) / params["macs_per_cycle"]  # 16^3 = 4096 cycles
compute_time_s = compute_cycles_per_matrix / (params["clock_MHz"] * 1e6)  # Convert to seconds
params["compute_time_s"] = compute_time_s

# Print all parameters for verification
print("Simulation parameters:")
for k,v in params.items():
    print(f"  {k}: {v}")
print(f"Derived: matrix_bytes={size_matrix_bytes}, link_bytes_per_tick={link_bytes_per_tick:.4f}, compute_time_s={compute_time_s:.6e}s")

# ============================================================================
# VPU STATE CLASS
# ============================================================================
# Represents the state of a single VPU including its buffers, memory, and
# computation status
# ============================================================================
class VPUState:
    def __init__(self, id, params):
        self.id = id  # Unique identifier for this VPU
        
        # FIFO buffer configuration (receives data from FPGA)
        self.fifo_capacity = params["fifo_bytes"]  # Max FIFO size (64 KB)
        self.fifo_bytes = 0.0                      # Current FIFO occupancy
        
        # Scratchpad Memory (SPM) configuration
        self.spm_capacity = params["vpu_spm_kb"] * 1024  # Max SPM size in bytes (512 KB)
        # Preload matrix A into SPM (A stays constant for all operations)
        # IMPORTANT: Matrix A is RESIDENT and never consumed
        self.spm_bytes = size_matrix_bytes  # Start with one matrix (A) loaded
        
        # Computation state tracking
        self.compute_busy = False      # Is VPU currently computing?
        self.compute_end_time = 0.0    # When will current computation finish?
        
        # Output buffer (holds result matrix C before sending back)
        self.out_bytes = 0.0           # Bytes waiting to be sent out
        
        # Statistics tracking
        self.total_in_bytes = 0.0      # Total bytes received from FPGA
        self.total_out_bytes = 0.0     # Total bytes sent back to FPGA
        self.dropped_in_bytes = 0.0    # Bytes dropped due to full FIFO
        self.ticks_stalled = 0         # Time steps when VPU was idle (no work)
        self.ticks_active = 0          # Time steps when VPU was active
        
        # Time-series data for analysis
        self.fifo_occupancy_samples = []  # Track FIFO usage over time
        self.spm_occupancy_samples = []   # Track SPM usage over time
        self.link_bytes_transferred = 0.0 # Total bytes moved on this VPU's link

    def sample(self):
        """Record current buffer occupancy for statistics"""
        self.fifo_occupancy_samples.append(self.fifo_bytes)
        self.spm_occupancy_samples.append(self.spm_bytes)

# ============================================================================
# MAIN SIMULATION FUNCTION
# ============================================================================
def run_sim(params):
    """
    Simulates the VPU system for the specified duration.
    Models data flow: FPGA → FIFO → SPM → Compute → Output → FPGA
    """
    # Initialize simulation
    num_vpus = params["num_vpus"]                           # 16 VPUs
    tick = params["tick_s"]                                 # Time step size
    total_ticks = int(math.ceil(params["sim_time_s"] / tick))  # Number of steps to simulate
    vpus = [VPUState(i, params) for i in range(num_vpus)]  # Create all VPUs
    t = 0.0                                                 # Current simulation time
    aggregate_bits_transferred = 0.0                        # Total system throughput

    # ========================================================================
    # MAIN SIMULATION LOOP - Execute one tick at a time
    # ========================================================================
    for step in range(total_ticks):
            # ====================================================================
    # PHASE 1: LINK DATA TRANSFER (per VPU)
    # --------------------------------------------------------------------
    # Each VPU has its own dedicated QSPI link (100 Mbps) to the FPGA.
    # The FPGA acts as the master I/O controller, streaming data (matrix B)
    # to the VPUs, which already hold matrix A locally in SPM.
    #
    # IMPORTANT:
    # - The FPGA is the *data source* on this link (master → slave).
    # - Each VPU has a local FIFO buffer that temporarily holds incoming data.
    # - If the FPGA pushes faster than the FIFO can accept, packets are dropped
    #   *at the FPGA ingress*, not by the VPU.
    # ====================================================================
        for v in vpus:
            # Bandwidth available per simulation tick for this link (bytes/tick)
            per_link_budget = link_bytes_per_tick  

            # ----------------------------------------------------------------
            # INCOMING DATA PATH: FPGA → VPU FIFO
            # ----------------------------------------------------------------
            # The FPGA tries to push a portion of matrix B to each VPU.
            # Whether it *respects* FIFO capacity depends on the mode:
            if params["unconditional_push"]:
                # Worst-case mode: FPGA blindly transmits at full QSPI rate
                # even if the VPU FIFO is already full.
                # This models a bandwidth saturation or link overrun scenario.
                incoming_attempt = per_link_budget
            else:
                # Flow-controlled mode: FPGA sends only what fits in the FIFO.
                incoming_attempt = min(per_link_budget, v.fifo_capacity - v.fifo_bytes)

            # Calculate available FIFO space on the VPU side
            free_fifo = v.fifo_capacity - v.fifo_bytes

            # The FIFO can only accept up to its free space.
            # Anything beyond this capacity is immediately *dropped* by the FPGA
            # (it has nowhere to buffer excess data).
            incoming_accepted = min(incoming_attempt, free_fifo)
            dropped = max(0.0, incoming_attempt - incoming_accepted)

            # Update FIFO occupancy and stats
            v.fifo_bytes += incoming_accepted
            v.total_in_bytes += incoming_accepted
            v.dropped_in_bytes += dropped

            # Note: 'dropped' here reflects *FPGA-side loss*.
            # The VPU never "sees" these lost packets; they never leave the FPGA.
            used_link_bytes = incoming_accepted

            # ----------------------------------------------------------------
            # OUTGOING DATA PATH: VPU → FPGA
            # ----------------------------------------------------------------
            # The VPU streams its result matrix C back to the FPGA,
            # using whatever bandwidth remains in the QSPI link after
            # the inbound (FPGA→VPU) traffic.
            remaining = max(0.0, per_link_budget - used_link_bytes)

            # Send as much of the result data as fits in the remaining budget.
            outgoing_sent = min(remaining, v.out_bytes)
            v.out_bytes -= outgoing_sent
            v.total_out_bytes += outgoing_sent
            used_link_bytes += outgoing_sent

            # Track total traffic for this simulation tick
            aggregate_bits_transferred += used_link_bytes * 8.0
            v.link_bytes_transferred += used_link_bytes


        # ====================================================================
        # PHASE 2: FIFO → SPM TRANSFER
        # Move complete matrices from FIFO to SPM when space available
        # ====================================================================
        for v in vpus:
            spm_free = v.spm_capacity - v.spm_bytes
            
            # How many complete matrices are in FIFO?
            avail_in_fifo_full = int(v.fifo_bytes // size_matrix_bytes)
            
            # How many complete matrices fit in SPM?
            fit_in_spm_full = int(spm_free // size_matrix_bytes)
            
            # Move as many as possible
            move_count = min(avail_in_fifo_full, fit_in_spm_full)
            if move_count > 0:
                move_bytes = move_count * size_matrix_bytes
                v.fifo_bytes -= move_bytes  # Remove from FIFO
                v.spm_bytes += move_bytes   # Add to SPM

        # ====================================================================
        # PHASE 3: START COMPUTATION
        # Begin matrix multiplication if VPU is idle and data is ready
        # ====================================================================
        for v in vpus:
            # Can start compute if: not busy AND at least one B matrix in SPM
            # Matrix A is resident (always present), so we need A + B in SPM
            # spm_bytes must be >= 2 * size_matrix_bytes (one A + at least one B)
            # Check for A + B, not just one matrix
            if not v.compute_busy and (v.spm_bytes >= 2 * size_matrix_bytes):
                # Consume one B matrix from SPM (A remains resident)
                v.spm_bytes -= size_matrix_bytes
                
                # Start computation
                v.compute_busy = True
                v.compute_end_time = t + params["compute_time_s"]
            
            # --- Track stall vs active time ---
            # Stalled = idle with no work available
            # Need to check if B matrix is available (SPM has room for A + B)
            has_B_available = (v.spm_bytes >= 2 * size_matrix_bytes)
            B_in_fifo = (v.fifo_bytes >= size_matrix_bytes)
            
            if (not v.compute_busy) and (not has_B_available) and (not B_in_fifo):
                v.ticks_stalled += 1
            else:
                v.ticks_active += 1

        # ====================================================================
        # PHASE 4: FINISH COMPUTATION
        # Check if any VPU has completed its matrix multiplication
        # ====================================================================
        for v in vpus:
            if v.compute_busy and t >= v.compute_end_time:
                # Computation complete!
                v.compute_busy = False
                v.compute_end_time = 0.0
                
                # Produce result matrix C and queue for output
                v.out_bytes += size_matrix_bytes

        # ====================================================================
        # PHASE 5: SAMPLE STATISTICS
        # Record current state for time-series analysis
        # ====================================================================
        for v in vpus:
            v.sample()

        # Advance simulation time
        t += tick

    # ========================================================================
    # POST-SIMULATION: COLLECT AND ANALYZE RESULTS
    # ========================================================================
    results = {"params": params, "per_vpu": [], "aggregate": {}}
    
    # Aggregate statistics (whole system)
    results["aggregate"]["total_bits_transferred"] = aggregate_bits_transferred
    
    # Calculate overall link utilization
    # Total capacity = 16 VPUs × 100 Mbps = 1600 Mbps
    total_link_bps = params["link_Mbps"] * 1e6 * params["num_vpus"]
    results["aggregate"]["aggregate_utilization"] = aggregate_bits_transferred / (params["sim_time_s"] * total_link_bps)

    # Per-VPU statistics
    for v in vpus:
        # Calculate averages from time-series data
        avg_fifo = statistics.mean(v.fifo_occupancy_samples) if v.fifo_occupancy_samples else 0.0
        avg_spm = statistics.mean(v.spm_occupancy_samples) if v.spm_occupancy_samples else 0.0
        
        # Calculate what fraction of time VPU was stalled
        stall_fraction = v.ticks_stalled / (v.ticks_stalled + v.ticks_active) if (v.ticks_stalled + v.ticks_active) > 0 else 0.0
        
        # Package per-VPU results
        per = {
            "id": v.id,
            "total_in_bytes": v.total_in_bytes,
            "total_out_bytes": v.total_out_bytes,
            "dropped_in_bytes": v.dropped_in_bytes,
            "avg_fifo_bytes": avg_fifo,
            "avg_spm_bytes": avg_spm,
            "stall_fraction": stall_fraction,
            "link_util_fraction": (v.link_bytes_transferred * 8.0) / (params["sim_time_s"] * params["link_Mbps"] * 1e6)
        }
        results["per_vpu"].append(per)
    
    return results

# ============================================================================
# EXECUTE SIMULATION
# ============================================================================
results = run_sim(params)

# ============================================================================
# SAVE RESULTS TO FILES
# ============================================================================
# Save JSON file with complete results
out_dir = tempfile.mkdtemp(prefix="vpu_sim_")  # Create temp directory
out_json = os.path.join(out_dir, "sim_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)  # Write formatted JSON
print("Results saved to", out_json)

# ============================================================================
# PRINT SUMMARY TO CONSOLE
# ============================================================================
print("\nAggregate summary:")
agg = results["aggregate"]
print(f"  Total bits transferred (approx): {agg['total_bits_transferred']:.0f} bits")
print(f"  Aggregate utilization (fraction of total {params['num_vpus']*params['link_Mbps']} Mbps): {agg['aggregate_utilization']:.3f}")

print("\nPer-VPU summary (first 6):")
for p in results["per_vpu"][:6]:
    print(f"  VPU{p['id']}: in={p['total_in_bytes']:.0f}B out={p['total_out_bytes']:.0f}B dropped={p['dropped_in_bytes']:.0f}B avg_fifo={p['avg_fifo_bytes']:.1f}B avg_spm={p['avg_spm_bytes']:.1f}B stall_frac={p['stall_fraction']:.3f} link_util={p['link_util_fraction']:.3f}")

# ============================================================================
# EXPORT PER-VPU STATISTICS TO CSV
# ============================================================================
downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
csv_file = os.path.join(downloads_dir, "per_vpu_stats.csv")
with open(csv_file, "w", newline="") as f:
    w = csv.writer(f)
    # Write header row
    w.writerow(["id","total_in_bytes","total_out_bytes","dropped_in_bytes","avg_fifo_bytes","avg_spm_bytes","stall_fraction","link_util_fraction"])
    # Write data rows
    for p in results["per_vpu"]:
        w.writerow([p["id"], p["total_in_bytes"], p["total_out_bytes"], p["dropped_in_bytes"], p["avg_fifo_bytes"], p["avg_spm_bytes"], p["stall_fraction"], p["link_util_fraction"]])

print("CSV saved to", csv_file)
out_dir, out_json, csv_file