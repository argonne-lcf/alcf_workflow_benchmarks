import os
import math
import argparse
import numpy as np
from time import sleep
from adios2 import Stream, Adios

import mpi4py.rc
mpi4py.rc.threads = True
mpi4py.rc.thread_level = 'multiple'
from mpi4py import MPI


# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--engine", type=str, choices=["bp5", "sst"], required=True)
parser.add_argument("--sst_mode", type=str, choices=["sync", "async"], default="sync")
parser.add_argument("--data_plane", type=str, choices=["WAN", "MPI", "UCX", "RDMA", "fabric"], default="WAN")
parser.add_argument("--io_mode", type=str, choices=["posix", "daos"], default="posix")
args = parser.parse_args()

# MPI
MPI.Init_thread(MPI.THREAD_MULTIPLE)
global_comm = MPI.COMM_WORLD
global_rank = global_comm.Get_rank()

color = 1234
comm = global_comm.Split(color, global_rank)
rank = comm.Get_rank()
size = comm.Get_size()
local_rank = int(os.getenv("PALS_LOCAL_RANKID", "0"))
local_size = int(os.getenv("PALS_LOCAL_SIZE", "1"))
host_name = MPI.Get_processor_name()

if rank == 0:
    print(f"[ML] Running with {size} MPI ranks (engine={args.engine}"
          f"{f' sst_mode={args.sst_mode} data_plane={args.data_plane}' if args.engine == 'sst' else ''}"
          f" io_mode={args.io_mode})", flush=True)

# Paths
path_prefix = "/tmp/datascience/balin/" if args.io_mode == "daos" else "./"
solution_path = path_prefix + "solution.bp"
ready_path = path_prefix + "solution.ready"
check_path = path_prefix + "check-run.done"

# ADIOS setup
adios = Adios(comm)
streamIO = adios.declare_io("solutionStream")
if args.engine == "bp5":
    streamIO.set_engine("BP5")
    open_path = solution_path
else:
    streamIO.set_engine("SST")
    params = {
        "DataTransport": args.data_plane,
        "OpenTimeoutSecs": "600",
    }
    streamIO.set_parameters(params)
    open_path = "solutionStream"

# For BP5, wait for the producer's sentinel file signaling the BP5 file is safe to open
if args.engine == "bp5":
    if rank == 0:
        print(f"[ML] Waiting for sentinel {ready_path}...", flush=True)
        while not os.path.exists(ready_path):
            sleep(1)
        print(f"[ML] Found sentinel, opening {solution_path}", flush=True)
    comm.Barrier()

# Open stream and read
workflow_steps = 20
get_time = 0.0
transfer_time = 0.0
completed_steps = 0
bytes_per_rank = None
N = None
start = None

stream = None
try:
    if rank == 0:
        print("[ML] Opening stream...", flush=True)
    stream = Stream(streamIO, open_path, "r", comm)

    sleep_time = 2.0  # seconds; > producer sleep so producer stays ahead
    for step in range(workflow_steps):
        sleep(sleep_time)
        status = stream.begin_step()

        # Discover buffer size on the first step
        if step == 0:
            bpr_arr = stream.read("bytes_per_rank")
            bytes_per_rank = int(bpr_arr.item() if hasattr(bpr_arr, "item") else bpr_arr)
            assert bytes_per_rank % 8 == 0, f"bytes_per_rank {bytes_per_rank} not multiple of 8"
            N = bytes_per_rank // 8
            start = rank * N
            if rank == 0:
                print(f"[ML] Producer sends {bytes_per_rank} bytes ({bytes_per_rank/1e9:.4f} GB) per rank", flush=True)

        # Bracket the read with barriers so tic_wrap..toc_wrap is the wall-clock
        # time for ALL ranks to finish reading (defines the transfer window for
        # the aggregate wall-clock BW). tic..toc is still just this rank's read.
        comm.Barrier()
        tic_wrap = MPI.Wtime()
        tic = MPI.Wtime()
        # for SST, stream.read() gets data now, Mode.Sync is default
        # see
        #   - https://github.com/ornladios/ADIOS2/blob/67f771b7a2f88ce59b6808cc4356159d86255f1d/python/adios2/stream.py#L331
        #   - https://github.com/ornladios/ADIOS2/blob/67f771b7a2f88ce59b6808cc4356159d86255f1d/python/adios2/engine.py#L123)
        train_data = stream.read("U", [start], [N])
        toc = MPI.Wtime()
        comm.Barrier()
        toc_wrap = MPI.Wtime()

        if step > 0:
            get_time += toc - tic
            transfer_time += toc_wrap - tic_wrap
        completed_steps = step + 1

        stream.end_step()
        if rank == 0:
            print(f"[ML] Iter {step}: {toc_wrap - tic_wrap:.6f} s", flush=True)

except Exception as e:
    print(f"[ML] Error on rank {rank}: {e}", flush=True)
finally:
    # Close the read stream, tolerating any teardown errors
    if stream is not None:
        try:
            stream.close()
        except Exception as e:
            print(f"[ML] Warning: stream.close() failed on rank {rank}: {e}", flush=True)

    # Always signal producer to quit, even if the read loop threw
    try:
        if rank == 0:
            with open(check_path, "w") as f:
                f.write("done\n")
            print(f"[ML] Wrote check-run sentinel {check_path}", flush=True)
        comm.Barrier()
    except Exception as e:
        print(f"[ML] Warning: failed to write check-run sentinel on rank {rank}: {e}", flush=True)

    # Metrics (only meaningful if we timed at least one non-warmup step)
    if completed_steps > 1 and bytes_per_rank is not None:
        get_time /= (completed_steps - 1)
        transfer_time /= (completed_steps - 1)
        avg_get_time = comm.allreduce(get_time, op=MPI.SUM) / size
        max_get_time = comm.allreduce(get_time, op=MPI.MAX)
        min_get_time = comm.allreduce(get_time, op=MPI.MIN)

        # Sum of per-rank rates
        local_rank_bw = (bytes_per_rank / 1e9) / get_time
        sum_of_rates = comm.allreduce(local_rank_bw, op=MPI.SUM)

        if rank == 0:
            gb_per_rank = bytes_per_rank / 1e9
            gb_per_iter = gb_per_rank * size
            print("\n=== Consumer Performance Summary ===")
            print(f"Consumer ranks: {size}")
            print(f"Data per rank per step: {gb_per_rank:.6f} GB")
            print(f"Total data per step: {gb_per_iter:.6f} GB")
            print(f"Steps timed: {completed_steps - 1} (step 0 = warmup)")
            print(f"Avg per-rank get time: {avg_get_time:.6f} s")
            print(f"Min per-rank get time: {min_get_time:.6f} s (fastest rank)")
            print(f"Max per-rank get time: {max_get_time:.6f} s (slowest rank)")
            print(f"Wall-clock transfer time (barrier-to-barrier): {transfer_time:.6f} s")
            print(f"Avg per-rank bandwidth (from get time): {gb_per_rank / avg_get_time:.6f} GB/s")
            print(f"Peak per-rank bandwidth (from min get time): {gb_per_rank / min_get_time:.6f} GB/s")
            print(f"Aggregate bandwidth (sum of per-rank rates): {sum_of_rates:.6f} GB/s")
            print(f"Aggregate bandwidth (from wall-clock barriers): {gb_per_iter / transfer_time:.6f} GB/s")
