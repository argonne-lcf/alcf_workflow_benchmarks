import os
import numpy as np
from time import sleep
import argparse

from smartredis import Client
from mpi4py import MPI


# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--db_nodes", type=int, default=1)
args = parser.parse_args()

# MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
local_rank = int(os.getenv("PALS_LOCAL_RANKID", "0"))
local_size = int(os.getenv("PALS_LOCAL_SIZE", "1"))
host_name = MPI.Get_processor_name()
if size < 1000:
    print(f"[ML] Hello from rank {rank}/{size} on {host_name} (local {local_rank}/{local_size})", flush=True)
if rank == 0:
    print(f"[ML] Running with {size} MPI ranks (db_nodes={args.db_nodes})", flush=True)

# Initialize SmartRedis client
SSDB = os.getenv("SSDB")
client = Client(address=SSDB, cluster=(args.db_nodes > 1))
comm.Barrier()

# Wait for data to be available in DB (discovers per-rank buffer size from the tensor shape)
if rank == 0:
    print("[ML] Waiting for data to be available in DB...", flush=True)
while True:
    if client.key_exists(f"y.{rank}"):
        first_read = client.get_tensor(f"y.{rank}")
        N = first_read.shape[0]
        bytes_per_rank = N * 8
        if rank == 0:
            print(f"[ML] Producer sends {bytes_per_rank} bytes ({bytes_per_rank/1e9:.4f} GB) per rank", flush=True)
        break
    sleep(1)
comm.Barrier()

# Receive training data
workflow_steps = 20
get_time = 0.0
completed_steps = 0

try:
    for step in range(workflow_steps):
        sleep(2)

        tic = MPI.Wtime()
        train_data = client.get_tensor(f"y.{rank}")
        toc = MPI.Wtime()

        if step > 0:
            get_time += toc - tic
        completed_steps = step + 1

        comm.Barrier()
        if rank == 0:
            print(f"[ML] Iter {step}: {toc - tic:.6f} s", flush=True)

except Exception as e:
    print(f"[ML] Error on rank {rank}: {e}", flush=True)
finally:
    # Always signal producer to quit, even if the read loop threw
    try:
        if rank % local_size == 0:
            arrMLrun = np.int32(np.zeros(1))
            client.put_tensor("check-run", arrMLrun)
        comm.Barrier()
        if rank == 0:
            print("[ML] Wrote check-run signal", flush=True)
    except Exception as e:
        print(f"[ML] Warning: failed to write check-run on rank {rank}: {e}", flush=True)

    # Metrics
    if completed_steps > 1:
        get_time /= (completed_steps - 1)
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
            print(f"Avg per-rank bandwidth (from get time): {gb_per_rank / avg_get_time:.6f} GB/s")
            print(f"Peak per-rank bandwidth (from min get time): {gb_per_rank / min_get_time:.6f} GB/s")
            print(f"Aggregate bandwidth (sum of per-rank rates): {sum_of_rates:.6f} GB/s")
            print(f"Aggregate bandwidth (from max get time): {gb_per_iter / max_get_time:.6f} GB/s")
