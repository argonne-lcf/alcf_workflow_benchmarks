import os
import logging
import numpy as np
from time import sleep
import argparse

import dragon
from dragon.data.ddict.ddict import DDict

from mpi4py import MPI

from custom_pickler import NumPy1DPickler, StringKeyPickler


# Parse args (do this first so --verify is available to the loop)
parser = argparse.ArgumentParser()
parser.add_argument("--launcher_mode", type=str, required=True,
                    choices=["colocated", "clustered", "mixed"])
parser.add_argument("--ddict_ser", type=str, required=True, help="Serialized DDict")
parser.add_argument("--verify", action="store_true",
                    help="Check received data byte-for-byte against expected pattern (off by default)")
args = parser.parse_args()

# MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
local_rank = int(os.getenv("PALS_LOCAL_RANKID", "0"))
local_size = int(os.getenv("PALS_LOCAL_SIZE", "1"))
host_name = MPI.Get_processor_name()

# Logging: use $CONSUMER_LOG if set (driver injects this), else consumer.out
LOG_FILE = os.getenv("CONSUMER_LOG", "consumer.out")
if rank == 0:
    open(LOG_FILE, "w").close()
comm.Barrier()

log = logging.getLogger("consumer")
log.setLevel(logging.INFO)
log.propagate = False
_handler = logging.FileHandler(LOG_FILE, mode="a")
_handler.setFormatter(
    logging.Formatter(
        fmt=f"%(asctime)s [rank {rank:>3d}/{size}] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
log.addHandler(_handler)

if rank == 0:
    log.info("[ML] Running with %d MPI ranks on head node %s (launcher_mode=%s)",
             size, host_name, args.launcher_mode)

# Attach to the Distributed Dictionary and switch to the C++-compatible pickler
if rank == 0:
    log.info("[ML] Attaching to DDict...")
dd = DDict.attach(args.ddict_ser, timeout=3600)
dd = dd.pickler(key_pickler=StringKeyPickler(), value_pickler=NumPy1DPickler(np.float64))
comm.Barrier()

# For colocated deployments, use the local manager to access local data only
if args.launcher_mode == "colocated":
    dd = dd.manager(dd.local_manager)

# Wait for data to appear (discovers per-rank buffer size from tensor shape)
if rank == 0:
    log.info("[ML] Waiting for data to be available in DDict...")
N = None
bytes_per_rank = None
while True:
    if f"y.{rank}" in dd.keys():
        first_read = dd[f"y.{rank}"]
        N = first_read.shape[0]
        bytes_per_rank = N * 8
        if rank == 0:
            log.info("[ML] Producer sends %d bytes (%.4f GB) per rank",
                     bytes_per_rank, bytes_per_rank / 1e9)
        break
    sleep(1)

# Receive training data
workflow_steps = 20
get_time = 0.0
transfer_time = 0.0
completed_steps = 0

try:
    for step in range(workflow_steps):
        sleep(2)

        # Bracket the read with barriers so tic_wrap..toc_wrap is the wall-clock
        # time for ALL ranks to finish reading (defines the transfer window for
        # the aggregate wall-clock BW). tic..toc is still just this rank's read.
        comm.Barrier()
        tic_wrap = MPI.Wtime()
        tic = MPI.Wtime()
        train_data = dd[f"y.{rank}"]
        toc = MPI.Wtime()
        comm.Barrier()
        toc_wrap = MPI.Wtime()

        if step > 0:
            get_time += toc - tic
            transfer_time += toc_wrap - tic_wrap
        completed_steps = step + 1

        if rank == 0:
            log.info("[ML] Iter %d: %.6f s", step, toc_wrap - tic_wrap)

        if args.verify:
            expected = np.arange(N, dtype=np.float64) + (1.0 / step if step > 0 else 0.0)
            if not np.allclose(train_data, expected):
                log.error("[ML] Data mismatch for step %d and rank %d", step, rank)
                comm.Abort(1)

except Exception:
    log.exception("[ML] Consumer failed with an unhandled exception")
finally:
    # Always signal producer to quit, even if the read loop threw
    try:
        if rank % local_size == 0:
            # C++ side expects a 1-element float64 vector; nonzero = keep going, 0 = quit
            arrMLrun = np.zeros((1,), dtype=np.float64)
            dd["check-run"] = arrMLrun
        comm.Barrier()
        if rank == 0:
            log.info("[ML] Wrote check-run signal")
    except Exception as e:
        log.warning("[ML] Failed to write check-run on rank %d: %s", rank, e)

    # Metrics
    if completed_steps > 1 and bytes_per_rank is not None:
        get_time /= (completed_steps - 1)
        transfer_time /= (completed_steps - 1)
        avg_get_time = comm.allreduce(get_time, op=MPI.SUM) / size
        max_get_time = comm.allreduce(get_time, op=MPI.MAX)
        min_get_time = comm.allreduce(get_time, op=MPI.MIN)

        local_rank_bw = (bytes_per_rank / 1e9) / get_time
        sum_of_rates = comm.allreduce(local_rank_bw, op=MPI.SUM)

        if rank == 0:
            gb_per_rank = bytes_per_rank / 1e9
            gb_per_iter = gb_per_rank * size
            log.info("=== Consumer Performance Summary ===")
            log.info("Consumer ranks: %d", size)
            log.info("Data per rank per step: %.6f GB", gb_per_rank)
            log.info("Total data per step: %.6f GB", gb_per_iter)
            log.info("Steps timed: %d (step 0 = warmup)", completed_steps - 1)
            log.info("Avg per-rank get time: %.6f s", avg_get_time)
            log.info("Min per-rank get time: %.6f s (fastest rank)", min_get_time)
            log.info("Max per-rank get time: %.6f s (slowest rank)", max_get_time)
            log.info("Wall-clock transfer time (barrier-to-barrier): %.6f s", transfer_time)
            log.info("Avg per-rank bandwidth (from get time): %.6f GB/s", gb_per_rank / avg_get_time)
            log.info("Peak per-rank bandwidth (from min get time): %.6f GB/s", gb_per_rank / min_get_time)
            log.info("Aggregate bandwidth (sum of per-rank rates): %.6f GB/s", sum_of_rates)
            log.info("Aggregate bandwidth (from wall-clock barriers): %.6f GB/s", gb_per_iter / transfer_time)
