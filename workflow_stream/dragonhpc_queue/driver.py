import sys
import os
import numpy as np
import argparse
from time import sleep, perf_counter

import dragon
import multiprocessing as mp
from dragon.native.process import Process, ProcessTemplate, MSG_PIPE, MSG_DEVNULL
from dragon.native.process_group import ProcessGroup
from dragon.infrastructure.facts import PMIBackend
from dragon.native.machine import System, Node
from dragon.infrastructure.policy import Policy
from dragon.infrastructure.connection import Connection

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

PRODUCER_ITERS = 20
CONSUMER_ITERS = 20
PRODUCER_SLEEP_S = 2.0
CONSUMER_SLEEP_S = 0.5

# CPU bindings, keyed by procs-per-node (ppn).
# Aurora: 2 sockets x 52 cores; avoid reserved cores (0, 52, 104, 156).
# No DDict here (queues are in-process), so bindings don't need to carve out
# DB cores.
#
# Colocated: producer + consumer share each node -> producer on socket 0,
# consumer on socket 1, one core per rank.
COLOCATED_BINDINGS = {
    1:  {"producer": [[1]],
         "consumer": [[53]]},
    2:  {"producer": [[1], [8]],
         "consumer": [[53], [60]]},
    6:  {"producer": [[1], [8], [16], [24], [32], [40]],
         "consumer": [[53], [60], [68], [76], [84], [92]]},
}

# Clustered: producer and consumer run on separate nodes, so each can use the
# full per-node core budget spread across both sockets.
CLUSTERED_BINDINGS = {
    1:  [[1]],
    2:  [[1], [8]],
    6:  [[1], [8], [16], [24], [32], [40]],
    8:  [[1], [8], [16], [24], [53], [60], [68], [76]],
    12: [[1], [8], [16], [24], [32], [40], [53], [60], [68], [76], [84], [92]],
}

## Read output from ProcessGroup
def read_output(stdout_conn: Connection) -> str:
    """Read stdout from the Dragon connection.

    :param stdout_conn: Dragon connection to rank 0's stdout
    :type stdout_conn: Connection
    :return: string with the output from stdout
    :rtype: str
    """
    output = ""
    try:
        while True:
            tmp = stdout_conn.recv()
            output += tmp
    except EOFError:
        pass
    finally:
        stdout_conn.close()
    return output

# Data producer
def producer(q: mp.queues.Queue, bytes_per_rank: int) -> None:
    """Generate a buffer and put it on the queue every iteration.

    :param q: Queue to add data to
    :type q: mp.queues.Queue
    :param bytes_per_rank: Buffer size per iteration in bytes (multiple of 8)
    :type bytes_per_rank: int
    """
    if not MPI.Is_initialized():
        MPI.Init()
    n = bytes_per_rank // 8
    for _ in range(PRODUCER_ITERS):
        sleep(PRODUCER_SLEEP_S)
        data = np.random.rand(n).astype(np.float64)
        MPI.COMM_WORLD.Barrier()
        q.put(data)
    MPI.Finalize()

## Data consumer
def consumer(q: mp.queues.Queue, bytes_per_rank: int) -> None:
    """Retrieve data from the queue every iteration.

    :param q: Queue to retrieve from
    :type q: mp.queues.Queue
    :param bytes_per_rank: Expected buffer size per iteration in bytes (multiple of 8)
    :type bytes_per_rank: int
    """
    if not MPI.Is_initialized():
        MPI.Init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n = bytes_per_rank // 8
    gb_per_rank = bytes_per_rank / 1e9
    gb_per_iter = gb_per_rank * size

    get_time = 0.0
    transfer_time = 0.0
    completed_steps = 0

    for step in range(CONSUMER_ITERS):
        # Bracket the Get with barriers so tic_wrap..toc_wrap is the wall-clock
        # time for ALL ranks to finish reading. tic..toc is just this rank's Get.
        while True:
            if not q.empty():
                break
            sleep(CONSUMER_SLEEP_S)
        comm.Barrier()
        tic_wrap = perf_counter()
        tic = perf_counter()
        data = q.get()
        toc = perf_counter()
        comm.Barrier()
        toc_wrap = perf_counter()

        if len(data) != n:
            sys.exit(1)

        if step > 0:
            get_time += toc - tic
            transfer_time += toc_wrap - tic_wrap
        completed_steps = step + 1

        if rank == 0:
            print(f"[ML] Iter {step}: {toc_wrap - tic_wrap:.6f} s", flush=True)

    # Metrics
    if completed_steps > 1:
        get_time /= (completed_steps - 1)
        transfer_time /= (completed_steps - 1)
        avg_get_time = comm.allreduce(get_time, op=MPI.SUM) / size
        max_get_time = comm.allreduce(get_time, op=MPI.MAX)
        min_get_time = comm.allreduce(get_time, op=MPI.MIN)

        # Sum of per-rank rates
        local_rank_bw = gb_per_rank / get_time
        sum_of_rates = comm.allreduce(local_rank_bw, op=MPI.SUM)

        if rank == 0:
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

    MPI.Finalize()


## Launch ProcessGroup
def launchProcessGroup(args, component_info, queues = None):
    """Launch a function with Dragon ProcessGroup
    """
    global_policy = Policy(distribution=Policy.Distribution.BLOCK)
    grp = ProcessGroup(policy=global_policy, pmi=PMIBackend.PMIX)
    if component_info["name"] == "producer":
        queues = []

    for node_num in range(component_info["node_count"]):
        node_name = Node(component_info["nodelist"][node_num]).hostname
        for proc in range(args.ppn):
            proc_id = node_num * args.ppn + proc
            local_policy = Policy(placement=Policy.Placement.HOST_NAME,
                                  host_name=node_name,
                                  cpu_affinity=component_info["cpu_bind"][proc])
            if component_info["name"] == "producer":
                q = mp.Queue()
                queues.append(q)
            if proc_id == 0:
                grp.add_process(nproc=1,
                            template=ProcessTemplate(target=component_info["function"],
                                                        args=(queues[proc_id], args.bytes_per_rank),
                                                        cwd=os.getcwd(),
                                                        policy=local_policy,
                                                        stdout=MSG_PIPE))
            else:
                grp.add_process(nproc=1,
                            template=ProcessTemplate(target=component_info["function"],
                                                        args=(queues[proc_id], args.bytes_per_rank),
                                                        cwd=os.getcwd(),
                                                        policy=local_policy,
                                                        stdout=MSG_DEVNULL))

    print(f"Starting Process Group for {component_info['name']}", flush=True)
    grp.init()
    grp.start()
    if component_info["name"] == "producer":
        return grp, queues
    else:
        return grp


def generate_cpu_bind_list(base_list, procs_per_node):
    """Create the list of cpu bindings
    """
    diffs = [base_list[i+1] - base_list[i] for i in range(len(base_list)-1)]
    max_procs_per_node = min(diffs) * len(base_list)
    if procs_per_node > max_procs_per_node:
        print(f"The maximum procs per node is {max_procs_per_node} and you selected {procs_per_node}.", flush=True)
        sys.exit(1)

    cpu_bindings = []
    base_list_len = len(base_list)
    for i in range(procs_per_node):
        block = i // base_list_len
        idx = i % base_list_len
        base_val = base_list[idx]
        cpu_bindings.append([base_val + block])

    return cpu_bindings


def get_colocated_binding(ppn, role):
    """Core list for a colocated (producer|consumer) rank at ppn."""
    if ppn not in COLOCATED_BINDINGS:
        raise ValueError(
            f"No colocated CPU binding defined for ppn={ppn}. "
            f"Available: {sorted(COLOCATED_BINDINGS.keys())}. "
            f"Add a new entry to COLOCATED_BINDINGS in driver.py."
        )
    return COLOCATED_BINDINGS[ppn][role]


def get_clustered_binding(ppn):
    """Core list for a clustered producer or consumer rank at ppn."""
    if ppn not in CLUSTERED_BINDINGS:
        raise ValueError(
            f"No clustered CPU binding defined for ppn={ppn}. "
            f"Available: {sorted(CLUSTERED_BINDINGS.keys())}. "
            f"Add a new entry to CLUSTERED_BINDINGS in driver.py."
        )
    return CLUSTERED_BINDINGS[ppn]

## Main function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", type=str, choices=["clustered", "colocated"], default="colocated", help="Deployment type")
    parser.add_argument("--ppn", type=int, default=12, help="Number of processes per node")
    parser.add_argument("--bytes_per_rank", type=int, default=8_000_000,
                        help="Bytes each rank puts on the queue per iteration (multiple of 8)")
    args = parser.parse_args()

    if args.bytes_per_rank % 8 != 0:
        print(f"ERROR: --bytes_per_rank ({args.bytes_per_rank}) must be a multiple of 8", flush=True)
        sys.exit(1)

    # Set dragon start method
    mp.set_start_method("dragon")

    # Get nodes of this allocation (job)
    alloc = System()
    num_tot_nodes = int(alloc.nnodes)
    if num_tot_nodes%2 !=0 and args.deployment == "clustered":
        print("Clustered deployment requires even number of nodes", flush=True)
        sys.exit(1)
    tot_nodelist = alloc.nodes
    print(f"Running on {num_tot_nodes} total nodes, {args.ppn} processes per node, "
          f"and with {args.bytes_per_rank/1e9:.4f} GB buffers per rank", flush=True)

    # Split nodes between the components
    producer_info = {"name": "producer", "function": producer}
    consumer_info = {"name": "consumer", "function": consumer}
    if (args.deployment == "colocated"):
        producer_info["node_count"] = num_tot_nodes
        consumer_info["node_count"] = num_tot_nodes

        producer_info["nodelist"] = tot_nodelist
        consumer_info["nodelist"] = tot_nodelist

        producer_info["cpu_bind"] = get_colocated_binding(args.ppn, "producer")
        consumer_info["cpu_bind"] = get_colocated_binding(args.ppn, "consumer")
    elif (args.deployment == "clustered"):
        num_half_nodes = num_tot_nodes // 2
        producer_info["node_count"] = num_half_nodes
        consumer_info["node_count"] = num_half_nodes

        producer_info["nodelist"] = tot_nodelist[:num_half_nodes]
        consumer_info["nodelist"] = tot_nodelist[num_half_nodes:]

        producer_info["cpu_bind"] = get_clustered_binding(args.ppn)
        consumer_info["cpu_bind"] = get_clustered_binding(args.ppn)
    print(f"Producer running on {[Node(node).hostname for node in producer_info['nodelist']]} nodes",flush=True)
    print(f"Consumer running on {[Node(node).hostname for node in consumer_info['nodelist']]} nodes",flush=True)

    # Launch producer and consumer with ProcessGroup
    prod_grp, queues = launchProcessGroup(args, producer_info)
    cons_grp = launchProcessGroup(args, consumer_info, queues)

    # Get output from consumer
    group_procs = [Process(None, ident=puid) for puid in cons_grp.puids]
    for proc in group_procs:
        if proc.stdout_conn:
            std_out = read_output(proc.stdout_conn)
            print(std_out, flush=True)

    prod_grp.join()
    prod_grp.close()
    cons_grp.join()
    cons_grp.close()
    print(f"Done", flush=True)


## Run main
if __name__ == "__main__":
   main()



