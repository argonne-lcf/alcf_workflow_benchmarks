# Task Launching Benchmark 

This benchmark measures the performance of a workflow framework at launching large ensembles of tasks on large-scale HPC systems. It is representative of many high-througput workflows, wherein the goal is to complete a large ensemble of calculations as efficiently as possible.

The tasks are implemented in both C++ and Python and are designed to simply sleep for a number of seconds determined by a command line argument. The tasks avoid performing any computation intentionally to avoid run-time variation across hardware. 
The types of tasks are organized under three motifs:

1. **Serial**: All tasks are serial applications. In this case, the workflow framework assigns one task to each GPU (or CPU core if GPUs are not available) available during a job. The maximum number of concurrent tasks is given by the number of GPUs (or CPUs) in the allocation.
2. **Multi-node**: All tasks are multi-node MPI applications. In this case, the workflow framework assigns one task per set of nodes required by MPI application. For the purposes of the benchmark, 2 nodes is sufficient. The maximum number of concurrent tasks is given by the number of nodes in the allocation divided by 2. 
3. **Heterogeneous**: The tasks are a mix of serial, single-node MPI and multi-node MPI applications. In this case, the workflow framework is responsible for packing the allocation with the mix of tasks as efficiently as possible to increase concurrency of the tasks. The ensemble of heterogeneous of tasks is created by sampling from pre-defined distributions with fixed seeds for reproducibility.

The benchmark measures the end-to-end run-time of the workflow framework at running the ensembles of tasks. The number of tasks is fixed at 100 and 1,000 for the small and large configurations, respectively. Gatherig data across multiple node counts therefore allows to measure the strong scaling efficiency of the various workflow tools. 


## Configurable Parameters

The benchmark has three main configuration options:

- **Ensemble size**: Two ensemble sizes are provided for the benchmarkm, namely `small` and `large` containing 100 and 1,000 tasks, respectively. 
- **Task motif**: The types of tasks to run in the benchmark are defined by the three ensemble motifs described above: `serial`, `multi-node`, and `heterogeneous`. 
- **Task duration**: The task duration in seconds of the tasks can be specified to any desired value. Note for for the `heterogeneous` ensemble, both mean and standard deviation are required.
- **Programming language**: The tasks can either be C++ or Python programs.

## Implementations

The same benchmark is implemented on top of several workflow frameworks and a
plain MPI baseline (where possible) for reference.

| Implementation | Notes |
|----------------|-------|
| [`mpi`](./mpi) | Baseline; considered peak achievable performance for other workflow tools. This implementation supports only the `serial` ensemble motif. |
| [`ensemble_launcher`](./ensemble_launcher) | ... |


## Results

### Runs on ALCF Aurora

