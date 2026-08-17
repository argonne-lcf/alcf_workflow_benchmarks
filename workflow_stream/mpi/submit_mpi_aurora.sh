#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N mpi_workflow_stream
#PBS -l select=2
#PBS -l place=scatter:group=tier1
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -A datascience
#PBS -q debug-scaling
#PBS -k doe
#PBS -j oe

cd $PBS_O_WORKDIR
export TZ='/usr/share/zoneinfo/US/Central'

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`
NODES=$(cat $PBS_NODEFILE | wc -l)

# Log directory
JOBID=$(cut -d. -f1 <<< "$PBS_JOBID")
LOG_DIR=logs_$JOBID
mkdir -p $LOG_DIR
echo Logs will be written to $LOG_DIR

# Load modules
module load frameworks
module list

# env variables

# Define bindings
declare -A CPU_BIND_MAP
CPU_BIND_MAP[1]="list:1"
CPU_BIND_MAP[2]="list:1:8"
CPU_BIND_MAP[8]="list:1:8:16:24:53:60:68:76"
CPU_BIND_MAP[12]="list:1:8:16:24:32:40:53:60:68:76:84:92"

# Run
for RANKS_PER_NODE in 1 2 8 12
do
  RANKS=$(( NODES * RANKS_PER_NODE ))
  CPU_BIND=${CPU_BIND_MAP[$RANKS_PER_NODE]}
  for BYTES in 262144 1048576 4194304 16777216 67108864 268435456 1073741824 4294967296
  do
    mpiexec -np $RANKS --ppn $RANKS_PER_NODE \
      --cpu-bind $CPU_BIND \
      numactl -m 2-3 \
      ./wkfl_stream_mpi $BYTES 2>&1 | tee $LOG_DIR/mpi_n${NODES}_N${RANKS_PER_NODE}_buff${BYTES}.log
  done
done

