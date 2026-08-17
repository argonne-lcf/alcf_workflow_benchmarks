#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N adios2_workflow_stream
#PBS -l select=2
#PBS -l place=scatter:group=tier1
#PBS -l walltime=0:30:00
#PBS -l filesystems=home:flare
#PBS -A datascience
#PBS -q debug
#PBS -k doe
#PBS -j oe

cd $PBS_O_WORKDIR
export TZ='/usr/share/zoneinfo/US/Central'

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`
NODES=$(cat $PBS_NODEFILE | wc -l)
if [ $((NODES % 2)) -ne 0 ]; then
    echo "Error: Need even number of nodes, got $NODES"
    exit 1
fi
COMPONENT_NODES=$(( NODES / 2))

# Log directory
JOBID=$(cut -d. -f1 <<< "$PBS_JOBID")
LOG_DIR=logs_$JOBID
mkdir -p $LOG_DIR
echo Logs will be written to $LOG_DIR

# Load modules
module load frameworks
module load adios2/2.11.0-cpu
module list

# Build code
./configAurora.sh

# env variables

# ADIOS2 env vars
# auto-locate the Python site-packages under the loaded adios2 module
ADIOS2_ROOT=$(dirname $(dirname $(command -v adios2-config)))
ADIOS2_SITE_PACKAGES=$(ls -d ${ADIOS2_ROOT}/lib/python*/site-packages 2>/dev/null | head -n 1)
if [ -z "$ADIOS2_SITE_PACKAGES" ]; then
    echo "ERROR: could not find adios2 site-packages under $ADIOS2_ROOT"
    exit 1
fi
export PYTHONPATH=$PYTHONPATH:$ADIOS2_SITE_PACKAGES
#export FABRIC_PROVIDER=cxi
#export FABRIC_IFACE=cxi
export SstVerbose=0
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Clean up run dir
cleanup_run_dir() {
    echo "Cleaning up old .sst, .bp, and sentinel files"
    rm -rf ./*.sst ./*.bp ./*.ready 2>/dev/null
    return 0
}
cleanup_run_dir

# Define bindings
declare -A CPU_BIND_MAP
CPU_BIND_MAP[1]="list:1"
CPU_BIND_MAP[2]="list:1:8"
CPU_BIND_MAP[8]="list:1:8:16:24:53:60:68:76"
CPU_BIND_MAP[12]="list:1:8:16:24:32:40:53:60:68:76:84:92"

# Run
ENGINE=bp5
SST_MODE=sync
DATA_PLANE=RDMA
IO_MODE=posix
for RANKS_PER_NODE in 1 2 8 12
do
  RANKS=$(( COMPONENT_NODES * RANKS_PER_NODE ))
  CPU_BIND=${CPU_BIND_MAP[$RANKS_PER_NODE]}
  for BYTES in 262144 1048576 4194304 16777216 67108864 268435456 1073741824 4294967296
  do
    # MPMD launch
    mpiexec -n $RANKS --ppn $RANKS_PER_NODE \
      --cpu-bind $CPU_BIND numactl -m 2-3 \
      ./producer $BYTES $ENGINE $SST_MODE $DATA_PLANE $IO_MODE \
      : -n $RANKS --ppn $RANKS_PER_NODE \
      python ./consumer.py --engine $ENGINE --sst_mode $SST_MODE --data_plane $DATA_PLANE --io_mode $IO_MODE \
      2>&1 | tee $LOG_DIR/adios_${ENGINE}_${SST_MODE}_${DATA_PLANE}_${IO_MODE}_n${NODES}_N${RANKS_PER_NODE}_buff${BYTES}.log
  
  cleanup_run_dir
  done
done
