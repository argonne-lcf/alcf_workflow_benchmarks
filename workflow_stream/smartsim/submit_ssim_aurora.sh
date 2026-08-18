#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N ssim_workflow_stream
#PBS -l select=3
#PBS -l place=scatter:group=tier1
#PBS -l walltime=0:30:00
#PBS -l filesystems=home:flare
#PBS -A datascience
#PBS -q debug-scaling
#PBS -k doe
#PBS -j oe

cd $PBS_O_WORKDIR
export TZ='/usr/share/zoneinfo/US/Central'
SSIM_ENV=${SSIM_ENV:-$PWD/_ssim_env}
DRIVER=${DRIVER:-$PWD/driver.py}

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
source "$SSIM_ENV/bin/activate"

# env variables

# SmartSim env vars
export SR_SOCKET_TIMEOUT=10000
export SR_LOG_FILE=stdout
export SR_LOG_LEVEL=QUIET

# Run
DEPLOYMENT=clustered
DB_NODES=1
COLOCATED_MAX_PPN=6   # colocated bindings in driver.py only go up to ppn=6

# For clustered, DB takes DB_NODES nodes; producer + consumer share the rest.
# For colocated, all nodes host both DB and workloads, so no DB nodes are set aside.
if [ "$DEPLOYMENT" = "clustered" ]; then
    COMPONENT_NODES=$(( NODES - DB_NODES ))
else
    COMPONENT_NODES=$NODES
fi

for RANKS_PER_NODE in 1 8 12
do
  if [ "$DEPLOYMENT" = "colocated" ] && [ "$RANKS_PER_NODE" -gt "$COLOCATED_MAX_PPN" ]; then
    echo "Skipping ppn=$RANKS_PER_NODE for colocated deployment (max is $COLOCATED_MAX_PPN)"
    continue
  fi

  for BYTES in 262144 1048576 4194304 16777216 67108864 268435456 #1073741824 4294967296
  do
    # Exp name encodes component-node count (matches MPI/ADIOS2 'n') and DB nodes as 'd'.
    EXP_NAME="${LOG_DIR}/ssim_${DEPLOYMENT}_n${COMPONENT_NODES}d${DB_NODES}_N${RANKS_PER_NODE}_buff${BYTES}"
    python $DRIVER --name $EXP_NAME \
      --deployment $DEPLOYMENT \
      --db_nodes $DB_NODES \
      --ppn $RANKS_PER_NODE \
      --producer_args $BYTES $DB_NODES
  done
done
