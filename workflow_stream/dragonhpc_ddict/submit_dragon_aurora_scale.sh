#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N dragon_workflow_stream
#PBS -l select=256
#PBS -l place=scatter:group=tier0
#PBS -l walltime=0:30:00
#PBS -l filesystems=home:flare
#PBS -A datascience
#PBS -q prod
#PBS -k doe
#PBS -j oe

cd $PBS_O_WORKDIR
export TZ='/usr/share/zoneinfo/US/Central'
DRAGON_ENV=${DRAGON_ENV:-$PWD/_dragon_env}
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
source "$DRAGON_ENV/bin/activate"

# env variables

# Dragon env variables
export DRAGON_DEFAULT_SEG_SZ=$((32 * 1024**3)) # increase default pool size

# Run
DEPLOYMENT=clustered
DDICT_NODES=32
DDICT_MEM=400         # DDict memory per node in GB
DEVICE="gpu"          # gpu/cpu (producer buffer location)
COLOCATED_MAX_PPN=6   # colocated bindings in driver.py only go up to ppn=6

# For clustered, DDict takes DDICT_NODES; producer + consumer share the rest.
# For colocated/mixed, DDict runs on every node alongside the workloads.
if [ "$DEPLOYMENT" = "clustered" ]; then
    COMPONENT_NODES=$(( NODES - DDICT_NODES ))
else
    COMPONENT_NODES=$NODES
fi

RANKS_PER_NODE=12 
BYTES=268435456
EXP_NAME="dragon_${DEPLOYMENT}_n${COMPONENT_NODES}d${DDICT_NODES}_${DEVICE}_N${RANKS_PER_NODE}_buff${BYTES}"
dragon $DRIVER --log_dir $LOG_DIR --exp_name $EXP_NAME \
  --deployment $DEPLOYMENT \
  --bytes_per_rank $BYTES \
  --ddict_nodes $DDICT_NODES \
  --procs_per_node $RANKS_PER_NODE \
  --ddict_mem_size_per_node $DDICT_MEM \
  --device $DEVICE

