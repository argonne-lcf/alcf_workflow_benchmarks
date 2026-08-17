#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N dragon_workflow_stream
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

# Run
DEPLOYMENT=clustered
DDICT_NODES=1
DDICT_MEM=500         # DDict memory per node in GB
COLOCATED_MAX_PPN=6   # colocated bindings in driver.py only go up to ppn=6
for RANKS_PER_NODE in 1 2 8 12
do
  if [ "$DEPLOYMENT" = "colocated" ] && [ "$RANKS_PER_NODE" -gt "$COLOCATED_MAX_PPN" ]; then
    echo "Skipping ppn=$RANKS_PER_NODE for colocated deployment (max is $COLOCATED_MAX_PPN)"
    continue
  fi

  for BYTES in 262144 1048576 4194304 16777216 67108864 268435456 1073741824 4294967296
  do
    EXP_NAME="dragon_${DEPLOYMENT}_n${NODES}_N${RANKS_PER_NODE}_buff${BYTES}"
    dragon $DRIVER --log_dir $LOG_DIR --exp_name $EXP_NAME \
      --deployment $DEPLOYMENT \
      --bytes_per_rank $BYTES \
      --ddict_nodes $DDICT_NODES \
      --procs_per_node $RANKS_PER_NODE \
      --ddict_mem_size_per_node $DDICT_MEM
  done
done
