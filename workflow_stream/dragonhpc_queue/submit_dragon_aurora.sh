#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N dragonq_workflow_stream
#PBS -l select=2
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

# Load modules and activate venv
module load frameworks
module list
source "$DRAGON_ENV/bin/activate"

# env variables

# Run
DEPLOYMENT=clustered
COLOCATED_MAX_PPN=6   # colocated bindings in driver.py only go up to ppn=6

for RANKS_PER_NODE in 1 8 12
do
  if [ "$DEPLOYMENT" = "colocated" ] && [ "$RANKS_PER_NODE" -gt "$COLOCATED_MAX_PPN" ]; then
    echo "Skipping ppn=$RANKS_PER_NODE for colocated deployment (max is $COLOCATED_MAX_PPN)"
    continue
  fi

  for BYTES in 262144 1048576 4194304 16777216 67108864 268435456 1073741824 #4294967296
  do
    LOG_FILE=$LOG_DIR/dragonq_${DEPLOYMENT}_n${NODES}_N${RANKS_PER_NODE}_buff${BYTES}.log
    dragon $DRIVER \
      --deployment $DEPLOYMENT \
      --bytes_per_rank $BYTES \
      --ppn $RANKS_PER_NODE \
      2>&1 | tee $LOG_FILE
  done
done
