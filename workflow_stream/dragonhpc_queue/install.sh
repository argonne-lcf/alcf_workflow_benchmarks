#!/bin/bash

set -eo pipefail

SYSTEM=${1:-"aurora"}
DRAGON_ENV=${DRAGON_ENV:-$PWD/_dragon_env}

if [ -d "$DRAGON_ENV" ]; then
    echo "Environment already exists at $DRAGON_ENV."
    echo "Delete it (rm -rf $DRAGON_ENV) or set DRAGON_ENV to a fresh path to reinstall."
    exit 0
fi

# Load modules and activate the Dragon venv
if [ "$SYSTEM" = "aurora" ]; then
    module load frameworks
elif [ "$SYSTEM" = "polaris" ]; then
    module use /soft/modulefiles
    module unload xalt
    module load conda
    conda activate base
fi

# Create venv and install dragon
python -m venv --clear "$DRAGON_ENV" --system-site-packages
source "$DRAGON_ENV/bin/activate"
pip install dragonhpc==0.14.0
if [ "$SYSTEM" = "aurora" ]; then
    dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64
elif [ "$SYSTEM" = "polaris" ]; then
    dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/2.2.0rc1/lib64
fi

echo
echo "Install complete."
echo "  venv:               $DRAGON_ENV"

