#!/bin/bash

set -eo pipefail

SSIM_ENV=${SSIM_ENV:-$PWD/_ssim_env}
SMARTREDIS_SRC=${SMARTREDIS_SRC:-$PWD/SmartRedis}
SMARTSIM_SRC=${SMARTSIM_SRC:-$PWD/SmartSim}

if [ -d "$SSIM_ENV" ]; then
    echo "Environment already exists at $SSIM_ENV."
    echo "Delete it (rm -rf $SSIM_ENV) or set SSIM_ENV to a fresh path to reinstall."
    exit 0
fi

# Load modules
module load frameworks

# Create venv
python -m venv --clear "$SSIM_ENV" --system-site-packages
source "$SSIM_ENV/bin/activate"

# Install SmartRedis (Python bindings + native lib)
if [ ! -d "$SMARTREDIS_SRC" ]; then
    git clone https://github.com/CrayLabs/SmartRedis.git "$SMARTREDIS_SRC"
fi
pushd "$SMARTREDIS_SRC"
pip install -e .
make lib
popd

# Install SmartSim
if [ ! -d "$SMARTSIM_SRC" ]; then
    git clone https://github.com/CrayLabs/SmartSim.git "$SMARTSIM_SRC"
fi
pushd "$SMARTSIM_SRC"
pip install -e .

# Aurora-specific config patch + smart build
export TORCH_CMAKE_PATH=$( python -c 'import torch; print(torch.utils.cmake_prefix_path)' )
export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH

curl -O https://gist.githubusercontent.com/rickybalin/fcf1d15a26dbbc120f42943041ada827/raw/e22485d53250b8a29ead537533bca7c8f229c362/aurora_config.patch
git apply aurora_config.patch
smart build -v --device cpu --skip-tensorflow --skip-onnx
smart validate
popd

echo
echo "Install complete."
echo "  venv:               $SSIM_ENV"
echo "  SmartRedis source:  $SMARTREDIS_SRC"
echo "  SmartRedis install: $SMARTREDIS_SRC/install"

