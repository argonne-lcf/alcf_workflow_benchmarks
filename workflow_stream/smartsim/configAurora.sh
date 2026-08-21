#!/bin/bash

set -eo pipefail

SSIM_ENV=${SSIM_ENV:-$PWD/_ssim_env}
SMARTREDIS_INSTALL_DIR=${SMARTREDIS_INSTALL_DIR:-$PWD/SmartRedis/install}

# Load modules and activate venv
module load frameworks
source "$SSIM_ENV/bin/activate"

# Sanity check
if [ ! -f "$SMARTREDIS_INSTALL_DIR/lib64/libsmartredis.so" ]; then
    echo "ERROR: libsmartredis.so not found under $SMARTREDIS_INSTALL_DIR/lib64"
    echo "Did you run ./installAurora.sh first?"
    exit 1
fi

# Build producer.cpp (with SYCL for GPU-resident producer buffers)
mpicxx -O3 -std=c++17 -fsycl \
  -I"$SMARTREDIS_INSTALL_DIR/include" \
  producer.cpp \
  -L"$SMARTREDIS_INSTALL_DIR/lib64" -Wl,-rpath,"$SMARTREDIS_INSTALL_DIR/lib64" \
  -lsmartredis \
  -o producer
