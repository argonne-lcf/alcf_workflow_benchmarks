#!/bin/bash

set -eo pipefail

SYSTEM=${1:-"aurora"}
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SRC"

# Load modules and activate the Dragon venv
if [ "$SYSTEM" = "aurora" ]; then
    module load frameworks
elif [ "$SYSTEM" = "polaris" ]; then
    module use /soft/modulefiles
    module unload xalt
    module load conda
    conda activate base
fi
module list

DRAGON_ENV="${SRC}/_dragon_env"
if [ ! -d "$DRAGON_ENV" ]; then
    echo "Error: Dragon venv not found at $DRAGON_ENV"
    echo "       Run ./install.sh first."
    exit 1
fi
source "${DRAGON_ENV}/bin/activate"

# Resolve the Dragon install root from the active venv
DRAGON_ROOT=$(python -c "import dragon, os; print(os.path.dirname(dragon.__file__))" 2>/dev/null)
if [ -z "$DRAGON_ROOT" ] || [ ! -d "$DRAGON_ROOT" ]; then
    echo "Error: could not locate dragonhpc install from the active Python."
    echo "       Make sure dragonhpc is installed in ${DRAGON_ENV}."
    exit 1
fi
echo "Using DRAGON_ROOT=$DRAGON_ROOT"

# Build producer.cpp
# _GLIBCXX_USE_CXX11_ABI=0 resolves an ABI conflict with the Dragon wheel's prebuilt library.
mpicxx -O3 -std=c++17 \
  -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I"$DRAGON_ROOT/include" \
  producer.cpp cpp_serializers.cpp \
  -L"$DRAGON_ROOT/lib" -Wl,-rpath,"$DRAGON_ROOT/lib" \
  -ldragon -ldl \
  -o producer
