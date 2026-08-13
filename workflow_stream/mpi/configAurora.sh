#!/bin/bash

# Load modules
module load frameworks

# Build sim.cpp
mpicxx -O3 -std=c++17 -o wkfl_stream_mpi wkfl_stream_mpi.cpp
