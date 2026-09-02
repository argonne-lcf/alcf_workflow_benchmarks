#!/bin/bash

# Load modules
module load frameworks

# Build wkfl_stream_mpi with SYCL 
mpicxx -O3 -std=c++17 -fsycl -o wkfl_stream_mpi wkfl_stream_mpi.cpp
