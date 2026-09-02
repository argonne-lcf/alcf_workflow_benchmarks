#!/bin/bash

# Load modules
module load frameworks
#module load adios2/2.11.0-cpu
module load adios2/2.11.0-sycl

# Build producer.cpp (with SYCL for GPU-resident producer buffers)
mpicxx -O3 -std=c++17 -fsycl \
  -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX \
  $(adios2-config --cxx-flags) \
  producer.cpp \
  $(adios2-config --cxx-libs) \
  -o producer
