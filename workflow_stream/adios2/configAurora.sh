#!/bin/bash

# Load modules
module load frameworks
module load adios2/2.11.0-cpu

# Build producer.cpp
mpicxx -O3 -std=c++17 \
  -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX \
  $(adios2-config --cxx-flags) \
  producer.cpp \
  $(adios2-config --cxx-libs) \
  -o producer
