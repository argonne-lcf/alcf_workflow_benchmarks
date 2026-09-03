#!/bin/bash

hostname=`hostname -f`
if [[ $hostname == *"aurora"* ]]; then
  echo Building for Aurora ...
  icpx -o parallel_app parallel_app.cpp -lmpi
  icpx -o serial_app serial_app.cpp
else
  echo No pre-defined build instructions for host $hostname 
fi
