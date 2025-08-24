#!/bin/bash

# Load the OpenMPI module
# module load openmpi/5.0.3-pbs
export PATH="$HOME/openmpi-cuda/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/openmpi-cuda/lib:$LD_LIBRARY_PATH"

# Suppress UCX variable warnings
export UCX_WARN_UNUSED_ENV_VARS=n
export JULIA_CPU_TARGET=native

# Run your Julia MPI program
N_GPUS=$(nvidia-smi -L | wc -l)
mpirun -np $(( N_GPUS + 1 )) julia --compile=all --project=. scripts/run_HPC.jl
#mpirun -np 3 julia --compile=all --project=. scripts/run_HPC.jl


