# HPC Code (no plotting, no interactive stuff)

using Pkg
Pkg.activate(dirname(@__DIR__))

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))



# BEGIN Code
# Rank 0: master. Runs all CPU initializations etc.
# Ranks 1, .., nprocs-1: workers. Each rank is dedicated to a GPU.

using MPI
using CUDA

@assert CUDA.functional() "ERROR: CUDA is not functional."

# Initialize MPI and specifically get current rank
MPI.Init();
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)


# Only on master process - commence simulation by initializing and setting everything up,
# and then call all other processes to perform the main algorithm

# Load packages for master worker to use
using MainCore
using Revise


# Run simulation
main_simulation_HPC(comm, rank, size);


# Wait until all processes are done
MPI.Barrier(comm);


# Finish and close the MPI
MPI.Finalize();