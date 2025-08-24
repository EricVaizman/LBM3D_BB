# if rank == 3

#     linear_to_xyz, _ = MPI.recv(0, 8, comm)

#     println("\n Rank $(rank) local_data:")
#     print("\n")
#     println(owned_idxs)
#     print("\n")
#     println([linear_to_xyz[i] for i in owned_idxs])
    
#     print("\n")
#     println(halo_nodes)
#     print("\n")
#     println([linear_to_xyz[i] for i in halo_nodes])
#     print("\n")
#     println(halo_dirs)
#     print("\n")
#     println(halo_ptr)

#     k = 6 # Halo node number of this rank
#     print("\n")
#     println("For halo node $k of rank $rank:")
#     println("Located at: $(linear_to_xyz[halo_nodes[k]])")
#     println("The dirs pointing from this halo node to relevant subdomain nodes:")
#     println(halo_dirs[(halo_ptr[k]):(halo_ptr[k+1]-1)])



#     k2 = 44 # Boundary node number of this rank
#     print("\n")
#     println("For boundary node $k2 of rank $rank:")
#     println("Located at: $(linear_to_xyz[bb_nodes_local[k2]] .- 0.5)")
#     println("The dirs pointing from this halo node to relevant subdomain nodes:")
#     println(bb_dirs_local[(bb_ptr_local[k2]):(bb_ptr_local[k2+1]-1)])

# end



# NOT WORKING PROPERLY:
#
# function build_linear_to_xyz(Nx, Ny, Nz)
#     linear_to_xyz = Dict{Int, NTuple{3, Int}}()
#     for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
#         lin = (ix - 1) + (iy - 1)*Nx + (iz - 1)*Nx*Ny + 1
#         linear_to_xyz[lin] = (ix, iy, iz)
#     end
#     return linear_to_xyz
# end

# function build_xyz_to_linear(Nx, Ny, Nz)
#     xyz_to_linear = Dict{NTuple{3, Int}, Int}()
#     for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
#         lin = (ix - 1) + (iy - 1)*Nx + (iz - 1)*Nx*Ny + 1
#         xyz_to_linear[(ix, iy, iz)] = lin
#     end
#     return xyz_to_linear
# end


# DEBUG PRINTING OF OWNED & HALO NODES
# print("I am rank $rank, and I have a total of $N_local nodes (owned + halo)\nI have $N_owned owned nodes:\n$(owned_idxs)\n$([linear_to_xyz_local[i] for i in owned_idxs])\nI have $N_halo halo nodes:\n$(halo_nodes)\n$([linear_to_xyz_local[i] for i in halo_nodes])\n\n")

# DEBUG PRINTING OF GPU GLOBAL TO LOCAL MAPPINGS AND VICE VERSA
# print("I'm rank $rank and my global_to_local is length $(length(global_to_local)) and is:\n$(global_to_local)\nMy GPU local_to_global is:\n$(local_to_global_gpu)\nBinary search keys and then vals:\n$(global_keys_gpu)\n$(local_vals_gpu)\n\n\n")

# GLOBAL LINEAR INDEX ON GPU FROM A GLOBAL 3D GPU INDEX
# g_j = (jx - 1)*Ny*Nz + (jy - 1)*Nz + jz
# IMPORTANT! this assumes column-major ordering



#=

    # for r in 0:(size-1)
    #     if (rank == r)
    #         if (r == 0)
    #             print("I'm rank $rank, MASTER.\n\n")
    #         else
    #             print("I am rank $rank, and I have a total of $N_local nodes (owned + halo)\nI have $N_owned owned nodes:\n$(Array(owned_idxs))\nI have $N_halo halo nodes with $(length(halo_dirs)) total dirs:\n$(halo_nodes)\n")
    #             print("I am rank $rank, my f after exchange is:\n$(Array(f_new_gpu))\n\n")
    #         end
    #     end
    #     MPI.Barrier(comm)
    # end

    # if (rank == 4)
    #     print("I am rank $(rank)! I have a total of $N_local nodes (owned + halo)\n\n")

    #     print("I have $N_owned owned nodes:\n$(Array(owned_idxs))\n$([linear_to_xyz_local[i] for i in Array(owned_idxs)])\n")
    #     print("The global to local mapping of my OWNED indices is:\n$([global_to_local[i] for i in Array(owned_idxs)]))\n\n")

    #     print("I have $N_halo halo nodes with $(length(halo_dirs)) total dirs:\n$(halo_nodes)\n$([linear_to_xyz_local[i] for i in halo_nodes])\n")
    #     print("The global to local mapping of my HALO indices is:\n$([global_to_local[i] for i in halo_nodes]))\n\n")

    #     print("My f after exchange is:\n$(Array(f_new_gpu))\n\n\n\n")

    #     print("halo_dirs length: $(length(halo_dirs))\n$(halo_dirs)\n")
    #     print("halo_ptr length: $(length(halo_ptr))\n$(halo_ptr)\n")

    #     print("neighs are:\n$(neighs)\n")
    #     print("and part_ptr is:\n$(part_ptr)\n\n")

    #     print("-------------------\n\n")

    #     for i in 1:length(halo_nodes)
    #         print("Halo node #$i in the list:\n")
    #         print("Local linear index: $(global_to_local[halo_nodes[i]])\n")
    #         print("Global linear index: $(halo_nodes[i])\n")
    #         print("Global 3D index: $(linear_to_xyz_local[halo_nodes[i]])\n")
    #         print("Dirs of this halo node: $(halo_dirs[halo_ptr[i]:(halo_ptr[i+1]-1)])\n\n")
    #     end

    # end

=#


# Create GPU index mappings
        # local_to_global_gpu, global_keys_gpu, local_vals_gpu = build_gpu_index_maps(global_to_local) # Create global to local linear index mapping for current rank
        # ix_gpu, iy_gpu, iz_gpu = build_local_coords_gpu(Array(local_to_global_gpu), linear_to_xyz_local) # Create a GPU local linear index to 3D global index mapping

    



# COMPUTE EDGE NODES
# function compute_edge_nodes(
#     owned_idxs::Vector{Int},
#     owned_set::Set{Int},
#     c_x::Vector{Int},
#     c_y::Vector{Int},
#     c_z::Vector{Int},
#     linear_to_xyz::Dict{Int, Tuple{Int,Int,Int}},
#     xyz_to_linear::Dict{Tuple{Int,Int,Int}, Int},
#     Nx::Int, Ny::Int, Nz::Int
#     )

#     q = length(c_x)
#     edge_nodes = Int[]

#     for idx in owned_idxs
#         ix, iy, iz = linear_to_xyz[idx]

#         for qdir in 1:q
#             jx = ix + c_x[qdir]
#             jy = iy + c_y[qdir]
#             jz = iz + c_z[qdir]

#             # Check if neighbor is within global bounds
#             if !(1 ≤ jx ≤ Nx && 1 ≤ jy ≤ Ny && 1 ≤ jz ≤ Nz)
#                 continue
#             end

#             neighbor_idx = xyz_to_linear[(jx, jy, jz)]

#             # If neighbor is not owned by this rank, this is an edge node
#             if !(neighbor_idx in owned_set)
#                 push!(edge_nodes, idx)
#                 break  # No need to check further directions
#             end
#         end
#     end

#     return edge_nodes
# end


# edge_nodes = compute_edge_nodes(
#     owned_idxs, owned_set, c_x, c_y, c_z,
#     linear_to_xyz_local, xyz_to_linear_local, Nx, Ny, Nz
# )



# # ---- SEND FINAL RESULTS TO LOCAL MACHINE FOR POST-PROCESSING ----
# write_fields_h5(
#     comm, rank,
#     owned_idxs,
#     Array(rho_gpu),
#     Array(u_gpu),
#     Array(v_gpu),
#     Array(w_gpu),
#     params.Nx, params.Ny, params.Nz,
#     "output/final_fields.h5"
# )




# # GPU UTILITIES
# function build_gpu_index_maps(global_to_local::Dict{Int, Int})
#     N_total = length(global_to_local)

#     # Build local_to_global (compact)
#     local_to_global_host = Vector{Int32}(undef, N_total)
#     for (g, l) in global_to_local
#         local_to_global_host[l] = Int32(g)
#     end
#     local_to_global_gpu = CuArray(local_to_global_host)

#     # Build sorted global_keys and corresponding local indices
#     global_keys = sort(collect(keys(global_to_local)))
#     local_vals = Int32[global_to_local[g] for g in global_keys]

#     global_keys_gpu = CuArray(Int32.(global_keys))
#     local_vals_gpu  = CuArray(local_vals)

#     return local_to_global_gpu, global_keys_gpu, local_vals_gpu
# end




# function build_local_coords_gpu(
#     local_to_global::Vector{Int32},
#     linear_to_xyz::Dict{Int, NTuple{3, Int}}
#     )

#     N = length(local_to_global)
#     ix = Vector{Int32}(undef, N)
#     iy = Vector{Int32}(undef, N)
#     iz = Vector{Int32}(undef, N)

#     for i in 1:N
#         gidx = local_to_global[i]
#         x, y, z = linear_to_xyz[gidx]
#         ix[i] = Int32(x)
#         iy[i] = Int32(y)
#         iz[i] = Int32(z)
#     end

#     return CuArray(ix), CuArray(iy), CuArray(iz)
# end




# BELOW IS THE TEST VERSION OF MAINCORE.JL THAT ALLOCATES THE F ARRAYS IN A SPECIAL WAY AND SUCH, AND SAVES THEM TOO
# module MainCore

# include("Parameters.jl")
# include("Geometry.jl")
# include("Partitioning.jl")
# include("Initialization.jl")
# include("ComputeMacro.jl")
# include("Collision.jl")
# include("Exchange.jl")
# include("BoundaryConditions.jl")
# include("Streaming.jl")
# include("I_O.jl")
# include("Utilities.jl")



# using .Parameters
# using .Geometry
# using .Partitioning
# using .Initialization
# using .ComputeMacro
# using .Collision
# using .Exchange
# using .BoundaryConditions
# using .Streaming
# using .I_O
# using .Utilities
# using MPI
# using CUDA

# export Parameters, main_simulation, I_O, Geometry, Partitioning, main_simulation_HPC




# function main_simulation_HPC(comm::MPI.Comm, rank::Int64, size::Int64)
    
#     nworkers = size - 1

#     # Each rank (master and workers) read input file into parameters struct and initialize simulation parameters
#     params = read_params_input(joinpath(dirname(@__DIR__), "input", "params.txt"));
#     cs2 = Float32(1//3)
#     num_snaps = ceil(Int, params.n_max / params.n_save) # For I/O saving

#     # Make sure all processes are done reading input data before advancing
#     # MPI.Barrier(comm)


#     # ---- HOST SETUP ----
#     # Initialize simulation setup on master process, and send local data to each worker
#     if rank == 0
#         N_local = 0
#         mask_s, mask_bb, partitions, xyz_to_linear =
#             initialize_distributed(comm, rank, nworkers, params)
            
#         owned_idxs = Int[]
#         rho_gpu, u_gpu, v_gpu, w_gpu = Float32[], Float32[], Float32[], Float32[]
#         f_old_gpu, f_new_gpu = Array{Float32}(undef, params.q, N_local), Array{Float32}(undef, params.q, N_local)

#         # Save partition figure once (just master)
#         output_path = joinpath(dirname(@__DIR__), "output", "partition_plot.png")
#         plot_mask_3D(mask_s, mask_bb, params, partitions, xyz_to_linear; savepath=output_path)
#     else
#         c_x, c_y, c_z, w, bbOpp,
#         local_data, owned_idxs, mask_s_local, mask_bb_local,
#         linear_to_xyz_local, xyz_to_linear_local,
#         global_to_local,
#         bb_ptr_local, bb_dirs_local, bb_nodes_local, bb_lidxs,
#         halo_nodes, halo_dirs, halo_ptr,
#         neighs, part_ptr,
#         gather_qdirs, gather_lidxs, gather_ptr,
#         scatter_qdirs, scatter_lidxs,
#         stream_lidxs, stream_int_tids, stream_halo_tids,
#         vel_lidxs, vel_dirs, pres_lidxs, pres_dirs,
#         per_dest, per_dir, per_src, per_int_tids, per_halo_tids = 
#             initialize_distributed(comm, rank, nworkers, params)   
#     end


#     # ---- MAIN CODE ----
#     if (rank != 0)

#         # print("Rank $rank entered main code.\n")

#         # ---- GPU SETUP ----
#         # Local parameters
#         N_owned = length(owned_idxs) # Number of owned nodes of current subdomain
#         N_halo  = length(halo_nodes) # Number of halo nodes of current subdomain
#         N_local = N_owned + N_halo   # Total number of nodes (owned + halo) of current subdomain
#         omega = params.dt/params.tau # Relaxation parameter

#     end

#     # after you compute N_local on every rank:
#     all_N = MPI.gather(N_local, comm, root=0)
#     if rank == 0
#         # all_N is length size; element 1 is the master’s N_local (unused)
#         N_locals = all_N[2:end]
#         N_local = 0
#     else
#         # Other ranks don’t need the full list
#         N_locals = Int[]
#     end

#     if (rank != 0)
#         # Pair GPU & rank, and allocate memory of GPU arrays per each worker
#         f_old_gpu, f_new_gpu, rho_gpu, u_gpu, v_gpu, w_gpu = worker_alloc(N_owned, N_halo, params.q, rank)
#         c_x_gpu, c_y_gpu, c_z_gpu = CuArray(Int8.(c_x)), CuArray(Int8.(c_y)), CuArray(Int8.(c_z))
#         wvec_gpu = CuArray(w)
#         bbOpp_gpu = CuArray(bbOpp)

#         total_halo = Int32(length(gather_lidxs))
#         send_buffer_gpu = CuArray{Float32}(undef, total_halo)
#         recv_buffer_gpu = CuArray{Float32}(undef, total_halo)
#         TAG = 0

#         f_old_test = Array{Float32}(undef, params.q, N_local)
#         f_new_test = Array{Float32}(undef, params.q, N_local)

#         n = 0
#         count = 1
#         for j = 1:N_local, i = 1:params.q
#             f_old_test[i, j] = Float32(rank)*1000 + count
#             f_new_test[i, j] = Float32(rank)*1000 + count
#             count += 1
#         end

#         f_old_gpu = CuArray(f_old_test)
#         f_new_gpu = CuArray(f_new_test)

#         # k = 1
#         # m = searchsortedlast(part_ptr, k)
#         # owner_rank = neighs[m]
#         # if (rank == 1)
#         #     print("\nI'm rank $rank, my halo node #$k:\n")
#         #     print("It originally belongs as an owned node to rank $owner_rank\n")
#         #     print("It has a global linear index of $(halo_nodes[k])\n")
#         #     print("It has a local linear index of $(global_to_local[halo_nodes[k]])\n")
#         #     print("It has a global 3D index of $(linear_to_xyz_local[halo_nodes[k]])\n")
#         #     print("The populations that stream from it into the subdomain are:\n")
#         #     curr_halo_dirs = halo_dirs[halo_ptr[k]:(halo_ptr[k+1]-1)];
#         #     for dir in curr_halo_dirs
#         #         print("Dir $dir: ($(c_x[dir]), $(c_y[dir]), $(c_z[dir]))\n")
#         #     end
            
#         #     # owned_values_of_halo = Float32(owner_rank)*1000 .+ 
#         #     print("\n")
#         # end

#         local_to_global = Dict( l => g for (g,l) in global_to_local )

#         # print("I am rank $rank and my owned nodes are:\n$(owned_idxs)\nTheir local indices are:\n$([global_to_local[owned_idxs[i]] for i in 1:length(owned_idxs)])\nMy 3D global indices are: $([linear_to_xyz_local[owned_idxs[i]] for i in 1:length(owned_idxs)])\nMy halo nodes are:\n$(halo_nodes)\nTheir local indices are:\n$([global_to_local[halo_nodes[i]] for i in 1:length(halo_nodes)])\nMy 3D global indices are: $([linear_to_xyz_local[halo_nodes[i]] for i in 1:length(halo_nodes)])\nMy gather_lidxs are:\n$(gather_lidxs)\nGather GLOBAL indices:\n$([local_to_global[glidx] for glidx in Array(gather_lidxs)])\nand my gather_qdirs are:\n$(gather_qdirs)\nMy scatter_lidxs are:\n$(scatter_lidxs)\nand my scatter_qdirs are:\n$(scatter_qdirs)\nScatter GLOBAL indices:\n$([local_to_global[slidx] for slidx in Array(scatter_lidxs)])\n\n\n")

#         # ---- INITIALIZE POPULATION ARRAYS ----
#         # initialize_f(
#         #     f_old_gpu, f_new_gpu,
#         #     c_x_gpu, c_y_gpu, c_z_gpu,
#         #     wvec_gpu, params.rho_inf,
#         #     params.U_inf, params.V_inf, params.W_inf,
#         #     N_owned, N_halo, params.q
#         # )
#         # print("Rank $rank finished initialization.\n")

#         # ---- COMPUTE MACRO QUANTITIES - ITERATION 0
#         compute_macros!(
#             f_old_gpu,
#             rho_gpu, u_gpu, v_gpu, w_gpu,
#             c_x_gpu, c_y_gpu, c_z_gpu
#         )


#     end

#     save_populations_h5!(
#         comm, rank,
#         Array(f_old_gpu), Array(f_new_gpu),
#         N_local,
#         N_locals,
#         params.q,
#         "output/populations.h5",
#         0,
#         params.n_save
#     )

#     # ---- MAIN LBM ALGORITHM
#     for n = 1:params.n_max

#         # Worker's job every time step
#         if (rank != 0)

#             # ---- COLLISION (+COMPUTE feq) ----
#             bgk_collision!(
#                 f_old_gpu, f_new_gpu,
#                 rho_gpu,
#                 u_gpu, v_gpu, w_gpu,
#                 c_x_gpu, c_y_gpu, c_z_gpu,
#                 wvec_gpu,
#                 omega, N_owned
#             )

#             # ---- BEGIN HALO EXCHANGE ----
#             send_reqs, recv_reqs = begin_halo_exchange!(
#                 comm,
#                 f_new_gpu, send_buffer_gpu, recv_buffer_gpu,
#                 gather_qdirs, gather_lidxs, gather_ptr,
#                 neighs, TAG
#             )

#             # ---- INTERIOR NODES STREAM ----
#             streaming_part!(
#                 f_new_gpu,
#                 f_old_gpu,
#                 stream_lidxs,
#                 stream_int_tids,
#                 params.q,
#                 N_owned
#             )

#             # ---- INTERIOR PERIODIC BC ----
#             periodic_BC_part!(
#                 f_old_gpu,
#                 per_dest,
#                 per_dir,
#                 per_src,
#                 per_int_tids 
#             )

#             # ---- FINISH HALO EXCHANGE ----
#             finish_halo_exchange!(
#                 send_reqs, recv_reqs,
#                 recv_buffer_gpu,
#                 scatter_qdirs, scatter_lidxs,
#                 f_new_gpu
#             )

#             # ---- HALO NODES STREAM ----
#             # streaming_part!(
#             #     f_new_gpu,
#             #     f_old_gpu,
#             #     stream_lidxs,
#             #     stream_halo_tids, 
#             #     params.q,
#             #     N_owned
#             # )

#             # ---- HALO PERIODIC BC ----
#             # periodic_BC_part!(
#             #     f_old_gpu,
#             #     per_dest,
#             #     per_dir,
#             #     per_src,
#             #     per_halo_tids
#             # )

#             # ---- BOUNCEBACK BC ----
#             # bounceback_BC!(
#             #     f_old_gpu,
#             #     f_new_gpu,
#             #     bb_lidxs,
#             #     bb_dirs_local,
#             #     bbOpp_gpu 
#             # )

#             # ---- APPLY VELOCITY BC AT FACES ----
#             # velocity_BC!(
#             #     f_old_gpu,
#             #     f_new_gpu,
#             #     vel_lidxs, vel_dirs,
#             #     pres_lidxs, pres_dirs,
#             #     bbOpp_gpu,
#             #     c_x_gpu, c_y_gpu, c_z_gpu,
#             #     wvec_gpu,
#             #     rho_gpu, u_gpu, v_gpu, w_gpu,
#             #     params.U_inf, params.V_inf, params.W_inf, 
#             #     cs2 
#             # )

#             # ---- COMPUTE MACRO QUANTITIES ----
#             # compute_macros!(
#             #     f_old_gpu,
#             #     rho_gpu, u_gpu, v_gpu, w_gpu,
#             #     c_x_gpu, c_y_gpu, c_z_gpu
#             # )


#             if (rank == 1)
#                 print("$n\n")
#             end

#         end # END OF WORKER FUNCTIONALITY

#         # ---- SAVE TO HOST EVERY FEW TIME STEPS ----
#         write_fields_h5_time!(
#             comm, rank,
#             owned_idxs,
#             Array(rho_gpu), Array(u_gpu),
#             Array(v_gpu), Array(w_gpu),
#             params.Nx, params.Ny, params.Nz,
#             "output/final_fields.h5",
#             n, params.n_save, num_snaps
#         )

#         save_populations_h5!(
#             comm, rank,
#             Array(f_old_gpu), Array(f_new_gpu),
#             N_local,
#             N_locals,
#             params.q,
#             "output/populations.h5",
#             n,
#             params.n_save
#         )
        

#     end # END OF TIME STEPPING LOOP


#     return nothing
# end




# end # module





# TEMP (WORKING FOR ONE WORKER MAINCORE.JL):
# module MainCore

# include("Parameters.jl")
# include("Geometry.jl")
# include("Partitioning.jl")
# include("Initialization.jl")
# include("ComputeMacro.jl")
# include("Collision.jl")
# include("Exchange.jl")
# include("BoundaryConditions.jl")
# include("Streaming.jl")
# include("I_O.jl")
# include("Utilities.jl")



# using .Parameters
# using .Geometry
# using .Partitioning
# using .Initialization
# using .ComputeMacro
# using .Collision
# using .Exchange
# using .BoundaryConditions
# using .Streaming
# using .I_O
# using .Utilities
# using MPI
# using CUDA

# export Parameters, main_simulation, I_O, Geometry, Partitioning, main_simulation_HPC




# function main_simulation_HPC(comm::MPI.Comm, rank::Int64, size::Int64)
    
#     nworkers = size - 1

#     # Each rank (master and workers) read input file into parameters struct and initialize simulation parameters
#     params = read_params_input(joinpath(dirname(@__DIR__), "input", "params.txt"));
#     cs2 = Float32(1//3)
#     num_snaps = ceil(Int, params.n_max / params.n_save) # For I/O saving

#     # Make sure all processes are done reading input data before advancing
#     # MPI.Barrier(comm)


#     # ---- HOST SETUP ----
#     # Initialize simulation setup on master process, and send local data to each worker
#     if rank == 0
#         mask_s, mask_bb, partitions, xyz_to_linear =
#             initialize_distributed(comm, rank, nworkers, params)
            
#         owned_idxs = Int[]
#         rho_gpu, u_gpu, v_gpu, w_gpu = Float32[], Float32[], Float32[], Float32[]

#         # Save partition figure once (just master)
#         output_path = joinpath(dirname(@__DIR__), "output", "partition_plot.png")
#         # plot_mask_3D(mask_s, mask_bb, params, partitions, xyz_to_linear; savepath=output_path)
#         print("I'm rank $rank, and tau = $(params.tau).\n")
#     else
#         c_x, c_y, c_z, w, bbOpp,
#         local_data, owned_idxs, mask_s_local, mask_bb_local,
#         linear_to_xyz_local, xyz_to_linear_local,
#         global_to_local,
#         bb_ptr_local, bb_dirs_local, bb_nodes_local, bb_lidxs,
#         halo_nodes, halo_dirs, halo_ptr,
#         neighs, part_ptr,
#         gather_qdirs, gather_lidxs, gather_ptr,
#         scatter_qdirs, scatter_lidxs,
#         stream_lidxs, stream_int_tids, stream_halo_tids,
#         vel_lidxs, vel_dirs, pres_lidxs, pres_dirs,
#         per_dest, per_dir, per_src, per_int_tids, per_halo_tids = 
#             initialize_distributed(comm, rank, nworkers, params)   
#     end


#     # ---- MAIN CODE ----
#     if (rank != 0)

#         print("Rank $rank entered main code.\n")

#         # ---- GPU SETUP ----
#         # Local parameters
#         N_owned = length(owned_idxs) # Number of owned nodes of current subdomain
#         N_halo  = length(halo_nodes) # Number of halo nodes of current subdomain
#         N_local = N_owned + N_halo   # Total number of nodes (owned + halo) of current subdomain
#         omega = params.dt/params.tau # Relaxation parameter

#         # Pair GPU & rank, and allocate memory of GPU arrays per each worker
#         f_old_gpu, f_new_gpu, rho_gpu, u_gpu, v_gpu, w_gpu = worker_alloc(N_owned, N_halo, params.q, rank)
#         c_x_gpu, c_y_gpu, c_z_gpu = CuArray(Int8.(c_x)), CuArray(Int8.(c_y)), CuArray(Int8.(c_z))
#         wvec_gpu = CuArray(w)
#         bbOpp_gpu = CuArray(bbOpp)

#         total_halo = Int32(length(gather_lidxs))
#         send_buffer_gpu = CuArray{Float32}(undef, total_halo)
#         recv_buffer_gpu = CuArray{Float32}(undef, total_halo)
#         TAG = 0

    
#         # ---- INITIALIZE POPULATION ARRAYS ----
#         initialize_f(
#             f_old_gpu, f_new_gpu,
#             c_x_gpu, c_y_gpu, c_z_gpu,
#             wvec_gpu, params.rho_inf,
#             params.U_inf, params.V_inf, params.W_inf,
#             N_owned, N_halo, params.q
#         )
#         print("Rank $rank finished initialization.\n")

#         # ---- COMPUTE MACRO QUANTITIES - ITERATION 0
#         compute_macros!(
#             f_old_gpu,
#             rho_gpu, u_gpu, v_gpu, w_gpu,
#             c_x_gpu, c_y_gpu, c_z_gpu
#         )

#     end


#     # ---- MAIN LBM ALGORITHM
#     for n = 1:params.n_max

#         # Worker's job every time step
#         if (rank != 0)

#             # ---- COLLISION (+COMPUTE feq) ----
#             bgk_collision!(
#                 f_old_gpu, f_new_gpu,
#                 rho_gpu,
#                 u_gpu, v_gpu, w_gpu,
#                 c_x_gpu, c_y_gpu, c_z_gpu,
#                 wvec_gpu,
#                 omega, N_owned
#             )


#             # ---- BEGIN HALO EXCHANGE ----
#             send_reqs, recv_reqs = begin_halo_exchange!(
#                 comm,
#                 f_new_gpu, send_buffer_gpu, recv_buffer_gpu,
#                 gather_qdirs, gather_lidxs, gather_ptr,
#                 neighs, TAG
#             )


#             # ---- FINISH HALO EXCHANGE ----
#             finish_halo_exchange!(
#                 send_reqs, recv_reqs,
#                 recv_buffer_gpu,
#                 scatter_qdirs, scatter_lidxs,
#                 f_new_gpu
#             )


#             # ---- INTERIOR NODES STREAM ----
#             streaming_part!(
#                 f_new_gpu,
#                 f_old_gpu,
#                 stream_lidxs,
#                 stream_int_tids,
#                 params.q,
#                 N_owned
#             )

#             # ---- HALO NODES STREAM ----
#             streaming_part!(
#                 f_new_gpu,
#                 f_old_gpu,
#                 stream_lidxs,
#                 stream_halo_tids, 
#                 params.q,
#                 N_owned
#             )

#             # ---- INTERIOR PERIODIC BC ----
#             periodic_BC_part!(
#                 f_old_gpu,
#                 per_dest,
#                 per_dir,
#                 per_src,
#                 per_int_tids 
#             )


#             # ---- HALO PERIODIC BC ----
#             periodic_BC_part!(
#                 f_old_gpu,
#                 per_dest,
#                 per_dir,
#                 per_src,
#                 per_halo_tids
#             )

#             # ---- BOUNCEBACK BC ----
#             bounceback_BC!(
#                 f_old_gpu,
#                 f_new_gpu,
#                 bb_lidxs,
#                 bb_dirs_local,
#                 bbOpp_gpu 
#             )

#             # ---- APPLY VELOCITY BC AT FACES ----
#             velocity_BC!(
#                 f_old_gpu,
#                 f_new_gpu,
#                 vel_lidxs, vel_dirs,
#                 pres_lidxs, pres_dirs,
#                 bbOpp_gpu,
#                 c_x_gpu, c_y_gpu, c_z_gpu,
#                 wvec_gpu,
#                 rho_gpu, u_gpu, v_gpu, w_gpu,
#                 params.U_inf, params.V_inf, params.W_inf, 
#                 cs2 
#             )

#             # ---- COMPUTE MACRO QUANTITIES ----
#             compute_macros!(
#                 f_old_gpu,
#                 rho_gpu, u_gpu, v_gpu, w_gpu,
#                 c_x_gpu, c_y_gpu, c_z_gpu
#             )


#             if (rank == 1)
#                 print("$n\n")
#             end

#         end # END OF WORKER FUNCTIONALITY

#         # ---- SAVE TO HOST EVERY FEW TIME STEPS ----
#         write_fields_h5_time!(
#             comm, rank,
#             owned_idxs,
#             Array(rho_gpu), Array(u_gpu),
#             Array(v_gpu), Array(w_gpu),
#             params.Nx, params.Ny, params.Nz,
#             "output/final_fields.h5",
#             n, params.n_save, num_snaps
#         )

#     end # END OF TIME STEPPING LOOP


#     return nothing
# end




# end # module


# WORKING VERSION:
# function write_fields_h5_time!(
#     comm::MPI.Comm,
#     rank::Int,
#     local_idxs::Vector{Int},
#     local_rho::Vector{Float32},
#     local_u::Vector{Float32},
#     local_v::Vector{Float32},
#     local_w::Vector{Float32},
#     Nx::Int, Ny::Int, Nz::Int,
#     fname::String,
#     step::Int,
#     n_save::Int,
#     num_snaps::Int
# )
#     # only save on multiples of n_save
#     if step % n_save != 0
#         return
#     end

#     snap = div(step, n_save)  # 1-based slice index

#     # — MPI gather as before —
#     Nloc = Int32(length(local_idxs))
#     nprocs = MPI.Comm_size(comm)

#     counts = MPI.gather(Nloc, comm; root=0)
#     # broadcast counts array
#     if rank == 0
#         counts_all = copy(counts)
#     else
#         counts_all = Vector{Int32}(undef, nprocs)
#     end
#     MPI.Bcast!(counts_all, 0, comm)

#     # compute displacements & total entries
#     offsets = cumsum([0; counts_all[1:end-1]])
#     Ntot    = sum(counts_all)

#     # prepare receive buffers on root
#     if rank == 0
#         all_idxs = Vector{Int32}(undef, Ntot)
#         all_rho  = Vector{Float32}(undef, Ntot)
#         all_u    = Vector{Float32}(undef, Ntot)
#         all_v    = Vector{Float32}(undef, Ntot)
#         all_w    = Vector{Float32}(undef, Ntot)
#     else
#         all_idxs = Vector{Int32}()
#         all_rho  = Vector{Float32}()
#         all_u    = Vector{Float32}()
#         all_v    = Vector{Float32}()
#         all_w    = Vector{Float32}()
#     end

#     # convert to Int32
#     local_idxs32 = Int32.(local_idxs)

#     # Gatherv into the big arrays
#     MPI.Gatherv!(local_idxs32, all_idxs, counts_all, 0, comm)
#     MPI.Gatherv!(local_rho,   all_rho,  counts_all, 0, comm)
#     MPI.Gatherv!(local_u,     all_u,    counts_all, 0, comm)
#     MPI.Gatherv!(local_v,     all_v,    counts_all, 0, comm)
#     MPI.Gatherv!(local_w,     all_w,    counts_all, 0, comm)

#     # — now rank 0 builds the global 4D file —
#     if rank == 0
#         # rebuild the 1D global arrays
#         global_rho = zeros(Float32, Nx*Ny*Nz)
#         global_u   = zeros(Float32, Nx*Ny*Nz)
#         global_v   = zeros(Float32, Nx*Ny*Nz)
#         global_w   = zeros(Float32, Nx*Ny*Nz)
#         for i in 1:length(all_idxs)
#             idx = Int(all_idxs[i])
#             global_rho[idx] = all_rho[i]
#             global_u[idx] = all_u[i]
#             global_v[idx] = all_v[i]
#             global_w[idx] = all_w[i]
#         end

#         # reshape into 3D
#         rho3d = reshape(global_rho, (Nx,Ny,Nz))
#         U3d = reshape(global_u,   (Nx,Ny,Nz))
#         V3d = reshape(global_v,   (Nx,Ny,Nz))
#         W3d = reshape(global_w,   (Nx,Ny,Nz))

#         # open HDF5—create or append
#         mode = (snap==1 ? "w" : "r+")
#         h5open(fname, mode) do h5
#             if snap == 1
#                 # create chunked 4D datasets of size (Nx,Ny,Nz,num_snaps)
#                 # you can tune chunk size if you like
#                 h5["rho"] = zeros(Float32, Nx,Ny,Nz,num_snaps)
#                 h5["u"]   = zeros(Float32, Nx,Ny,Nz,num_snaps)
#                 h5["v"]   = zeros(Float32, Nx,Ny,Nz,num_snaps)
#                 h5["w"]   = zeros(Float32, Nx,Ny,Nz,num_snaps)
#                 h5["steps"] = zeros(Int32, num_snaps)
#             end

#             # write slice #snap into the 4th dimension
#             h5["rho"][:,:,:,snap] = rho3d
#             h5["u"][:,:,:,snap] = U3d
#             h5["v"][:,:,:,snap] = V3d
#             h5["w"][:,:,:,snap] = W3d
#             h5["steps"][snap] = step
#         end
#     end

#     if (rank == 0) print("Saving snapshot for n = $step.\n") end

#     return nothing
# end




# module MainCore

# include("Parameters.jl")
# include("Geometry.jl")
# include("Partitioning.jl")
# include("Initialization.jl")
# include("ComputeMacro.jl")
# include("Collision.jl")
# include("Exchange.jl")
# include("BoundaryConditions.jl")
# include("Streaming.jl")
# include("I_O.jl")
# include("Utilities.jl")



# using .Parameters
# using .Geometry
# using .Partitioning
# using .Initialization
# using .ComputeMacro
# using .Collision
# using .Exchange
# using .BoundaryConditions
# using .Streaming
# using .I_O
# using .Utilities
# using MPI
# using CUDA

# export Parameters, main_simulation, I_O, Geometry, Partitioning, main_simulation_HPC




# function main_simulation_HPC(comm::MPI.Comm, rank::Int64, size::Int64)
    
#     nworkers = size - 1

#     # Each rank (master and workers) read input file into parameters struct and initialize simulation parameters
#     params = read_params_input(joinpath(dirname(@__DIR__), "input", "params.txt"));
#     cs2 = Float32(1//3)
#     num_snaps = ceil(Int, params.n_max / params.n_save) # For I/O saving



#     # ---- HOST SETUP ----
#     # Initialize simulation setup on master process, and send local data to each worker
#     if rank == 0
#         N_local, N_owned, omega = 0, 0, 0f0
#         mask_s, mask_bb, partitions, xyz_to_linear =
#             initialize_distributed(comm, rank, nworkers, params)
            
#         owned_idxs = Int[]
#         rho_gpu, u_gpu, v_gpu, w_gpu = CuArray{Float32}(undef, N_local), CuArray{Float32}(undef, N_local), CuArray{Float32}(undef, N_local), CuArray{Float32}(undef, N_local)
#         f_old_gpu, f_new_gpu = CuArray{Float32}(undef, params.q, N_local), CuArray{Float32}(undef, params.q, N_local)
#         c_x_gpu, c_y_gpu, c_z_gpu = CuArray{Int8}(undef, params.q), CuArray{Int8}(undef, params.q), CuArray{Int8}(undef, params.q)
#         wvec_gpu, bbOpp_gpu = CuArray{Float32}(undef, params.q), CuArray{Int8}(undef, params.q)

#         send_buffer_gpu = CuArray{Float32}(undef, 0)
#         recv_buffer_gpu = CuArray{Float32}(undef, 0)
#         # Dummy halo‐exchange metadata
#         gather_qdirs   = CuArray{Int8}(undef, N_local)
#         gather_lidxs   = CuArray{Int32}(undef, N_local)
#         gather_ptr     = Array{Int64}(undef, N_local)
#         scatter_qdirs  = CuArray{Int8}(undef, N_local)
#         scatter_lidxs  = CuArray{Int32}(undef, N_local)
#         neighs         = Int[]  # neighbor ranks
#         TAG            = 0      # can be any Int

#         # Dummy streaming indices
#         stream_lidxs       = CuArray{Int32}(undef, N_local)
#         stream_int_tids    = CuArray{Int32}(undef, N_local)
#         stream_halo_tids   = CuArray{Int32}(undef, N_local)

#         # Dummy BC indices
#         per_dest      = CuArray{Int32}(undef, N_local)
#         per_dir       = CuArray{Int8}(undef, N_local)
#         per_src       = CuArray{Int32}(undef, N_local)
#         per_int_tids  = CuArray{Int32}(undef, N_local)
#         per_halo_tids = CuArray{Int32}(undef, N_local)

#         vel_lidxs    = CuArray{Int32}(undef, N_local)
#         vel_dirs     = CuArray{Int8}(undef, N_local)
#         pres_lidxs   = CuArray{Int32}(undef, N_local)
#         pres_dirs    = CuArray{Int8}(undef, N_local)

#         bb_lidxs      = CuArray{Int32}(undef, N_local)
#         bb_dirs_local = CuArray{Int8}(undef, N_local)

#         # Save partition figure once (just master)
#         output_path = joinpath(dirname(@__DIR__), "output", "partition_plot.png")
#         if (nworkers > 1)
#             plot_mask_3D(mask_s, mask_bb, params, partitions, xyz_to_linear; savepath=output_path)
#         end
#         print("I'm rank $rank, and tau = $(params.tau).\n")
#     else
#         c_x, c_y, c_z, w, bbOpp,
#         local_data, owned_idxs, mask_s_local, mask_bb_local,
#         linear_to_xyz_local, xyz_to_linear_local,
#         global_to_local,
#         bb_ptr_local, bb_dirs_local, bb_nodes_local, bb_lidxs,
#         halo_nodes, halo_dirs, halo_ptr,
#         neighs, part_ptr,
#         gather_qdirs, gather_lidxs, gather_ptr,
#         scatter_qdirs, scatter_lidxs,
#         stream_lidxs, stream_int_tids, stream_halo_tids,
#         vel_lidxs, vel_dirs, pres_lidxs, pres_dirs,
#         per_dest, per_dir, per_src, per_int_tids, per_halo_tids = 
#             initialize_distributed(comm, rank, nworkers, params)   
#     end


#     # ---- MAIN CODE ----
#     if (rank != 0)

#         print("Rank $rank entered main code.\n")

#         # ---- GPU SETUP ----
#         # Local parameters
#         N_owned = length(owned_idxs) # Number of owned nodes of current subdomain
#         N_halo  = length(halo_nodes) # Number of halo nodes of current subdomain
#         N_local = N_owned + N_halo   # Total number of nodes (owned + halo) of current subdomain
#         omega = params.dt/params.tau # Relaxation parameter

#         # Pair GPU & rank, and allocate memory of GPU arrays per each worker
#         f_old_gpu, f_new_gpu, rho_gpu, u_gpu, v_gpu, w_gpu = worker_alloc(N_owned, N_halo, params.q, rank)
#         c_x_gpu, c_y_gpu, c_z_gpu = CuArray(Int8.(c_x)), CuArray(Int8.(c_y)), CuArray(Int8.(c_z))
#         wvec_gpu = CuArray(w)
#         bbOpp_gpu = CuArray(bbOpp)

#         total_halo = Int32(length(gather_lidxs))
#         send_buffer_gpu = CuArray{Float32}(undef, total_halo)
#         recv_buffer_gpu = CuArray{Float32}(undef, total_halo)
#         TAG = 0

#         local_to_global = Dict(v=>k for (k,v) in global_to_local)
#         # f_old_test = Array{Float32}(undef, params.q, N_local)
#         # f_new_test = Array{Float32}(undef, params.q, N_local)

#         # for j = 1:N_local, i = 1:params.q
#         #     f_old_test[i,j] = local_to_global[j]*1000 + i
#         #     f_new_test[i,j] = local_to_global[j]*1000 + i
#         # end

#         # f_old_gpu = CuArray(f_old_test)
#         # f_new_gpu = CuArray(f_new_test)

#         N_tids_int = length(stream_int_tids)
#         N_tids_halo = length(stream_halo_tids)

#         # print("\nI am rank $rank and my owned nodes are:\n$(owned_idxs)\nTheir local indices are:\n$([global_to_local[owned_idxs[i]] for i in 1:length(owned_idxs)])\nMy 3D global indices are: $([linear_to_xyz_local[owned_idxs[i]] for i in 1:length(owned_idxs)])\nMy halo nodes are:\n$(halo_nodes)\nTheir local indices are:\n$([global_to_local[halo_nodes[i]] for i in 1:length(halo_nodes)])\nMy 3D global indices are: $([linear_to_xyz_local[halo_nodes[i]] for i in 1:length(halo_nodes)])\nMy gather_lidxs are:\n$(gather_lidxs)\nGather GLOBAL indices:\n$([local_to_global[glidx] for glidx in Array(gather_lidxs)])\nand my gather_qdirs are:\n$(gather_qdirs)\nMy gather_ptr is:\n$(Array(gather_ptr))\nMy scatter_lidxs are:\n$(scatter_lidxs)\nand my scatter_qdirs are:\n$(scatter_qdirs)\nScatter GLOBAL indices:\n$([local_to_global[slidx] for slidx in Array(scatter_lidxs)])\nThe stream_lidxs is:\n$(Array(stream_lidxs))\nAnd in global stream_lidxs:\n$([local_to_global[sidx] for sidx in Array(stream_lidxs)])\n\n\n")

        
#         # ---- INITIALIZE POPULATION ARRAYS ----
#         initialize_f(
#             f_old_gpu, f_new_gpu,
#             c_x_gpu, c_y_gpu, c_z_gpu,
#             wvec_gpu, params.rho_inf,
#             params.U_inf, params.V_inf, params.W_inf,
#             N_owned, N_halo, params.q
#         )
#         print("Rank $rank finished initialization.\n")

#         # ---- COMPUTE MACRO QUANTITIES - ITERATION 0
#         compute_macros!(
#             f_old_gpu,
#             rho_gpu, u_gpu, v_gpu, w_gpu,
#             c_x_gpu, c_y_gpu, c_z_gpu, rank
#         )

#     end


#     # ---- MAIN LBM ALGORITHM
#     for n = 1:params.n_max

#         # ---- STAGE 1: pre-collision ----
#         #save_full_f!(comm, f_old_gpu, owned_idxs, n, 1, "old", nworkers)
#         #save_full_f!(comm, f_new_gpu, owned_idxs, n, 1, "new", nworkers)
        
#         # ---- COLLISION (+COMPUTE feq) ----
#         bgk_collision!(
#             f_old_gpu, f_new_gpu,
#             rho_gpu,
#             u_gpu, v_gpu, w_gpu,
#             c_x_gpu, c_y_gpu, c_z_gpu,
#             wvec_gpu,
#             omega, N_owned, rank
#         )


#         # ---- STAGE 2: post-collision, pre-halo-exchange ----
#         #save_full_f!(comm, f_old_gpu, owned_idxs, n, 2, "old", nworkers)
#         #save_full_f!(comm, f_new_gpu, owned_idxs, n, 2, "new", nworkers)

#         # ---- BEGIN HALO EXCHANGE ----
#         send_reqs, recv_reqs = begin_halo_exchange!(
#             comm,
#             f_new_gpu, send_buffer_gpu, recv_buffer_gpu,
#             gather_qdirs, gather_lidxs, gather_ptr,
#             neighs, TAG, rank
#         )

#         # ---- FINISH HALO EXCHANGE ----
#         finish_halo_exchange!(
#             send_reqs, recv_reqs,
#             recv_buffer_gpu,
#             scatter_qdirs, scatter_lidxs,
#             f_new_gpu, rank
#         )

#         # ---- STAGE 3: post-halo-exchange, pre-streaming ----
#         #save_full_f!(comm, f_old_gpu, owned_idxs, n, 3, "old", nworkers)
#         #save_full_f!(comm, f_new_gpu, owned_idxs, n, 3, "new", nworkers)

        
#         # # ---- PERIODIC BC ----
#         # periodic_BC!(
#         #     f_new_gpu,
#         #     per_dest,
#         #     per_dir,
#         #     per_src,
#         #     rank
#         # )

#         # ---- STREAMING ----
#         streaming!(
#             f_new_gpu,
#             f_old_gpu,
#             stream_lidxs,
#             params.q,
#             N_owned,
#             rank
#         )

#         # ---- STAGE 4: post-streaming, pre-BC ----
#         #save_full_f!(comm, f_old_gpu, owned_idxs, n, 4, "old", nworkers)
#         #save_full_f!(comm, f_new_gpu, owned_idxs, n, 4, "new", nworkers)


#         # ---- BOUNCEBACK BC ----
#         bounceback_BC!(
#             f_old_gpu,
#             f_new_gpu,
#             bb_lidxs,
#             bb_dirs_local,
#             bbOpp_gpu,
#             rank
#         )

#         # ---- APPLY VELOCITY BC AT FACES ----
#         velocity_BC!(
#             f_old_gpu,
#             f_new_gpu,
#             vel_lidxs, vel_dirs,
#             pres_lidxs, pres_dirs,
#             bbOpp_gpu,
#             c_x_gpu, c_y_gpu, c_z_gpu,
#             wvec_gpu,
#             rho_gpu, u_gpu, v_gpu, w_gpu,
#             params.U_inf, params.V_inf, params.W_inf, 
#             cs2, rank
#         )

#         # ---- COMPUTE MACRO QUANTITIES ----
#         compute_macros!(
#             f_old_gpu,
#             rho_gpu, u_gpu, v_gpu, w_gpu,
#             c_x_gpu, c_y_gpu, c_z_gpu,
#             rank
#         )


#         if (rank == 1)
#             print("$n\n")
#         end


#         # ---- SAVE TO HOST EVERY FEW TIME STEPS ----
#         write_fields_h5_time!(
#             comm, rank,
#             owned_idxs,
#             Array(rho_gpu), Array(u_gpu),
#             Array(v_gpu), Array(w_gpu),
#             params.Nx, params.Ny, params.Nz,
#             "output/final_fields.h5",
#             n, params.n_save, num_snaps
#         )

#     end # END OF TIME STEPPING LOOP

#     return nothing

# end




# end # module
