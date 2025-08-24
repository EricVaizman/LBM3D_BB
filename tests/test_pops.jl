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

#         print("\nI am rank $rank and my owned nodes are:\n$(owned_idxs)\nTheir local indices are:\n$([global_to_local[owned_idxs[i]] for i in 1:length(owned_idxs)])\nMy 3D global indices are: $([linear_to_xyz_local[owned_idxs[i]] for i in 1:length(owned_idxs)])\nMy halo nodes are:\n$(halo_nodes)\nTheir local indices are:\n$([global_to_local[halo_nodes[i]] for i in 1:length(halo_nodes)])\nMy 3D global indices are: $([linear_to_xyz_local[halo_nodes[i]] for i in 1:length(halo_nodes)])\nMy gather_lidxs are:\n$(gather_lidxs)\nGather GLOBAL indices:\n$([local_to_global[glidx] for glidx in Array(gather_lidxs)])\nand my gather_qdirs are:\n$(gather_qdirs)\nMy gather_ptr is:\n$(Array(gather_ptr))\nMy scatter_lidxs are:\n$(scatter_lidxs)\nand my scatter_qdirs are:\n$(scatter_qdirs)\nScatter GLOBAL indices:\n$([local_to_global[slidx] for slidx in Array(scatter_lidxs)])\n\n\n")#\nThe stream_lidxs is:\n$(Array(stream_lidxs))\nAnd in global stream_lidxs:\n$([local_to_global[sidx] for sidx in Array(stream_lidxs)])\n\n\n")


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
#             # bgk_collision!(
#             #     f_old_gpu, f_new_gpu,
#             #     rho_gpu,
#             #     u_gpu, v_gpu, w_gpu,
#             #     c_x_gpu, c_y_gpu, c_z_gpu,
#             #     wvec_gpu,
#             #     omega, N_owned
#             # )

#             # ---- BEGIN HALO EXCHANGE ----
#             send_reqs, recv_reqs = begin_halo_exchange!(
#                 comm,
#                 f_new_gpu, send_buffer_gpu, recv_buffer_gpu,
#                 gather_qdirs, gather_lidxs, gather_ptr,
#                 neighs, TAG
#             )

#             # ---- INTERIOR NODES STREAM ----
#             # streaming_part!(
#             #     f_new_gpu,
#             #     f_old_gpu,
#             #     stream_lidxs,
#             #     stream_int_tids,
#             #     params.q,
#             #     N_owned
#             # )

#             # ---- INTERIOR PERIODIC BC ----
#             # periodic_BC_part!(
#             #     f_old_gpu,
#             #     per_dest,
#             #     per_dir,
#             #     per_src,
#             #     per_int_tids 
#             # )

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