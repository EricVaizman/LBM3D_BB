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

#         N_tids_int = length(stream_int_tids)
#         N_tids_halo = length(stream_halo_tids)

    
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


#             # ---- STREAMING ----
#             streaming!(
#                 f_new_gpu,
#                 f_old_gpu,
#                 stream_lidxs,
#                 params.q,
#                 N_owned
#             )


#             # ---- PERIODIC BC ----
#             periodic_BC!(
#                 f_old_gpu,
#                 per_dest,
#                 per_dir,
#                 per_src
#             )


#             # # ---- INTERIOR NODES STREAM ----
#             # streaming_part!(
#             #     f_new_gpu,
#             #     f_old_gpu,           
#             #     stream_lidxs,
#             #     stream_int_tids,
#             #     params.q,
#             #     N_owned 
#             # )


#             # # ---- INTERIOR PERIODIC BC ----
#             # periodic_BC_part!(
#             #     f_old_gpu,
#             #     per_dest,
#             #     per_dir,
#             #     per_src,
#             #     per_int_tids 
#             # )


#             # # ---- HALO NODES STREAM ----
#             # streaming_part!(
#             #     f_new_gpu,   
#             #     f_old_gpu,         
#             #     stream_lidxs, 
#             #     stream_halo_tids,
#             #     params.q,
#             #     N_owned
#             # )


#             # # ---- HALO PERIODIC BC ----
#             # periodic_BC_part!(
#             #     f_old_gpu,
#             #     per_dest,
#             #     per_dir,
#             #     per_src,
#             #     per_halo_tids
#             # )

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