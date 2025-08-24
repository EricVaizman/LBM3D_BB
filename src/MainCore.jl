module MainCore
# The MainCore.jl module includes the main simulation function, and incorporates together all the other functionalities of the project.
# This is the main module of the project.

include("Parameters.jl")
include("Collision.jl")
include("SolidBC.jl")
include("Geometry.jl")
include("Partitioning.jl")
include("Initialization.jl")
include("ComputeMacro.jl")
include("Exchange.jl")
include("OtherBC.jl")
include("Streaming.jl")
include("I_O.jl")
include("Utilities.jl")
include("MEA.jl")
include("Motion.jl")



using .Parameters
using .Collision
using .SolidBC
using .Geometry
using .Partitioning
using .ComputeMacro
using .Exchange
using .OtherBC
using .Initialization
using .Streaming
using .I_O
using .Utilities
using .MEA
using .Motion
using MPI
using CUDA

export Parameters, main_simulation, I_O, Geometry, Partitioning, main_simulation_HPC




#=
main_simulation_HPC runs the simulation setup of main host & device workers, generating local data per worker, and conducting the time loop.
Finally, results are written to file and the function returns.
=#
function main_simulation_HPC(comm::MPI.Comm, rank::Int64, size::Int64)
    
    nworkers = size - 1

    # BEGIN MEASURING TOTAL RUN TIME
    MPI.Barrier(comm)
    if rank == 0
        t_ini_start = time_ns()
    end

    # Each rank (master and workers) read input file into parameters struct and initialize simulation parameters
    params = read_params_input(joinpath(dirname(@__DIR__), "input", "params.txt"));
    Nx, Ny, Nz, h, R, x_c, y_c, theta0,
    q, Re, U_inf, V_inf, W_inf, rho_inf, dt, Lambda,
    n_max, n_save, omega_p, omega_m,
    BC, coll, shape, naca_code, chord =
        params.Nx, params.Ny, params.Nz, params.h, params.R, params.x_c, params.y_c, params.theta0,
        params.q, params.Re, params.U_inf, params.V_inf, params.W_inf, params.rho_inf, params.dt, params.Lambda,
        params.n_max, params.n_save, params.omega_p, params.omega_m,
        params.bc, params.coll, params.shape, params.naca_code, params.chord
    num_snaps = ceil(Int, n_max/n_save) # For I/O saving


    # Setup motion law
    Ae = 0.2f0 * 2*R
    fe = 0.154f0 * U_inf / (2*R)
    we = 2f0*pi * fe
    
    mot_law = MotionLaw(
        t -> x_c,                    t -> 0f0,                 # x, xdot
        t -> y_c - Ae*sin(we*t),     t -> -Ae*we*cos(we*t),    # y, ydot
        t -> theta0,                 t -> 0f0                  # theta, thetadot
    )

    # k = 0.25f0
    # U_phys = U_inf * h / dt
    # f      = k * U_phys / (π * chord)           # Hz
    # ω      = 2f0 * π * f
    # alpha_0     = deg2rad(15f0)          # mean pitch [rad]
    # alpha_1     = deg2rad(10f0)          # amplitude  [rad]

    # mot_law = MotionLaw(
    #     t -> x_c,                  t -> 0f0,              # x, xdot
    #     t -> y_c,                  t -> 0f0,              # y, ydot
    #     t -> alpha_0 + alpha_1*cos(ω*t),       t -> -alpha_1*ω*sin(ω*t)    # theta, thetadot
    # )

    # mot_law = MotionLaw(
    #     t -> x_c,     t -> 0f0,               
    #     t -> y_c,     t -> 0f0,    
    #     t -> 0f0,     t -> 0f0                
    # )



    # ---- HOST SETUP ----
    # Initialize simulation setup on master process, and send local data to each worker
    if rank == 0
        N_local, N_owned, N_halo, N_bb = 0, 0, 0, 0
        A_ref_lat = ref_area_lat(shape, R, chord, Nz, h) # Lattice surface units, for an infinite body in the Z direction
        fc_aero = 1f0 / (0.5f0 * rho_inf * U_inf^2 * A_ref_lat) # Coefficient for normalizing MEA

        # Real & empty dummy data for master
        partitions, xyz_to_linear,
        ix_lli, iy_lli, iz_lli,
        x_lli, y_lli, nbr_lli,
        owned_idxs, halo_nodes,
        rho_gpu, u_gpu, v_gpu, w_gpu,
        f_old_gpu, f_new_gpu,
        c_x, c_y, c_z,
        wvec_gpu, bbOpp_gpu,
        send_buffer_gpu, recv_buffer_gpu,
        gather_qdirs, gather_lidxs, gather_ptr,
        scatter_qdirs, scatter_lidxs, neighs, TAG,
        stream_lidxs, stream_int_tids, stream_halo_tids,
        per_dest, per_src, per_dir, per_int_tids, per_halo_tids, per_mask,
        vel_lidxs, vel_dirs, pres_lidxs, pres_dirs,
        CL_gpu, CD_gpu, CZ_gpu,
        bcstruct =
            initialize_distributed(comm, rank, nworkers, params)

        c_x_gpu, c_y_gpu, c_z_gpu = CuArray(Int8.(c_x)), CuArray(Int8.(c_y)), CuArray(Int8.(c_z))

        mask_s_local, mask_bb_local, cnt =
            init_bb_buffers!(bcstruct, 0, 0; init_cap = 0)

        state = init_solid_mask_state(params)

        x_lli_lat = x_lli ./ h
        y_lli_lat = y_lli ./ h

        # Save partition figure once (just master)
        output_path = joinpath(dirname(@__DIR__), "output", "partition_plot.png")
        #mask_s_tot, mask_bb_tot = generate_plotmask(params, Array(c_x_gpu), Array(c_y_gpu), Array(c_z_gpu))
        #plot_mask_3D(mask_s_tot, mask_bb_tot, params, partitions, xyz_to_linear; savepath=output_path)
    
    else

        # Different return pattern for worker (local data)
        c_x, c_y, c_z, w, bbOpp,
        local_data, owned_idxs,
        ix_lli, iy_lli, iz_lli,
        x_lli, y_lli, nbr_lli,
        halo_nodes, halo_dirs, halo_ptr,
        neighs, part_ptr,
        gather_qdirs, gather_lidxs, gather_ptr,
        scatter_qdirs, scatter_lidxs,
        stream_lidxs, stream_int_tids, stream_halo_tids,
        vel_lidxs, vel_dirs, pres_lidxs, pres_dirs,
        per_dest, per_dir, per_src, per_int_tids, per_halo_tids, per_mask,
        bcstruct = 
            initialize_distributed(comm, rank, nworkers, params)
            
    end


    # ---- MAIN CODE ----
    if (rank != 0)

        print("Rank $rank entered main code.\n")

        # Local parameters
        N_owned = length(owned_idxs)     # Number of owned nodes of current subdomain
        N_halo  = length(halo_nodes)     # Number of halo nodes of current subdomain
        N_local = N_owned + N_halo       # Total number of nodes (owned + halo) of current subdomain

        # Aerodynamic coefficient calculation helper factors
        A_ref_lat = ref_area_lat(shape, R, chord, Nz, h) # Lattice surface units, for an infinite body in the Z direction
        fc_aero = 1f0 / (0.5f0 * rho_inf * U_inf^2 * A_ref_lat) # Coefficient for normalizing MEA


        # ---- GPU WORKER SETUP ----
        # Pair GPU & rank, and allocate memory of GPU arrays per each worker
        f_old_gpu, f_new_gpu, rho_gpu, u_gpu, v_gpu, w_gpu = worker_alloc(N_owned, N_halo, q, rank)
        c_x_gpu, c_y_gpu, c_z_gpu = CuArray(Int8.(c_x)), CuArray(Int8.(c_y)), CuArray(Int8.(c_z))
        wvec_gpu = CuArray(w)
        bbOpp_gpu = CuArray(bbOpp)
        bcstruct.bbOpp = bbOpp_gpu

        total_halo = Int32(length(gather_lidxs))
        send_buffer_gpu = CuArray{Float32}(undef, total_halo)
        recv_buffer_gpu = CuArray{Float32}(undef, total_halo)
        TAG = 0

        # Initialize BB-metadata CuArrays on worker
        mask_s_local, mask_bb_local, cnt =
            init_bb_buffers!(bcstruct, N_owned, N_halo; init_cap = 8*N_owned)

        # Initialize solid mask state
        state = init_solid_mask_state(params)

        x_lli_lat = x_lli ./ h
        y_lli_lat = y_lli ./ h


        # Dedicate empty worker arrays for coefficient saving
        CL_gpu = CUDA.zeros(Float32, 0)
        CD_gpu = CUDA.zeros(Float32, 0)
        CZ_gpu = CUDA.zeros(Float32, 0)

   
        # ---- INITIALIZE POPULATION ARRAYS ----
        initialize_f(
            f_old_gpu, f_new_gpu,
            c_x_gpu, c_y_gpu, c_z_gpu,
            wvec_gpu, rho_inf,
            U_inf, V_inf, W_inf,
            N_owned, N_halo, q
        )

        print("Rank $rank finished initialization.\n")

        # ---- COMPUTE MACRO QUANTITIES - ITERATION 0
        compute_macros!(
            f_old_gpu,
            rho_gpu, u_gpu, v_gpu, w_gpu,
            c_x_gpu, c_y_gpu, c_z_gpu, rank
        )

    end

    # BEGIN MEASURING COMPUTATION RUN TIME
    MPI.Barrier(comm)
    if rank == 0
        t_master_start = time_ns()
        #print("\nColl method: $coll, omega_p = $omega_p, omega_m = $omega_m, Lambda = $Lambda \n")
    end


    if (rank != 0)
        #print("I'm rank $rank.\nMy owned nodes in their global indices are: $owned_idxs\nand in their local indices: $([global_to_local[owned_idxs[i]] for i in 1:length(owned_idxs)])\nMy halo nodes in their global indices are: $halo_nodes\nand in their local indices: $([global_to_local[halo_nodes[i]] for i in 1:length(halo_nodes)])\n\n")
        #print("\n\n\n\nI'm rank $rank. My ix_lli, iy_lli, iz_lli are:\n$(ix_lli)\n$(iy_lli)\n$(iz_lli)\nMy x_lli, y_lli are:\n$(x_lli)\n$(y_lli)\nMy nbr_lli is:\n$(nbr_lli)\n\n\n\n")
    end

    # ----------------------------
    # ---- MAIN LBM ALGORITHM ----
    # ----------------------------
    for n = 1:n_max

        # ---- MOTION (UPDATE CHANGE IN BOUNDARY) ----
        motion!(
            x_lli, y_lli, nbr_lli, state,
            mask_s_local, mask_bb_local, cnt,
            R, x_c, y_c,
            dt, h,
            bcstruct, mot_law,
            n, rank
        )


        # ---- COLLISION (+COMPUTE feq) ----
        collision!(
            f_old_gpu, f_new_gpu,
            rho_gpu,
            u_gpu, v_gpu, w_gpu,
            c_x_gpu, c_y_gpu, c_z_gpu,
            wvec_gpu, bbOpp_gpu,
            omega_p, omega_m,
            coll, 
            N_owned, rank
        )


        # ---- BEGIN HALO EXCHANGE ----
        send_reqs, recv_reqs = begin_halo_exchange!(
            comm,
            f_new_gpu, send_buffer_gpu, recv_buffer_gpu,
            gather_qdirs, gather_lidxs, gather_ptr,
            neighs, TAG, rank
        )


        # ---- FINISH HALO EXCHANGE ----
        finish_halo_exchange!(
            send_reqs, recv_reqs,
            recv_buffer_gpu,
            scatter_qdirs, scatter_lidxs,
            f_new_gpu, rank
        )


        # ---- STREAMING ----
        streaming!(
            f_new_gpu,
            f_old_gpu,
            stream_lidxs,
            per_mask,
            q,
            N_owned,
            rank
        )


        # ---- PERIODIC BC ----
        periodic_BC!(
            f_old_gpu,
            f_new_gpu,
            per_dest,
            per_dir,
            per_src,
            rank
        )


        # ---- SOLID BOUNDARY BC ----
        solid_BC!(
            f_old_gpu,
            f_new_gpu,
            rho_gpu,           # local density per node (lattice)
            wvec_gpu,          # weights length q
            c_x_gpu, c_y_gpu,  # lattice directions
            x_lli_lat, y_lli_lat,  # node coords in lattice units (precompute once)
            bcstruct,
            params,
            BC,
            rank
        )


        # ---- APPLY VELOCITY BC AT FACES ----
        velocity_BC!(
            f_old_gpu,
            f_new_gpu,
            vel_lidxs, vel_dirs,
            pres_lidxs, pres_dirs,
            bbOpp_gpu,
            c_x_gpu, c_y_gpu, c_z_gpu,
            wvec_gpu,
            rho_gpu, u_gpu, v_gpu, w_gpu,
            U_inf, V_inf, W_inf, 
            rank
        )


        # ---- COMPUTE AERODYNAMIC QUANTITIES ----
        compute_aero!(
            n, n_save, n_max,
            f_new_gpu, f_old_gpu,
            bcstruct,
            bbOpp_gpu, c_x_gpu, c_y_gpu, c_z_gpu,
            CL_gpu, CD_gpu, CZ_gpu,
            fc_aero,
            comm, rank
        )


        # ---- COMPUTE MACRO QUANTITIES ----
        compute_macros!(
            f_old_gpu,
            rho_gpu, u_gpu, v_gpu, w_gpu,
            c_x_gpu, c_y_gpu, c_z_gpu,
            rank
        )


        # Print iteration number
        if (rank == 1)
            print("$n\n")
        end

        # ---- SAVE TO HOST EVERY n_save TIME STEPS ----
        write_fields_h5_time!(
            comm, rank,
            owned_idxs,
            Array(rho_gpu), Array(u_gpu),
            Array(v_gpu), Array(w_gpu),
            Nx, Ny, Nz,
            "output/final_fields.h5",
            n, n_save, num_snaps
        )

    end # END OF TIME STEPPING LOOP


    # RUNTIME DIAGNOSTICS
    MPI.Barrier(comm)
    if rank == 0
        t_master_end = time_ns()
        measure_Mlups(
            t_master_start, t_master_end, t_ini_start,
            Nx, Ny, Nz,
            n_max
        )
    end


    return nothing
end




end # module