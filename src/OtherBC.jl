module OtherBC

using CUDA
using ..Collision: CS2, INV_CS2, INV_CS4

export velocity_BC!, periodic_BC!, periodic_BC_part!




function periodic_BC_part!_gpu(
    f_post    :: CuDeviceMatrix{Float32},
    per_dest  :: CuDeviceVector{Int32},
    per_dir   :: CuDeviceVector{Int8},
    per_src   :: CuDeviceVector{Int32},
    tids      :: CuDeviceVector{Int32},
    N_tids    :: Int32
)
    th = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if th ≤ N_tids
        tid = Int(tids[th])       # which entry in per_dest/per_src/per_dir
        dest = Int(per_dest[tid])
        src  = Int(per_src[tid])
        dir  = Int(per_dir[tid])
        @inbounds f_post[dir, dest] = f_post[dir, src]
    end
    return
end




# Host wrapper: call partial periodic BC on the given tid‐list
function periodic_BC_part!(
    f_post_gpu     :: CuArray{Float32,2},
    per_dest_gpu   :: CuArray{Int32,1},
    per_dir_gpu    :: CuArray{Int8,1},
    per_src_gpu    :: CuArray{Int32,1},
    periodic_tids  :: CuArray{Int32,1}
    )
    
    N_tids  = Int32(length(periodic_tids))

    # In case no periodic tids in my current subdomain
    if N_tids == 0
        return        # nothing to do
    end

    threads = 256
    blocks  = cld(N_tids, threads)
    @cuda threads=threads blocks=blocks periodic_BC_part!_gpu(
        f_post_gpu,
        per_dest_gpu,
        per_dir_gpu,
        per_src_gpu,
        periodic_tids,
        N_tids
    )
    return nothing
end




function periodic_BC!_gpu(
    f_old     :: CuDeviceMatrix{Float32},  # post-stream buffer
    f_new     :: CuDeviceMatrix{Float32},
    per_dest  :: CuDeviceVector{Int32},    # where to write
    per_dir   :: CuDeviceVector{Int8},     # which direction
    per_src   :: CuDeviceVector{Int32},    # where to read from
    N_per     :: Int32                     # length of lists
    )

    tid = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if tid ≤ N_per
        dest = Int(per_dest[tid])
        src  = Int(per_src[tid])
        dir  = Int(per_dir[tid])
        @inbounds f_old[dir, dest] = f_new[dir, src]
    end
    return
end




function periodic_BC!(
    f_old_gpu    :: CuArray{Float32,2},
    f_new_gpu    :: CuArray{Float32,2},
    per_dest_gpu :: CuArray{Int32,1},
    per_dir_gpu  :: CuArray{Int8,1},
    per_src_gpu  :: CuArray{Int32,1},
    rank::Int
    )

    if (rank == 0)
        return
    end

    N_per   = Int32(length(per_dest_gpu))
    threads = 256
    blocks  = cld(N_per, threads)
    @cuda threads=threads blocks=blocks periodic_BC!_gpu(
        f_old_gpu, f_new_gpu, per_dest_gpu, per_dir_gpu, per_src_gpu, N_per
    )
    return nothing
end




function velocity_BC!_gpu(
    f_old            ::CuDeviceMatrix{Float32},
    f_new            ::CuDeviceMatrix{Float32},
    vel_lidxs        ::CuDeviceVector{Int32},
    vel_dirs         ::CuDeviceVector{Int8},
    pres_lidxs       ::CuDeviceVector{Int32},
    pres_dirs        ::CuDeviceVector{Int8},
    bbOpp            ::CuDeviceVector{Int8},
    c_x              ::CuDeviceVector{Int8},
    c_y              ::CuDeviceVector{Int8},
    c_z              ::CuDeviceVector{Int8},
    wvec             ::CuDeviceVector{Float32},
    rho              ::CuDeviceVector{Float32},
    u                ::CuDeviceVector{Float32},
    v                ::CuDeviceVector{Float32},
    w                ::CuDeviceVector{Float32},
    U_inf            ::Float32,
    V_inf            ::Float32,
    W_inf            ::Float32,
    N_vel            ::Int32,
    N_pres           ::Int32
    )

    tid = (blockIdx().x-1)*blockDim().x + threadIdx().x

    #— moving‐wall (Zou–He) on vel faces
    if tid ≤ N_vel
        i   = Int(vel_lidxs[tid])
        dir = Int(vel_dirs[tid])
        opp = Int(bbOpp[dir])
        rho_l = rho[i]
        cu = Float32(c_x[dir])*(-U_inf) +
             Float32(c_y[dir])*(-V_inf) +
             Float32(c_z[dir])*(-W_inf)
        w_i = wvec[dir]
        @inbounds f_old[dir,i] = f_new[opp,i] -
                                2f0 * w_i * rho_l * cu * INV_CS2
    end

    #— pressure (anti–bounce‐back) on outlet
    if tid ≤ N_pres
        i   = Int(pres_lidxs[tid])
        dir = Int(pres_dirs[tid])
        opp = Int(bbOpp[dir])
        rho_l = rho[i]
        w_i = wvec[dir]
        uw = u[i]; vw = v[i]; ww = w[i]
        cu = c_x[dir]*uw + c_y[dir]*vw + c_z[dir]*ww
        cu2 = cu*cu
        uu2 = uw*uw + vw*vw + ww*ww

        @inbounds f_old[dir,i] = -f_new[opp,i] +
               2f0 * w_i * rho_l *
               (1f0 + 0.5f0 * cu2 * INV_CS4 - 0.5f0 * uu2 * INV_CS2)
    end

    return
end




function velocity_BC!(
    f_old_gpu::CuArray{Float32,2},
    f_new_gpu::CuArray{Float32,2},
    vel_lidxs_gpu::CuArray{Int32,1},
    vel_dirs_gpu::CuArray{Int8,1},
    pres_lidxs_gpu::CuArray{Int32,1},
    pres_dirs_gpu::CuArray{Int8,1},
    bbOpp_gpu::CuArray{Int8,1},
    c_x_gpu::CuArray{Int8,1},
    c_y_gpu::CuArray{Int8,1},
    c_z_gpu::CuArray{Int8,1},
    wvec_gpu::CuArray{Float32,1},
    rho_gpu::CuArray{Float32,1},
    u_gpu::CuArray{Float32,1},
    v_gpu::CuArray{Float32,1},
    w_gpu::CuArray{Float32,1},
    U_inf::Float32,
    V_inf::Float32,
    W_inf::Float32,
    rank::Int
    )

    if (rank == 0)
        return
    end

    N_vel  = Int32(length(vel_lidxs_gpu))
    N_pres = Int32(length(pres_lidxs_gpu))
    total  = max(N_vel, N_pres)
    threads = 256
    blocks  = cld(total, threads)

    @cuda threads=threads blocks=blocks velocity_BC!_gpu(
        f_old_gpu, f_new_gpu,
        vel_lidxs_gpu, vel_dirs_gpu,
        pres_lidxs_gpu, pres_dirs_gpu,
        bbOpp_gpu,
        c_x_gpu, c_y_gpu, c_z_gpu,
        wvec_gpu,
        rho_gpu, u_gpu, v_gpu, w_gpu,
        U_inf, V_inf, W_inf,
        N_vel, N_pres
    )

    return nothing
end




end # module