module Initialization

using CUDA
using ..SolidBC
using ..Collision: CS2, INV_CS2, INV_CS4

export initialize_f, worker_alloc, ensure_bb_capacity!, init_bb_buffers!


# GPU kernel: initialize the equilibrium LBM distribution f (feq) for each direction and node.
function initialize_f_gpu(
    f, rho_inf::Float32, U_inf::Float32, V_inf::Float32, W_inf::Float32,
    c_x::CuDeviceVector{Int8}, c_y::CuDeviceVector{Int8}, c_z::CuDeviceVector{Int8},
    w::CuDeviceVector{Float32},
    q::Int32, N_tot::Int32,
    rand_node::CuDeviceVector{Float32}
)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    qI  = Int(q)
    Nt  = qI * Int(N_tot)

    if tid <= Nt
        i0   = tid - 1
        node = (i0 ÷ qI) + 1              # 1..N_tot
        dir  = (i0 % qI) + 1              # 1..q

        # 20% multiplicative disturbance per node ([-1,1] already)
        r  = rand_node[node]
        fac = 0.2f0
        ux = U_inf * (1f0 + fac*r)
        uy = V_inf * (1f0 + fac*r)
        uz = W_inf * (1f0 + fac*r)

        # promote lattice velocities once
        cx = Float32(c_x[dir]); cy = Float32(c_y[dir]); cz = Float32(c_z[dir])

        cu  = cx*ux + cy*uy + cz*uz
        u2  = ux*ux + uy*uy + uz*uz
        cu2 = cu*cu

        feq = w[dir] * rho_inf *
              (1f0 + cu*INV_CS2 + 0.5f0*cu2*INV_CS4 - 0.5f0*u2*INV_CS2)

        @inbounds f[dir, node] = feq
    end
    return
end


#=
Allocate and initialize the LBM distribution arrays `f_old_gpu` and `f_new_gpu`.
`f_old_gpu` is filled with the equilibrium distribution for density `rho_inf` and velocity `(u_inf, v_inf, w_inf)`. `f_new_gpu` is a copy of `f_old_gpu`.

Returns:
- `f_old_gpu`, `f_new_gpu`: CuArray{Float32,2} of size (q, N_owned+N_halo)
=#
function initialize_f(
    f_old_gpu::CuArray{Float32,2},
    f_new_gpu::CuArray{Float32,2},
    c_x_gpu::CuArray{Int8,1},
    c_y_gpu::CuArray{Int8,1},
    c_z_gpu::CuArray{Int8,1},
    wvec_gpu::CuArray{Float32,1},
    rho_inf::Float32, U_inf::Float32, V_inf::Float32, W_inf::Float32,
    N_owned::Int, N_halo::Int, q::Int
)
    N_tot = N_owned + N_halo

    # precompute randomness ON GPU (ok on host side)
    rand_node = 2f0 .* CUDA.rand(Float32, N_tot) .- 1f0

    threads = 256
    blocks  = cld(q * N_tot, threads)
    @cuda threads=threads blocks=blocks initialize_f_gpu(
        f_old_gpu, rho_inf, U_inf, V_inf, W_inf,
        c_x_gpu, c_y_gpu, c_z_gpu, wvec_gpu,
        Int32(q), Int32(N_tot),
        rand_node
    )

    copy!(f_new_gpu, f_old_gpu)
    return f_old_gpu, f_new_gpu
end




# Initialize a worker process
function worker_alloc(N_owned::Int64, N_halo::Int64, q::Int64, rank::Int64)
    
    # Set CUDA device for current worker
    CUDA.device!(rank-1)
    dev = CUDA.device()

    # Allocate memory on GPU
    f_old = CUDA.zeros(Float32, q, (N_owned + N_halo))
    f_new = CUDA.zeros(Float32, q, (N_owned + N_halo))
    rho   = CUDA.zeros(Float32, N_owned)
    u     = CUDA.zeros(Float32, N_owned)
    v     = CUDA.zeros(Float32, N_owned)
    w     = CUDA.zeros(Float32, N_owned)

    return (f_old, f_new, rho, u, v, w)
end




function init_bb_buffers!(
    bcstruct,
    N_owned::Integer,
    N_halo::Integer;
    init_cap::Integer = 8 * N_owned
)
    N_local = N_owned + N_halo
    cap     = max(init_cap, 1)

    # --- masks & scratch (returned) ---
    mask_s_local  = CuArray{Bool}(undef, N_local)
    mask_bb_local = CuArray{Bool}(undef, N_owned)
    cnt           = CuArray{Int32}(undef, N_owned)

    # --- BB metadata (mutate bcstruct fields) ---
    bcstruct.bb_ptr   = CuArray{Int32}(undef, N_owned + 1)
    bcstruct.bb_dirs  = CuArray{Int8}(undef,  cap)
    bcstruct.bb_lidxs = CuArray{Int32}(undef, cap)
    bcstruct.nnz      = Int32(0)

    return mask_s_local, mask_bb_local, cnt
end




function ensure_bb_capacity!(
    bcstruct::SolidBCStruct,
    rank::Int,
    needed::Integer;
    slack::Float64 = 1.2,
    growth::Int    = 2
)
    cap = length(bcstruct.bb_dirs)
    if needed <= cap
        #print("\nRank $rank, no need to increase cap.\n")
        return cap
    end
    newcap = max(Int(ceil(slack * needed)), growth * cap, 1)
    bcstruct.bb_dirs  = CuArray{Int8}(undef,  newcap)
    bcstruct.bb_lidxs = CuArray{Int32}(undef, newcap)
    print("\nRank $rank, cap needs to be increased. Old one was $cap, new one is $newcap.\n")
    return newcap
end




end # module Initialization
