module Collision

using CUDA

export collision!, CS2, INV_CS2, INV_CS4

const CS2     = 1f0/3f0       # c_s^2
const INV_CS2 = 3f0           # 1/CS2
const INV_CS4 = 9f0           # 1/CS2^2




function collision!(
    f_old::CuArray{Float32,2},
    f_new::CuArray{Float32,2},
    rho::CuArray{Float32,1},
    u::CuArray{Float32,1},
    v::CuArray{Float32,1},
    w::CuArray{Float32,1},
    c_x::CuArray{Int8,1},
    c_y::CuArray{Int8,1},
    c_z::CuArray{Int8,1},
    wvec::CuArray{Float32,1},
    bbOpp::CuArray{Int8,1},
    omega_p::Float32,
    omega_m::Float32,
    coll::String,
    N_owned::Int,
    rank::Int
)
    if haskey(coll_dispatch, coll)
        if coll == "BGK"
            coll_dispatch[coll](f_old, f_new, rho, u, v, w,
                                c_x, c_y, c_z, wvec,
                                omega_p, N_owned, rank)
        elseif coll == "TRT"
            coll_dispatch[coll](f_old, f_new, rho, u, v, w,
                                c_x, c_y, c_z, wvec, bbOpp,
                                omega_p, omega_m, N_owned, rank)
        end
    else
        error("\nUnknown collision method: $coll.\n")
    end
end




const coll_dispatch = Dict{String, Function}(
    "BGK" => (f_old, f_new, rho, u, v, w, c_x, c_y, c_z, wvec,
              omega_p, N_owned, rank) -> BGK_collision!(
        f_old, f_new,
        rho, u, v, w,
        c_x, c_y, c_z, wvec,
        omega_p,
        N_owned, rank
    ),

    "TRT" => (f_old, f_new, rho, u, v, w, c_x, c_y, c_z, wvec, bbOpp,
              omega_p, omega_m, N_owned, rank) -> TRT_collision!(
        f_old, f_new,
        rho, u, v, w,
        c_x, c_y, c_z, wvec,
        bbOpp,
        omega_p, omega_m,
        N_owned, rank
    )
)




function TRT_collision!(
    f_old::CuArray{Float32,2},
    f_new::CuArray{Float32,2},
    rho::CuArray{Float32,1},
    u::CuArray{Float32,1},
    v::CuArray{Float32,1},
    w::CuArray{Float32,1},
    c_x::CuArray{Int8,1},
    c_y::CuArray{Int8,1},
    c_z::CuArray{Int8,1},
    wvec::CuArray{Float32,1},
    bbOpp::CuArray{Int8,1},
    omega_p::Float32,
    omega_m::Float32,
    N_owned::Int,
    rank::Int
)
    if rank == 0
        return
    end

    threads = 256
    blocks  = cld(N_owned, threads)
    Q = Int32(length(wvec))

    @cuda threads=threads blocks=blocks TRT_collision!_gpu(
        f_old, f_new,
        rho, u, v, w,
        c_x, c_y, c_z, wvec, bbOpp,
        omega_p, omega_m,
        Int32(N_owned), Q
    )
    return
end




function TRT_collision!_gpu(
    f_old::CuDeviceMatrix{Float32},
    f_new::CuDeviceMatrix{Float32},
    rho::CuDeviceVector{Float32},
    u::CuDeviceVector{Float32},
    v::CuDeviceVector{Float32},
    w::CuDeviceVector{Float32},
    c_x::CuDeviceVector{Int8},
    c_y::CuDeviceVector{Int8},
    c_z::CuDeviceVector{Int8},
    wvec::CuDeviceVector{Float32},
    bbOpp::CuDeviceVector{Int8},
    omega_p::Float32,
    omega_m::Float32,
    N_owned::Int32,
    Q::Int32
)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if tid > N_owned
        return
    end

    @inbounds begin
        # Macroscopics
        rho_l = rho[tid]
        ux = u[tid];  uy = v[tid];  uz = w[tid]
        u2 = ux*ux + uy*uy + uz*uz

        Qi = Int(Q)  # fixed loop bound

        for d in 1:Qi
            # Opposite index (avoid dynamic widening)
            qopp = Int(Int32(bbOpp[d]))

            # Old populations
            fi     = f_old[d,    tid]
            fi_opp = f_old[qopp, tid]

            # Direction & weight (match BGK style)
            cx = Int32(c_x[d]); cy = Int32(c_y[d]); cz = Int32(c_z[d])
            wi = wvec[d]

            # Projection
            cu  = float(cx)*ux + float(cy)*uy + float(cz)*uz
            cu2 = cu*cu

            feq_plus  = wi * rho_l * (1f0 + 0.5f0 * cu2 * INV_CS4 - 0.5f0 * u2 * INV_CS2)  # even
            feq_minus = wi * rho_l * (cu * INV_CS2)  

            # Current split
            f_plus  = 0.5f0 * (fi + fi_opp)
            f_minus = 0.5f0 * (fi - fi_opp)

            # TRT relaxation
            f_plus_new  = f_plus  - omega_p * (f_plus  - feq_plus)
            f_minus_new = f_minus - omega_m * (f_minus - feq_minus)

            # Recombine
            f_new[d, tid] = f_plus_new + f_minus_new
        end
    end
    return
end




function BGK_collision!(
    f_old::CuArray{Float32,2},
    f_new::CuArray{Float32,2},
    rho::CuArray{Float32,1},
    u::CuArray{Float32,1},
    v::CuArray{Float32,1},
    w::CuArray{Float32,1},
    c_x::CuArray{Int8,1},
    c_y::CuArray{Int8,1},
    c_z::CuArray{Int8,1},
    wvec::CuArray{Float32,1},
    omega::Float32,
    N_owned::Int,
    rank::Int
    )

    if (rank == 0)
        return
    end

    threads, blocks = 256, cld(N_owned,256)
    @cuda threads=threads blocks=blocks BGK_collision!_gpu(
        f_old, f_new,
        rho, u, v, w,
        c_x, c_y, c_z, wvec, omega,
        Int32(N_owned)
    )
    return
end




function BGK_collision!_gpu(
    f_old::CuDeviceMatrix{Float32},
    f_new::CuDeviceMatrix{Float32},
    rho::CuDeviceVector{Float32},
    u::CuDeviceVector{Float32},
    v::CuDeviceVector{Float32},
    w::CuDeviceVector{Float32},
    c_x::CuDeviceVector{Int8},
    c_y::CuDeviceVector{Int8},
    c_z::CuDeviceVector{Int8},
    wvec::CuDeviceVector{Float32},
    omega::Float32,
    N_owned::Int32
    )
    
    tid = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if tid ≤ N_owned
        rho_l = rho[tid]
        ux,uy,uz = u[tid], v[tid], w[tid]
        q = size(f_old,1)
        @inbounds for d in 1:q
            cx, cy, cz = Int32(c_x[d]), Int32(c_y[d]), Int32(c_z[d])
            cu = cx*ux + cy*uy + cz*uz
            cu2 = cu*cu
            u2 = ux*ux + uy*uy + uz*uz
            feq = wvec[d] * rho_l * (1f0 + cu * INV_CS2 + 0.5f0 * cu2 * INV_CS4 - 0.5f0 * u2  * INV_CS2)      
            f_new[d,tid] = f_old[d,tid] - omega*(f_old[d,tid] - feq)
        end
    end
    return
end




end # module
