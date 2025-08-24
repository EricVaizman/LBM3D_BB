module ComputeMacro

using CUDA

export compute_macros!




# GPU kernel: compute macroscopic density and velocity from LBM distributions.
function compute_macros!_gpu(
    f::CuDeviceMatrix{Float32},
    rho::CuDeviceVector{Float32},
    u::CuDeviceVector{Float32},
    v::CuDeviceVector{Float32},
    w::CuDeviceVector{Float32},
    c_x::CuDeviceVector{Int8},
    c_y::CuDeviceVector{Int8},
    c_z::CuDeviceVector{Int8},
    q::Int32,
    N_owned::Int32
)
    tid = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if tid ≤ N_owned
        # accumulate sums
        local_rho = 0f0
        mx = 0f0
        my = 0f0
        mz = 0f0
        for d in 1:q
            fval = f[d, tid]
            local_rho += fval
            # cast direction to Float32 for multiplication
            local_cx = Float32(c_x[d])
            local_cy = Float32(c_y[d])
            local_cz = Float32(c_z[d])
            mx += fval * local_cx
            my += fval * local_cy
            mz += fval * local_cz
        end
        rho[tid] = local_rho
        # avoid division by zero guard, assume density > 0
        u[tid] = mx / local_rho
        v[tid] = my / local_rho
        w[tid] = mz / local_rho
    end
    return
end




function compute_macros!(
    f::CuArray{Float32,2},
    rho::CuArray{Float32,1},
    u::CuArray{Float32,1},
    v::CuArray{Float32,1},
    w::CuArray{Float32,1},
    c_x::CuArray{Int8,1},
    c_y::CuArray{Int8,1},
    c_z::CuArray{Int8,1},
    rank::Int
)

    if (rank == 0)
        return
    end

    # number of directions and owned nodes
    q = size(f, 1)
    N_owned = length(rho)

    threads = 256
    blocks  = cld(N_owned, threads)
    @cuda threads=threads blocks=blocks compute_macros!_gpu(
        f,
        rho, u, v, w,
        c_x, c_y, c_z,
        Int32(q), Int32(N_owned)
    )

    return nothing
end




end # module ComputeMacro
