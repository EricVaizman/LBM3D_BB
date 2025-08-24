module Streaming

using CUDA

export streaming!, streaming_part!, streaming_test!, streaming_test!_gpu




function streaming_part!_gpu(
    f_pre::CuDeviceMatrix{Float32},
    f_post::CuDeviceMatrix{Float32},
    stream_lidxs::CuDeviceVector{Int32},
    mask::CuDeviceVector{UInt8},
    tids::CuDeviceVector{Int32},
    q::Int32,
    N_owned::Int32,
    N_tids::Int32
)
    th = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if th ≤ N_tids
        # lookup which global tid to run
        tid = Int(tids[th])

        # skip any z-wrap slot
        if mask[tid] == 1
            return
        end

        # decode into direction & local index
        i0  = tid - 1
        dir = (i0 % q) + 1
        i   = fld(i0, q) + 1
        # pull and deposit
        j   = stream_lidxs[tid]
        @inbounds f_post[dir, i] = f_pre[dir, j]
    end
    return
end




# host‐side wrapper for the partial stream
function streaming_part!(
    f_pre_gpu::CuArray{Float32,2},
    f_post_gpu::CuArray{Float32,2},
    stream_lidxs_gpu::CuArray{Int32,1},
    stream_mask::CuArray{UInt8, 1},
    tids_gpu::CuArray{Int32,1},
    q::Int,
    N_owned::Int,
    rank::Int
)

    if (rank == 0)
        return
    end

    N_tids  = Int32(length(tids_gpu))
    if N_tids == 0
        return
    end
    threads = 256
    blocks  = cld(N_tids, threads)
    @cuda threads=threads blocks=blocks streaming_part!_gpu(
        f_pre_gpu, f_post_gpu,
        stream_lidxs_gpu, stream_mask,
        tids_gpu,
        Int32(q), Int32(N_owned), N_tids
    )
    return nothing
end




#=
Pull-style streaming: for each direction & owned node, look up the
neighbor’s population and write it into `f_post`.
=#
function streaming!_gpu(
    f_pre::CuDeviceMatrix{Float32},
    f_post::CuDeviceMatrix{Float32},
    stream_lidxs::CuDeviceVector{Int32},
    mask::CuDeviceVector{UInt8},
    q::Int32,
    N_owned::Int32
)
    tid = (blockIdx().x-1)*blockDim().x + threadIdx().x
    total = q * N_owned
    if tid ≤ total
        # skip any z-wrap slot
        if mask[tid] == 1
            return
        end

        # otherwise do regular streaming
        # decode direction & local index
        i0  = tid - 1
        dir = (i0 % q) + 1
        i   = fld(i0, q) + 1

        # lookup precomputed target column
        j = stream_lidxs[tid]

        # perform streaming pull
        @inbounds f_post[dir, i] = f_pre[dir, j]
    end
    return
end




function streaming!(
    f_pre_gpu::CuArray{Float32,2},
    f_post_gpu::CuArray{Float32,2},
    stream_lidxs_gpu::CuArray{Int32,1},
    stream_mask::CuArray{UInt8, 1},
    q::Int,
    N_owned::Int,
    rank::Int
)
    if (rank == 0)
        return
    end

    threads = 256
    blocks  = cld(q * N_owned, threads)
    @cuda threads=threads blocks=blocks streaming!_gpu(
        f_pre_gpu, f_post_gpu,
        stream_lidxs_gpu, stream_mask,
        Int32(q), Int32(N_owned)
    )
    return nothing
end




end