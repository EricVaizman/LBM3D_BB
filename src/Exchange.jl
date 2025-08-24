module Exchange

using MPI
using CUDA

export begin_halo_exchange!, finish_halo_exchange!




function begin_halo_exchange!(
    comm::MPI.Comm,
    f_new_gpu::CuArray{Float32,2},
    send_buffer_gpu::CuArray{Float32,1},
    recv_buffer_gpu::CuArray{Float32,1},
    gather_qdirs::CuArray{Int8,1},
    gather_lidxs::CuArray{Int32,1},
    gather_ptr::Vector{Int},   # CSR pointers on the host
    neighs::Vector{Int},
    tag::Integer,
    rank::Int
)
    if (rank == 0)
        return MPI.Request[], MPI.Request[]
    end

    # total number of halo entries
    total_halo = length(gather_lidxs)

    # 2) pack post-collision halo into send_buffer_gpu
    if (total_halo > 0)
        threads = 256
        blocks  = cld(total_halo, threads)
        @cuda threads=threads blocks=blocks gather_halo!(
        f_new_gpu,
        gather_qdirs,
        gather_lidxs,
        send_buffer_gpu,
        total_halo
        )
        CUDA.synchronize()
    end

    # 3a) post non-blocking recvs into recv_buffer_gpu
    recv_reqs = MPI.Request[]
    for (p, nbr) in enumerate(neighs)
        off = gather_ptr[p]               # start index (1-based)
        len = gather_ptr[p+1] - off       # number of entries
        # view the 1D recv buffer from off .. off+len-1
        buf = @view recv_buffer_gpu[off:(off+len-1)]
        push!(recv_reqs, MPI.Irecv!(buf, nbr, tag, comm))
    end

    # 3b) post non-blocking sends from send_buffer_gpu
    send_reqs = MPI.Request[]
    for (p, nbr) in enumerate(neighs)
        off = gather_ptr[p]
        len = gather_ptr[p+1] - off
        buf = @view send_buffer_gpu[off:(off+len-1)]
        push!(send_reqs, MPI.Isend(buf, nbr, tag, comm))
    end

    return send_reqs, recv_reqs
end




function finish_halo_exchange!(
    send_reqs::Vector{MPI.Request},
    recv_reqs::Vector{MPI.Request},
    recv_buffer_gpu::CuArray{Float32,1},
    scatter_qdirs::CuArray{Int8,1},
    scatter_lidxs::CuArray{Int32,1},
    f_new_gpu::CuArray{Float32,2},
    rank::Int
    )

    if (rank == 0)
        return
    end

    total_halo = length(scatter_lidxs)
    threads = 256
    blocks = cld(total_halo, threads)

    # 5) Wait for halos to arrive
    MPI.Waitall!(recv_reqs)
    MPI.Waitall!(send_reqs)
    CUDA.synchronize()

    # 6) SCATTER: unpack recv_buffer_gpu back into f_new_gpu halo
    if (total_halo > 0)
        @cuda threads=threads blocks=blocks scatter_halo!(
            recv_buffer_gpu,
            f_new_gpu, 
            scatter_qdirs, scatter_lidxs,
            total_halo
        )
        CUDA.synchronize()
    end
end




# ==== GPU KERNELS ====




function gather_halo!(
    f_snap::CuDeviceMatrix{Float32},
    qdirs::CuDeviceVector{Int8},
    lidxs::CuDeviceVector{Int32},
    sendbuf::CuDeviceVector{Float32},
    N::Int
)
    tid = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if tid ≤ N
        q  = Int(qdirs[tid])
        ll = Int(lidxs[tid])
        sendbuf[tid] = f_snap[q,ll]
    end
    return
end




function scatter_halo!(
    recvbuf::CuDeviceVector{Float32},
    f_new::CuDeviceMatrix{Float32},
    qdirs::CuDeviceVector{Int8},
    lidxs::CuDeviceVector{Int32},
    N::Int
)
    tid = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if tid ≤ N
        q  = Int(qdirs[tid])
        ll = Int(lidxs[tid])
        f_new[q,ll] = recvbuf[tid]
    end
    return
end




end # module Exchange
