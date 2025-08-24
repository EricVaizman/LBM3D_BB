module MEA

using MPI
using CUDA
using HDF5
using ..I_O
using ..SolidBC

export compute_aero!




function compute_aero!(step, n_save, n_max,
                       f_new_gpu, f_old_gpu,
                       bcstruct::SolidBCStruct,
                       bbOpp, c_x, c_y, c_z,
                       CL_gpu, CD_gpu, CZ_gpu,
                       fc_aero::Float32, comm::MPI.Comm, rank::Int)

    bb_ptr   = bcstruct.bb_ptr
    bb_lidxs = bcstruct.bb_lidxs
    bb_dirs  = bcstruct.bb_dirs

    # dense CSR ⇒ rows = owned nodes
    N_owned = Int(length(bb_ptr) - 1)

    Px_loc = 0f0; Py_loc = 0f0; Pz_loc = 0f0

    if rank != 0 && N_owned > 0
        P_x_rank = CUDA.zeros(Float32, N_owned)
        P_y_rank = CUDA.zeros(Float32, N_owned)
        P_z_rank = CUDA.zeros(Float32, N_owned)

        threads = 256
        blocks  = cld(N_owned, threads)
        @cuda threads=threads blocks=blocks momentum_exchange_gpu(
            P_x_rank, P_y_rank, P_z_rank,
            f_new_gpu, f_old_gpu,
            bb_ptr, bb_lidxs, bb_dirs, bbOpp,
            c_x, c_y, c_z,
            Int32(N_owned)
        )
        CUDA.synchronize()

        # Sum over all owned rows (non-BB rows contributed zeros)
        Px_loc = sum(P_x_rank)
        Py_loc = sum(P_y_rank)
        Pz_loc = sum(P_z_rank)
    end

    Px_tot = MPI.Allreduce(Px_loc, MPI.SUM, comm)
    Py_tot = MPI.Allreduce(Py_loc, MPI.SUM, comm)
    Pz_tot = MPI.Allreduce(Pz_loc, MPI.SUM, comm)

    if rank == 0
        Cl_val = Py_tot * fc_aero
        Cd_val = Px_tot * fc_aero
        Cz_val = Pz_tot * fc_aero
        idx = ((step - 1) % n_save) + 1
        CUDA.@allowscalar begin
            CL_gpu[idx] = Cl_val
            CD_gpu[idx] = Cd_val
            CZ_gpu[idx] = Cz_val
        end

        save_coefficients!(step, n_save, n_max, CL_gpu, CD_gpu, CZ_gpu)
    end

    return
end




function momentum_exchange_gpu(
    P_x_rank :: CuDeviceVector{Float32},
    P_y_rank :: CuDeviceVector{Float32},
    P_z_rank :: CuDeviceVector{Float32},
    f_new    :: CuDeviceMatrix{Float32},
    f_old    :: CuDeviceMatrix{Float32},
    bb_ptr   :: CuDeviceVector{Int32},   # length N_owned+1
    bb_lidxs :: CuDeviceVector{Int32},   # first nnz entries valid
    bb_dirs  :: CuDeviceVector{Int8},    # first nnz entries valid
    bbOpp    :: CuDeviceVector{Int8},
    c_x      :: CuDeviceVector{Int8},
    c_y      :: CuDeviceVector{Int8},
    c_z      :: CuDeviceVector{Int8},
    N_owned  :: Int32                    # NOTE: N_owned, not N_bb
)
    a = (blockIdx().x-1)*blockDim().x + threadIdx().x  # owned LLI (row index)
    if 1 <= a <= N_owned
        start_j = bb_ptr[a]
        stop_j  = bb_ptr[a+1] - 1

        Px_val = 0f0; Py_val = 0f0; Pz_val = 0f0
        if start_j <= stop_j
            @inbounds for jj in start_j:stop_j
                dir       = Int(bb_dirs[jj])
                opp       = Int(bbOpp[dir])
                i         = Int(bb_lidxs[jj])      # owned LLI for this link (== a)
                f_in_val  = f_new[dir, i]          # post-collision, pre-stream
                f_out_val = f_old[opp, i]          # post-bounce-back
                tmp       = f_in_val + f_out_val
                Px_val   += tmp * Float32(c_x[dir])
                Py_val   += tmp * Float32(c_y[dir])
                Pz_val   += tmp * Float32(c_z[dir])
            end
        end

        P_x_rank[a] = Px_val
        P_y_rank[a] = Py_val
        P_z_rank[a] = Pz_val
    end
    return
end




# function momentum_exchange(
#     P_x_rank  :: CuArray{Float32,1},
#     P_y_rank  :: CuArray{Float32,1},
#     P_z_rank  :: CuArray{Float32,1},
#     f_new_gpu :: CuArray{Float32,2},
#     f_old_gpu :: CuArray{Float32,2},
#     bb_ptr    :: CuArray{Int32,1},
#     bb_lidxs  :: CuArray{Int32,1},
#     bb_dirs   :: CuArray{Int8,1},
#     bbOpp     :: CuArray{Int8,1},
#     c_x       :: CuArray{Int8,1},
#     c_y       :: CuArray{Int8,1},
#     c_z       :: CuArray{Int8,1},
#     N_owned   :: Int32,
#     )

#     threads, blocks = 256, cld(N_owned,256)
#     @cuda threads=threads blocks=blocks momentum_exchange_gpu(
#         P_x_rank, P_y_rank, P_z_rank,
#         f_new_gpu, f_old_gpu,
#         bb_ptr, bb_lidxs, bb_dirs, bbOpp,
#         c_x, c_y, c_z,
#         Int32(N_owned)
#     )
#     return
# end




end # module


