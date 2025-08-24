module I_O

using ColorSchemes
using CairoMakie          # Switched from GLMakie
CairoMakie.activate!()    # Activate non-interactive backend
using HDF5
using CUDA
using MPI
using DelimitedFiles
using Printf

using Makie: Axis3, Figure, scatter!, axislegend, text!, Point3f, Vec3
using Rotations
using GeometryBasics
using LinearAlgebra: norm
using ..Parameters

export save_pops_debug, plot_mask_3D, plot_mask_3D_interactive, write_fields_h5, write_fields_h5_time!, save_populations_h5!, save_full_f!, save_coefficients!




#=
    plot_mask_3D(mask_s, mask_bb, params, partitions, xyz_to_linear; savepath="")

Visualizes a 3D binary mask where `1` indicates solid nodes.
Plots scatter markers for all grid cells, colored by partition number.
If `savepath` is provided, saves the figure there.
=#
function plot_mask_3D(
    mask_s::BitArray{3},
    mask_bb::BitArray{3},
    params::LBMParams,
    partitions::Vector{Int32},
    xyz_to_linear::Dict{NTuple{3, Int}, Int};
    savepath::String = ""
)
    Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
    h = params.h

    # Fluid node coordinates + partition labels
    xs_p, ys_p, zs_p = Float32[], Float32[], Float32[]
    part_colors = Int[]
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        push!(xs_p, (ix - 0.5f0) * h)
        push!(ys_p, (iy - 0.5f0) * h)
        push!(zs_p, (iz - 0.5f0) * h)
        idx = xyz_to_linear[(ix, iy, iz)]
        push!(part_colors, partitions[idx])
    end

    # Optional: solid geometry
    xs_s, ys_s, zs_s = Float32[], Float32[], Float32[]
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        if mask_s[ix, iy, iz]
            push!(xs_s, (ix - 0.5f0) * h)
            push!(ys_s, (iy - 0.5f0) * h)
            push!(zs_s, (iz - 0.5f0) * h)
        end
    end

    # Optional: bounce-back / boundary nodes
    xs_bb, ys_bb, zs_bb = Float32[], Float32[], Float32[]
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        if mask_bb[ix, iy, iz]
            push!(xs_bb, (ix - 0.5f0) * h)
            push!(ys_bb, (iy - 0.5f0) * h)
            push!(zs_bb, (iz - 0.5f0) * h)
        end
    end

    fig = Figure(size = (800, 600))
    ax = Axis3(fig[1, 1];
        title = "METIS 3D Partitioned Domain",
        aspect = :data,
        limits = ((0, Nx*h), (0, Ny*h), (0, Nz*h)),
        xlabel = "x", ylabel = "y", zlabel = "z"
    )

    # Unique partition labels
    ranks = sort(unique(part_colors))

    # Generate colors (handle single-rank case)
    if length(ranks) == 1
        colors = [:steelblue]
    else
        cmap = get(ColorSchemes.colorschemes, :viridis, ColorSchemes.viridis)
        colors = CairoMakie.resample_cmap(cmap, length(ranks))
    end

    # Map each rank to a consistent color
    rank_to_color = Dict(ranks[i] => colors[i] for i in eachindex(ranks))

    # Plot fluid nodes by partition
    for r in ranks
        idxs = findall(part_colors .== r)
        scatter!(ax, xs_p[idxs], ys_p[idxs], zs_p[idxs];
            color = rank_to_color[r],
            label = "Rank $r",
            markersize = 12
        )
    end

    # Plot solid nodes
    if !isempty(xs_s)
        scatter!(ax, xs_s, ys_s, zs_s;
            markersize = 16,
            color = :red,
            label = "Solid"
        )
    end

    # Plot bounce-back / boundary nodes
    if !isempty(xs_bb)
        scatter!(ax, xs_bb, ys_bb, zs_bb;
            markersize = 16,
            color = :darkblue,
            label = "Boundary"
        )
    end

    axislegend(ax; position = :rt, framevisible = true)

    if savepath != "" 
        save(savepath, fig)
    end

    return fig, ax
end




function plot_mask_3D_interactive(
    mask_s::BitArray{3},
    params::LBMParams,
    partitions::Vector{Int32},
    xyz_to_linear::Dict{NTuple{3, Int}, Int}
    )

    Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
    h = params.h

    # Prepare coordinates
    xs_p, ys_p, zs_p = Float32[], Float32[], Float32[]
    part_colors = Int[]
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        push!(xs_p, (ix - 0.5f0) * h)
        push!(ys_p, (iy - 0.5f0) * h)
        push!(zs_p, (iz - 0.5f0) * h)
        idx = xyz_to_linear[(ix, iy, iz)]
        push!(part_colors, partitions[idx])
    end

    xs_s, ys_s, zs_s = Float32[], Float32[], Float32[]
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        if mask_s[ix, iy, iz]
            push!(xs_s, (ix - 0.5f0) * h)
            push!(ys_s, (iy - 0.5f0) * h)
            push!(zs_s, (iz - 0.5f0) * h)
        end
    end

    ranks = sort(unique(part_colors))
    colors = GLMakie.resample_cmap(get(ColorSchemes.colorschemes, :viridis, ColorSchemes.viridis), length(ranks))
    rank_to_color = Dict(ranks[i] => colors[i] for i in eachindex(ranks))

    fig = Figure(size = (1200, 900))
    ax = Axis3(fig[1, 1];
        title = "Interactive METIS Partitioned Domain",
        aspect = :data,
        xlabel = "x", ylabel = "y", zlabel = "z"
    )

    for r in ranks
        idxs = findall(part_colors .== r)
        scatter!(ax, xs_p[idxs], ys_p[idxs], zs_p[idxs];
            color = rank_to_color[r],
            markersize = 16,
            label = "Rank $r"
        )
    end

    if !isempty(xs_s)
        scatter!(ax, xs_s, ys_s, zs_s; markersize = 24, color = :red, label = "Solid")
    end

    axislegend(ax, position = :rt, framevisible = true)

    display(fig)
    wait()

    return fig, ax
end




function write_fields_h5(
    comm::MPI.Comm,
    rank::Int,
    local_idxs::Vector{Int},
    local_rho::Vector{Float32},
    local_u::Vector{Float32},
    local_v::Vector{Float32},
    local_w::Vector{Float32},
    Nx::Int, Ny::Int, Nz::Int,
    fname::String
)
    # number of local entries
    Nloc = Int32(length(local_idxs))
    size = MPI.Comm_size(comm)

    # gather counts to root
    counts = MPI.gather(Nloc, comm; root=0)
    # broadcast counts to all ranks
    if rank == 0
        counts_all = copy(counts)
    else
        counts_all = Vector{Int32}(undef, size)
    end
    MPI.Bcast!(counts_all, 0, comm)

    # compute displacements
    offsets = Vector{Int32}(undef, size)
    offsets[1] = 0
    for i in 2:size
        offsets[i] = offsets[i-1] + counts_all[i-1]
    end
    Ntot = sum(counts_all)

    # prepare receive buffers on root
    if rank == 0
        all_idxs = Vector{Int32}(undef, Ntot)
        all_rho  = Vector{Float32}(undef, Ntot)
        all_u    = Vector{Float32}(undef, Ntot)
        all_v    = Vector{Float32}(undef, Ntot)
        all_w    = Vector{Float32}(undef, Ntot)
    else
        all_idxs = Vector{Int32}()
        all_rho  = Vector{Float32}()
        all_u    = Vector{Float32}()
        all_v    = Vector{Float32}()
        all_w    = Vector{Float32}()
    end

    # convert local_idxs to Int32
    local_idxs32 = Vector{Int32}(undef, Nloc)
    for i in 1:Nloc
        local_idxs32[i] = Int32(local_idxs[i])
    end

    # MPI Gatherv calls
    MPI.Gatherv!(local_idxs32, all_idxs, counts_all, 0, comm)
    MPI.Gatherv!(local_rho,   all_rho,   counts_all, 0, comm)
    MPI.Gatherv!(local_u,     all_u,     counts_all, 0, comm)
    MPI.Gatherv!(local_v,     all_v,     counts_all, 0, comm)
    MPI.Gatherv!(local_w,     all_w,     counts_all, 0, comm)

    # on root, reconstruct full arrays and write HDF5
    if rank == 0
        Ntot_entries = length(all_idxs)
        # allocate full 1D arrays
        global_rho = zeros(Float32, Nx*Ny*Nz)
        global_u   = zeros(Float32, Nx*Ny*Nz)
        global_v   = zeros(Float32, Nx*Ny*Nz)
        global_w   = zeros(Float32, Nx*Ny*Nz)
        # scatter contributions
        for k in 1:Ntot_entries
            idx = Int(all_idxs[k])
            global_rho[idx] = all_rho[k]
            global_u[idx]   = all_u[k]
            global_v[idx]   = all_v[k]
            global_w[idx]   = all_w[k]
        end
        # reshape into 3D
        rho = reshape(global_rho, (Nx,Ny,Nz))
        U = reshape(global_u,   (Nx,Ny,Nz))
        V = reshape(global_v,   (Nx,Ny,Nz))
        W = reshape(global_w,   (Nx,Ny,Nz))
        # write file
        h5open(fname, "w") do h5
            write(h5, "/rho", rho)
            write(h5, "/u",   U)
            write(h5, "/v",   V)
            write(h5, "/w",   W)
        end
    end

    return nothing
end




function write_fields_h5_time!(
    comm::MPI.Comm,
    rank::Int,
    local_idxs::Vector{Int},
    local_rho::Vector{Float32},
    local_u::Vector{Float32},
    local_v::Vector{Float32},
    local_w::Vector{Float32},
    Nx::Int, Ny::Int, Nz::Int,
    fname::String,
    step::Int,
    n_save::Int,
    num_snaps::Int
)
    # only save on multiples of n_save
    if step % n_save != 0
        return
    end

    snap = div(step, n_save)  # 1-based slice index

    # — MPI gather as before —
    Nloc = Int32(length(local_idxs))
    nprocs = MPI.Comm_size(comm)

    counts = MPI.gather(Nloc, comm; root=0)
    # broadcast counts array
    if rank == 0
        counts_all = copy(counts)
    else
        counts_all = Vector{Int32}(undef, nprocs)
    end
    MPI.Bcast!(counts_all, 0, comm)

    # compute displacements & total entries
    offsets = cumsum([0; counts_all[1:end-1]])
    Ntot    = sum(counts_all)

    # prepare receive buffers on root
    if rank == 0
        all_idxs = Vector{Int32}(undef, Ntot)
        all_rho  = Vector{Float32}(undef, Ntot)
        all_u    = Vector{Float32}(undef, Ntot)
        all_v    = Vector{Float32}(undef, Ntot)
        all_w    = Vector{Float32}(undef, Ntot)
    else
        all_idxs = Vector{Int32}()
        all_rho  = Vector{Float32}()
        all_u    = Vector{Float32}()
        all_v    = Vector{Float32}()
        all_w    = Vector{Float32}()
    end

    # convert to Int32
    local_idxs32 = Int32.(local_idxs)

    # Gatherv into the big arrays
    MPI.Gatherv!(local_idxs32, all_idxs, counts_all, 0, comm)
    MPI.Gatherv!(local_rho,   all_rho,  counts_all, 0, comm)
    MPI.Gatherv!(local_u,     all_u,    counts_all, 0, comm)
    MPI.Gatherv!(local_v,     all_v,    counts_all, 0, comm)
    MPI.Gatherv!(local_w,     all_w,    counts_all, 0, comm)

    # — now rank 0 builds the global 4D file —
    if rank == 0
        # rebuild the 1D global arrays
        global_rho = zeros(Float32, Nx*Ny*Nz)
        global_u   = zeros(Float32, Nx*Ny*Nz)
        global_v   = zeros(Float32, Nx*Ny*Nz)
        global_w   = zeros(Float32, Nx*Ny*Nz)
        for i in 1:length(all_idxs)
            idx = Int(all_idxs[i])
            global_rho[idx] = all_rho[i]
            global_u[idx] = all_u[i]
            global_v[idx] = all_v[i]
            global_w[idx] = all_w[i]
        end

        # reshape into 3D
        rho3d = reshape(global_rho, (Nx,Ny,Nz))
        U3d = reshape(global_u,   (Nx,Ny,Nz))
        V3d = reshape(global_v,   (Nx,Ny,Nz))
        W3d = reshape(global_w,   (Nx,Ny,Nz))

        # open HDF5—create or append
        mode = (snap==1 ? "w" : "r+")
        h5open(fname, mode) do h5
            if snap == 1
                # create chunked 4D datasets of size (Nx,Ny,Nz,num_snaps)
                # you can tune chunk size if you like
                h5["rho"] = zeros(Float32, Nx,Ny,Nz,num_snaps)
                h5["u"]   = zeros(Float32, Nx,Ny,Nz,num_snaps)
                h5["v"]   = zeros(Float32, Nx,Ny,Nz,num_snaps)
                h5["w"]   = zeros(Float32, Nx,Ny,Nz,num_snaps)
                h5["steps"] = zeros(Int32, num_snaps)
            end

            # write slice #snap into the 4th dimension
            h5["rho"][:,:,:,snap] = rho3d
            h5["u"][:,:,:,snap] = U3d
            h5["v"][:,:,:,snap] = V3d
            h5["w"][:,:,:,snap] = W3d
            h5["steps"][snap] = step
        end
    end

    if (rank == 0) print("Saving snapshot for n = $step.\n") end

    return nothing
end




function save_populations_h5!(
    comm::MPI.Comm,
    rank::Int,
    f_old::Array{Float32,2},
    f_new::Array{Float32,2},
    N_local::Int,
    N_locals::Vector{Int},
    q::Int,
    output_file::String,
    n::Int,
    n_save::Int
)
    # Only save at the specified interval
    if n % n_save != 0
        return
    end

    TAG_F_OLD = 101
    TAG_F_NEW = 102

    if rank != 0
        MPI.Send(f_old, 0, TAG_F_OLD, comm)
        MPI.Send(f_new, 0, TAG_F_NEW, comm)
        return
    end

    # Rank 0: receive and write
    n_workers = length(N_locals)
    mode = isfile(output_file) ? "r+" : "w"
    h5open(output_file, mode) do h5
        for worker in 1:n_workers
            count = N_locals[worker]
            # receive buffers
            buf_old = Vector{Float32}(undef, q*count)
            buf_new = Vector{Float32}(undef, q*count)
            MPI.Recv!(buf_old, worker, TAG_F_OLD, comm)
            MPI.Recv!(buf_new, worker, TAG_F_NEW, comm)

            # dataset paths
            step_grp = "/step_$n"
            rank_grp = "$step_grp/rank_$worker"
            path_old = "$rank_grp/f_old"
            path_new = "$rank_grp/f_new"

            # ensure groups exist as before…
            for grp in (step_grp, rank_grp)
                if !haskey(h5, grp)
                    create_group(h5, grp)
                end
            end

            # open-or-create & write helper
            write_or_replace = function(path, data)
                if haskey(h5, path)
                    ds = h5[path]
                    write(ds, data)
                else
                    h5[path] = data
                end
            end

            # write both f_old and f_new
            write_or_replace(path_old, reshape(buf_old, q, count))
            write_or_replace(path_new, reshape(buf_new, q, count))
        end
    end
end




function save_pops_debug(f_gpu::CuArray{Float32,2},
                         kind::String,
                         rank::Integer,
                         iter::Integer,
                         stage::Integer)
    
    if (rank == 0)
        return
    end

    # pull down to CPU
    data = Array(f_gpu)

    # ensure output folder exists
    outdir = "f_debug_saves"
    mkpath(outdir)

    # build filename
    fname = @sprintf("f_%s_r%d_n%d_s%d.txt", kind, rank, iter, stage)
    path  = joinpath(outdir, fname)

    # write comma-separated values: one row per direction, one col per node
    writedlm(path, data, ' ')
end




function save_full_f!(
    comm::MPI.Comm,
    f_gpu::CuArray{Float32,2},
    owned_idxs::Vector{Int},
    n::Int,
    stage::Int,
    tag::String,
    R::Int
    )

    rank = MPI.Comm_rank(comm)
    q, _ = size(f_gpu)

    # Build this rank’s local piece
    Nloc       = length(owned_idxs)
    f_host     = Array(f_gpu[:, 1:Nloc])    # q×Nloc
    perm       = sortperm(owned_idxs)
    idxs_sorted= owned_idxs[perm]           # sorted global indices
    mat_loc    = transpose(f_host[:, perm]) # Nloc×q
    sendbuf    = vec(mat_loc')              # length = q*Nloc

    TAG_IDXS = 200
    TAG_DATA = 201

    if rank != 0
        # Worker: send sorted indices, then data, then return
        MPI.Send(Int32.(idxs_sorted),  0, TAG_IDXS, comm)
        MPI.Send(Float32.(sendbuf),    0, TAG_DATA, comm)
        return nothing
    end

    # — Master (rank==0): receive all R workers’ data —
    all_idxs = Vector{Vector{Int32}}()   # will hold each worker’s index list
    all_bufs = Vector{Vector{Float32}}() # will hold each worker’s data buffer

    for src in 1:R
        # receive index list
        stat    = MPI.Probe(src, TAG_IDXS, comm)
        cnt_i   = MPI.Get_count(stat, MPI.Datatype(Int32))
        buf_i   = Vector{Int32}(undef, cnt_i)
        MPI.Recv!(buf_i, src, TAG_IDXS, comm)
        push!(all_idxs, buf_i)

        # receive data buffer
        stat    = MPI.Probe(src, TAG_DATA, comm)
        cnt_d   = MPI.Get_count(stat, MPI.Datatype(Float32))
        buf_d   = Vector{Float32}(undef, cnt_d)
        MPI.Recv!(buf_d, src, TAG_DATA, comm)
        push!(all_bufs, buf_d)
    end

    # Compute total number of global nodes
    Nnodes = sum(length.(all_idxs))
    # Allocate full array
    full_f = zeros(Float32, Nnodes, q)

    # Fill it rank by rank
    for r in 1:R
        buf_i = all_idxs[r]
        buf_d = all_bufs[r]
        Ni    = length(buf_i)
        mat_r = reshape(buf_d, q, Ni)'   # Ni×q
        idxs  = Int.(buf_i)
        full_f[idxs, :] = mat_r
    end

    # Write one combined f matrix
    debug_dir = joinpath(@__DIR__, "..", "f_debug_saves")
    mkpath(debug_dir)
    fname = "f_$(tag)_r$(R)_n$(n)_s$(stage).txt"
    writedlm(joinpath(debug_dir, fname), full_f)

    return nothing
end




function save_coefficients!(
    step::Int,
    n_save::Int,
    n_max::Int,
    CL_gpu::CuArray{Float32,1},
    CD_gpu::CuArray{Float32,1},
    CZ_gpu::CuArray{Float32,1}
)
    # 1) Ensure the output/ folder exists
    output_dir = "output"
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    filepath = joinpath(output_dir, "aero_coeffs.h5")

    # 2) On the very first step, create the file and zero‐initialize the datasets
    if step == 1
        h5open(filepath, "w") do fid
            # Create three 1D datasets of length n_max filled with zeros
            fid["CL"] = zeros(Float32, n_max)
            fid["CD"] = zeros(Float32, n_max)
            fid["CZ"] = zeros(Float32, n_max)
        end
    end

    # 3) Compute position within the current n_save block
    idx = ((step - 1) % n_save) + 1

    # 4) If not at the end of a full block and not the final step, do nothing
    if idx != n_save && step < n_max
        return
    end

    # 5) Make sure all GPU work is done before we copy back
    CUDA.synchronize()

    # 6) Open the file in read/write mode and write the appropriate slice
    h5open(filepath, "r+") do fid
        cl_ds = fid["CL"]
        cd_ds = fid["CD"]
        cz_ds = fid["CZ"]

        if idx == n_save
            # A full block: copy all n_save values back to host
            CL_block = Array(CL_gpu)  # length = n_save
            CD_block = Array(CD_gpu)
            CZ_block = Array(CZ_gpu)

            # Compute the slice in [1 : n_max] we want to overwrite
            start = step - (n_save - 1)
            stop  = step

            # Write into each dataset
            cl_ds[start:stop] = CL_block
            cd_ds[start:stop] = CD_block
            cz_ds[start:stop] = CZ_block

            # Clear the GPU buffers for the next block
            fill!(CL_gpu, 0f0)
            fill!(CD_gpu, 0f0)
            fill!(CZ_gpu, 0f0)

        else
            # Final partial block (step == n_max, idx < n_save)
            rem = idx
            CL_partial = Array(view(CL_gpu, 1:rem))
            CD_partial = Array(view(CD_gpu, 1:rem))
            CZ_partial = Array(view(CZ_gpu, 1:rem))

            start = n_max - rem + 1
            stop  = n_max

            cl_ds[start:stop] = CL_partial
            cd_ds[start:stop] = CD_partial
            cz_ds[start:stop] = CZ_partial
        end
    end

    return
end




end # module
