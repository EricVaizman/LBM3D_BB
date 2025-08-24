module Utilities


using MPI
using CUDA
using ..Parameters
using ..Geometry
using ..Partitioning
using ..SolidBC

export initialize_distributed, build_gpu_index_maps, build_local_coords_gpu, measure_Mlups, ref_area_lat




function initialize_distributed(comm::MPI.Comm, rank::Int, nworkers::Int, params::LBMParams)
    # Generate lattice info (global, same for all ranks)
    c_x, c_y, c_z, w, bbOpp = gen_lattice_sets(params)

    bcstruct = init_SolidBCStruct(
        params.bc,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        rank
    )

    if rank == 0
        # === MASTER PROCESS === #

        # Partition domain
        partitions, xyz_to_linear, linear_to_xyz = Partitioning.partition_domain_metis(params, c_x, c_y, c_z, nworkers)


        # Send local data to each worker 
        for r in 1:nworkers
            local_data = generate_local_data(r, partitions, xyz_to_linear, linear_to_xyz, Int.(c_x), Int.(c_y), Int.(c_z), bbOpp, params.h, params.Nx, params.Ny, params.Nz)
            MPI.send(local_data, r, 0, comm)
        end

        Q = params.q
        N_local = 0

        #–– owned indices ––
        owned_idxs = Int[]
        halo_nodes = Int[]

        ix_lli = CuArray{Int32}(undef, N_local)
        iy_lli = similar(ix_lli)
        iz_lli = similar(ix_lli)
        x_lli = CuArray{Float32}(undef, N_local)
        y_lli = similar(x_lli)
        nbr_lli = CuArray{Int32}(undef, Q, N_local)

        #–– core fields ––
        rho_gpu    = CuArray{Float32}(undef, N_local)
        u_gpu      = similar(rho_gpu)
        v_gpu      = similar(rho_gpu)
        w_gpu      = similar(rho_gpu)

        f_old_gpu  = CuArray{Float32}(undef, Q, N_local)
        f_new_gpu  = similar(f_old_gpu)

        wvec_gpu   = CuArray{Float32}(undef, Q)
        bbOpp_gpu  = CuArray{Int8}(undef, Q)

        #–– halo-exchange buffers ––
        send_buffer_gpu = CuArray{Float32}(undef, 0)
        recv_buffer_gpu = similar(send_buffer_gpu)

        gather_qdirs  = CuArray{Int8}(undef, N_local)
        gather_lidxs  = CuArray{Int32}(undef, N_local)
        gather_ptr    = Array{Int64}(undef, N_local)

        scatter_qdirs = CuArray{Int8}(undef, N_local)
        scatter_lidxs = CuArray{Int32}(undef, N_local)

        neighs = Int[]
        TAG    = 0

        #–– streaming indices ––
        stream_lidxs     = CuArray{Int32}(undef, N_local)
        stream_int_tids  = CuArray{Int32}(undef, N_local)
        stream_halo_tids = CuArray{Int32}(undef, N_local)

        #–– periodic-BC indices ––
        per_dest      = CuArray{Int32}(undef, N_local)
        per_src       = CuArray{Int32}(undef, N_local)
        per_dir       = CuArray{Int8}(undef, N_local)
        per_int_tids  = CuArray{Int32}(undef, N_local)
        per_halo_tids = CuArray{Int32}(undef, N_local)
        per_mask      = CuArray{UInt8}(undef, N_local)

        #–– velocity / pressure-BC indices ––
        vel_lidxs  = CuArray{Int32}(undef, N_local)
        vel_dirs   = CuArray{Int8}(undef, N_local)
        pres_lidxs = CuArray{Int32}(undef, N_local)
        pres_dirs  = CuArray{Int8}(undef, N_local)

        CL_gpu = CUDA.zeros(Float32, params.n_save)
        CD_gpu = CUDA.zeros(Float32, params.n_save)
        CZ_gpu = CUDA.zeros(Float32, params.n_save)

        xyz_to_linear_local = Dict{NTuple{3,Int}, Int}()
        linear_to_xyz_local = Dict{Int, NTuple{3,Int}}()

        global_to_local = Dict{Int, Int}()


        return (
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
            bcstruct
        )

    else
        # === WORKER PROCESS === #
        local_data, _ = MPI.recv(0, 0, comm)
        owned_idxs = local_data.owned_idxs
        ix_lli = CuArray(local_data.ix_lli)
        iy_lli = CuArray(local_data.iy_lli)
        iz_lli = CuArray(local_data.iz_lli)
        x_lli = CuArray(local_data.x_lli)
        y_lli = CuArray(local_data.y_lli)
        nbr_lli = CuArray(local_data.nbr_lli)
        halo_nodes = local_data.halo_nodes
        halo_dirs = local_data.halo_dirs
        halo_ptr = local_data.halo_ptr
        neighs = local_data.neighs
        part_ptr = local_data.part_ptr
        gather_qdirs = CuArray(local_data.gather_qdirs)
        gather_lidxs = CuArray(local_data.gather_lidxs)
        gather_ptr = local_data.gather_ptr
        scatter_qdirs = CuArray(local_data.scatter_qdirs)
        scatter_lidxs = CuArray(local_data.scatter_lidxs)
        stream_lidxs = CuArray(local_data.stream_lidxs)
        stream_int_tids = CuArray(local_data.stream_int_tids)
        stream_halo_tids = CuArray(local_data.stream_halo_tids)
        vel_lidxs = CuArray(local_data.vel_lidxs)
        vel_dirs = CuArray(local_data.vel_dirs)
        pres_lidxs = CuArray(local_data.pres_lidxs)
        pres_dirs = CuArray(local_data.pres_dirs)
        per_dest = CuArray(local_data.per_dest)
        per_dir = CuArray(local_data.per_dir)
        per_src = CuArray(local_data.per_src)
        per_int_tids = CuArray(local_data.per_int_tids)
        per_halo_tids = CuArray(local_data.per_halo_tids)
        per_mask = CuArray(local_data.per_mask)


        return (
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
            bcstruct
        )
    end
end




struct LocalDomainData
    owned_idxs::Vector{Int}

    ix_lli::Vector{Int32}
    iy_lli::Vector{Int32}
    iz_lli::Vector{Int32}
    x_lli::Vector{Float32}
    y_lli::Vector{Float32}
    nbr_lli::Matrix{Int32}

    halo_nodes::Vector{Int}
    halo_dirs::Vector{Int8}
    halo_ptr::Vector{Int}
    neighs::Vector{Int}
    part_ptr::Vector{Int}

    gather_qdirs::Vector{Int8} 
    gather_lidxs::Vector{Int32}
    gather_ptr::Vector{Int}
    scatter_qdirs::Vector{Int8} 
    scatter_lidxs::Vector{Int32}

    stream_lidxs::Vector{Int32}
    stream_int_tids::Vector{Int32}
    stream_halo_tids::Vector{Int32}

    vel_lidxs::Vector{Int32}
    vel_dirs::Vector{Int8}
    pres_lidxs::Vector{Int32}
    pres_dirs::Vector{Int8}

    per_dest::Vector{Int32}
    per_dir::Vector{Int8}
    per_src::Vector{Int32}
    per_int_tids::Vector{Int32}
    per_halo_tids::Vector{Int32}
    per_mask::Vector{UInt8}
end




function generate_local_data(
    rank::Int,
    partitions::Vector{Int32},
    xyz_to_linear::Dict{NTuple{3,Int}, Int},
    linear_to_xyz::Dict{Int, NTuple{3,Int}},
    c_x::Vector{Int}, c_y::Vector{Int}, c_z::Vector{Int},
    bbOpp::Vector{Int8},
    h::Float32,
    Nx::Int, Ny::Int, Nz::Int
    )

    q = length(c_x)

    owned_idxs = findall(partitions .== rank) # Owned nodes of the worker

    # -- Build the local bb, solid & dict data --
    owned_idxs, N_owned, global_to_local = build_owned(
        partitions, rank
    )


    # -- Build the local halo data --
    hd = build_halo_data(
        owned_idxs,
        linear_to_xyz,
        c_x, c_y, c_z,
        Nx, Ny, Nz,
        xyz_to_linear,
        partitions, rank,
        bbOpp,
        global_to_local
    )

    # Unpack
    halo_nodes = hd.halo_nodes
    halo_dirs  = hd.halo_dirs
    halo_ptr   = hd.halo_ptr
    neighs     = hd.neighs
    part_ptr   = hd.part_ptr

    # print("I'm rank $rank and neighs is: $(neighs)\n")


    # -- Build local dicts --
    linear_to_xyz_local, xyz_to_linear_local = build_local_index_maps(
        owned_idxs,
        halo_nodes,
        linear_to_xyz
    )


    # -- Build local "GPU dict" tables (constant even for moving boundaries)
    ix_lli, iy_lli, iz_lli,
    x_lli, y_lli, nbr_lli = build_local_tables(
        owned_idxs, halo_nodes,
        linear_to_xyz_local,
        xyz_to_linear_local,
        global_to_local,
        c_x, c_y, c_z,
        Nx, Ny, Nz,
        h
    )


    # -- Build local lists for local indices and qdirs for both gather and scatter of halo nodes exchange --
    gather_qdirs, gather_lidxs, gather_ptr,
    scatter_qdirs, scatter_lidxs = make_exchange_indices(
        owned_idxs,
        halo_nodes,
        halo_dirs,
        halo_ptr,
        global_to_local,
        partitions,
        linear_to_xyz_local,
        xyz_to_linear_local,
        c_x, c_y, c_z,
        Nx, Ny, Nz,
        rank,
        neighs, bbOpp
    )


    stream_lidxs, stream_int_tids, stream_halo_tids, periodic_tids = build_stream_lists(
        owned_idxs,
        linear_to_xyz_local,
        xyz_to_linear_local,
        global_to_local,
        c_x, c_y, c_z,
        Nx, Ny, Nz,
        q, bbOpp, rank
    )
    N_owned = length(owned_idxs)
    total_tids = q * N_owned
    per_mask = zeros(UInt8, total_tids)
    per_mask[periodic_tids] .= 1


    # -- Build local velocity BC local indices -- 
    vel_lidxs, vel_dirs, pres_lidxs, pres_dirs = compile_face_BC(
        owned_idxs,
        linear_to_xyz_local,
        c_x, c_y, c_z,
        Nx, Ny, Nz
    )


    # -- Build local periodic BC indices --
    per_dest, per_src, per_dir = compile_periodic_BC(
        owned_idxs,
        linear_to_xyz_local,
        global_to_local,
        c_x, c_y, c_z,
        Nx, Ny, Nz, bbOpp
    )

    per_int_tids = Int32[];
    per_halo_tids = Int32[];
    


    # Create local domain data struct and return
    return LocalDomainData(
        owned_idxs,
        ix_lli,
        iy_lli,
        iz_lli,
        x_lli,
        y_lli,
        nbr_lli,
        halo_nodes,
        halo_dirs,
        halo_ptr,
        neighs,
        part_ptr,
        gather_qdirs,
        gather_lidxs,
        gather_ptr,
        scatter_qdirs,
        scatter_lidxs,
        stream_lidxs,
        stream_int_tids,
        stream_halo_tids,
        vel_lidxs,
        vel_dirs,
        pres_lidxs,
        pres_dirs,
        per_dest,
        per_dir,
        per_src,
        per_int_tids,
        per_halo_tids,
        per_mask
    )
end




function build_owned(
    partitions::Vector{Int32}, rank::Int
)
    # — owned nodes setup —
    owned_idxs    = findall(partitions .== rank)
    N_owned       = length(owned_idxs)
    global_to_local = Dict{Int,Int}()

    for (i, g) in enumerate(owned_idxs)
        global_to_local[g] = i
    end

    return owned_idxs, N_owned, global_to_local
        
end




function build_halo_data(
    owned_idxs::Vector{Int},
    linear_to_xyz::Dict{Int64, Tuple{Int64, Int64, Int64}},
    c_x::Vector{Int}, c_y::Vector{Int}, c_z::Vector{Int},
    Nx::Int, Ny::Int, Nz::Int,
    xyz_to_linear::Dict{Tuple{Int,Int,Int},Int},
    partitions::Vector{Int32}, rank::Int,
    bbOpp::Vector{Int8},
    global_to_local::Dict{Int,Int}
    )

    q = length(c_x)

    # 1) Build halo_map: for each owned global idx, record off-rank neighbors
    halo_map = Dict{Int,Vector{Int8}}()
    for g in owned_idxs
        (ix,iy,iz) = linear_to_xyz[g]
        for qdir in 1:q
            jx, jy, jz = ix + c_x[qdir], iy + c_y[qdir], iz + c_z[qdir]
            if 1 ≤ jx ≤ Nx && 1 ≤ jy ≤ Ny && 1 ≤ jz ≤ Nz # If neighbor is in the domain
                nbr = xyz_to_linear[(jx,jy,jz)]
                if partitions[nbr] != rank
                    push!( get!(halo_map, nbr, Int8[]), Int8(bbOpp[qdir]) )
                end
            else # The neighbor is not in the domain (maybe z wrap?)
                if ((jz < 1) || (jz > Nz)) # The neighbor has specifically gone out of z bounds
                    jz = mod1(jz, Nz)
                    if (1 ≤ jx ≤ Nx) && (1 ≤ jy ≤ Ny) # The neighbor is still in x,y bounds
                        nbr = xyz_to_linear[(jx,jy,jz)]
                        if partitions[nbr] != rank
                            push!( get!(halo_map, nbr, Int8[]), Int8(bbOpp[qdir]) )
                        end
                    end
                end
            end
        end
    end

    # 2) Sort halo nodes by destination rank
    halo_nodes_unsorted = collect(keys(halo_map))
    halo_parts = [partitions[h] for h in halo_nodes_unsorted]
    perm = sortperm(halo_parts)
    halo_nodes  = halo_nodes_unsorted[perm]
    sorted_parts = halo_parts[perm]

    # 3) CSR-flatten the direction lists
    halo_dirs = Int8[]
    halo_ptr  = Int[]
    cnt = 1
    for h in halo_nodes
        push!(halo_ptr, cnt)
        append!(halo_dirs, halo_map[h])
        cnt += length(halo_map[h])
    end
    push!(halo_ptr, cnt)

    # 4) Compute unique neighbor ranks and their pointers
    neighs   = unique(sorted_parts)
    part_ptr = Int[]
    lastp = nothing
    for (i,p) in enumerate(sorted_parts)
        if p !== lastp
            push!(part_ptr, i)
            lastp = p
        end
    end
    push!(part_ptr, length(sorted_parts) + 1)

    # 5) Extend global_to_local for halo slots
    N_owned = length(owned_idxs)
    for (i,g) in enumerate(halo_nodes)
        global_to_local[g] = N_owned + i
    end

    return (
        halo_nodes = halo_nodes,
        halo_dirs  = halo_dirs,
        halo_ptr   = halo_ptr,
        neighs     = neighs,
        part_ptr   = part_ptr,
    )
end




function build_local_index_maps(
    owned_idxs::Vector{Int},
    halo_nodes::Vector{Int},
    linear_to_xyz::Dict{Int, Tuple{Int,Int,Int}}
    )
    # allocate the two maps
    linear_to_xyz_local = Dict{Int, NTuple{3,Int}}()
    xyz_to_linear_local = Dict{NTuple{3,Int}, Int}()

    # fill in owned nodes
    for g in owned_idxs
        xyz = linear_to_xyz[g]
        linear_to_xyz_local[g] = xyz
        xyz_to_linear_local[xyz] = g
    end

    # extend to halo nodes
    for h in halo_nodes
        xyz = linear_to_xyz[h]
        linear_to_xyz_local[h] = xyz
        xyz_to_linear_local[xyz] = h
    end

    return linear_to_xyz_local, xyz_to_linear_local
end




function build_local_tables(
    owned_idxs::Vector{Int}, halo_nodes::Vector{Int},
    linear_to_xyz_local::Dict{Int,Tuple{Int,Int,Int}},
    xyz_to_linear_local::Dict{Tuple{Int,Int,Int},Int},
    global_to_local::Dict{Int,Int},
    c_x::Vector{Int}, c_y::Vector{Int}, c_z::Vector{Int},
    Nx::Int, Ny::Int, Nz::Int, h::Float32
)
    all = vcat(owned_idxs, halo_nodes)
    N_owned = length(owned_idxs)
    N_local = length(all)
    q = length(c_x)

    ix_of_lli = Vector{Int32}(undef, N_local)
    iy_of_lli = Vector{Int32}(undef, N_local)
    iz_of_lli = Vector{Int32}(undef, N_local)
    x_of_lli  = Vector{Float32}(undef, N_local)
    y_of_lli  = Vector{Float32}(undef, N_local)

    # Neighbor table ONLY for owned columns; entries can point to owned or halo LLIs
    nbr_lli = fill(Int32(-1), q, N_owned)   # -1 = outside global domain

    # Build a quick local map (ix,iy,iz) -> LLI (1..N_local), matching [owned; halo] column order
    pos_to_lli = Dict{Tuple{Int,Int,Int},Int32}()

    @inbounds for (a, g) in enumerate(all)   # a == LLI (column in f)
        ix, iy, iz = linear_to_xyz_local[g]
        ix_of_lli[a] = ix
        iy_of_lli[a] = iy
        iz_of_lli[a] = iz
        x_of_lli[a]  = (ix - 0.5f0) * h
        y_of_lli[a]  = (iy - 0.5f0) * h
        pos_to_lli[(ix,iy,iz)] = Int32(a)
    end

    # Neighbors: local (owned or halo) LLI if inside global domain; else -1
    @inbounds for (a, g) in enumerate(owned_idxs)  # a ranges 1:N_owned (owned columns only)
        ix = ix_of_lli[a]; iy = iy_of_lli[a]; iz = iz_of_lli[a]
        for qdir in 1:q
            jx = ix + c_x[qdir]; jy = iy + c_y[qdir]; jz = iz + c_z[qdir]
            if 1 ≤ jx ≤ Nx && 1 ≤ jy ≤ Ny && 1 ≤ jz ≤ Nz
                # Prefer local map; fallback via provided dicts (still build-time only)
                if haskey(pos_to_lli, (jx,jy,jz))
                    nbr_lli[qdir, a] = pos_to_lli[(jx,jy,jz)]              # 1..N_local (owned or halo)
                elseif haskey(xyz_to_linear_local, (jx,jy,jz))
                    ng = xyz_to_linear_local[(jx,jy,jz)]
                    nbr_lli[qdir, a] = Int32(global_to_local[ng])           # should resolve to a halo LLI
                else
                    nbr_lli[qdir, a] = Int32(-1)
                end
            end
        end
    end

    return ix_of_lli, iy_of_lli, iz_of_lli, x_of_lli, y_of_lli, nbr_lli
end




function sort_and_streak(gl::Vector{Int32},
                         glg::Vector{Int32},
                         gq::Vector{Int8})
    @assert length(gl)  == length(glg) == length(gq)

    # 1) global sort by glg
    perm = sortperm(glg)                   # Vector{Int}
    glg_sorted = glg[perm]                 # Vector{Int32}
    gl_sorted  = gl[perm]                  # Vector{Int32}
    gq_sorted  = gq[perm]                  # Vector{Int8}

    # 2) within each streak of equal gl values, sort that block of gq
    n = length(gl_sorted)
    i = 1
    while i <= n
        # find end of streak j
        j = i
        while j < n && gl_sorted[j+1] == gl_sorted[i]
            j += 1
        end

        # sort only the gq entries in this streak
        gq_sorted[i:j] = sort(gq_sorted[i:j])

        # move to the next streak
        i = j + 1
    end

    return gl_sorted, glg_sorted, gq_sorted
end




function make_exchange_indices(
    owned_idxs::Vector{Int},
    halo_nodes::Vector{Int},
    halo_dirs::Vector{Int8},
    halo_ptr::Vector{Int},
    global_to_local::Dict{Int,Int},
    partitions::Vector{Int32},
    linear_to_xyz::Dict{Int, NTuple{3,Int}},
    xyz_to_linear::Dict{Tuple{Int,Int,Int},Int},
    c_x::Vector{Int}, c_y::Vector{Int}, c_z::Vector{Int},
    Nx::Int, Ny::Int, Nz::Int,
    myrank::Int,
    neighs::Vector{Int32},
    bbOpp::Vector{Int8}
)
    q = length(c_x)

    # 0) Inject periodic‐BC halos into tmp lists
    tmp_q = Int8[]
    tmp_l = Int32[]
    tmp_n = Int32[]
    tmp_g = Int[]

    for (li, g) in enumerate(owned_idxs)
        x, y, z = linear_to_xyz[g]
        for d in 1:q
            nx = x + c_x[d]
            ny = y + c_y[d]
            nz = z + c_z[d]
            # only wrap in Z
            if (nz < 1 || nz > Nz) && (1 ≤ nx ≤ Nx) && (1 ≤ ny ≤ Ny)
                wz = mod1(nz, Nz)
                g2 = xyz_to_linear[(nx, ny, wz)]
                p  = partitions[g2]
                if p != myrank
                    push!(tmp_q, Int8(d))
                    push!(tmp_l, Int32(li))
                    push!(tmp_n, Int32(p))
                    push!(tmp_g, Int(g2))
                end
            end
        end
    end

    # 1) Build raw gather lists (existing logic)
    for (li, g) in enumerate(owned_idxs)
        x, y, z = linear_to_xyz[g]
        for d in 1:q
            nx, ny, nz = x + c_x[d], y + c_y[d], z + c_z[d]
            if (1 ≤ nx ≤ Nx) && (1 ≤ ny ≤ Ny) && (1 ≤ nz ≤ Nz)
                g2 = xyz_to_linear[(nx, ny, nz)]
                p  = partitions[g2]
                if p != myrank
                    push!(tmp_q, Int8(d))
                    push!(tmp_l, Int32(li))
                    push!(tmp_n, Int32(p))
                    push!(tmp_g, Int(g2))
                end
            end
        end
    end

    # 2) Pack into neighbor‐ordered CSR lists
    gather_qdirs = Int8[]
    gather_lidxs = Int32[]
    gather_gids  = Int[]
    gather_ptr   = Int32[]
    idx = 1
    for nbr in neighs
        push!(gather_ptr, Int32(idx))
        for i in eachindex(tmp_n)
            if tmp_n[i] == nbr
                push!(gather_qdirs, tmp_q[i])
                push!(gather_lidxs, tmp_l[i])
                push!(gather_gids,  tmp_g[i])
                idx += 1
            end
        end
    end
    push!(gather_ptr, Int32(idx))

    # 3) Create scatter lists mirroring gather
    scatter_qdirs = bbOpp[gather_qdirs]
    scatter_lidxs = Int32[ global_to_local[g] for g in gather_gids ]

    # 4) Sort per‐neighbor CSR blocks
    local_to_global = Dict(v=>k for (k,v) in global_to_local)
    gather_global   = Int32[ local_to_global[l] for l in gather_lidxs ]
    scatter_global  = Int32[ local_to_global[l] for l in scatter_lidxs ]
    for b in 1:length(neighs)
        s = Int(gather_ptr[b])
        e = Int(gather_ptr[b+1]) - 1

        # sort gather block
        gl_s, _, q_s = sort_and_streak(
            gather_lidxs[s:e], gather_global[s:e], gather_qdirs[s:e]
        )
        gather_lidxs[s:e] = gl_s
        gather_qdirs[s:e] = q_s

        # sort scatter block
        sc_s, _, sq_s = sort_and_streak(
            scatter_lidxs[s:e], scatter_global[s:e], scatter_qdirs[s:e]
        )
        scatter_lidxs[s:e] = sc_s
        scatter_qdirs[s:e] = sq_s
    end

    return (
        gather_qdirs,
        gather_lidxs,
        gather_ptr,
        scatter_qdirs,
        scatter_lidxs,
    )
end




function build_stream_lists(
    owned_idxs::Vector{Int},
    linear_to_xyz_local::Dict{Int,NTuple{3,Int}},
    xyz_to_linear_local::Dict{NTuple{3,Int},Int},  # unused here
    global_to_local::Dict{Int,Int},
    c_x::Vector{Int}, c_y::Vector{Int}, c_z::Vector{Int},
    Nx::Int, Ny::Int, Nz::Int,
    q::Int, bbOpp::Vector{Int8}, rank::Int
)
    N_owned      = length(owned_idxs)
    total_tids   = q * N_owned

    stream_lidxs = Vector{Int32}(undef, total_tids)
    int_tids     = Int32[]
    halo_tids    = Int32[]
    periodic_tids = Int32[]   # collects the Z-wrap tids

    for local_i in 1:N_owned
        gx, gy, gz = linear_to_xyz_local[owned_idxs[local_i]]
        base = (local_i - 1) * q

        for dir in 1:q
            # neighbor coords
            nx = gx - c_x[dir]
            ny = gy - c_y[dir]
            nz = gz - c_z[dir]

            # compute this tid once
            tid = Int32(base + dir)

            if 1 ≤ nx ≤ Nx && 1 ≤ ny ≤ Ny && 1 ≤ nz ≤ Nz
                # normal in-domain neighbor
                gidx = (nz-1)*Nx*Ny + (ny-1)*Nx + nx
                src  = global_to_local[gidx]
            else
                if (nz < 1 || nz > Nz) && (1 ≤ nx ≤ Nx) && (1 ≤ ny ≤ Ny)
                    # Z-wrap → leave as bounce-back here but record it
                    src = local_i
                    push!(periodic_tids, tid)
                else
                    # X/Y out-of-bounds → true bounce-back
                    src = local_i
                end
            end

            stream_lidxs[tid] = src

            if src ≤ N_owned
                push!(int_tids, tid)
            else
                push!(halo_tids, tid)
            end
        end
    end

    return stream_lidxs, int_tids, halo_tids, periodic_tids
end




function compile_face_BC(
    owned_idxs::Vector{Int},                    # local index of every node
    linear_to_xyz_local::Dict{Int,NTuple{3,Int}},  # maps global idx → (x,y,z)
    c_x::Vector{Int}, c_y::Vector{Int}, c_z::Vector{Int},
    Nx::Int, Ny::Int, Nz::Int
)
    q = length(c_x)

    vel_lidxs  = Int32[]   # moving-wall (velocity)   → inlet + y-faces
    vel_dirs   = Int8[]
    pres_lidxs = Int32[]   # pressure (anti–BB)     → outlet only
    pres_dirs  = Int8[]

    for (local_i, g_i) in enumerate(owned_idxs)
        x,y,z = linear_to_xyz_local[g_i]

        # — inlet at x=1 (moving-wall → u = U∞)
        if x == 1
            for dir in 1:q
                if c_x[dir] > 0
                    push!(vel_lidxs, Int32(local_i))
                    push!(vel_dirs,    Int8(dir))
                end
            end
        end

        # — south wall at y=1 (moving-wall → u = U∞)
        if y == 1
            for dir in 1:q
                if c_y[dir] > 0
                    push!(vel_lidxs, Int32(local_i))
                    push!(vel_dirs,    Int8(dir))
                end
            end
        end

        # — north wall at y=Ny (moving-wall → u = U∞)
        if y == Ny
            for dir in 1:q
                if c_y[dir] < 0
                    push!(vel_lidxs, Int32(local_i))
                    push!(vel_dirs,    Int8(dir))
                end
            end
        end

        # — outlet at x=Nx (pressure BC → anti–BB)
        if x == Nx
            for dir in 1:q
                if c_x[dir] < 0
                    push!(pres_lidxs, Int32(local_i))
                    push!(pres_dirs,    Int8(dir))
                end
            end
        end

        # — NOTE: no z=1 or z=Nz case here, because z is periodic!
    end

    return vel_lidxs, vel_dirs, pres_lidxs, pres_dirs
end




function compile_periodic_BC(
    owned_idxs,
    linear_to_xyz_local,
    global_to_local,
    c_x, c_y, c_z,
    Nx, Ny, Nz, bbOpp
)
    per_dest = Int32[]
    per_src  = Int32[]
    per_dir  = Int8[]

    for (local_i, g_i) in enumerate(owned_idxs)
        x,y,z = linear_to_xyz_local[g_i]

        for dir in 1:length(c_x)
            dx, dy, dz = c_x[dir], c_y[dir], c_z[dir]

            # only wrap if this pop would leave in z
            if (z + dz < 1) || (z + dz > Nz)
                # perform periodic wrap in z
                zs = mod1(z + dz, Nz)
                xs = x + dx
                ys = y + dy

                # must still lie inside x,y domain
                if 1 ≤ xs ≤ Nx && 1 ≤ ys ≤ Ny
                    gsrc = (zs-1)*Nx*Ny + (ys-1)*Nx + xs
                    if haskey(global_to_local, gsrc)
                        push!(per_dest, Int32(local_i))
                        push!(per_src,  Int32(global_to_local[gsrc]))
                        push!(per_dir,  Int8(bbOpp[dir]))
                    end
                end
            end
        end
    end

    return per_dest, per_src, per_dir
end




function measure_Mlups(
    t_start :: UInt64,
    t_end   :: UInt64,
    t_ini   :: UInt64,
    Nx      :: Int,
    Ny      :: Int,
    Nz      :: Int,
    n_max   :: Int
)

    total_comp_master_s = (t_end - t_start) / 1e9
    N_global       = Nx * Ny * Nz 
    total_updates  = N_global * n_max
    MLUPS_master   = total_updates / (total_comp_master_s * 1e6)
    tot_sim_time = (t_end - t_ini) / 1e9

    println("────────────────────────────────────────────────────────")
    println("  Simulation finished.             ")
    println("  Master wall-clock (incl. I/O)  : ", round(tot_sim_time, digits=4), " s")
    println("  MLUPS (master timing)          : ", round(MLUPS_master, digits=2), " MLUPS")
    println("────────────────────────────────────────────────────────")

    return
end




@inline function ref_area_lat(shape::Symbol, R::Float32, chord::Float32, Nz::Int, h::Float32)
    Lz_cells = Float32(Nz)                  # span in cells
    if shape === :CYL
        return (2f0*R / h) * Lz_cells       # (diameter in cells) * (span in cells)
    else
        return (chord / h) * Lz_cells       # (chord in cells) * (span in cells)
    end
end




# Generate lattice sets, c_x, c_y, c_z, w
# LATTICE SETS ARE DEFINED AS PRESENTED IN BOOK FIG. 3.4 (P. 87)
function gen_lattice_sets(params::LBMParams)
    
    q = params.q

    if (q == 19) # D3Q19
        c_x = [ 1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  0,  0,  1, -1,  1, -1,  0,  0,  0];
        c_y = [ 0,  0,  1, -1,  0,  0,  1, -1,  0,  0,  1, -1, -1,  1,  0,  0,  1, -1,  0];
        c_z = [ 0,  0,  0,  0,  1, -1,  0,  0,  1, -1,  1, -1,  0,  0, -1,  1, -1,  1,  0];
        
        W = [
            1//18, 1//18, 1//18, 1//18, 1//18, 1//18,
            1//36, 1//36, 1//36, 1//36, 1//36, 1//36,
            1//36, 1//36, 1//36, 1//36, 1//36, 1//36,
            1//3]
        
        bbOpp = [2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 19];
                
    elseif (q == 27) # D3Q27
        c_x = [ 1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  0,  0,  1, -1,  1, -1,  0,  0,  1, -1,  1, -1,  1, -1, -1,  1,  0];
        c_y = [ 0,  0,  1, -1,  0,  0,  1, -1,  0,  0,  1, -1, -1,  1,  0,  0,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  0];
        c_z = [ 0,  0,  0,  0,  1, -1,  0,  0,  1, -1,  1, -1,  0,  0, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  0];

        W = [
            2//27, 2//27, 2//27, 2//27, 2//27, 2//27,
            1//54, 1//54, 1//54, 1//54, 1//54, 1//54,
            1//54, 1//54, 1//54, 1//54, 1//54, 1//54,
            1//216, 1//216, 1//216, 1//216,
            1//216, 1//216, 1//216, 1//216,
            8//27
            ];

        bbOpp = [2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 27];
    
    elseif (q == 3) # D1Q3
        c_x = [ 1,  0,  -1];
        c_y = [];
        c_z = [];

        W = [1//6, 1//6, 2//3];

        bbOpp = [2, 1, 3];

    else
        error("Invalid q value!")
    end

    return Int8.(c_x), Int8.(c_y), Int8.(c_z), Float32.(W), Int8.(bbOpp)

end




end # module
