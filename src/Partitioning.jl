module Partitioning

using SparseArrays
using Metis
using ..Parameters

export build_connectivity_graph, compute_node_weights, partition_domain_metis




function build_connectivity_graph(params::LBMParams,
    c_x::Vector{Int8}, c_y::Vector{Int8}, c_z::Vector{Int8})

    Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
    q = length(c_x)
    num_nodes = Nx * Ny * Nz

    linear_to_xyz = Dict{Int, NTuple{3, Int}}()
    xyz_to_linear = Dict{NTuple{3, Int}, Int}()

    # Create linear index mapping
    counter = 1
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        xyz_to_linear[(ix, iy, iz)] = counter
        linear_to_xyz[counter] = (ix, iy, iz)
        counter += 1
    end

    # Adjacency list: neighbors per node
    adj_list = Vector{Vector{Int}}(undef, num_nodes)

    for k in 1:num_nodes
        ix, iy, iz = linear_to_xyz[k]
        neighbors = Int[]

        for qdir in 1:q
            jx = ix + c_x[qdir]
            jy = iy + c_y[qdir]
            jz = iz + c_z[qdir]

            if 1 <= jx <= Nx && 1 <= jy <= Ny && 1 <= jz <= Nz
                neighbor_idx = xyz_to_linear[(jx, jy, jz)]
                if neighbor_idx != k
                    push!(neighbors, neighbor_idx)
                end
            end
        end

        adj_list[k] = neighbors
    end

    return adj_list, linear_to_xyz, xyz_to_linear
end




#=
compute_node_weights creates a weights array with the same size as the 3D grid, giving each node an appropriate
weight to later be considered in the METIS partitioning process. The idea is that boundary nodes are more
computationally expensive than bulk nodes and thus the partitioning process must consider that in order to
yield a workload-based parition, rather than a size-based one.
=#
function compute_node_weights(mask_bb::BitArray{3}, xyz_to_linear::Dict{NTuple{3,Int}, Int})
    Nx, Ny, Nz = size(mask_bb)
    num_nodes = Nx * Ny * Nz
    node_weights = Vector{Int32}(undef, num_nodes)

    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        node_id = xyz_to_linear[(ix, iy, iz)]

        if mask_bb[ix, iy, iz]
            node_weights[node_id] = 2  # boundary node
        else
            node_weights[node_id] = 1  # all others (fluid + solid)
        end
    end

    return node_weights
end




# adj_to_sparse converts an adjacency list (representing a graph) into a SparseMatrix when then can be converted to a graph entity by Metis
function adj_to_sparse(adj_list)
    rows = Int[]
    cols = Int[]
    vals = Int[]

    for (i, neighbors) in enumerate(adj_list)
        for j in neighbors
            push!(rows, i)
            push!(cols, j)
            push!(vals, 1)  # Weight of the edge (1 for unweighted graphs)
        end
    end

    N = length(adj_list)  # Number of nodes
    return sparse(rows, cols, vals, N, N)  # Convert to sparse matrix
end




function partition_domain_metis(params::LBMParams, c_x::Vector{Int8}, 
    c_y::Vector{Int8}, c_z::Vector{Int8}, n_parts::Int)

    adj_list, linear_to_xyz, xyz_to_linear = build_connectivity_graph(params, c_x, c_y, c_z)

    adj_mat = adj_to_sparse(adj_list)

    graph = Metis.graph(adj_mat)

    partitions = Metis.partition(graph, n_parts)

    return partitions, xyz_to_linear, linear_to_xyz
end




end # module