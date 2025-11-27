


"""
	 make_adj(c::CausalSets.AbstractCauset; type::Type{M} = SparseArrays.SparseMatrixCSC, eltype::Type{T} = Float32)::AbstractMatrix{T} where {T<:Number,M<:AbstractArray}

Creates an adjacency matrix from a causet's future relations.

# Arguments
- `c::CausalSets.AbstractCauset`: The causet object containing future relations
- `type::Type{M}`: Array type for the adjacency matrix entries. Must be a subtype of AbstractArray
- `eltype::Type{T}`: Numeric element type to store the elements. Ignored if you pass in a BitMatrix, which has no element type.

# Returns
- `AbstractMatrix{T}`: Adjacency matrix of the specified matrix type (as given by the `type` parameter), constrained to subtypes of `AbstractMatrix`.

# Notes
Converts the causet's future_relations to a matrix format by
horizontally concatenating, transposing, and converting to the specified type.
"""
function make_adj(
    c::CausalSets.AbstractCauset;
    type::Type{M} = SparseArrays.SparseMatrixCSC,
    eltype::Type{T} = Float32,
)::AbstractMatrix{T} where {T<:Number,M<:AbstractArray}
    if c.atom_count == 0
        throw(ArgumentError("The causal set must not be empty."))
    end

    adj = transpose(hcat(c.future_relations...))

    if type <: BitArray # BitArray doesn't have any kind of eltype
        return adj |> type
    else
        return adj |> type{eltype}
    end
end

"""
	maxpathlen(adj_matrix::AbstractMatrix, topo_order::AbstractVector, source::Int) -> Int32

Calculates the maximum path length from a source node in a directed acyclic graph.

# Arguments
- `adj_matrix::AbstractMatrix`: Adjacency matrix representing the directed graph
- `topo_order::AbstractVector{Int}`: Topologically sorted order of vertices
- `source::Int`: Source vertex index to calculate distances from

# Returns
- `Int32`: Maximum finite distance from the source node, or 0 if no paths exist

# Notes
Uses dynamic programming with topological ordering for efficient longest path computation.
Processes vertices in topological order to ensure optimal substructure property.
Returns 0 if no finite distances exist from the source.
"""
function max_pathlen(adj_matrix::AbstractMatrix, topo_order::AbstractVector, source::Int)
    n = size(adj_matrix, 1)

    # Dynamic programming for longest paths
    dist = fill(-Inf, n)
    dist[source] = 0

    # Process vertices in topological order
    for u in topo_order
        if dist[u] != -Inf
            if adj_matrix isa SparseArrays.AbstractSparseMatrix
                indices = SparseArrays.findnz(adj_matrix[u, :])[1]
            else
                indices = [v for v ∈ 1:n if adj_matrix[u, v] != 0]
            end

            for v in indices
                dist[v] = max(dist[v], dist[u] + 1)
            end
        end
    end

    # Return max finite distance
    finite_dists = filter(d -> d != -Inf, dist)
    return isempty(finite_dists) ? 0 : maximum(finite_dists)
end

"""
	transitive_reduction!(mat::AbstractMatrix)

Compute the transitive reduction of the input matrix, such that only connections (i,j) remain that have a max pathlen of 1. This assumes that the input matrix is upper triangular, i.e., a topologically ordered DAG. If that is not the case, the results will be incorrect.

# Arguments
- `mat::AbstractMatrix`: The input matrix to compute the transitive reduction on
"""
function transitive_reduction!(mat::AbstractMatrix)

    # transitive reduction
    n = size(mat, 1)
    @inbounds for i ∈ 1:n
        for j ∈ (i+1):n
            if mat[i, j] == 1
                # If any intermediate node k exists with i → k and k → j, remove i → j
                for k ∈ (i+1):(j-1)
                    if mat[i, k] == 1 && mat[k, j] == 1
                        mat[i, j] = 0 # remove intermediate nodes
                        break
                    end
                end
            end
        end
    end
end
