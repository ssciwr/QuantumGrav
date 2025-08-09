function bv_to_view(bv::BitVector)::Vector{UInt64}
    nwords = cld(length(bv), 64) # number of 64 bit elements in bitvector
    ptr = Base.unsafe_convert(Ptr{UInt}, bv.chunks)
    GC.@preserve bv begin
        Base.unsafe_wrap(Vector{UInt64}, ptr, nwords)
    end
end

"""
    transitive_closure!(adj::Vector{BitVector})

Compute the transitive closure of a DAG represented by vector of future relations `adj` by
successively adding reachable nodes in the future of a node to it'
"""
function transitive_closure!(mat::Vector{BitVector})::Nothing
    n = size(mat, 1)
    @inbounds for i in 1:n
        for j in (i + 1):n  # only look at future nodes
            if mat[i][j]
                mat[i] .= mat[i] .|| mat[j] # OR operation to include all reachable nodes from node j --> adding the future
            end
        end
    end
end

function transitive_closure_fast!(adj::Vector{BitVector})::Nothing
    # FIXME: not reliable yet. Gives different result from the above
    n = length(adj)
    @inbounds for i in n:-1:1 # go from last to first node
        wi = bv_to_view(adj[i])
        @inbounds for j in (i + 1):n
            wj = bv_to_view(adj[j])
            for k in eachindex(wi)
                wi[k] |= wj[k]
            end
        end
    end
end

"""TODO: change to vector{bit}
    transitive_reduction!(adj::Vector{BitVector})

Compute the transitive reduction of a DAG represented by matrix `mat` by
removing intermediate nodes in the future of a node that are reachable from the past of that node via the node itself.
This function modifies the input matrix in place.
The matrix `mat` is assumed to be in topological order, i.e., the rows
and columns correspond to the nodes in a topological order.
"""
function transitive_reduction!(mat::BitMatrix)
    n = size(mat, 1)
    @inbounds for i in 1:n
        for j in (i + 1):n
            if mat[i, j]
                # If any intermediate node k exists with i → k and k → j, remove i → j
                for k in (i + 1):(j - 1)
                    if mat[i, k] && mat[k, j]
                        mat[i, j] = false # remove intermediate nodes
                        break
                    end
                end
            end
        end
    end
end

"""
    mat_to_cs(adj::Vector{BitVector})

Convert a graph given by the matrix `adj`, assumed to be a transitively closed 
DAG, to a `CausalSets.BitArrayCauset`.
The matrix `adj` is assumed to be in topological order, i.e., the rows
and columns correspond to the nodes in a topological order.

# Arguments:
- `adj`: A `BitMatrix` representing the adjacency matrix of a DAG.

# Returns:
- A `CausalSets.BitArrayCauset` constructed from the adjacency matrix.
"""
function mat_to_cs(adj::Vector{BitVector})::CausalSets.BitArrayCauset
    n = length(adj)
    future_relations = Vector{BitVector}(undef, n)
    past_relations = Vector{BitVector}(undef, n)

    # assume topological ordering
    for node_idx in 1:n
        future_relations[node_idx] = adj[node_idx]
        past_relations[node_idx] = [adj[i][node_idx] for i in 1:n]
    end

    return CausalSets.BitArrayCauset(n, future_relations, past_relations)
end

"""
    mat_to_cs(adj::SparseMatrixCSC)

Convert a graph given by the sparse matrix `adj`, assumed to be a transitively closed
DAG, to a `CausalSets.SparseArrayCauset`.
The matrix `adj` is assumed to be in topological order, i.e., the rows
and columns correspond to the nodes in a topological order. 

# Arguments:
- `adj`: A `SparseMatrixCSC` representing the adjacency matrix of a DAG.

# Returns:
- A `CausalSets.SparseArrayCauset` constructed from the adjacency matrix.
"""
function mat_to_cs(adj::SparseArrays.SparseMatrixCSC)::CausalSets.SparseArrayCauset
    n = size(adj, 1) # assume topological ordering

    future_relations = Vector{Vector{Int64}}(undef, n)
    past_relations = Vector{Vector{Int64}}(undef, n)

    nodelist = 1:n  # assume this to be a topological ordering of the nodes. This is fine, because for generation, any strong ordering will do.
    for node_idx in nodelist
        future_relations[node_idx] = nodelist[Bool.(adj[node_idx, :])]
        past_relations[node_idx] = nodelist[Bool.(adj[:, node_idx])]
    end

    return CausalSets.SparseArrayCauset(n, future_relations, past_relations), adj
end

"""
    create_random_cset(atom_count::Int64, future_deg::Function, link_prob::Function, rng::Random.AbstractRNG; type::Type{T})

Create a random causal set with `atom_count` atoms, where the in-degree and link probabilities are defined by the provided functions.

# Arguments:
- `atom_count`: The number of atoms in the causal set.
- `future_deg`: A function that takes the random number generator, the current node index, the past nodes, and the total number of atoms, and returns the future degree of a node, i.e., to how many future nodes it should connect. 
- `link_prob`: A function that takes the random number generator, the current node index, the past nodes, and returns the link probability for a node to connect to any 'future' node (ahead of it in the topological order). This is an unnormalized probability, i.e., it does not need to sum to 1.
- `rng`: A random number generator.
- `type`: The type of causal set to create (either `CausalSets.SparseArrayCauset` or `CausalSets.BitArrayCauset`).


# Returns:
- A tuple `(cset, adj)` where:
  - `cset`: The generated causal set.
  - `adj`: The adjacency matrix of the causal set. This is transitively closed, i.e., contains all transitive links in order to represent the full causal set
"""
function create_random_cset(atom_count::Int64,
                            future_deg::Function,
                            link_prob::Function,
                            rng::Random.AbstractRNG;
                            type::Type{T}=CausalSets.SparseArrayCauset,) where {T<:CausalSets.AbstractCauset}
    if atom_count <= 0
        throw(ArgumentError("n_atoms must be greater than 0, got $atom_count"))
    end

    if type != CausalSets.SparseArrayCauset && type != CausalSets.BitArrayCauset
        throw(ArgumentError("Unsupported CausalSet type: $type"))
    end

    # we interpret the random ordering as a topo-order and continue building the causet from there
    nodelist = 1:atom_count # assume this to be a topological ordering of the nodes. This is fine, because for generation, any strong ordering will do.

    adj = [falses(atom_count) for _ in 1:atom_count]

    raw_weights = zeros(Float64, atom_count)  # preallocate weights for connection sampling, later fill in the part we need

    for i in nodelist
        future_connection_number = future_deg(rng, i, (i + 1):atom_count, atom_count)

        for j in (i + 1):atom_count
            raw_weights[j] = link_prob(rng, i, j)
        end

        rw = @view raw_weights[(i + 1):atom_count]
        sum_rw = sum(rw)

        weights = StatsBase.Weights(rw, sum_rw) # TODO allocation that needs to go

        try
            # Sample future connections based on the weights and assign them in the adjacency matrix. This will result in the presence of some transitive edges, but not all of them. 
            # In order to reliably transitively reduce this DAG, we must hence first transitively close it. --> see below

            out_edges = StatsBase.sample(rng,
                                         (i + 1):atom_count,
                                         weights,
                                         future_connection_number;
                                         replace=false,)

            # @assert length(out_edges) == future_connection_number "Sampled $future_connection_number edges, but got $(length(out_edges)) edges instead."

            adj[i][out_edges] .= true

            raw_weights[(i + 1):atom_count] .= 0.0  # reset weights for the next iteration

        catch e
            @warn "Sampling in edges failed for node $i with future connections $future_connection_number: $e"
            @warn "Weights: $weights"
            break
        end
    end

    # # make true adj matrix including missing transitives. This will result in the existing transitives being added again, but this is fine wrt correctness because they will just be a no-op. y
    # transitive_closure!(adj)

    return mat_to_cs(adj)
end
