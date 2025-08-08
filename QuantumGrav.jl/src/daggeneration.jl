
"""
    transitive_closure!(mat::BitMatrix)

Compute the transitive closure of a DAG represented by matrix `mat` by 
successively adding reachable nodes in the future of a node to it'
"""
function transitive_closure!(mat::BitMatrix)
    n = size(mat, 1)
    @inbounds for i = 1:n
        for j = (i+1):n  # only look at future nodes
            if mat[i, j]
                mat[i, :] .= mat[i, :] .|| mat[j, :]  # OR operation to include all reachable nodes from node j --> adding the future
            end
        end
    end
end

"""
    transitive_reduction!(mat::BitMatrix)

Compute the transitive reduction of a DAG represented by matrix `mat` by
removing intermediate nodes in the future of a node that are reachable from the past of that node via the node itself.
This function modifies the input matrix in place.
The matrix `mat` is assumed to be in topological order, i.e., the rows
and columns correspond to the nodes in a topological order.
"""
function transitive_reduction!(mat::BitMatrix)
    n = size(mat, 1)
    @inbounds for i = 1:n
        for j = (i+1):n
            if mat[i, j]
                # If any intermediate node k exists with i → k and k → j, remove i → j
                for k = (i+1):(j-1)
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
    mat_to_cs(adj::BitMatrix)

Convert a graph given by the matrix `adj`, assumed to be a transitively closed 
DAG, to a `CausalSets.BitArrayCauset`.
The matrix `adj` is assumed to be in topological order, i.e., the rows
and columns correspond to the nodes in a topological order.

# Arguments:
- `adj`: A `BitMatrix` representing the adjacency matrix of a DAG.

# Returns:
- A `CausalSets.BitArrayCauset` constructed from the adjacency matrix.
"""
function mat_to_cs(adj::BitMatrix)::CausalSets.BitArrayCauset
    n = size(adj, 1)
    future_relations = Vector{BitVector}(undef, n)
    past_relations = Vector{BitVector}(undef, n)

    # assume topological ordering
    for node_idx = 1:n
        future_relations[node_idx] = adj[node_idx, :]
        past_relations[node_idx] = adj[:, node_idx]
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

    for i = 1:n
        future_relations[i] = []
        past_relations[i] = []
    end
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
function create_random_cset(
    atom_count::Int64,
    future_deg::Function,
    link_prob::Function,
    rng::Random.AbstractRNG;
    type::Type{T} = CausalSets.SparseArrayCauset,
) where {T<:CausalSets.AbstractCauset}

    if atom_count <= 0
        throw(ArgumentError("n_atoms must be greater than 0, got $atom_count"))
    end

    if type != CausalSets.SparseArrayCauset && type != CausalSets.BitArrayCauset
        throw(ArgumentError("Unsupported CausalSet type: $type"))
    end

    # we interpret the random ordering as a topo-order and continue building the causet from there
    nodelist = 1:atom_count # assume this to be a topological ordering of the nodes. This is fine, because for generation, any strong ordering will do.

    adj = falses(atom_count, atom_count)

    raw_weights = zeros(Float64, atom_count)  # preallocate weights 

    for i in nodelist
        future = view(nodelist, (i+1):atom_count)  # future nodes are all nodes after the current one

        future_connection_number = future_deg(rng, i, future, atom_count)
        raw_weights[(i+1):atom_count] .= link_prob.(rng, i, future)
        weights = StatsBase.weights(raw_weights[(i+1):atom_count])
        try
            out_edges = StatsBase.sample(
                rng,
                (i+1):atom_count,
                weights,
                future_connection_number;
                replace = false,
            )
            adj[i, out_edges] .= 1
            raw_weights[(i+1):atom_count] .= 0.0  # reset weights for the next iteration
        catch e
            @warn "Sampling in edges failed for node $i with future connections $future_connection_number: $e"
            @warn "Weights: $weights"
            break
        end
    end

    # # make true adj matrix including transitives
    transitive_closure!(adj)

    return mat_to_cs(adj), adj

end
