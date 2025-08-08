
"""
    transitive_closure!(mat::BitMatrix)

Compute the transitive closure of a DAG represented by matrix `mat` by 
successively adding reachable nodes in the future of a node to it'
"""
function transitive_closure!(mat::BitMatrix)
    # we are operating on the 'past relations matrix here'
    # transitive closure achieved by 'adding the future more than one step out'
    n = size(mat, 1)
    @inbounds for i = 1:n
        for j = 1:(i-1)
            if mat[i, j]

                # FIXME: I think this is doing the indexing wrong. the row gives the past, the column gives the future!
                mat[i, :] .= mat[i, :] .|| mat[j, :]  # OR operation to include all reachable nodes from node j --> adding the future
            end
        end
    end
end

"""
    transitive_reduction!(mat::BitMatrix)

DOCSTRING
"""
function transitive_reduction!(mat::BitMatrix)
    n = size(mat, 1)
    @inbounds for i = 1:n
        for j = (i+1):n # get the future node i
            if mat[j, i]
                # go over the past of node i. if there is 
                # a connection from the past of node i to the future of node i # via j, we remove it
                # FIXME: what about the indexing here?
                for k = 1:(i-1)
                    if mat[j, k] && mat[i, k]
                        mat[j, i] = false  # remove the intermediate j to i link
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
        past_relations[node_idx] = adj[node_idx, :]
        future_relations[node_idx] = adj[:, node_idx]
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
function mat_to_cs(adj::SparseMatrixCSC)::CausalSets.SparseArrayCauset
    n = size(adj, 1) # assume topological ordering

    future_relations = Vector{Vector{Int64}}(undef, n)
    past_relations = Vector{Vector{Int64}}(undef, n)

    for i = 1:n
        future_relations[i] = []
        past_relations[i] = []
    end
    nodelist = 1:n  # assume this to be a topological ordering of the nodes. This is fine, because for generation, any strong ordering will do.
    for node_idx in nodelist
        # FIXME: I think this is doing the indexing wrong
        past_relations[node_idx] = nodelist[Bool.(adj[node_idx, :])]
        future_relations[node_idx] = nodelist[Bool.(adj[:, node_idx])]
    end

    return CausalSets.SparseArrayCauset(n, future_relations, past_relations), adj
end

"""
    create_random_cset(atom_count::Int64, indeg_prob::Function, link_prob::Function, rng::Random.AbstractRNG; type::Type{T})

Create a random causal set with `atom_count` atoms, where the in-degree and link probabilities are defined by the provided functions.

# Arguments:
- `atom_count`: The number of atoms in the causal set.
- `indeg_prob`: A function that takes the random number generator, the current node index, the past nodes, and the total number of atoms, and returns the in-degree probability.
- `link_prob`: A function that takes the random number generator, the current node index, the past nodes, and returns the link probability.
- `rng`: A random number generator.
- `type`: The type of causal set to create (either `CausalSets.SparseArrayCauset` or `CausalSets.BitArrayCauset`).


# Returns:
- A tuple `(cset, adj)` where:
  - `cset`: The generated causal set.
  - `adj`: The adjacency matrix of the causal set. This is transitively closed, i.e., contains all transitive links
"""
function create_random_cset(
    atom_count::Int64,
    indeg_prob::Function,
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
        if i == 1
            continue  # skip the first node, no incoming relations
        end
        past = view(nodelist, 1:(i-1))  # past nodes are all nodes before the current one
        in_degree = indeg_prob(rng, i, past, atom_count) # using the in-degree results in a matrix containing the past relations of the system as columns, not the future. out_degree would do the latter

        raw_weights[1:(i-1)] .= link_prob.(rng, i, past)
        weights = StatsBase.weights(raw_weights[1:(i-1)])
        try
            in_edges = StatsBase.sample(rng, 1:(i-1), weights, in_degree; replace = false)
            adj[i, in_edges] .= 1
            raw_weights[1:(i-1)] .= 0.0  # reset weights for the next iteration
        catch e
            @warn "Sampling in edges failed for node $n with in-degree $in_degree: $e"
            @warn "Weights: $weights"
            break
        end
    end

    # make true adj matrix including transitives
    transitive_closure!(adj)

    return mat_to_cs(adj), adj

end
