using Random
using CausalSets
using CairoMakie

""" 
Generate a full adjacency matrix for a graph of given size.

# Parameters
- `count::Int`: Number of nodes in the graph.

# Returns
- `Vector{BitVector}`: A vector of bitvectors representing the adjacency matrix where all edges from the diagonal to the end are true.
"""
function full_adjacency_matrix(count::Int)
    adj = Vector{BitVector}(undef, count)
    for i in 1:count
        # true from diagonal (i) to the end
        adj[i] = trues(count)
        adj[i][1:i-1] .= false
    end
    return adj
end

""" 
Create a full directed acyclic graph (DAG) with all possible edges from lower to higher indices.

# Parameters
- `count::Int64`: Number of nodes in the graph.

# Returns
- `CausalSets.ToposortedDAG`: A fully connected DAG with edges from node i to nodes j > i.
"""
function full_graph(count::Int64)
    adj = full_adjacency_matrix(count)
    return CausalSets.ToposortedDAG(
                         count,
                         adj
                        )
end

""" 
Sample a causet using an alternative Markov Chain Monte Carlo method with bitarray representation.

# Parameters
- `size::Int64`: Number of nodes in the causet.
- `connectivity_goal::Float64`: Target connectivity ratio for the causet.
- `markov_steps::Int64`: Number of Markov chain steps to perform.
- `rng::AbstractRNG`: Random number generator instance.
- `flips_per_step::Int64`: Number of edge flips to attempt per Markov step.

# Returns
- `BitArray`: A bitarray causet sampled according to the connectivity goal.
"""
function alt_sample_bitarray_causet(size::Int64, connectivity_goal::Float64, markov_steps::Int64, rng::AbstractRNG, flips_per_step::Int64)

    # Initialize graph and transitive closures depending on connectivity goal
    if connectivity_goal < 0.9 # this is a pretty random number, and we'll have to find out which value is best for convergence
        # Start from empty graphs if connectivity goal is low
        graph = CausalSets.empty_graph(size)
        tcg = CausalSets.empty_graph(size)
        trg = CausalSets.empty_graph(size)
        tcg_new = CausalSets.empty_graph(size)
        trg_new = CausalSets.empty_graph(size)
    else
        # Start from full graphs if connectivity goal is high
        graph = full_graph(size)
        tcg = full_graph(size)
        trg = full_graph(size)
        tcg_new = full_graph(size)
        trg_new = full_graph(size)
    end

    # Compute initial transitive closure and connectivity
    CausalSets.transitive_closure!(graph, tcg)
    prev_connectivity = CausalSets.count_edges(tcg)/(size*(size-1)/2)

    for step in 1:markov_steps
        # Randomly select edges to flip
        i = [rand(rng, 1:size-1) for flip in 1:flips_per_step]
        j = [rand(rng, i[flip]+1:size) for flip in 1:flips_per_step]
        # Store previous edge states for possible rollback
        prev_edge = [graph.edges[i[flip]][j[flip]] for flip in 1:flips_per_step]
        
        # Flip selected edges
        for flip in 1:flips_per_step
            graph.edges[i[flip]][j[flip]] = !prev_edge[flip]
        end

        # Check if transitive closure needs to be recomputed
        if any([prev_edge[flip] || !tcg.edges[i[flip]][j[flip]] for flip in 1:flips_per_step])
            CausalSets.transitive_closure!(graph, tcg_new)
        end
        
        # Compute new connectivity after flips
        new_connectivity = CausalSets.count_edges(tcg_new)/(size*(size-1)/2)
        
        # Decide whether to accept or reject the new state based on connectivity and Metropolis criterion
        if (new_connectivity - connectivity_goal)^2 <= (prev_connectivity - connectivity_goal)^2 || rand(rng) < 2. ^(1e5*(abs(prev_connectivity - connectivity_goal)-abs(new_connectivity - connectivity_goal)))
            # Accept the modification:
            tcg = tcg_new
            trg = trg_new
            prev_connectivity = new_connectivity
        else
            # Reject the modification: revert flipped edges
            for flip in 1:flips_per_step
                graph.edges[i[flip]][j[flip]] = prev_edge[flip]
            end
        end
    end
    # Return the sampled causet as a bitarray
    return CausalSets.to_bitarray_causet(tcg)
end