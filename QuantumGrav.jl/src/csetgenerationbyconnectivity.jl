
"""
# `flip_param_determiner`

A 2D spline interpolator (from `Dierckx.jl`) that maps a given `(connectivity_goal, size)` pair to an estimated `flip_param`.

# input
- `connectivity_goal` (`Float64`): target connectivity ratio in `[0,1]`.
- `size` (`Int64`): number of nodes in the causet.

# Returns 
- interpolated `flip_param` (`Float64`).

This spline is built on a full grid from `optim_values.csv` with shape `(13, 6)`, using `kx=1, ky=1` (piecewise-linear) and `s=0.0` (exact interpolation).
"""

values = CSV.read(joinpath(@__DIR__,"optim_values.csv"), DataFrames.DataFrame);
flip_param_determiner = Dierckx.Spline2D(sort(unique(values.connectivity_goal)), sort(unique(values.size)), reshape(values.flip_param,(13,6)); kx=1, ky=1, s=0.0);

""" 
Sample a causet with given connectivity using a Markov Chain Monte Carlo method with adaptive number of edge flips.

# Parameters
- `size::Int64`: Number of nodes in the causet.
- `connectivity_goal::Float64`: Target connectivity ratio for the causet.
- `markov_steps::Int64`: Number of Markov chain steps to perform.
- `rng::AbstractRNG`: Random number generator instance.
- `flips_param::Float64`: Parameter of algorithm that relates distance from connectivity goal with number of edge flips.
- `rel_tol::Float64`: Relative distance between connectivity and connectivity_goal beyond which the algorithm stops.
- `abs_tol::Float64`: Absolute distance between connectivity and connectivity_goal beyond which the algorithm stops.

# Returns
- A bitarray causet sampled according to the connectivity goal.
"""

function sample_bitarray_causet_by_connectivity(size::Int64, connectivity_goal::Float64, markov_steps::Int64, rng::Random.AbstractRNG; rel_tol::Union{Float64,Nothing}=nothing, abs_tol::Union{Float64,Nothing}=0.01)
    if size < 1
        throw(ArgumentError("size must be larger than 0, is $(size)"))
    end

    if connectivity_goal > 1 || connectivity_goal < 0
        throw(ArgumentError("connectivity_goal has to be in [0,1], is $(connectivity_goal)"))
    end

    if markov_steps < 1
        throw(ArgumentError("markov_steps has to be at least 1, is $(markov_steps)"))
    end
    
    if !isnothing(rel_tol) && rel_tol < 0
        throw(ArgumentError("rel_tol has to be in [0,1], is $(rel_tol)"))
    end

    if !isnothing(abs_tol) && abs_tol < 0
        throw(ArgumentError("abs_tol has to be in [0,1], is $(abs_tol)"))
    end

    flip_param = flip_param_determiner(connectivity_goal, size)

    if flip_param <= 0 
        throw(ArgumentError("flip_param has to be larger than 0, is $(flip_param)"))
    end

    # Start from empty graphs if connectivity goal is low
    graph = CausalSets.empty_graph(size)
    tcg = CausalSets.empty_graph(size)
    trg = CausalSets.empty_graph(size)
    tcg_new = CausalSets.empty_graph(size)
    trg_new = CausalSets.empty_graph(size)

    # Compute initial transitive closure and connectivity
    CausalSets.transitive_closure!(graph, tcg)
    prev_connectivity = CausalSets.count_edges(tcg) / (size * (size - 1) / 2)

    step = 1
    while step < markov_steps

        flips_per_step = Int64(ceil(flip_param * abs(prev_connectivity - connectivity_goal) * size * (size-1) / 2))
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
        
        if !isnothing(abs_tol) && abs(new_connectivity - connectivity_goal) < abs_tol
            return CausalSets.to_bitarray_causet(tcg_new)
        end

        if !isnothing(rel_tol) && abs(new_connectivity - connectivity_goal) / connectivity_goal < rel_tol
            return CausalSets.to_bitarray_causet(tcg_new)
        end

        # Decide whether to accept or reject the new state based on connectivity and Metropolis criterion
        if (new_connectivity - connectivity_goal)^2 <= (prev_connectivity - connectivity_goal)^2 || rand(rng) < 2. ^(5e5*(abs(prev_connectivity - connectivity_goal)-abs(new_connectivity - connectivity_goal)))
            # Accept the modification:
            tcg = tcg_new
            prev_connectivity = new_connectivity
        else
            # Reject the modification: revert flipped edges
            for flip in 1:flips_per_step
                graph.edges[i[flip]][j[flip]] = prev_edge[flip]
            end
        end
        step += 1
    end
    @warn "Relative precision $(rel_tol) or absolute precision $(abs_tol) not reached after $(markov_steps) steps. Final connectivity error: $(abs(prev_connectivity - connectivity_goal))"
    # Return the sampled causet
    return CausalSets.to_bitarray_causet(tcg)
end