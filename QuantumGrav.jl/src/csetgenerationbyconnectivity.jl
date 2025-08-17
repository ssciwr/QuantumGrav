"""
# `flip_param_determiner`

A 2D spline interpolator that maps a given `(connectivity_goal, size)` pair to an estimated `flip_param`.

# input
- `connectivity_goal` (`Float64`): target connectivity ratio in `[0,1]`.
- `size` (`Int64`): number of nodes in the causet.

# Returns 
- interpolated `flip_param` (`Float64`).

This piecewise-linear, exact spline is built on a full grid with shape `(13, 6)`.
"""
struct FlipParamDeterminer
    grid_connectivity_goal::Vector{Float64}
    grid_size::Vector{Int64}
    grid_flip_param::Matrix{Float64}
end

function FlipParamDeterminer()
    grid_connectivity_goal = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.95,
                              0.99]  # connectivity_goal grid (ascending)
    grid_size = [128, 256, 512, 1024, 2048, 4096]  # size grid (ascending)
    grid_flip_param = [0.05 0.04 0.025 0.015 0.013 0.012;
                       0.03 0.035 0.026 0.016 0.013 0.007;
                       0.02 0.03 0.023 0.012 0.01 0.005;
                       0.025 0.02 0.024 0.015 0.01 0.006;
                       0.025 0.025 0.025 0.02 0.014 0.008;
                       0.03 0.035 0.03 0.025 0.015 0.008;
                       0.06 0.04 0.04 0.027 0.016 0.009;
                       0.1 0.07 0.05 0.032 0.019 0.01;
                       0.17 0.1 0.065 0.042 0.025 0.014;
                       0.27 0.18 0.11 0.07 0.04 0.025;
                       0.28 0.19 0.125 0.075 0.045 0.027;
                       0.45 0.32 0.19 0.11 0.07 0.047;
                       1.2 0.7 0.45 0.3 0.3 0.2]
    return FlipParamDeterminer(grid_connectivity_goal, grid_size, grid_flip_param)
end

# Fixed flip_param_determiner (piecewise-bilinear interpolation).
# Clamps out-of-range queries to the nearest grid boundaries.
# Inputs: connectivity_goal::Float64 in [0,1], size::Int64 (number of nodes)
# Returns: Float64 flip_param
@inline function flip_param_determiner(connectivity_goal::Float64, size::Int64)::Float64
    c = FlipParamDeterminer()
    x = clamp(connectivity_goal, c.grid_connectivity_goal[1], c.grid_connectivity_goal[end])
    y = clamp(Float64(size), c.grid_size[1], c.grid_size[end])

    i = clamp(searchsortedlast(c.grid_connectivity_goal, x),
              1,
              length(c.grid_connectivity_goal) - 1)
    j = clamp(searchsortedlast(c.grid_size, y), 1, length(c.grid_size) - 1)

    x1 = c.grid_connectivity_goal[i]
    x2 = c.grid_connectivity_goal[i+1]
    y1 = c.grid_size[j]
    y2 = c.grid_size[j+1]
    tx = x2 == x1 ? 0.0 : (x - x1) / (x2 - x1)
    ty = y2 == y1 ? 0.0 : (y - y1) / (y2 - y1)

    z11 = c.grid_flip_param[i, j]
    z21 = c.grid_flip_param[i+1, j]
    z12 = c.grid_flip_param[i, j+1]
    z22 = c.grid_flip_param[i+1, j+1]

    return (1 - tx) * (1 - ty) * z11 +
           tx * (1 - ty) * z21 +
           (1 - tx) * ty * z12 +
           tx * ty * z22
end
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
- `acceptance::Float64`: Acceptance parameter for the Metropolis criterion.
# Returns
- A bitarray causet sampled according to the connectivity goal.
- A boolean indicating whether the sampling was successful.
"""

function sample_bitarray_causet_by_connectivity(size::Int64,
                                                connectivity_goal::Float64,
                                                markov_steps::Int64,
                                                rng::Random.AbstractRNG;
                                                rel_tol::Union{Float64,Nothing}=nothing,
                                                abs_tol::Union{Float64,Nothing}=nothing,
                                                acceptance::Float64=5e5,)::Tuple{CausalSets.BitArrayCauset,
                                                                                 Bool}
    if size < 1
        throw(ArgumentError("size must be larger than 0, is $(size)"))
    end

    if connectivity_goal > 1 || connectivity_goal < 1e-9 # use small number here -> everything below is considered zero
        throw(ArgumentError("connectivity_goal has to be in (0,1], is $(connectivity_goal)"))
    end

    if markov_steps < 1
        throw(ArgumentError("markov_steps has to be at least 1, is $(markov_steps)"))
    end

    if abs_tol === nothing && rel_tol === nothing
        @warn "Neither abs_tol nor rel_tol set, using markov_steps as stopping criterion."
    end

    if abs_tol !== nothing && rel_tol !== nothing
        throw(ArgumentError("Only one of abs_tol or rel_tol can be set, got abs_tol=$(abs_tol) and rel_tol=$(rel_tol)"))
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
    tcg_new = CausalSets.empty_graph(size)

    # Compute initial connectivity
    prev_connectivity = CausalSets.count_edges(tcg) / (size * (size - 1) / 2)

    step = 1
    while step < markov_steps
        flips_per_step = Int64(ceil(flip_param *
                                    abs(prev_connectivity - connectivity_goal) *
                                    size *
                                    (size - 1) / 2))

        # Randomly select edges to flip
        i = [rand(rng, 1:(size - 1)) for flip in 1:flips_per_step]
        j = [rand(rng, (i[flip] + 1):size) for flip in 1:flips_per_step]

        # Store previous edge states for possible rollback
        prev_edges = [graph.edges[i[flip]][j[flip]] for flip in 1:flips_per_step]

        # Flip selected edges
        for flip in 1:flips_per_step
            graph.edges[i[flip]][j[flip]] = !prev_edges[flip]
        end

        # Restore transitivity
        CausalSets.transitive_closure!(graph, tcg_new)

        # Compute new connectivity after flips
        new_connectivity = CausalSets.count_edges(tcg_new) / (size * (size - 1) / 2)

        if abs_tol !== nothing && abs(new_connectivity - connectivity_goal) < abs_tol
            return CausalSets.to_bitarray_causet(tcg_new), true
        end

        if rel_tol !== nothing &&
           abs(new_connectivity - connectivity_goal) / connectivity_goal < rel_tol
            return CausalSets.to_bitarray_causet(tcg_new), true
        end

        # Decide whether to accept or reject the new state based on connectivity and Metropolis criterion
        # Decide whether to accept or reject the new state based on connectivity and Metropolis criterion
        e_old = abs(prev_connectivity - connectivity_goal)
        e_new = abs(new_connectivity - connectivity_goal)
        delta_e = e_old - e_new

        if e_new <= e_old || rand(rng) < min(1.0, exp(acceptance * delta_e))
            # Accept the modification:
            tcg = tcg_new
            prev_connectivity = new_connectivity
        else
            # Reject the modification: revert flipped edges
            for flip in 1:flips_per_step
                graph.edges[i[flip]][j[flip]] = prev_edges[flip]
            end
        end
        step += 1
    end

    @warn "Relative precision $(rel_tol) or absolute precision $(abs_tol) not reached after $(markov_steps) steps. Final connectivity error: $(abs(prev_connectivity - connectivity_goal))"

    # Return the sampled causet
    return CausalSets.to_bitarray_causet(tcg), false
end
