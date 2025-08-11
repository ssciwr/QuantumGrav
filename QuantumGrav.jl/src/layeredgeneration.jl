"""
    gaussian_dist_cuts(N, n, σ; rng=Random.GLOBAL_RNG)

Create cut points between layers drawn from a Gaussian distribution centered on equal partition sizes.

Inputs:
    N :: Int — total number of elements to partition
    n :: Int — number of layers
    σ :: Float64 — standard deviation for Gaussian offsets from equal partitioning
    rng :: AbstractRNG — random number generator (default: Random.GLOBAL_RNG)

Returns:
    cuts :: Vector{Int} — list of cut indices separating layers (length n-1)
"""
function gaussian_dist_cuts(N, n, σ; rng=Random.GLOBAL_RNG)
    μs = zeros(n-1)
    μs[1] = N/n
    for i in 2:n-1
        μs[i] = μs[i-1] + N/n
    end
    cuts = [round(Int,μs[i] + σ * randn(rng)) for i in 1:n-1]
    return cuts
end


"""
    layered_causet(N, n; p=0.5, rng=Random.GLOBAL_RNG)

Generate an n-layered causal set with N elements,
randomly partitioned into n layers,
with each potential link between adjacent layers
included independently with probability p.
Layer sizes are drawn from a Gaussian distribution 
centered around equal partition size.

Inputs:
    N :: Int — total number of elements in the causal set
    n :: Int — number of layers
    p :: Float64 — probability for a link to exist between elements in successive layers
    rng :: AbstractRNG — random number generator to use (default: Random.GLOBAL_RNG)
    standard_deviation :: Float64: standard deviation of the Gaussian around equal partitioning

Returns:
    tcg :: BitArrayCauset - randomly produced layered causal set
    atoms_per_layer :: Vector{Int64} - number of atoms per layer sorted from past to future

Notes:
    The `standard_deviation` keyword controls the spread of the Gaussian in partitioning; if not provided, defaults to `0.1 * N / n`.
    Layer sizes are resampled until within bounds to avoid bias from clamping.
"""

function create_random_layered_causet(N, n; p::Float64 = 0.5, rng::AbstractRNG = Random.GLOBAL_RNG, standard_deviation::Union{Float64,Nothing} = nothing)
    
    @assert N >= n  "N (number of atoms) must be at least n (number of layers)."
    @assert N >= 1  "N (number of atoms) must be ≥ 1, is $N."
    @assert n >= 1  "n (number of layers) must be ≥ 1., is $n."
    @assert 0 < p <= 1  "p must be in (0,1], is $p."
    @assert 0 < standard_deviation "standard_deviation must be >0, is $standard_deviation."

    σ = isnothing(standard_deviation) ? 0.1 * N / n : standard_deviation

    # Random partition into n layers, Gaussian centered around equal partition
    cuts = gaussian_dist_cuts(N, n, σ; rng = rng)

    sizes = diff([0; cuts; N])

    layers = Vector{Vector{Int}}(undef, n)
    idx = 1
    for i in 1:n
        layers[i] = collect(idx:(idx + sizes[i] - 1))
        idx += sizes[i]
    end

    graph = CausalSets.empty_graph(N)   # link matrix
    tcg = CausalSets.empty_graph(N)     # covering relations

    # Random links between successive layers
    for i in 1:(n-1)
        for a in layers[i], b in layers[i+1]
            if rand(rng) < p
                graph.edges[a][b] = true
            end
        end
    end

    # number of atoms per layer

    atoms_per_layer = length.(layers)

    # Transitive closure
    CausalSets.transitive_closure!(graph,tcg)

    return CausalSets.to_bitarray_causet(tcg), atoms_per_layer
end