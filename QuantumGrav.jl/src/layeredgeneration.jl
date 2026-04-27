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
function gaussian_dist_cuts(N::Int64, n::Int64, σ::Float64; rng = Random.GLOBAL_RNG)
    if N < 2 * σ * n
        @warn "N is less than 2×σ×n; partitions may be biased to have more points in earlier layers than in later ones."
    end
    i = 1
    cuts = zeros(n)
    while i < n
        μ = cuts[i]+(N - cuts[i])/(n - i + 1)
        cuts[i+1] = round(Int, μ + σ * randn(rng))
        if N + i - n > cuts[i+1] > cuts[i]
            i+=1
        end
    end
    popfirst!(cuts)
    return cuts
end

"""
    create_KR_order(N; rng=Random.GLOBAL_RNG)

Generate a Kleitman–Rothschild (KR) order with N elements.

The causal set consists of exactly three layers with approximately fixed proportions
(1/4, 1/2, 1/4) of the total size, and links are placed
independently between adjacent layers with probability 1/2.

Inputs:
    N :: Int — total number of elements
    rng :: AbstractRNG — random number generator (default: Random.GLOBAL_RNG)

Returns:
    tcg :: BitArrayCauset — generated KR-order causal set
    atoms_per_layer :: Vector{Int64} — number of atoms per layer (length 3)
"""
function create_KR_order(
    N::Int64;
    rng::Random.AbstractRNG = Random.GLOBAL_RNG,
)
    if N < 3
        throw(ArgumentError("N must be at least 3 to construct a KR order, is $N."))
    end

    # assign each element independently to layers with probabilities (1/4, 1/2, 1/4)
    layers = [Int[] for _ in 1:3]

    for i in 1:N
        u = rand(rng)
        if u ≤ 0.25
            push!(layers[1], i)
        elseif u ≤ 0.75
            push!(layers[2], i)
        else
            push!(layers[3], i)
        end
    end

    atoms_per_layer = length.(layers)

    # impose layer-respecting topological ordering
    perm = vcat(layers[1], layers[2], layers[3])
    invperm = zeros(Int, N)
    for (new, old) in enumerate(perm)
        invperm[old] = new
    end

    graph = CausalSets.empty_graph(N)
    tcg = CausalSets.empty_graph(N)

    # link probability 1/2 between adjacent layers, relabeled by invperm
    for i = 1:2
        for a in layers[i], b in layers[i+1]
            if rand(rng) < 0.5
                graph.edges[invperm[a]][invperm[b]] = true
            end
        end
    end

    CausalSets.transitive_closure!(graph, tcg)

    return CausalSets.to_bitarray_causet(tcg), atoms_per_layer
end

"""
    create_random_layered_causet(N, n; p=0.5, rng=Random.GLOBAL_RNG)

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
function create_random_layered_causet(
    N::Int64,
    n::Int64;
    p::Float64 = 0.5,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG,
    standard_deviation::Union{Float64,Nothing} = nothing,
)

    if N < n
        throw(ArgumentError("N (number of atoms) must be at least n (number of layers)."))
    end
    if N < 1
        throw(ArgumentError("N (number of atoms) must be ≥ 1, is $N."))
    end
    if n < 1
        throw(ArgumentError("n (number of layers) must be ≥ 1, is $n."))
    end
    if !(0 < p <= 1)
        throw(ArgumentError("p must be in (0,1], is $p."))
    end
    if !(isnothing(standard_deviation) || standard_deviation > 0)
        throw(ArgumentError("standard_deviation must be >0, is $standard_deviation."))
    end

    σ = isnothing(standard_deviation) ? 0.1 * N / n : standard_deviation

    # Random partition into n layers, Gaussian centered around equal partition
    cuts = gaussian_dist_cuts(N, n, σ; rng = rng)

    sizes = diff([0; cuts; N])

    layers = Vector{Vector{Int}}(undef, n)
    idx = 1
    for i = 1:n
        layers[i] = collect(idx:(idx+sizes[i]-1))
        idx += sizes[i]
    end

    graph = CausalSets.empty_graph(N)   # link matrix
    tcg = CausalSets.empty_graph(N)     # covering relations

    # Random links between successive layers
    for i = 1:(n-1)
        for a in layers[i], b in layers[i+1]
            if rand(rng) < p
                graph.edges[a][b] = true
            end
        end
    end

    # number of atoms per layer

    atoms_per_layer = length.(layers)

    # Transitive closure
    CausalSets.transitive_closure!(graph, tcg)

    return CausalSets.to_bitarray_causet(tcg), atoms_per_layer
end
