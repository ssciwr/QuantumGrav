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
    bool_mul2(A::BitMatrix, B::BitMatrix) -> BitMatrix

Compute the Boolean matrix product of `A` and `B`.

# Arguments
- `A`: Left Boolean matrix.
- `B`: Right Boolean matrix.

# Throws
- `ArgumentError`: If the inner matrix dimensions do not agree.
"""
function bool_mul2(A::BitMatrix, B::BitMatrix)
    mA, nA = size(A)
    mB, nB = size(B)
    nA == mB || throw(ArgumentError(
        "inner dimensions must agree for Boolean matrix multiplication, got $(size(A)) and $(size(B)).",
    ))
    AB = BitArray(undef, mA, nB)
    for i in 1:mA, j in 1:nB
        AB[i,j] = any(A[i,k] && B[k,j] for k in 1:nA)
    end
    AB
end

function KR_poset_from_blocks(
    n::Int64,
    layer_counts::AbstractVector{<:Integer},
    bottom_to_middle::BitMatrix,
    middle_to_top::BitMatrix,
    bottom_to_top::BitMatrix,
)
    n_bottom, n_middle, n_top = layer_counts
    mid_start = n_bottom + 1
    top_start = n_bottom + n_middle + 1

    future_relations = [falses(n) for _ in 1:n]
    past_relations = [falses(n) for _ in 1:n]

    @inbounds for i in 1:n_bottom
        if n_middle > 0
            future_relations[i][mid_start:(top_start - 1)] = bottom_to_middle[i, :]
        end
        if n_top > 0
            future_relations[i][top_start:n] = bottom_to_top[i, :]
        end
    end

    @inbounds for m_local in 1:n_middle
        m = n_bottom + m_local
        if n_top > 0
            future_relations[m][top_start:n] = middle_to_top[m_local, :]
        end
    end

    @inbounds for m_local in 1:n_middle
        m = n_bottom + m_local
        if n_bottom > 0
            past_relations[m][1:n_bottom] = bottom_to_middle[:, m_local]
        end
    end

    @inbounds for t_local in 1:n_top
        t = top_start + t_local - 1
        if n_bottom > 0
            past_relations[t][1:n_bottom] = bottom_to_top[:, t_local]
        end
        if n_middle > 0
            past_relations[t][mid_start:(top_start - 1)] = middle_to_top[:, t_local]
        end
    end

    return CausalSets.BitArrayCauset(n, future_relations, past_relations)
end

function normalized_KR_order_size(N::Int64)::Int64
    N > 0 || throw(ArgumentError("N must be positive to construct a KR order, is $N."))
    if N < 3
        @warn "KR orders need at least 3 elements; using 3 instead of $N."
        return 3
    end
    return N
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
    N = normalized_KR_order_size(N)

    atoms_per_layer = rand(rng, Distributions.Multinomial(N, [0.25, 0.5, 0.25]))
    bottom_to_middle = Random.bitrand(rng, atoms_per_layer[1], atoms_per_layer[2])
    middle_to_top = Random.bitrand(rng, atoms_per_layer[2], atoms_per_layer[3])
    bottom_to_top = bool_mul2(bottom_to_middle, middle_to_top)

    return KR_poset_from_blocks(
        N,
        atoms_per_layer,
        bottom_to_middle,
        middle_to_top,
        bottom_to_top,
    ),
    atoms_per_layer
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
