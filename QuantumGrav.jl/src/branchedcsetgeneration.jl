"""
    generate_random_branch_points(sprinkling::Vector{Coordinates{N}}, n::Int; rng::AbstractRNG = Random.default_rng()) -> Vector{Coordinates{N}}

Randomly selects `n` distinct coordinate points from the given `sprinkling` to serve as branch points.  
The returned branch points are sorted by coordinate time.

# Arguments
- `sprinkling::Vector{Coordinates{N}}`: List of sprinkled points in spacetime.
- `n::Int`: Number of branch points to select (must be ≥ 0 and ≤ length of sprinkling).
- `rng::AbstractRNG`: Optional RNG.

# Returns
- `Vector{Coordinates{N}}`: `n` randomly selected and time-ordered branch points.

# Throws
- `ArgumentError` if `n > length(sprinkling)` or `n < 0`
"""
function generate_random_branch_points(sprinkling::Vector{CausalSets.Coordinates{N}}, n::Int; rng::Random.AbstractRNG = Random.default_rng()) where {N}
    n > length(sprinkling) && throw(ArgumentError("Requested $n branch points, but sprinkling only has $(length(sprinkling)) elements."))
    n < 0 && throw(ArgumentError("n must be at least 0, is $(n)."))
    
    idxs = rand(rng, 1:length(sprinkling), n)
    selected = sprinkling[idxs]
    return sort(selected, by = x -> x[1])  # sort by coordinate time
end

"""
    in_past_of(manifold::ConformallyTimesliceableManifold{2}, x::Coordinates{2}, y::Coordinates{2}, branch_points::Vector{Coordinates{2}}) -> Bool

Determines whether the point `x` is in the causal past of `y` in a 2D conformally timesliceable manifold with possible branching singularities.

This function extends the standard causal relation by enforcing topological constraints from vertical branch cuts:
- It first checks whether `x` is causally related to `y` in the usual spacetime sense (i.e., `x ≺ y`).
- Then, it examines whether any `branch_point ∈ branch_points` lies strictly spatially between `x` and `y`.
- If such a branch point exists and is **not in the causal future of `x`**, the causal path from `x` to `y` is considered obstructed and the function returns `false`.

This models scenarios like topology change (e.g., trousers) where causal curves cannot cross unconnected regions without violating causal consistency.

# Arguments
- `manifold::ConformallyTimesliceableManifold{2}`: The spacetime background.
- `x::Coordinates{2}`: Potential past event.
- `y::Coordinates{2}`: Potential future event.
- `branch_points::Vector{Coordinates{2}}`: Locations of topological branch points, assumed to induce vertical (time-directed) branch cuts.

# Returns
- `Bool`: `true` if `x` is causally in the past of `y` and no branch cut obstructs the path; `false` otherwise.

# Throws
- Nothing, but assumes `branch_points` are valid coordinate tuples in 2D spacetime.
"""
function CausalSets.in_past_of(
    manifold::CausalSets.ConformallyTimesliceableManifold{2},
    x::CausalSets.Coordinates{2},
    y::CausalSets.Coordinates{2},
    branch_points::Vector{CausalSets.Coordinates{2}},
)::Bool
    # Check standard causal relation
    CausalSets.in_past_of(manifold, x, y) || return false

    # Check for branch cuts that obstruct causality
    for b in branch_points
        if min(x[2], y[2]) < b[2] < max(x[2], y[2])
            CausalSets.in_past_of(manifold, x, b) || return false
        end
    end
    return true
end

struct BranchedManifoldCauset{N, M} <: CausalSets.AbstractCauset where {N, M<:CausalSets.AbstractManifold}
    atom_count::Int64
    manifold::M
    sprinkling::Vector{CausalSets.Coordinates{N}}
    branch_points::Vector{CausalSets.Coordinates{N}}
end

"""
    BranchedManifoldCauset(manifold, branched_sprinkling, branch_relations)

Create a `BranchedManifoldCauset` from a `manifold`, a list of `BranchedCoord{N}`, and a branch causality matrix.
"""
function BranchedManifoldCauset(
    manifold::M,
    sprinkling::Vector{CausalSets.Coordinates{N}},
    branch_points::Vector{CausalSets.Coordinates{N}},
)::BranchedManifoldCauset{N, M} where {N, M<:CausalSets.AbstractManifold{N}}
    return BranchedManifoldCauset{N, M}(length(sprinkling), manifold, sprinkling, branch_points)
end

"""
    in_past_of_unchecked(causet::BranchedManifoldCauset, i::Int, j::Int) -> Bool

Returns `true` if element `i` is in the past of element `j`, based on both spacetime and branch relations.
"""
function CausalSets.in_past_of_unchecked(causet::BranchedManifoldCauset, i::Int, j::Int)::Bool
    x = causet.sprinkling[i]
    y = causet.sprinkling[j]
    return CausalSets.in_past_of(causet.manifold, x, y, causet.branch_points)
end

"""
    CausalSets.BitArrayCauset(causet::BranchedManifoldCauset)

Constructs a `BitArrayCauset` by computing causal relations from a `BranchedManifoldCauset`.
"""
function CausalSets.BitArrayCauset(causet::BranchedManifoldCauset{N})::CausalSets.BitArrayCauset where {N}
    return convert(CausalSets.BitArrayCauset, causet)
end

"""
    convert(BitArrayCauset, causet::BranchedManifoldCauset)

Computes the causal matrix for a `BranchedManifoldCauset` and returns a `BitArrayCauset`.
"""
function Base.convert(::Type{CausalSets.BitArrayCauset}, causet::BranchedManifoldCauset{N})::CausalSets.BitArrayCauset where {N}
    atom_count = causet.atom_count

    future_relations = Vector{BitVector}(undef, atom_count)
    past_relations = Vector{BitVector}(undef, atom_count)

    for i in 1:atom_count
        future_relations[i] = falses(atom_count)
        past_relations[i] = falses(atom_count)
    end

    Threads.@threads for i in 1:atom_count
        for j in i+1:atom_count
            if CausalSets.in_past_of(causet.manifold, causet.sprinkling[i], causet.sprinkling[j], causet.branch_points)
                future_relations[i][j] = true
                past_relations[j][i] = true
            end
        end
    end

    return CausalSets.BitArrayCauset(atom_count, future_relations, past_relations)
end

"""
    make_branched_manifold_cset(
        npoints::Int64,
        nbranchpoints::Int64,
        rng::Random.AbstractRNG,
        order::Int64,
        r::Float64;
        d::Int64 = 2,
        type::Type{T} = Float32,
    )::Tuple{CausalSets.BitArrayCauset, Vector{Tuple{T,Vararg{T}}}, Matrix{T}}

Generates a causal set embedded into a branched 2D polynomial manifold with `nbranchpoints` topological branch points. These divide the spacetime into causally separated sectors.

# Arguments
- `npoints::Int64`: Number of sprinkled points in the spacetime. Must be > 0.
- `nbranchpoints::Int64`: Number of branch points to insert. Must be ≥ 1 and ≤ `npoints`.
- `rng::AbstractRNG`: Random number generator used for reproducibility.
- `order::Int64`: Truncation order for the Chebyshev expansion. Must be > 0.
- `r::Float64`: Decay base for the Chebyshev coefficients. Must be > 1.
- `d::Int64`: Dimension of the spacetime. Only `d = 2` is currently supported.
- `type::Type{T}`: Type of the output coordinates and coefficient matrix.

# Returns
- A tuple `(cset, sprinkling, chebyshev_coefs)` where:
    - `cset`: The `CausalSets.BitArrayCauset` for the branched manifold.
    - `sprinkling`: The original list of coordinates (before branch assignment).
    - `chebyshev_coefs`: The matrix of Chebyshev coefficients used to define the manifold.

# Throws
- `ArgumentError` if:
    - `npoints <= 0`
    - `nbranchpoints < 1` or `nbranchpoints > npoints`
    - `order <= 0`
    - `r <= 1`
    - `d ≠ 2`
"""
function make_branched_manifold_cset(
    npoints::Int64,
    nbranchpoints::Int64,
    rng::Random.AbstractRNG,
    order::Int64,
    r::Float64;
    d::Int64 = 2,
    type::Type{T} = Float32,
)::Tuple{CausalSets.BitArrayCauset,Vector{Tuple{T,Vararg{T}}},Matrix{T}} where {T<:Number}

    if npoints <= 0
        throw(ArgumentError("npoints must be greater than 0, got $npoints"))
    end

    if nbranchpoints < 0 || nbranchpoints > npoints
        throw(ArgumentError("nbranchpoints must be between 0 and npoints = $npoints, got $nbranchpoints"))
    end

    if order <= 0
        throw(ArgumentError("order must be greater than 0, got $order"))
    end

    if r <= 1
        throw(
            ArgumentError(
                "r must be greater than 1 for exponential convergence of the Chebyshev series, got $r",
            ),
        )
    end

    if d != 2
        throw(ArgumentError("Currently, only 2D is supported, got $d"))
    end

    # Generate a matrix of random Chebyshev coefficients that decay exponentially with base r
    # it has to be a (order x order)-matrix because we describe a function of two variables
    chebyshev_coefs = zeros(Float64, order, order)
    for i = 1:order
        for j = 1:order
            chebyshev_coefs[i, j] = r^(-i - j) * Random.randn(rng)
        end
    end

    # Construct the Chebyshev-to-Taylor transformation matrix
    cheb_to_taylor_mat = CausalSets.chebyshev_coef_matrix(order - 1)

    # Transform Chebyshev coefficients to Taylor coefficients
    taylorcoefs = CausalSets.transform_polynomial(chebyshev_coefs, cheb_to_taylor_mat)

    # Square the polynomial to ensure positivity
    squaretaylorcoefs = CausalSets.polynomial_pow(taylorcoefs, 2)

    # Create a polynomial manifold from the squared Taylor coefficients
    polym = CausalSets.PolynomialManifold{d}(squaretaylorcoefs)

    # Define the square box boundary in 2D -- this works only in 2D and with square box boundary at the moment
    boundary = CausalSets.BoxBoundary{d}(((-1.0, -1.0), (1.0, 1.0)))

    # Generate a sprinkling of npoints in the manifold within the boundary
    sprinkling = CausalSets.generate_sprinkling(polym, boundary, npoints)

    # Randomly promote nbranchpoints of the sprinkling points to branch points
    branch_points = generate_random_branch_points(sprinkling, nbranchpoints)

    # Construct the causal set from the manifold and sprinkling
    cset = BranchedManifoldCauset(polym, sprinkling, branch_points)

    return CausalSets.BitArrayCauset(cset), sprinkling, type.(chebyshev_coefs)
end