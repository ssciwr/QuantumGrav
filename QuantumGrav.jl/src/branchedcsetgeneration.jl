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
    assign_branch(point::Coordinates{2}, branch_points::Vector{Coordinates{2}}, manifold::ConformallyTimesliceableManifold{2}) -> Int

Assigns a branch index to a spacetime point `point` relative to a list of `branch_points`, which represent singular vertical branch cuts starting at each branch point and extending toward the future (i.e. increasing time at fixed spatial coordinate).

This function defines a piecewise-topological structure on spacetime as follows:

1. If `point` lies in the causal past of one of the `branch_points`, then it belongs to the branch indexed by the **earliest such branch point** in time. That is, iterate through the `branch_points` in increasing coordinate time and assign the branch index `i` as soon as `point ≺ branch_points[i]`.

2. If `point` is **not** in the causal past of any branch point, then it lies in one of the n+1 **future spatial sectors**, determined by the spatial (x-coordinate) position of the `branch_points`.

   - First, the `branch_points` are sorted by spatial coordinate.
   - If `point.x < branch_points[1].x`, assign branch index `n + 1`.
   - If `point.x ∈ [branch_points[i].x, branch_points[i+1].x)`, assign branch index `n + 1 + i`.
   - If `point.x > branch_points[end].x`, assign branch index `2n + 1`.

The total number of branches is therefore `2n + 1`: 
- `n` for the temporal pasts of the branch points, 
- `n+1` for the spatial sectors above the branch cuts.

# Arguments
- `point::Coordinates{2}`: The spacetime point to classify.
- `branch_points::Vector{Coordinates{2}}`: Branch points sorted by coordinate time (first component). Must not be empty.
- `manifold::ConformallyTimesliceableManifold{2}`: The manifold on which to evaluate causal relations.

# Returns
- `Int`: The branch index of the point (in 1 : 2n+1).

# Throws
- `ArgumentError` if `branch_points` are not sorted by coordinate time.
"""
function assign_branch(point::CausalSets.Coordinates{2}, branch_points::Vector{CausalSets.Coordinates{2}}, manifold::CausalSets.ConformallyTimesliceableManifold{2})::Int
    n = length(branch_points)

    # Check monotonicity in time
    for i in 2:n
        if branch_points[i-1][1] > branch_points[i][1]
            throw(ArgumentError("branch_points must be sorted by coordinate time (increasing first component)"))
        end
    end

    # 1. Temporal assignment
    for i in 1:n
        if CausalSets.in_past_of(manifold, point, branch_points[i])
            return i
        end
    end

    # 2. Spatial assignment
    sorted = sort(branch_points, by = x -> x[2])  # sort by x (spatial)
    px = point[2]

    if px < sorted[1][2]
        return n + 1
    end
    for i in 1:(n - 1)
        if sorted[i][2] ≤ px < sorted[i + 1][2]
            return n + 1 + i
        end
    end
    return 2n + 1
end

struct BranchedCoordinates{N}
    coord::CausalSets.Coordinates{N}
    branch::Int 
end

"""
    compute_branch_relations(manifold::ConformallyTimesliceableManifold{2}, branch_points::Vector{Coordinates{2}}) -> Vector{BitVector}

Computes the causal relationships between branches (only in 2D):
- For branches 1 to n, branch i is in the past of j if it's causally in the past and no spatially intermediate branch cut blocks the path.
- For branches n+1 to 2n+1, the branch inherits the full causal past of its mother branches (either one or two, based on spatial sector).

# Arguments
- `manifold::ConformallyTimesliceableManifold{2}`: Manifold to evaluate causal structure.
- `branch_points::Vector{Coordinates{2}}`: Sorted list of branch points (by time).

# Returns
- `Vector{BitVector}`: Past-relation matrix `B[i][j] = true` iff branch `i` is in the past of branch `j`.

# Throws
- `ArgumentError` if `branch_points` are not sorted by time (first coordinate).
"""
function compute_branch_relations(
    manifold::CausalSets.ConformallyTimesliceableManifold{2},
    branch_points::Vector{CausalSets.Coordinates{2}},
)::Vector{BitVector}
    n = length(branch_points)

    # Check monotonicity in time
    for i in 2:n
        if branch_points[i-1][1] > branch_points[i][1]
            throw(ArgumentError("branch_points must be sorted by coordinate time (increasing first component)"))
        end
    end

    total = 2n + 1
    B = [falses(total) for _ in 1:total]  # Vector{BitVector}

    # === Step 1: base causal relations for branches 1..n ===
    B[1] .= true  # everything is in future of branch 1

    for i in 2:n
        B[i][i] = true
        xᵢ = branch_points[i]
        for j in (i + 1):n
            xⱼ = branch_points[j]
            CausalSets.in_past_of(manifold, xᵢ, xⱼ) || continue
            obstructed = any(
                k -> min(xᵢ[2], xⱼ[2]) < branch_points[k][2] < max(xᵢ[2], xⱼ[2]) &&
                     !CausalSets.in_past_of(manifold, xᵢ, branch_points[k]),
                (i + 1):(j - 1)
            )
            B[i][j] = !obstructed
        end
    end

    # === Step 2: assign future sectors (branches n+1..2n+1) ===
    sorted_idxs = sort(1:length(branch_points); by = i -> branch_points[i][2])

    for b in (n + 1):(2n + 1)
        sector = b - n
        if sector == 1
            mothers = (sorted_idxs[1][1],)
        elseif sector == n + 1
            mothers = (sorted_idxs[end][1],)
        else
            mothers = (sorted_idxs[sector - 1][1], sorted_idxs[sector][1])
        end

        # Reflexivity
        B[b][b] = true

        # Add links from all mothers
        for m in mothers
            B[m][b] = true
        end

        # Inherit pasts of both mothers
        for k in 1:n
            B[k][b] = any(m -> B[k][m], mothers)
        end
    end

    return B
end

"""
    in_past_of(manifold, x::BranchedCoordinates{N}, y::BranchedCoordinates{N}, branch_relations::Vector{BitVector}) -> Bool

Returns `true` if `x` is in the causal past of `y`, considering both the spacetime lightcone structure and the causal structure of the branches.

# Arguments
- `manifold::ConformallyTimesliceableManifold{N}`: The underlying spacetime.
- `x, y::BranchedCoordinates{N}`: The two spacetime points with branch labels.
- `branch_relations::Vector{BitVector}`: Branch causal relations.

# Returns
- `Bool`: `true` if `x ≺ y` according to both spacetime and branch topology.

# Throws
- `ArgumentError` if either `x.branch` or `y.branch` is out of bounds (i.e., < 1 or > length(branch_relations)).
"""
function CausalSets.in_past_of(
    manifold::CausalSets.ConformallyTimesliceableManifold{N},
    x::BranchedCoordinates{N},
    y::BranchedCoordinates{N},
    branch_relations::Vector{BitVector},
)::Bool where {N}
    total_branches = length(branch_relations)
    if x.branch < 1 || x.branch > total_branches
        throw(ArgumentError("Invalid branch index x.branch = $(x.branch). Must be in 1:$(total_branches)."))
    end
    if y.branch < 1 || y.branch > total_branches
        throw(ArgumentError("Invalid branch index y.branch = $(y.branch). Must be in 1:$(total_branches)."))
    end
    # Check branch causality
    branch_relations[x.branch][y.branch] || return false

    # Check standard spacetime causality
    return CausalSets.in_past_of(manifold, x.coord, y.coord)
end

struct BranchedManifoldCauset{N, M} <: CausalSets.AbstractCauset where {N, M<:CausalSets.AbstractManifold}
    atom_count::Int64
    manifold::M
    sprinkling::Vector{BranchedCoordinates{N}}
    branch_relations::Vector{BitVector}
end

"""
    BranchedManifoldCauset(manifold, branched_sprinkling, branch_relations)

Create a `BranchedManifoldCauset` from a `manifold`, a list of `BranchedCoord{N}`, and a branch causality matrix.
"""
function BranchedManifoldCauset(
    manifold::M,
    sprinkling::Vector{BranchedCoordinates{N}},
    branch_relations::Vector{BitVector},
)::BranchedManifoldCauset{N, M} where {N, M<:CausalSets.AbstractManifold{N}}
    return BranchedManifoldCauset{N, M}(length(sprinkling), manifold, sprinkling, branch_relations)
end

"""
    in_past_of_unchecked(causet::BranchedManifoldCauset, i::Int, j::Int) -> Bool

Returns `true` if element `i` is in the past of element `j`, based on both spacetime and branch relations.
"""
function CausalSets.in_past_of_unchecked(causet::BranchedManifoldCauset, i::Int, j::Int)::Bool
    x = causet.sprinkling[i]
    y = causet.sprinkling[j]
    return CausalSets.in_past_of(causet.manifold, x, y, causet.branch_relations)
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
            if CausalSets.in_past_of(causet.manifold, causet.sprinkling[i], causet.sprinkling[j], causet.branch_relations)
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

    if nbranchpoints < 1 || nbranchpoints > npoints
        throw(ArgumentError("nbranchpoints must be between 1 and npoints = $npoints, got $nbranchpoints"))
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
    branch_points = generate_branch_points(sprinkling, nbranchpoints)

    # assign a branch to every point in sprinkling
    branched_sprinkling = [BranchedCoordinates{d}(sprinkling[i], assign_branch(sprinkling[i], branch_points, polym)) for i in 1:length(sprinkling)]

    # compute branch-relation matrix
    rel = compute_branch_relations(polym, branch_points)

    # Construct the causal set from the manifold and sprinkling
    cset = BranchedManifoldCauset(polym, branched_sprinkling, rel)

    return cset, sprinkling, type.(chebyshev_coefs)
end