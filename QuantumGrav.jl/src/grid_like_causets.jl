"""
    generate_grid_from_brillouin_cell(num_atoms, edges; origin=nothing) -> Vector{CausalSets.Coordinates{N}}

Construct an N‑dimensional regular grid of approximately `num_atoms` points that fills the Brillouin cell spanned by `N` edge vectors `edges`.

# Arguments
- `num_atoms::Int`: desired number of grid points (≥1). The implementation lays down a Cartesian parameter grid with ≈ `num_atoms` candidates and trims to exactly `num_atoms`.
- `edges::NTuple{N,CausalSets.Coordinates{N}}`: the `N` spanning edge vectors of a *parallelepiped* cell. Points are of the form
  `x = origin + sum_i t_i * edges[i]` with `t_i ∈ [0,1]`.
- `origin`: optional `NTuple{N,Float64}` origin (vertex) of the cell. If `nothing`, zero is used.

# Returns
- `Vector{CausalSets.Coordinates{N}}`: grid points inside the cell, ordered lexicographically in the parameter tuple `(t₁,…,t_N)`, trimmed to length `num_atoms`.

# Errors
- Throws `ArgumentError` if `num_atoms < 1`.
- Throws `ArgumentError` if the `edges` are not linearly independent (near‑zero determinant).
"""
function generate_grid_from_brillouin_cell(
    num_atoms::Int,
    edges::NTuple{N,CausalSets.Coordinates{N}};
    origin = nothing,
)::Vector{CausalSets.Coordinates{N}} where {N}
    num_atoms > 0 || throw(ArgumentError("num_atoms must be ≥ 1"))

    # Normalize origin keyword (cannot parametrize keyword type with N reliably)
    local origin_nt::NTuple{N,Float64}
    if isnothing(origin)
        origin_nt = ntuple(_->0.0, N)
    else
        origin_nt = convert(NTuple{N,Float64}, origin)
    end

    # Build edge matrix and check linear independence
    E = Matrix{Float64}(undef, N, N)
    for j = 1:N
        for i = 1:N
            E[i, j] = edges[j][i]
        end
    end
    if abs(LinearAlgebra.det(E)) ≤ eps(Float64)
        throw(
            ArgumentError(
                "Brillouin-cell edges must be linearly independent (determinant ≈ 0)",
            ),
        )
    end

    # Choose per-dimension resolution ~ num_atoms^(1/N)
    k = max(2, ceil(Int, num_atoms^(1/N)))
    # total candidate points
    total = k^N

    # Parameter grids in [0,1]
    params = ntuple(_ -> range(0.0, 1.0; length = k), N)

    pts = Vector{CausalSets.Coordinates{N}}(undef, total)
    idx = 0
    for T in Iterators.product(params...)
        # x = origin + E * t
        x = Vector{Float64}(undef, N)
        for i = 1:N
            xi = origin_nt[i]
            @inbounds for j = 1:N
                xi += E[i, j] * T[j]
            end
            x[i] = xi
        end
        idx += 1
        pts[idx] = tuple(x...)
    end

    # Trim or return exactly num_atoms
    return length(pts) == num_atoms ? pts : pts[1:min(num_atoms, length(pts))]
end

"""
    generate_grid_2d(num_atoms, lattice; a=1.0, b=0.5, gamma_deg=60.0, rotate_deg=nothing, origin=(0.0,0.0))
        -> Vector{CausalSets.Coordinates{2}}

Wrapper that builds a 2D Bravais cell from a lattice name and calls
`generate_grid_from_brillouin_cell` to return `num_atoms` points.

Supported `lattice` names (case‑insensitive, with aliases):
- "square", "quadratic":         edges `((a,0), (0,a))`
- "rectangular":                  edges `((a,0), (0,b))`
- "centered-rectangular", "rhombic", "c-rect": primitive edges `((a/2,b/2), (a/2,-b/2))`
- "hexagonal", "triangular":      edges `((a,0), (a/2, a*sqrt(3)/2))`
- "oblique", "monoclinic":       edges `((a,0), (b*cosγ, b*sinγ))`, with `γ = gamma_deg` (degrees)

# Keywords
- `a::Float64=1.0`, `b::Float64=0.5`: lattice constants (for square, only `a` is used).
- `gamma_deg::Float64=60.0`: angle (degrees) between edges for the oblique lattice.
- `rotate_deg=nothing`: rotation (degrees) applied to the basis before grid generation. If `nothing`, an automatic rotation aligns the net growth direction `e₁+e₂` with `+y`.
- `origin::CausalSets.Coordinates{2}=(0.0,0.0)`: origin (vertex) of the cell.
"""
function generate_grid_2d(
    num_atoms::Int,
    lattice::AbstractString;
    a::Float64 = 1.0,
    b::Float64 = 0.5,
    gamma_deg::Float64 = 60.0,
    rotate_deg = nothing,
    origin = (0.0, 0.0),
)::Vector{CausalSets.Coordinates{2}}
    lname = lowercase(strip(lattice))
    if lname in ("square", "quadratic")
        edges = ((a, 0.0), (0.0, a))
    elseif lname == "rectangular"
        edges = ((a, 0.0), (0.0, b))
    elseif lname in ("centered-rectangular", "rhombic", "c-rect")
        edges = ((0.5a, 0.5b), (0.5a, -0.5b))
    elseif lname in ("hexagonal", "triangular")
        edges = ((a, 0.0), (0.5a, 0.5a*sqrt(3.0)))
    elseif lname in ("oblique", "monoclinic")
        γ = deg2rad(gamma_deg)
        edges = ((a, 0.0), (b*cos(γ), b*sin(γ)))
    else
        throw(ArgumentError("Unsupported 2D lattice name: '$lattice'"))
    end

    # Apply rotation: either user-provided (degrees) or auto so that (e1+e2) points upward
    local θ::Float64
    if isnothing(rotate_deg)
        v = (edges[1][1] + edges[2][1], edges[1][2] + edges[2][2])
        φ = atan(v[2], v[1])              # current angle of net growth
        θ = (pi/2) - φ                     # rotate to +y
    else
        θ = deg2rad(Float64(rotate_deg))
    end
    c = cos(θ);
    s = sin(θ)
    R = ((c, -s), (s, c))  # rotation matrix
    rot = x -> (R[1][1]*x[1] + R[1][2]*x[2], R[2][1]*x[1] + R[2][2]*x[2])
    edges = (rot(edges[1]), rot(edges[2]))

    return generate_grid_from_brillouin_cell(num_atoms, edges; origin = origin)
end

"""
    sort_grid_by_time_from_manifold(manifold, preset_atoms) -> Vector{CausalSets.Coordinates{N}}

Stable sort of `preset_atoms` using the manifold‑provided time ordering `CausalSets.isless_coord_time`.
Useful to impose a pseudo‑sprinkling order before constructing a causet.
"""
function sort_grid_by_time_from_manifold(
    manifold::CausalSets.AbstractManifold{N},
    preset_atoms::Vector{CausalSets.Coordinates{N}},
)::Vector{CausalSets.Coordinates{N}} where {N}
    atoms = sort(preset_atoms, lt = (x, y) -> CausalSets.isless_coord_time(manifold, x, y))
    return atoms
end

"""
    center_and_rescale_grid_to_box(grid::Vector{CausalSets.Coordinates{2}}, 
                                   box::Tuple{CausalSets.Coordinates{2}, CausalSets.Coordinates{2}})
        -> Vector{CausalSets.Coordinates{2}}
Translates and rescales the given grid so that it fits maximally inside the rectangular `box`, centered around the box's midpoint.
# Arguments
- `grid`: Vector of 2D coordinates representing the pseudo-sprinkling or grid.
- `box`: Tuple of lower-left and upper-right coordinates defining the bounding box.
# Returns
- Transformed grid that fits symmetrically and maximally into the given box.
"""
function center_and_rescale_grid_to_box(
    grid::Vector{CausalSets.Coordinates{2}},
    box::Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}},
)::Vector{CausalSets.Coordinates{2}}

    # Extract box info
    (lower, upper) = box
    box_center = ((lower[1] + upper[1]) / 2, (lower[2] + upper[2]) / 2)
    box_width = upper[1] - lower[1]
    box_height = upper[2] - lower[2]

    # Compute grid bounding box
    ts = first.(grid)
    xs = last.(grid)
    min_t, max_t = extrema(ts)
    min_x, max_x = extrema(xs)
    grid_width = max_x - min_x
    grid_height = max_t - min_t

    # Determine scaling factor
    scale = min(box_width / grid_width, box_height / grid_height)

    # Translate to origin, scale, then shift to box center
    transformed = map(grid) do (t, x)
        ct = (t - (min_t + max_t)/2) * scale + box_center[1]
        cx = (x - (min_x + max_x)/2) * scale + box_center[2]
        (ct, cx)
    end

    return transformed
end


"""
    create_grid_causet_2D(size, lattice, manifold; 
                          type=Float32, a=1.0, b=0.5, 
                          gamma_deg=60.0, rotate_deg=nothing, 
                          origin=(0.0, 0.0)) 
        -> Tuple{CausalSets.BitArrayCauset, Bool, Matrix{T}}

Construct a 2D grid of `size` points based on the given Bravais lattice, sort them by the time function of `manifold`, and build a `BitArrayCauset` from the resulting pseudo‑sprinkling.

# Arguments
- `size::Int`: number of atoms to generate (≥ 1).
- `lattice::AbstractString`: name of the 2D Bravais lattice. Supported names (case-insensitive):
    • "square", "quadratic" → edges ((a,0), (0,a))
    • "rectangular" → edges ((a,0), (0,b))
    • "centered-rectangular", "rhombic", "c-rect" → edges ((a/2,b/2), (a/2,-b/2))
    • "hexagonal", "triangular" → edges ((a,0), (a/2, a*sqrt(3)/2))
    • "oblique", "monoclinic" → edges ((a,0), (b*cos(γ), b*sin(γ))) with `γ = gamma_deg`
- `manifold::CausalSets.AbstractManifold{2}`: 2D manifold defining the causal structure on the grid.

# Keywords
- `type::Type{T}=Float32`: numeric type used for the returned coordinate matrix.
- `a::Float64=1.0`, `b::Float64=0.5`: lattice constants.
- `gamma_deg::Float64=60.0`: angle between lattice vectors (for "oblique"/"monoclinic" only).
- `rotate_deg=nothing`: if set, rotate the lattice by the given angle (in degrees). If `nothing`, automatically aligns `(e₁ + e₂)` with the positive y-axis.
- `origin::Tuple{Float64,Float64}`: coordinate of the origin vertex.

# Returns
- `Tuple{BitArrayCauset, Bool, Matrix{T}}`:    
    - `BitArrayCauset` the constructed causet,
    - `Bool` - `true` immitates the "converged" return option for random_csets
    - `Matrix{T}` - the coordinate matrix of atoms on the grid, immitating the pseudo-sprinkling in random_csets
"""
function create_grid_causet_2D(
    size::Int64,
    lattice::AbstractString,
    manifold::CausalSets.AbstractManifold;
    type::Type{T} = Float32,
    a::Float64 = 1.0,
    b::Float64 = 0.5,
    gamma_deg::Float64 = 60.0,
    rotate_deg = nothing,
    origin = (0.0, 0.0),
)::Tuple{CausalSets.BitArrayCauset,Bool,Matrix{T}} where {T<:Number}
    grid = generate_grid_2d(
        size,
        lattice;
        a = a,
        b = b,
        gamma_deg = gamma_deg,
        rotate_deg = rotate_deg,
        origin = origin,
    )
    pseudosprinkling = sort_grid_by_time_from_manifold(manifold, grid)

    return CausalSets.BitArrayCauset(manifold, pseudosprinkling),
    true,
    type.(stack(collect.(pseudosprinkling), dims = 1))
end

"""
    create_grid_causet_2D_polynomial_manifold(size, lattice, rng, order, r;
                                              type=Float32, a=1.0, b=0.5, 
                                              gamma_deg=60.0, rotate_deg=nothing, 
                                              origin=(0.0, 0.0)) 
        -> Tuple{CausalSets.BitArrayCauset, Bool, Matrix{T}}

Construct a 2D grid of `size` points based on the given Bravais lattice, generate a random 2D polynomial time function, and build a `BitArrayCauset` from the resulting pseudo-sprinkling order.

# Arguments
- `size::Int`: number of atoms to generate (≥ 1).
- `lattice::AbstractString`: name of the 2D Bravais lattice. Supported names (case-insensitive):
    • "square", "quadratic" → edges ((a,0), (0,a))
    • "rectangular" → edges ((a,0), (0,b))
    • "centered-rectangular", "rhombic", "c-rect" → edges ((a/2,b/2), (a/2,-b/2))
    • "hexagonal", "triangular" → edges ((a,0), (a/2, a*sqrt(3)/2))
    • "oblique", "monoclinic" → edges ((a,0), (b*cos(γ), b*sin(γ))) with `γ = gamma_deg`
- `rng::Random.AbstractRNG`: random number generator for coefficient sampling.
- `order::Int`: order of the polynomial (≥ 1).
- `r::Float64`: exponential decay rate for Chebyshev coefficients.

# Keywords
- `type::Type{T}=Float32`: numeric type used for the returned coordinate matrix.
- `a::Float64=1.0`, `b::Float64=0.5`: lattice constants.
- `gamma_deg::Float64=60.0`: angle between lattice vectors (for "oblique"/"monoclinic" only).
- `rotate_deg=nothing`: if set, rotate the lattice by the given angle (in degrees). If `nothing`, automatically aligns `(e₁ + e₂)` with the positive y-axis.
- `origin::Tuple{Float64,Float64}`: coordinate of the origin vertex.

# Returns
- `Tuple{BitArrayCauset, Bool, Matrix{T}}`:    
    - `BitArrayCauset` — the constructed causet from the polynomial time ordering,
    - `Bool` — always `true`, for compatibility
    - `Matrix{T}` — coordinate matrix of atoms on the grid.
"""
function create_grid_causet_2D_polynomial_manifold(
    size::Int64,
    lattice::AbstractString,
    rng::Random.AbstractRNG,
    order::Int64,
    r::Float64;
    type::Type{T} = Float32,
    a::Float64 = 1.0,
    b::Float64 = 0.5,
    gamma_deg::Float64 = 60.0,
    rotate_deg = nothing,
    origin = (0.0, 0.0),
)::Tuple{CausalSets.BitArrayCauset,Bool,Matrix{T},Matrix{T}} where {T<:Number}

    size ≥ 2 || throw(ArgumentError("size must be ≥ 2, is $(size)"))
    order ≥ 0 || throw(ArgumentError("order must be ≥ 0, is $(order)"))
    r > 1 || throw(ArgumentError("r must be > 1, is $(r)"))

    grid = generate_grid_2d(
        size,
        lattice;
        a = a,
        b = b,
        gamma_deg = gamma_deg,
        rotate_deg = rotate_deg,
        origin = origin,
    )

    # Generate a matrix of random Chebyshev coefficients that decay exponentially with base r
    # it has to be a (order + 1 x order + 1)-matrix because we describe a function of two variables
    chebyshev_coefs = zeros(Float64, order + 1, order + 1)
    for i = 1:order
        for j = 1:order
            chebyshev_coefs[i, j] = r^(-i - j) * Random.randn(rng)
        end
    end

    # Construct the Chebyshev-to-Taylor transformation matrix
    cheb_to_taylor_mat = CausalSets.chebyshev_coef_matrix(order)

    # Transform Chebyshev coefficients to Taylor coefficients
    taylorcoefs = CausalSets.transform_polynomial(chebyshev_coefs, cheb_to_taylor_mat)

    # Square the polynomial to ensure positivity
    squaretaylorcoefs = CausalSets.polynomial_pow(taylorcoefs, 2)

    # Create a polynomial manifold from the squared Taylor coefficients
    polym = CausalSets.PolynomialManifold{2}(squaretaylorcoefs)

    # Create grid
    grid = sort_grid_by_time_from_manifold(polym, grid)

    # Rescale and translate grid so it fits into Chebyshev domain
    pseudosprinkling = center_and_rescale_grid_to_box(grid, ((-1.0, -1.0), (1.0, 1.0)))

    return CausalSets.BitArrayCauset(polym, pseudosprinkling),
    true,
    type.(stack(collect.(pseudosprinkling), dims = 1)),
    type.(chebyshev_coefs)
end

"""
    lattice_points_in_box(
        edges::NTuple{N,CausalSets.Coordinates{N}},
        box::Tuple{CausalSets.Coordinates{N},CausalSets.Coordinates{N}},
        ℓ::Float64,
    ) -> Vector{CausalSets.Coordinates{N}}

Enumerate **all** lattice points generated by integer translations of the
Bravais lattice defined by `edges`, scaled by `ℓ`, that lie inside the
axis-aligned box `box`.

No trimming or approximation is performed: a point is included iff it lies
inside the box.
"""
function lattice_points_in_box(
    edges::NTuple{N,CausalSets.Coordinates{N}},
    box::Tuple{CausalSets.Coordinates{N},CausalSets.Coordinates{N}},
    ℓ::Float64,
)::Vector{CausalSets.Coordinates{N}} where {N}

    lower, upper = box
    lengths = ntuple(i -> upper[i] - lower[i], N)

    # Build edge matrix
    E = [ℓ * edges[j][i] for i in 1:N, j in 1:N]

    # Conservative bound on integer translations
    max_extent = maximum(LinearAlgebra.norm(E[:, j]) for j in 1:N)
    kmax = ceil(Int, maximum(lengths) / max_extent) + 1

    pts = CausalSets.Coordinates{N}[]
    for k in Iterators.product(ntuple(_ -> -kmax:kmax, N)...)
        x = ntuple(i -> sum(E[i, j] * k[j] for j in 1:N), N)
        all(lower[i] ≤ x[i] ≤ upper[i] for i in 1:N) && push!(pts, x)
    end
    return pts
end
import CausalSets
import LinearAlgebra
edges = ((a, 0.0), (0.0, a))
test = lattice_points_in_box(((1., 0.0), (0.0, 1.)),((-1.,-1.),(1.,1.)), .1)
using CairoMakie

"""
    count_lattice_points_in_box(edges, box, ℓ) -> Int

Return the number of lattice points inside `box` for scale `ℓ`
without allocating the full point vector.
"""
function count_lattice_points_in_box(
    edges::NTuple{N,CausalSets.Coordinates{N}},
    box::Tuple{CausalSets.Coordinates{N},CausalSets.Coordinates{N}},
    ℓ::Float64,
)::Int where {N}

    return length(lattice_points_in_box(edges, box, ℓ))
end

"""
    boundary_shell_indices(points, box, thickness) -> Vector{Int}

Return indices of points lying within distance `thickness` of any face
of the axis-aligned box `box`.
"""
function boundary_shell_indices(
    points::Vector{CausalSets.Coordinates{N}},
    box::Tuple{CausalSets.Coordinates{N},CausalSets.Coordinates{N}},
    thickness::Float64,
)::Vector{Int} where {N}

    lower, upper = box
    idxs = Int[]
    for (i, x) in enumerate(points)
        any(
            min(abs(x[d] - lower[d]), abs(upper[d] - x[d])) ≤ thickness
            for d in 1:N
        ) && push!(idxs, i)
    end
    return idxs
end

"""
    generate_grid_in_box_from_bravais(
        n,
        edges,
        box;
        rng = Random.default_rng(),
        shell_thickness = nothing,
    ) -> Vector{CausalSets.Coordinates{N}}

Generate a Bravais grid inside `box` with **exactly `n` points**.

The grid is generated at the largest possible lattice spacing such that the
number of lattice points inside the box is ≥ `n`. Any excess points are then
randomly removed **only from a boundary shell**, preserving bulk regularity.

This function generates all interior lattice points explicitly.
"""
function generate_grid_in_box_from_bravais(
    n::Int,
    edges::NTuple{N,CausalSets.Coordinates{N}},
    box::Tuple{CausalSets.Coordinates{N},CausalSets.Coordinates{N}};
    rng::Random.AbstractRNG = Random.default_rng(),
    shell_thickness::Union{Nothing,Float64} = nothing,
)::Vector{CausalSets.Coordinates{N}} where {N}

    n ≥ 1 || throw(ArgumentError("n must be ≥ 1"))

    lower, upper = box
    lengths = ntuple(i -> upper[i] - lower[i], N)

    # Analytic estimate for lattice spacing ℓ from density
    detE = abs(LinearAlgebra.det(reduce(hcat, edges)))
    Vbox = prod(lengths)
    ℓ = (Vbox / (n * detE))^(1 / N) * 1.05  # safety factor

    # Enumerate once at estimated ℓ; increase ℓ if needed
    points = lattice_points_in_box(edges, box, ℓ)
    if length(points) < n
        ℓ *= 1.2
        points = lattice_points_in_box(edges, box, ℓ)
    end

    Δn = length(points) - n
    Δn == 0 && return points

    thickness = isnothing(shell_thickness) ? ℓ : shell_thickness
    shell = boundary_shell_indices(points, box, thickness)

    length(shell) ≥ Δn ||
        throw(ArgumentError("Boundary shell too small to remove Δn = $Δn points"))

    remove = rand(rng, shell, Δn; replace = false)
    keep = trues(length(points))
    keep[remove] .= false

    return points[keep]
end

"""
    generate_grid_2d_in_box(
        num_atoms,
        lattice,
        box;
        a=1.0,
        b=0.5,
        gamma_deg=60.0,
        rotate_deg=nothing,
        rng=Random.default_rng(),
        shell_thickness=nothing,
    ) -> Vector{CausalSets.Coordinates{2}}

Construct a 2D Bravais grid inside the rectangular `box` with **exactly**
`num_atoms` points.

This is the boundary-aware analogue of `generate_grid_2d`. The lattice type,
lattice parameters, and rotation conventions are identical, but the grid is
generated by filling the physical boundary and thinning boundary points if
necessary.
"""
function generate_grid_2d_in_box(
    num_atoms::Int,
    lattice::AbstractString,
    box::Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}};
    a::Float64 = 1.0,
    b::Float64 = 0.5,
    gamma_deg::Float64 = 60.0,
    rotate_deg = nothing,
    rng::Random.AbstractRNG = Random.default_rng(),
    shell_thickness::Union{Nothing,Float64} = nothing,
)::Vector{CausalSets.Coordinates{2}}

    lname = lowercase(strip(lattice))
    if lname in ("square", "quadratic")
        edges = ((a, 0.0), (0.0, a))
    elseif lname == "rectangular"
        edges = ((a, 0.0), (0.0, b))
    elseif lname in ("centered-rectangular", "rhombic", "c-rect")
        edges = ((0.5a, 0.5b), (0.5a, -0.5b))
    elseif lname in ("hexagonal", "triangular")
        edges = ((a, 0.0), (0.5a, 0.5a*sqrt(3.0)))
    elseif lname in ("oblique", "monoclinic")
        γ = deg2rad(gamma_deg)
        edges = ((a, 0.0), (b*cos(γ), b*sin(γ)))
    else
        throw(ArgumentError("Unsupported 2D lattice name: '$lattice'"))
    end

    # Rotation: identical convention to generate_grid_2d
    local θ::Float64
    if isnothing(rotate_deg)
        v = (edges[1][1] + edges[2][1], edges[1][2] + edges[2][2])
        φ = atan(v[2], v[1])
        θ = (pi/2) - φ
    else
        θ = deg2rad(Float64(rotate_deg))
    end

    c = cos(θ)
    s = sin(θ)
    R = ((c, -s), (s, c))
    rot = x -> (
        R[1][1]*x[1] + R[1][2]*x[2],
        R[2][1]*x[1] + R[2][2]*x[2],
    )
    edges = (rot(edges[1]), rot(edges[2]))

    return generate_grid_in_box_from_bravais(
        num_atoms,
        edges,
        box;
        rng = rng,
        shell_thickness = shell_thickness,
    )
end

"""
    create_grid_causet_in_boundary_2D(
        size,
        lattice,
        boundary,
        manifold;
        type=Float32,
        a=1.0,
        b=0.5,
        gamma_deg=60.0,
        rotate_deg=nothing,
        rng=Random.default_rng(),
        shell_thickness=nothing,
    ) -> Tuple{BitArrayCauset, Bool, Matrix{T}}

Construct a 2D grid causet with exactly `size` points inside a given
`boundary::AbstractBoundary{2}`.

Supported boundaries:
- `BoxBoundary{2}`: grid is generated directly in `(t,x)` coordinates.
- `CausalDiamondBoundary{2}`: grid is generated in null coordinates `(u,v)`
  with `u,v ∈ [0,duration]` and then mapped to `(t,x)` via
  `t = (u+v)/2`, `x = (v-u)/2`.

The rest of the construction parallels `create_grid_causet_2D`.
"""

function create_grid_causet_in_boundary_2D(
    size::Int64,
    lattice::AbstractString,
    boundary::CausalSets.AbstractBoundary{2},
    manifold::CausalSets.AbstractManifold{2};
    type::Type{T} = Float32,
    a::Float64 = 1.0,
    b::Float64 = 0.5,
    gamma_deg::Float64 = 60.0,
    rotate_deg = nothing,
    rng::Random.AbstractRNG = Random.default_rng(),
    shell_thickness::Union{Nothing,Float64} = nothing,
)::Tuple{CausalSets.BitArrayCauset,Bool,Matrix{T}} where {T<:Number}

    size ≥ 1 || throw(ArgumentError("size must be ≥ 1"))

    # ------------------------------------------------------------
    # Determine box and coordinate interpretation from boundary
    # ------------------------------------------------------------
    if boundary isa CausalSets.BoxBoundary{2}
        box = boundary.edges
        grid = generate_grid_2d_in_box(
            size,
            lattice,
            box;
            a = a,
            b = b,
            gamma_deg = gamma_deg,
            rotate_deg = rotate_deg,
            rng = rng,
            shell_thickness = shell_thickness,
        )

    elseif boundary isa CausalSets.CausalDiamondBoundary{2}
        Tdur = boundary.duration
        # work in null coordinates (u,v)
        box = ((0.0, 0.0), (Tdur, Tdur))

        uv_grid = generate_grid_2d_in_box(
            size,
            lattice,
            box;
            a = a,
            b = b,
            gamma_deg = gamma_deg,
            rotate_deg = rotate_deg,
            rng = rng,
            shell_thickness = shell_thickness,
        )

        # map (u,v) -> (t,x)
        grid = map(uv_grid) do (u, v)
            ((u + v) / 2, (v - u) / 2)
        end

    else
        throw(ArgumentError("Unsupported boundary type: $(typeof(boundary))"))
    end

    # ------------------------------------------------------------
    # Sort by manifold time and build causet
    # ------------------------------------------------------------
    pseudosprinkling = sort_grid_by_time_from_manifold(manifold, grid)

    return CausalSets.BitArrayCauset(manifold, pseudosprinkling),
           true,
           type.(stack(collect.(pseudosprinkling), dims = 1))
end


"""
    create_grid_causet_in_boundary_2D_polynomial_manifold(
        size,
        lattice,
        boundary,
        rng,
        order,
        r;
        type=Float32,
        a=1.0,
        b=0.5,
        gamma_deg=60.0,
        rotate_deg=nothing,
        shell_thickness=nothing,
    ) -> Tuple{BitArrayCauset, Bool, Matrix{T}, Matrix{T}}

Construct a 2D grid causet with exactly `size` points inside a given
`boundary::AbstractBoundary{2}` and impose a random polynomial time function.

Supported boundaries:
- `BoxBoundary{2}`: grid generated directly in `(t,x)` coordinates.
- `CausalDiamondBoundary{2}`: grid generated in null coordinates `(u,v)`
  and mapped to `(t,x)` via `t=(u+v)/2`, `x=(v-u)/2`.

This is the boundary-aware analogue of
`create_grid_causet_2D_polynomial_manifold`.
"""
function create_grid_causet_in_boundary_2D_polynomial_manifold(
    size::Int64,
    lattice::AbstractString,
    boundary::CausalSets.AbstractBoundary{2},
    rng::Random.AbstractRNG,
    order::Int64,
    r::Float64;
    type::Type{T} = Float32,
    a::Float64 = 1.0,
    b::Float64 = 0.5,
    gamma_deg::Float64 = 60.0,
    rotate_deg = nothing,
    shell_thickness::Union{Nothing,Float64} = nothing,
)::Tuple{CausalSets.BitArrayCauset,Bool,Matrix{T},Matrix{T}} where {T<:Number}

    size ≥ 2 || throw(ArgumentError("size must be ≥ 2, is $(size)"))
    order ≥ 0 || throw(ArgumentError("order must be ≥ 0, is $(order)"))
    r > 1 || throw(ArgumentError("r must be > 1, is $(r)"))

    # ------------------------------------------------------------
    # Generate grid inside boundary
    # ------------------------------------------------------------
    if boundary isa CausalSets.BoxBoundary{2}
        box = boundary.edges
        grid = generate_grid_2d_in_box(
            size,
            lattice,
            box;
            a = a,
            b = b,
            gamma_deg = gamma_deg,
            rotate_deg = rotate_deg,
            rng = rng,
            shell_thickness = shell_thickness,
        )

    elseif boundary isa CausalSets.CausalDiamondBoundary{2}
        Tdur = boundary.duration
        box = ((0.0, 0.0), (Tdur, Tdur))

        uv_grid = generate_grid_2d_in_box(
            size,
            lattice,
            box;
            a = a,
            b = b,
            gamma_deg = gamma_deg,
            rotate_deg = rotate_deg,
            rng = rng,
            shell_thickness = shell_thickness,
        )

        grid = map(uv_grid) do (u, v)
            ((u + v) / 2, (v - u) / 2)
        end

    else
        throw(ArgumentError("Unsupported boundary type: $(typeof(boundary))"))
    end

    # ------------------------------------------------------------
    # Polynomial manifold construction (identical to non-boundary version)
    # ------------------------------------------------------------
    chebyshev_coefs = zeros(Float64, order + 1, order + 1)
    for i = 1:order
        for j = 1:order
            chebyshev_coefs[i, j] = r^(-i - j) * Random.randn(rng)
        end
    end

    cheb_to_taylor_mat = CausalSets.chebyshev_coef_matrix(order)
    taylorcoefs = CausalSets.transform_polynomial(chebyshev_coefs, cheb_to_taylor_mat)
    squaretaylorcoefs = CausalSets.polynomial_pow(taylorcoefs, 2)
    polym = CausalSets.PolynomialManifold{2}(squaretaylorcoefs)

    # ------------------------------------------------------------
    # Impose time ordering and rescale to Chebyshev domain
    # ------------------------------------------------------------
    grid = sort_grid_by_time_from_manifold(polym, grid)
    pseudosprinkling =
        center_and_rescale_grid_to_box(grid, ((-1.0, -1.0), (1.0, 1.0)))

    return CausalSets.BitArrayCauset(polym, pseudosprinkling),
           true,
           type.(stack(collect.(pseudosprinkling), dims = 1)),
           type.(chebyshev_coefs)
end