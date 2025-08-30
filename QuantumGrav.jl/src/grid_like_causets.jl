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
function generate_grid_from_brillouin_cell(num_atoms::Int, edges::NTuple{N,CausalSets.Coordinates{N}}; origin=nothing)::Vector{CausalSets.Coordinates{N}} where {N}
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
    for j in 1:N
        for i in 1:N
            E[i, j] = edges[j][i]
        end
    end
    if abs(LinearAlgebra.det(E)) ≤ eps(Float64)
        throw(ArgumentError("Brillouin-cell edges must be linearly independent (determinant ≈ 0)"))
    end

    # Choose per-dimension resolution ~ num_atoms^(1/N)
    k = max(2, ceil(Int, num_atoms^(1/N)))
    # total candidate points
    total = k^N

    # Parameter grids in [0,1]
    params = ntuple(_ -> range(0.0, 1.0; length=k), N)

    pts = Vector{CausalSets.Coordinates{N}}(undef, total)
    idx = 0
    for T in Iterators.product(params...)
        # x = origin + E * t
        x = Vector{Float64}(undef, N)
        for i in 1:N
            xi = origin_nt[i]
            @inbounds for j in 1:N
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
function generate_grid_2d(num_atoms::Int, lattice::AbstractString; a::Float64=1.0, b::Float64=0.5, gamma_deg::Float64=60.0, rotate_deg=nothing, origin=(0.0,0.0))::Vector{CausalSets.Coordinates{2}}
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
    c = cos(θ); s = sin(θ)
    R = ((c, -s), (s, c))  # rotation matrix
    rot = x -> (R[1][1]*x[1] + R[1][2]*x[2], R[2][1]*x[1] + R[2][2]*x[2])
    edges = (rot(edges[1]), rot(edges[2]))

    return generate_grid_from_brillouin_cell(num_atoms, edges; origin=origin)
end

"""
    sort_grid_by_time_from_manifold(manifold, preset_atoms) -> Vector{CausalSets.Coordinates{N}}

Stable sort of `preset_atoms` using the manifold‑provided time ordering `CausalSets.isless_coord_time`.
Useful to impose a pseudo‑sprinkling order before constructing a causet.
"""
function sort_grid_by_time_from_manifold(manifold::CausalSets.AbstractManifold{N}, preset_atoms::Vector{CausalSets.Coordinates{N}})::Vector{CausalSets.Coordinates{N}} where N
    atoms = sort(preset_atoms, lt = (x, y) -> CausalSets.isless_coord_time(manifold, x, y))
    return atoms
end

"""
    create_grid_causet_2D(size, lattice, manifold; a=1.0, b=0.5, gamma_deg=60.0, rotate_deg=nothing, origin=(0.0,0.0))
        -> CausalSets.BitArrayCauset

Construct a 2D grid of `size` points for the given Bravais lattice name via `generate_grid_2d`,
sort the points by the manifold’s time ordering, and return a `BitArrayCauset` built on that order.

# Keywords
See `generate_grid_2d` for lattice parameters and rotation.
"""
function create_grid_causet_2D( size::Int64, 
                                lattice::AbstractString,
                                manifold::CausalSets.AbstractManifold; 
                                type::Type{T} = Float32, 
                                a::Float64=1.0, b::Float64=0.5, 
                                gamma_deg::Float64=60.0, 
                                rotate_deg=nothing, 
                                origin=(0.0,0.0))::Tuple{CausalSets.BitArrayCauset, Bool, Matrix{T}} where {T<:Number}
    grid = generate_grid_2d(size, lattice; a=a, b=b, gamma_deg=gamma_deg, rotate_deg=rotate_deg, origin=origin)
    pseudosprinkling = sort_grid_by_time_from_manifold(manifold, grid)

    return CausalSets.BitArrayCauset(manifold, pseudosprinkling), true, type.(stack(collect.(pseudosprinkling), dims = 1))
end