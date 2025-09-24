"""
    make_manifold(i::Int, d::Int; size::Float64=1.0) -> CausalSets.AbstractManifold

Creates a manifold object based on an integer encoding and dimension.

# Arguments
- `i`: Integer encoding of the manifold type (1-6)
- `d`: Dimension of the manifold

# Keyword Arguments 
- `size`: Manifold size parameter (dependent on desired manifold, see docs of CausalSets.jl)

# Returns
- `CausalSets.AbstractManifold`: The constructed manifold object

# Manifold Encodings
- 1: Minkowski manifold
- 2: Hypercylinder manifold  
- 3: De Sitter manifold
- 4: Anti-de Sitter manifold
- 5: Torus manifold

# Throws
- `ErrorException`: If manifold encoding `i` is not supported (not 1-6)
"""
function make_manifold(i::Int, d::Int; size::Float64 = 1.0)::CausalSets.AbstractManifold

    if i < 1 || i > 5
        throw(ArgumentError("Unsupported manifold encoding: $i"))
    end

    if d < 2 || d > 4
        throw(ArgumentError("Unsupported manifold dimension: $d"))
    end

    Dict(
        1 => () -> CausalSets.MinkowskiManifold{d}(),
        2 => () -> CausalSets.HypercylinderManifold{d}(size),
        3 => () -> CausalSets.DeSitterManifold{d}(size),
        4 => () -> CausalSets.AntiDeSitterManifold{d}(size),
        5 => () -> CausalSets.TorusManifold{d}(size),
    )[i]()
end

"""
    make_manifold(name::String, d::Int; size::Float64 = 1.0)

Creates a manifold object based on an integer encoding and dimension.

# Arguments:
- `name`: Name of the manifold to be created. not case sensitive
- `d`: Dimension of the manifold

# Keyword Arguments 
- `size`: Manifold size parameter (dependent on desired manifold, see docs of CausalSets.jl)

# Returns
- `CausalSets.AbstractManifold`: The constructed manifold object

# Manifold Encodings
- minkowski: Minkowski manifold
- hypercylinder: Hypercylinder manifold  
- desitter: De Sitter manifold
- antidesitter: Anti-de Sitter manifold
- torus: Torus manifold

# Throws
- `ErrorException`: If manifold encoding `i` is not supported (not 1-6)
"""
function make_manifold(
    name::String,
    d::Int;
    size::Float64 = 1.0,
)::CausalSets.AbstractManifold

    namedict = Dict(
        "minkowski" => 1,
        "hypercylinder" => 2,
        "desitter" => 3,
        "antidesitter" => 4,
        "torus" => 5,
    )

    if haskey(namedict, lowercase(name)) == false
        throw(ArgumentError("Unsupported manifold name: $name"))
    end

    return make_manifold(namedict[lowercase(name)], d; size = size)

end


"""
    make_boundary(i::Int, d::Int; limits::Union{Tuple{Vararg{Float64}}, Nothing, Float64} = nothing)

Create a CausalSets.AbstractBoundary object from given parameters

# Arguments:
- `i`: boundary indicator
- `d`: dimensionality of the underlying manifold

# Keyword arguments: 
- `limits`: Boundary edges, defaults to nothing. Dependent on the boundary type to be created. Defaults are: 
   - CausalDiamond: 1.0
   - TimeBoundary: (-1.0, 1.0)
   - BoxBoundary: d-dimensional box of size (-0.49, 0.49) in each dimension

# Boundary encodings
- 1: CausalDiamondBoundary
- 2: TimeBoundary
- 3: BoxBoundary
"""
function make_boundary(
    i::Int,
    d::Int;
    limits::Union{Tuple{Vararg{Float64}},Nothing,Float64} = nothing,
)::CausalSets.AbstractBoundary
    if d < 2 || d > 4
        throw(ArgumentError("Unsupported boundary dimension: $d"))
    end

    if i < 1 || i > 3
        throw(ArgumentError("Unsupported boundary encoding: $i"))
    end

    if limits === nothing
        if i == 1
            limits = 1.0
        elseif i == 2
            limits = (-1.0, 1.0)
        else
            limits = ((([-0.49 for i = 1:d]...,), ([0.49 for i = 1:d]...,)))
        end
    end

    return Dict(
        1 => () -> CausalSets.CausalDiamondBoundary{d}(limits),
        2 => () -> CausalSets.TimeBoundary{d}(limits...),
        3 => () -> CausalSets.BoxBoundary{d}(limits),
    )[i]()
end


"""
    make_boundary(name::String, d::Int; limits::Union{Tuple{Vararg{Float64}}, Nothing, Float64} = nothing)

Create a CausalSets.AbstractBoundary object from given parameters

# Arguments:
- `name`: name of the boundary type to create. not case sensitive
- `d`: dimensionality of the underlying manifold

# Keyword arguments:
- `limits`: Boundary edges, defaults to nothing. Dependent on the boundary type to be created. Defaults are: 
   - CausalDiamond: 1.0
   - TimeBoundary: (-1.0, 1.0)
   - BoxBoundary: d-dimensional box of size (-0.49, 0.49) in each dimension

# Boundary names
- causaldiamond: CausalDiamondBoundary
- timeboundary: TimeBoundary
- boxboundary: BoxBoundary
"""
function make_boundary(
    name::String,
    d::Int;
    limits::Union{Tuple{Vararg{Float64}},Nothing,Float64} = nothing,
)::CausalSets.AbstractBoundary

    namedict = Dict("causaldiamond" => 1, "timeboundary" => 2, "boxboundary" => 3)

    if haskey(namedict, lowercase(name)) == false
        throw(ArgumentError("Unsupported boundary name: $name"))
    end

    return make_boundary(namedict[lowercase(name)], d; limits = limits)
end


"""
    make_pseudosprinkling(n, d, box_min, box_max, type; rng) -> Vector{Vector{T}}

Generates random points uniformly distributed in a d-dimensional box.

# Arguments
- `n::Int64`: Number of points to generate
- `d::Int64`: Dimension of each point
- `box_min::Float64`: Minimum coordinate value
- `box_max::Float64`: Maximum coordinate value
- `type::Type{T}`: Numeric type for coordinates
- `rng`: Random number generator (default: Xoshiro(1234))

# Returns
- `Vector{Vector{T}}`: Vector of n points, each point is a d-dimensional vector

# Notes
Used for creating pseudo-sprinklings in PseudoManifold when geometric
manifold sprinkling is not applicable.
"""
function make_pseudosprinkling(
    n::Int64,
    d::Int64,
    box_min::Float64,
    box_max::Float64,
    type::Type{T};
    rng = Random.Xoshiro(1234),
)::Vector{Vector{T}} where {T<:Number}
    if box_min >= box_max
        throw(ArgumentError("box_min must be less than box_max"))
    end
    distr = Distributions.Uniform(box_min, box_max)

    return [[type(rand(rng, distr)) for _ = 1:d] for _ = 1:n]
end
