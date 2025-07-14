
"""
    PseudoManifold{N}

A pseudo-manifold structure representing a random manifold in N dimensions.
This is used as an alternative to the geometric manifolds provided by CausalSets
when working with randomly generated causets.

# Type Parameters
- `N`: The dimension of the manifold
"""
struct PseudoManifold{N} <: CausalSets.AbstractManifold{N} end

"""
    get_manifold_name(type::Type, d)

Returns the string name of a manifold type for a given dimension.

# Arguments
- `type`: The manifold type (e.g., CausalSets.MinkowskiManifold{d})
- `d`: The dimension of the manifold

# Returns
- `String`: The name of the manifold ("Minkowski", "DeSitter", etc.)
"""
function get_manifold_name(type::Type)
    Dict(
        CausalSets.MinkowskiManifold{2} => "Minkowski",
        CausalSets.DeSitterManifold{2} => "DeSitter",
        CausalSets.AntiDeSitterManifold{2} => "AntiDeSitter",
        CausalSets.HypercylinderManifold{2} => "HyperCylinder",
        CausalSets.TorusManifold{2} => "Torus",
        PseudoManifold{2} => "Random",
        CausalSets.MinkowskiManifold{3} => "Minkowski",
        CausalSets.DeSitterManifold{3} => "DeSitter",
        CausalSets.AntiDeSitterManifold{3} => "AntiDeSitter",
        CausalSets.HypercylinderManifold{3} => "HyperCylinder",
        CausalSets.TorusManifold{3} => "Torus",
        PseudoManifold{3} => "Random",
        CausalSets.MinkowskiManifold{4} => "Minkowski",
        CausalSets.DeSitterManifold{4} => "DeSitter",
        CausalSets.AntiDeSitterManifold{4} => "AntiDeSitter",
        CausalSets.HypercylinderManifold{4} => "HyperCylinder",
        CausalSets.TorusManifold{4} => "Torus",
        PseudoManifold{4} => "Random",
    )[type]
end

"""
    make_manifold(name::String, d::Int) -> CausalSets.AbstractManifold
Creates a manifold object based on its name and dimension.
# Arguments
- `name::String`: Name of the manifold ("Minkowski", "DeSitter", etc.)
- `d::Int`: Dimension of the manifold (2, 3, or 4)
# Returns
- `CausalSets.AbstractManifold`: The constructed manifold object    
"""
function make_manifold(name::String, d::Int)::CausalSets.AbstractManifold
    if d < 2 || d > 4
        throw(ArgumentError("Unsupported manifold dimension: $d"))
    end

    return Dict(
        "Minkowski" => CausalSets.MinkowskiManifold{d}(),
        "DeSitter" => CausalSets.DeSitterManifold{d}(1.0),
        "AntiDeSitter" => CausalSets.AntiDeSitterManifold{d}(1.0),
        "HyperCylinder" => CausalSets.HypercylinderManifold{d}(1.0),
        "Torus" => CausalSets.TorusManifold{d}(1.0),
        "Random" => PseudoManifold{d}(),
    )[name]
end

"""
    get_manifold_encoding

A dictionary mapping manifold names to their integer encodings.
Used for converting between string representations and numeric codes
for different spacetime manifolds.

# Mappings
- "Minkowski" => 1
- "HyperCylinder" => 2  
- "DeSitter" => 3
- "AntiDeSitter" => 4
- "Torus" => 5
- "Random" => 6
"""
get_manifold_encoding = Dict(
    "Minkowski" => 1,
    "DeSitter" => 3,
    "AntiDeSitter" => 4,
    "HyperCylinder" => 2,
    "Torus" => 5,
    "Random" => 6,
)

"""
    make_manifold(i::Int, d::Int) -> CausalSets.AbstractManifold

Creates a manifold object based on an integer encoding and dimension.

# Arguments
- `i`: Integer encoding of the manifold type (1-6)
- `d`: Dimension of the manifold

# Returns
- `CausalSets.AbstractManifold`: The constructed manifold object

# Manifold Encodings
- 1: Minkowski manifold
- 2: Hypercylinder manifold  
- 3: De Sitter manifold
- 4: Anti-de Sitter manifold
- 5: Torus manifold
- 6: Pseudo manifold (random)

# Throws
- `ErrorException`: If manifold encoding `i` is not supported (not 1-6)
"""
function make_manifold(i::Int, d::Int)::CausalSets.AbstractManifold

    if d < 2 || d > 4
        throw(ArgumentError("Unsupported manifold dimension: $d"))
    end

    Dict(
        1 => CausalSets.MinkowskiManifold{d}(),
        2 => CausalSets.HypercylinderManifold{d}(1.0),
        3 => CausalSets.DeSitterManifold{d}(1.0),
        4 => CausalSets.AntiDeSitterManifold{d}(1.0),
        5 => CausalSets.TorusManifold{d}(1.0),
        6 => PseudoManifold{d}(),
    )[i]
end

"""
    make_boundary(name::String, d::Int) -> CausalSets.AbstractBoundary
Creates a boundary object based on its name and dimension.
# Arguments
- `name::String`: Name of the boundary ("CausalDiamond", "TimeBoundary", "BoxBoundary")
- `d::Int`: Dimension of the boundary (2, 3, or 4)
# Returns
- `CausalSets.AbstractBoundary`: The constructed boundary object
"""
function make_boundary(name::String, d::Int)::CausalSets.AbstractBoundary
    if d < 2 || d > 4
        throw(ArgumentError("Unsupported boundary dimension: $d"))
    end

    return Dict(
        "CausalDiamond" => CausalSets.CausalDiamondBoundary{d}(1.0),
        "TimeBoundary" => CausalSets.TimeBoundary{d}(-1.0, 1.0), # check if this makes sense
        "BoxBoundary" => CausalSets.BoxBoundary{d}((
            ([-0.49 for i = 1:d]...,),
            ([0.49 for i = 1:d]...,),
        )),
    )[name]
end

"""
    make_boundary(i::Int, d::Int) -> CausalSets.AbstractBoundary
Creates a boundary object based on an integer encoding and dimension.
# Arguments
- `i`: Integer encoding of the boundary type (1-3)
- `d`: Dimension of the boundary (2, 3, or 4)
# Returns
- `CausalSets.AbstractBoundary`: The constructed boundary object
"""
function make_boundary(i::Int, d::Int)::CausalSets.AbstractBoundary
    if d < 2 || d > 4
        throw(ArgumentError("Unsupported boundary dimension: $d"))
    end

    Dict(
        1 => CausalSets.CausalDiamondBoundary{d}(1.0),
        2 => CausalSets.TimeBoundary{d}(-1.0, 1.0), # check if this makes sense
        3 => CausalSets.BoxBoundary{d}((
            ([-0.49 for i = 1:d]...,),
            ([0.49 for i = 1:d]...,),
        )),
    )[i]
end


"""
    resize(m::AbstractArray{T}, new_size::Tuple) -> AbstractArray{T}

Resizes an array to a new size, either by truncating or zero-padding.

# Arguments
- `m::AbstractArray{T}`: The input array to resize
- `new_size::Tuple`: The target dimensions

# Returns
- `AbstractArray{T}`: Resized array of the same type as input

# Notes
- If new size is larger, pads with zeros (preserving sparsity for sparse arrays)
- If new size is smaller, truncates the array
- Maintains the original array type (dense or sparse)
"""
function resize(m::AbstractArray{T}, new_size::Tuple)::AbstractArray{T} where {T<:Number}

    if any(size(m) .< new_size) && any(size(m) .> new_size)
        throw(
            ArgumentError(
                "Cannot resize array to a smaller size in some dimensions while enlarging it in others.",
            ),
        )
    end

    if any(size(m) .< new_size)
        resized_m =
            m isa SparseArrays.AbstractSparseArray ? SparseArrays.spzeros(T, new_size...) :
            zeros(T, new_size...)
        @inbounds resized_m[tuple([1:n for n in size(m)]...)...] .= m
        return resized_m
    else
        return @inbounds m[tuple([1:n for n in new_size]...)...]
    end
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
- `rng`: Random number generator (default: MersenneTwister(1234))

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
    rng = Random.MersenneTwister(1234),
)::Vector{Vector{T}} where {T<:Number}
    if box_min >= box_max
        throw(ArgumentError("box_min must be less than box_max"))
    end
    distr = Distributions.Uniform(box_min, box_max)

    return [[type(rand(rng, distr)) for _ = 1:d] for _ = 1:n]
end
