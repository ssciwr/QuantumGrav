
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
function get_manifold_name(type::Type, d)
    Dict(
        CausalSets.MinkowskiManifold{d} => "Minkowski",
        CausalSets.DeSitterManifold{d} => "DeSitter",
        CausalSets.AntiDeSitterManifold{d} => "AntiDeSitter",
        CausalSets.HypercylinderManifold{d} => "HyperCylinder",
        CausalSets.TorusManifold{d} => "Torus",
        PseudoManifold{2} => "Random",
    )[type]
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

    if i == 1
        return CausalSets.MinkowskiManifold{d}()
    elseif i == 2
        return CausalSets.HypercylinderManifold{d}(1.0)
    elseif i == 3
        return CausalSets.DeSitterManifold{d}(1.0)
    elseif i == 4
        return CausalSets.AntiDeSitterManifold{d}(1.0)
    elseif i == 5
        return CausalSets.TorusManifold{d}(1.0)
    elseif i == 6
        return PseudoManifold{d}()
    else
        throw(ArgumentError("Unsupported manifold: $i"))
    end
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
    distr = Distributions.Uniform(box_min, box_max)

    return [[rand(distr) for i = 1:d] for _ = 1:n]
end

"""
    topsort(adj_matrix, in_degree::AbstractVector{T}) -> Vector{Int}

Performs topological sorting on a directed graph using Kahn's algorithm.

# Arguments
- `adj_matrix`: Adjacency matrix representing the directed graph
- `in_degree::AbstractVector{T}`: Vector containing the in-degree of each vertex

# Returns
- `Vector{Int}`: Topologically sorted order of vertices

# Notes
Uses Kahn's algorithm for topological sorting. This will be needed later
for determining the topological order of causets. The algorithm maintains
a queue of vertices with zero in-degree and processes them iteratively.
"""
function topsort(
    adj_matrix::AbstractMatrix{T},
    in_degree::AbstractVector{T},
)::Vector{Int} where {T<:Number}
    n = size(adj_matrix, 1)
    # TODO: check this again

    # Topological sort using Kahn's algorithm --> will be needed later for the topo order of the CausalSets
    queue = Vector{Int64}()
    sizehint!(queue, n)
    for i = 1:n
        if isapprox(in_degree[i], zero(T))
            @inbounds push!(queue, i)
        end
    end

    topo_order = Vector{Int64}()
    sizehint!(topo_order, n)
    while !isempty(queue)
        @inbounds u = popfirst!(queue)
        @inbounds push!(topo_order, u)

        if adj_matrix isa SparseArrays.AbstractSparseMatrix
            # For sparse matrices, use findnz to get non-zero neighbors
            @inbounds for v in SparseArrays.findnz(adj_matrix[u, :])[1]
                @inbounds in_degree[v] -= 1
                if in_degree[v] == 0
                    @inbounds push!(queue, v)
                end
            end
        else
            # For dense matrices, iterate over the row directly
            @inbounds for v = 1:n
                if adj_matrix[u, v] != zero(T)
                    @inbounds in_degree[v] -= 1
                    if isapprox(in_degree[v], zero(T))
                        @inbounds push!(queue, v)
                    end
                end
            end
        end
    end

    return topo_order
end
