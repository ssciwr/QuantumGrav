
"""
    PseudoManifold{N}

DOCSTRING

"""
struct PseudoManifold{N} <: CSets.AbstractManifold{N} end


"""
    get_manifold_name(type::Type, d)

DOCSTRING
"""
function get_manifold_name(type::Type, d)
    Dict(
        CSets.MinkowskiManifold{d} => "Minkowski",
        CSets.DeSitterManifold{d} => "DeSitter",
        CSets.AntiDeSitterManifold{d} => "AntiDeSitter",
        CSets.HypercylinderManifold{d} => "HyperCylinder",
        CSets.TorusManifold{d} => "Torus",
        PseudoManifold{2} => "Random")[type]
end

"""
    get_manifold_encoding

"""
get_manifold_encoding = Dict(
    "Minkowski" => 1,
    "DeSitter" => 3,
    "AntiDeSitter" => 4,
    "HyperCylinder" => 2,
    "Torus" => 5,
    "Random" => 6
)

"""
    make_manifold(i::Int, d::Int)

DOCSTRING
"""
function make_manifold(i::Int, d::Int)::CSets.AbstractManifold
    if i == 1
        return CSets.MinkowskiManifold{d}()
    elseif i == 2
        return CSets.HypercylinderManifold{d}(1.0)
    elseif i == 3
        return CSets.DeSitterManifold{d}(1.0)
    elseif i == 4
        return CSets.AntiDeSitterManifold{d}(1.0)
    elseif i == 5
        return CSets.TorusManifold{d}(1.0)
    elseif i == 6
        return PseudoManifold{d}()
    else
        error("Unsupported manifold: $i")
    end
end

"""
    make_cset(manifold::CSets.AbstractManifold, boundary::CSets.AbstractBoundary, n::Int64, d::Int, rng::Random.AbstractRNG, type::Type{T})

DOCSTRING

# Arguments:
- `manifold`: DESCRIPTION
- `boundary`: DESCRIPTION
- `n`: DESCRIPTION
- `d`: DESCRIPTION
- `rng`: DESCRIPTION
- `type`: DESCRIPTION
"""
function make_cset(
        manifold::CSets.AbstractManifold, boundary::CSets.AbstractBoundary, n::Int64,
        d::Int, rng::Random.AbstractRNG, type::Type{T}) where {T <: Number}
    if manifold isa PseudoManifold
        return CSets.sample_random_causet(CSets.BitArrayCauset, n, 300, rng),
        stack(make_pseudosprinkling(n, d, -0.49, 0.49, type; rng = rng), dims = 1)
    else
        sprinkling = CSets.generate_sprinkling(manifold, boundary, n; rng = rng)
        cset = CSets.BitArrayCauset(manifold, sprinkling)
        return cset, stack(collect.(sprinkling), dims = 1)
    end
end

"""
    resize(m::AbstractArray{T}, new_size::Tuple)

DOCSTRING
"""
function resize(m::AbstractArray{T}, new_size::Tuple)::AbstractArray{T} where {T <: Number}
    if any(size(m) .< new_size)
        resized_m = m isa SparseArrays.AbstractSparseArray ?
                    SparseArrays.spzeros(T, new_size...) : zeros(T, new_size...)
        @inbounds resized_m[tuple([1:n for n in size(m)]...)...] .= m
        return resized_m
    else
        return @inbounds m[tuple([1:n for n in new_size]...)...]
    end
end

"""
    make_pseudosprinkling(n::Int64, d::Int64, box_min::Float64, box_max::Float64, type::Type{T}; rng = Random.MersenneTwister(1234))

DOCSTRING

# Arguments:
- `n`: DESCRIPTION
- `d`: DESCRIPTION
- `box_min`: DESCRIPTION
- `box_max`: DESCRIPTION
- `type`: DESCRIPTION
- `rng`: DESCRIPTION
"""
function make_pseudosprinkling(
        n::Int64, d::Int64, box_min::Float64, box_max::Float64, type::Type{T};
        rng = Random.MersenneTwister(1234))::Vector{Vector{T}} where {T <: Number}
    distr = Distributions.Uniform(box_min, box_max)

    return [[rand(distr) for i in 1:d] for _ in 1:n]
end

"""
    topsort(adj_matrix, in_degree::AbstractVector{T})

DOCSTRING
"""
function topsort(adj_matrix, in_degree::AbstractVector{T})::Vector{Int} where {T <: Number}
    n = size(adj_matrix, 1)

    # Topological sort using Kahn's algorithm --> will be needed later for the topo order of the csets
    queue = Vector{Int64}()
    sizehint!(queue, n)
    for i in 1:n
        if isapprox(in_degree[i], zero(T))
            @inbounds push!(queue, i)
        end
    end

    topo_order = Vector{Int64}()
    sizehint!(topo_order, n)
    while !isempty(queue)
        @inbounds u = popfirst!(queue)
        @inbounds push!(topo_order, u)

        # For each neighbor v of u
        @inbounds for v in SparseArrays.findnz(adj_matrix[u, :])[1]
            @inbounds in_degree[v] -= 1
            if in_degree[v] == 0
                @inbounds push!(queue, v)
            end
        end
    end

    return topo_order
end
