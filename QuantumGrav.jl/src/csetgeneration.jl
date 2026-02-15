"""
    weighted_simplex_indices(r_vec, npoints)

Build weighted-simplex multi-indices `α` satisfying `sum(α[i] * log(r_vec[i])) <= Λ`
with `Λ = 2 * log(npoints) + 1`. Returns `(order_inds, αmax)`, where `αmax[i]` is the
largest admissible degree in dimension `i`.
"""
function weighted_simplex_indices(
    r_vec::NTuple{D,Float64},
    npoints::Int,
)::Tuple{Vector{NTuple{D,Int}},Vector{Int}} where D
    if npoints <= 0
        throw(ArgumentError("npoints must be greater than 0, got $npoints"))
    end
    if any(r_vec .<= 1.0)
        throw(ArgumentError("all components of r_vec must be > 1, got $r_vec"))
    end

    logr = ntuple(i -> log(r_vec[i]), D)
    Λ = 2 * log(npoints) + 1
    αmax = [floor(Int, Λ / logr[i]) for i in 1:D]

    order_inds = NTuple{D,Int}[]
    for I in CartesianIndices(ntuple(i -> 0:αmax[i], D))
        α = Tuple(I)
        w = sum(α[i] * logr[i] for i in 1:D)
        if w <= Λ
            push!(order_inds, α)
        end
    end
    return order_inds, αmax
end

"""
    random_chebyshev_coefficients(rng, order, r, d)

Sample a rank-`d` tensor of Chebyshev coefficients on degrees `0:order` in each axis,
with isotropic envelope `r^(-|α|)` where `|α| = α₁ + ... + α_d`.
"""
function random_chebyshev_coefficients(
    rng::Random.AbstractRNG,
    order::Int64,
    r::Float64,
    d::Int64,
)::Array{Float64}
    dims = ntuple(_ -> order + 1, d)
    chebyshev_coefs = zeros(Float64, dims)

    inv_r = inv(r)
    decay = ones(Float64, order + 1)
    for k = 2:(order + 1)
        decay[k] = decay[k - 1] * inv_r
    end

    for I in CartesianIndices(chebyshev_coefs)
        coeff = randn(rng)
        for idx in Tuple(I)
            coeff *= decay[idx]
        end
        chebyshev_coefs[I] = coeff
    end

    return chebyshev_coefs
end

"""
    random_weighted_chebyshev_coefficients(rng, αmax, logr, order_inds)

Sample a tensor of Chebyshev coefficients on a weighted admissible index set.
Entries outside `order_inds` are zero.
"""
function random_weighted_chebyshev_coefficients(
    rng::Random.AbstractRNG,
    αmax::NTuple{D,Int},
    logr::NTuple{D,Float64},
    order_inds::AbstractVector{NTuple{D,Int}},
)::Array{Float64,D} where D
    coefs = zeros(Float64, ntuple(i -> αmax[i] + 1, D))
    for α in order_inds
        w = sum(α[i] * logr[i] for i in 1:D)
        coefs[(α .+ 1)...] = exp(-w) * randn(rng)
    end
    return coefs
end

function CausalSets.eval_polynomial(
    coefs::CausalSets.PolynomialCoefs{N},
    x::CausalSets.Coordinates,
    integrate::NTuple{N,Bool},
    order_inds::Vector{NTuple{N,Int}},
)::Float64 where N
    val = 0.0
    for α in order_inds
        var_prod = 1.0
        pow_law_quotient = 1.0
        for i in 1:N
            p = α[i] + (integrate[i] ? 1 : 0)
            var_prod *= x[i]^p
            if integrate[i]
                pow_law_quotient *= (α[i] + 1)
            end
        end
        val += var_prod * coefs.c[(α .+ 1)...] / pow_law_quotient
    end
    return val
end

function CausalSets.transform_polynomial(
    cheb_coefs::Array{Float64, D},
    cheb_to_taylor_mat::AbstractMatrix,
    order_inds::Vector{NTuple{D,Int}};
    αmax::Union{Nothing,AbstractVector{<:Integer}},
)::Array{Float64, D} where D

    αmax_t = isnothing(αmax) ? ntuple(i -> maximum(α[i] for α in order_inds), D) : Tuple(αmax)
    taylor_coefs = zeros(Float64, ntuple(i -> αmax_t[i] + 1, D))

    for α in order_inds
        Iα = α .+ 1
        cα = cheb_coefs[Iα...]
        for β in CartesianIndices(ntuple(i -> 0:α[i], D))
            βt = Tuple(β)
            coef = cα
            for i in 1:D
                coef *= cheb_to_taylor_mat[βt[i] + 1, α[i] + 1]
            end
            taylor_coefs[(βt .+ 1)...] += coef
        end
    end

    return taylor_coefs
end

function CausalSets.polynomial_pow(
    coefs::Array{Float64, D},
    n::Int64,
    order_inds::Vector{NTuple{D,Int}};
    αmax::Union{Nothing,AbstractVector{<:Integer}} = nothing,
)::Array{Float64, D} where D

    @assert n > 1

    αmax_t = isnothing(αmax) ?
        ntuple(i -> maximum(α[i] for α in order_inds), D) :
        Tuple(αmax)

    new_coefs = zeros(Float64, ntuple(i -> αmax_t[i] + 1, D))
    admissible = Set(order_inds)

    # Convolution restricted to simplex structure
    for inds in Iterators.product(ntuple(_ -> order_inds, n)...)
        coef_prod = 1.0
        αsum = zeros(Int, D)
        for α in inds
            coef_prod *= coefs[(α .+ 1)...]
            αsum .+= α
        end
        if all(αsum[i] <= αmax_t[i] for i in 1:D)
            αsum_t = ntuple(i -> αsum[i], D)
            if αsum_t in admissible
                new_coefs[(αsum .+ 1)...] += coef_prod
            end
        end
    end

    return new_coefs
end

"""
    TruncatedPolynomialManifold{N}(coefs)

An ``N``-dimensional conformally flat manifold with the metric

```math
\\mathrm{d}s^2 = F^(2/d)(x) (-\\mathrm{d}t^2 + \\mathrm{d}x_i^2)
```

where ``F(x)`` is a nontrivially truncated Taylor polynomial with coefficients 
determined by the order_ind entries of the ``N``-dimensional array `coefs`. 
Since Julia indices start with 1, the powers will be shifted by one - 
for instance, for a 3D manifold, the array element `coefs[3, 2, 5]` will correspond 
to the coefficient of the monomial ``x_0^2 x_1^1 x_2^4``.
"""
struct TruncatedPolynomialManifold{N} <: CausalSets.ConformallyMinkowskianManifold{N}
    coefs::CausalSets.PolynomialCoefs
    order_inds::Vector{NTuple{N,Int}}
end

function TruncatedPolynomialManifold{N}(coefs::AbstractArray{Float64, N}, order_inds::Vector{NTuple{N,Int}}) where N
    return TruncatedPolynomialManifold{N}(CausalSets.PolynomialCoefs(coefs), order_inds)
end

function is_in_boundary(
    manifold::TruncatedPolynomialManifold{N},
    boundary::CausalSets.BoxBoundary{N},
    coords::CausalSets.Coordinates{N},
) where N
    return all(boundary.edges[1] .< coords .&& coords .< boundary.edges[2])
end

sampling_primary_coord(::TruncatedPolynomialManifold) = 1

struct TruncatedPolynomialRectangularSegment{N} <: CausalSets.SprinklingSegment where N
    edges::Tuple{CausalSets.Coordinates{N}, CausalSets.Coordinates{N}}
    coefs::CausalSets.PolynomialCoefs
    order_inds::Vector{NTuple{N,Int}}
end

function CausalSets.definite_integral(
    coefs::CausalSets.PolynomialCoefs{N}, 
    params::CausalSets.DimLimits, 
    integrate::NTuple{N, Bool},
    order_inds::Vector{NTuple{N,Int}}
    )::Float64 where N
    d = length(params.l)
    if all(isa.(params.l, Float64))
        return CausalSets.eval_polynomial(coefs, params.l, integrate, order_inds)
    else
        for i in 1:d
            if isa(params.l[i], Tuple)
                upper_params = CausalSets.DimLimits(ntuple(j -> j == i ? params.l[j][2] : params.l[j], d))
                lower_params = CausalSets.DimLimits(ntuple(j -> j == i ? params.l[j][1] : params.l[j], d))
                return CausalSets.definite_integral(coefs, upper_params, integrate, order_inds) - 
                CausalSets.definite_integral(coefs, lower_params, integrate, order_inds)
            end
        end
    end
end

function CausalSets.sample_step(
    rng::Random.AbstractRNG, 
    coefs::CausalSets.PolynomialCoefs{N}, 
    params::CausalSets.DimLimits, 
    i::Int64,
    order_inds::Vector{NTuple{N,Int}};
    αmax::Union{Nothing,Vector{Int64}} = nothing,
    )::CausalSets.DimLimits where N
    d = length(params.l)
    lower_bound = CausalSets.DimLimits(ntuple(j -> j == i ? params.l[j][1] : params.l[j], d))
    upper_bound = CausalSets.DimLimits(ntuple(j -> j == i ? params.l[j][2] : params.l[j], d))
    integrate = ntuple(j -> j >= i, N)
    upper_vol = CausalSets.definite_integral(coefs, upper_bound, integrate, order_inds)
    lower_vol = CausalSets.definite_integral(coefs, lower_bound, integrate, order_inds)
    partial_vol = (upper_vol - lower_vol) * rand(rng) + lower_vol
    coord = CausalSets.bisect_inverse(x -> CausalSets.definite_integral(coefs, CausalSets.DimLimits(ntuple(j -> j == i ? x : params.l[j], d)), integrate, order_inds), partial_vol, params.l[i][1], params.l[i][2])
    return CausalSets.DimLimits(ntuple(j -> j == i ? coord : params.l[j], d))
end

CausalSets.min_max_primary_coords(seg::TruncatedPolynomialRectangularSegment) = (seg.edges[1][1], seg.edges[2][1])
function CausalSets.partial_volume(seg::TruncatedPolynomialRectangularSegment{N}, t) where N
    params = CausalSets.DimLimits(((seg.edges[1][1], t), ntuple(i -> (seg.edges[1][i+1], seg.edges[2][i+1]), N-1)...))
    return CausalSets.definite_integral(seg.coefs, params, ntuple(i -> true, N), seg.order_inds)
end

function CausalSets.inverse_partial_volume(seg::TruncatedPolynomialRectangularSegment{N}, vol) where N
    volf = t -> CausalSets.definite_integral(seg.coefs, CausalSets.DimLimits(((seg.edges[1][1], t), ntuple(i -> (seg.edges[1][i+1], seg.edges[2][i+1]), N-1)...)), ntuple(i -> true, N), seg.order_inds)
    return CausalSets.bisect_inverse(volf, vol, seg.edges[1][1], seg.edges[2][1])
end

function CausalSets.sample_at_primary_coord(
    rng::Random.AbstractRNG,
    seg::TruncatedPolynomialRectangularSegment{N},
    t,
)::CausalSets.Coordinates{N} where N
    params = CausalSets.DimLimits((t, ntuple(i -> (seg.edges[1][i+1], seg.edges[2][i+1]), N-1)...))
    for i in 2:N
        params = CausalSets.sample_step(rng, seg.coefs, params, i, seg.order_inds)
    end
    return params.l
end

function CausalSets.sprinkling_segment_sequence(
    manifold::TruncatedPolynomialManifold{N},
    boundary::CausalSets.BoxBoundary,
) where N
    return CausalSets.SprinklingSegmentSequence(
                                     TruncatedPolynomialRectangularSegment(boundary.edges, manifold.coefs, manifold.order_inds)
                                    )
end

"""
    make_polynomial_manifold_cset(
    npoints::Int64, 
    rng::Random.AbstractRNG, 
    order::Int64, 
    r::Float64; 
    d::Int64=2, 
    type::Type{T}=Float32)::Tuple{
    CausalSets.BitArrayCauset,
    Vector{Tuple{T,Vararg{T}}},
    Array{T,d}
} where {T<:Number}

Generate a causal set by sampling from a positive polynomial constructed via a truncated 
Chebyshev series with exponentially decaying coefficients.

# Arguments
- `npoints::Int`: Number of elements to sprinkle into the causal set. Must be > 0.
- `seed::Int`: Seed for pseudo-random number generation to ensure reproducibility.
- `order::Int`: Truncation order of the Chebyshev expansion (number of basis functions in each direction). Must be > 0.
- `r::Float64`: Decay base for Chebyshev coefficients. Must be > 1 to ensure exponential convergence; defines the radius of analyticity in the complex plane.

# Keyword arguments
- `d::Int64`: Dimension of the manifold, defaults to 2.
- `type::Type{T}`: Type to which the sprinkling coordinates will be converted (default is Float32).

# Returns
- A tuple `(cset, sprinkling, chebyshev_coefs)` where:
    - `cset`: The generated causal set.
    - `sprinkling`: The list of sprinkled points.
    - `chebyshev_coefs`: The rank-d tensor of Chebyshev coefficients used to construct the manifold.

# Throws
- `ArgumentError` if `npoints <= 0`
- `ArgumentError` if `order <= 0`
- `ArgumentError` if `r <= 1`
- `ArgumentError` if `d < 1`
"""
function make_polynomial_manifold_cset(
    npoints::Int64,
    rng::Random.AbstractRNG,
    order::Int64,
    r::Float64;
    d::Int64 = 2,
    type::Type{T} = Float32,
)::Tuple{
    CausalSets.BitArrayCauset,
    Vector{Tuple{T,Vararg{T}}},
    Array{T,d}
} where {T<:Number}

    if npoints <= 0
        throw(ArgumentError("npoints must be greater than 0, got $npoints"))
    end

    if order < 0
        throw(ArgumentError("order must be greater than -1, got $order"))
    end

    if r <= 1
        throw(
            ArgumentError(
                "r must be greater than 1 for exponential convergence of the Chebyshev series, got $r",
            ),
        )
    end

    if d < 1
        throw(ArgumentError("dimension d must be at least 1, got $d"))
    end

    # Sample Chebyshev coefficients on degrees 0:order with isotropic decay envelope.
    chebyshev_coefs = random_chebyshev_coefficients(rng, order, r, d)


    # Construct the Chebyshev-to-Taylor transformation matrix
    cheb_to_taylor_mat = CausalSets.chebyshev_coef_matrix(order)

    # Transform Chebyshev coefficients to Taylor coefficients
    taylorcoefs = CausalSets.transform_polynomial(chebyshev_coefs, cheb_to_taylor_mat)

    # Square the polynomial to ensure positivity
    squaretaylorcoefs = CausalSets.polynomial_pow(taylorcoefs, 2)

    # Create a polynomial manifold from the squared Taylor coefficients
    polym = CausalSets.PolynomialManifold{d}(squaretaylorcoefs)

    # Define the square box boundary
    boundary = CausalSets.BoxBoundary{d}((ntuple(_ -> -1.0, d), ntuple(_ -> 1.0, d)))

    # Generate a sprinkling of npoints in the manifold within the boundary
    sprinkling = CausalSets.generate_sprinkling(polym, boundary, npoints; rng = rng)

    # Construct the causal set from the manifold and sprinkling
    cset = CausalSets.BitArrayCauset(polym, sprinkling)

    return cset, sprinkling, type.(chebyshev_coefs)
end

"""
    make_polynomial_manifold_cset(
    npoints::Int64, 
    rng::Random.AbstractRNG, 
    order::Int64, 
    r::Float64; 
    d::Int64=2, 
    type::Type{T}=Float32)::Tuple{
    CausalSets.BitArrayCauset,
    Vector{Tuple{T,Vararg{T}}},
    Array{T,d}
} where {T<:Number}

Generate a causal set by sampling from a positive polynomial constructed via a truncated 
Chebyshev series with exponentially decaying coefficients.

# Arguments
- `npoints::Int`: Number of elements to sprinkle into the causal set. Must be > 0.
- `seed::Int`: Seed for pseudo-random number generation to ensure reproducibility.
- `r_vec::Vector{Float64}`: Decay base for Chebyshev coefficients (independent in each dimension). 
Each must be > 1 to ensure exponential convergence; defines the radius of analyticity in the complex plane.

# Keyword arguments
- `d::Int64`: Dimension of the manifold, defaults to 2.
- `type::Type{T}`: Type to which the sprinkling coordinates will be converted (default is Float32).

# Returns
- A tuple `(cset, sprinkling, chebyshev_coefs)` where:
    - `cset`: The generated causal set.
    - `sprinkling`: The list of sprinkled points.
    - `chebyshev_coefs`: The rank-d tensor of Chebyshev coefficients used to construct the manifold.

# Throws
- `ArgumentError` if `npoints <= 0`
- `ArgumentError` if `order <= 0`
- `ArgumentError` if `r <= 1`
- `ArgumentError` if `d < 1`
"""
function make_anisotropically_weighted_polynomial_manifold_cset(
    npoints::Int64,
    rng::Random.AbstractRNG,
    r_vec::NTuple{D,Float64};
    d::Int64 = D,
    type::Type{T} = Float32,
)::Tuple{
    CausalSets.BitArrayCauset,
    Vector{Tuple{T,Vararg{T}}},
    Array{T,D}
} where {T<:Number,D}

    if npoints <= 0
        throw(ArgumentError("npoints must be greater than 0, got $npoints"))
    end

    if any(r_vec .<= 1)
        throw(
            ArgumentError(
                "components of r_vec must be greater than 1 for exponential convergence of the Chebyshev series, got $r_vec",
            ),
        )
    end

    if d < 1
        throw(ArgumentError("dimension d must be at least 1, got $d"))
    end

    order_inds, αmax = weighted_simplex_indices(r_vec, npoints)
    αmax_t = Tuple(αmax)

    logr = Tuple(log.(r_vec))
    chebyshev_coefs =
        random_weighted_chebyshev_coefficients(rng, αmax_t, logr, order_inds)

    # Construct the Chebyshev-to-Taylor transformation matrices for each dimension
    cheb_to_taylor_mat = CausalSets.chebyshev_coef_matrix(maximum(αmax_t))

    taylorcoefs = CausalSets.transform_polynomial(
        chebyshev_coefs,
        cheb_to_taylor_mat,
        order_inds;
        αmax = αmax,
    )

    squaretaylorcoefs = CausalSets.polynomial_pow(
        taylorcoefs,
        2,
        order_inds;
        αmax = αmax,
    )

    # Create a polynomial manifold from the squared Taylor coefficients
    polym = TruncatedPolynomialManifold{d}(squaretaylorcoefs, order_inds)

    # Define the square box boundary
    boundary = CausalSets.BoxBoundary{d}((ntuple(_ -> -1.0, d), ntuple(_ -> 1.0, d)))

    # Generate a sprinkling of npoints in the manifold within the boundary
    sprinkling = CausalSets.generate_sprinkling(polym, boundary, npoints; rng = rng)

    # Construct the causal set from the manifold and sprinkling
    cset = CausalSets.BitArrayCauset(polym, sprinkling)

    return cset, sprinkling, type.(chebyshev_coefs)
end
