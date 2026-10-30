function validate_regularization_epsilon(epsilon::Real)::Float64
    value = Float64(epsilon)
    if !isfinite(value) || value <= 0
        throw(ArgumentError("epsilon must be finite and strictly positive, got $epsilon"))
    end
    return value
end

"""
    regularized_polynomial_manifold(coefs::Array{Float64,d}, epsilon::Real)

Construct a `CausalSets.PolynomialManifold{d}` with volume density
``F = P^2 + \\epsilon``, where `coefs` contains the tensor-product Chebyshev
coefficients of ``P`` on ``[-1,1]^d``. The metric is
``g = F^{2/d} \\eta``. In two dimensions this is ``g = (P^2 + \\epsilon) \\eta``.

`epsilon` must be finite and strictly positive. All coefficient-array axes must
have the same nonzero length, and the coefficients must be finite. The input
array is not modified. Positivity holds mathematically; expanded Taylor
polynomials remain subject to floating-point cancellation at high orders.
"""
function regularized_polynomial_manifold(coefs::Array{Float64,d}, epsilon::Real) where {d}
    epsilon = validate_regularization_epsilon(epsilon)
    if d < 1 || isempty(coefs) || !all(==(size(coefs, 1)), size(coefs))
        throw(ArgumentError("Chebyshev coefficient axes must have the same nonzero length"))
    end
    if !all(isfinite, coefs)
        throw(ArgumentError("Chebyshev coefficients must be finite"))
    end

    order = size(coefs, 1) - 1
    # CausalSets.chebyshev_coef_matrix assumes that at least T_1 is present.
    taylorcoefs =
        order == 0 ? copy(coefs) :
        CausalSets.transform_polynomial(coefs, CausalSets.chebyshev_coef_matrix(order))
    densitycoefs = CausalSets.polynomial_pow(taylorcoefs, 2)
    densitycoefs[ntuple(_->1, d)...] += epsilon
    return CausalSets.PolynomialManifold{d}(densitycoefs)
end

"""
    make_regularized_polynomial_manifold_cset(npoints, rng, order, r, epsilon;
                                            d=2, type=Float32)

Generate a causal set on ``[-1,1]^d`` with volume density ``P^2 + \\epsilon``.
The random polynomial ``P`` uses the same exponentially decaying Chebyshev
coefficients as [`make_polynomial_manifold_cset`](@ref). `epsilon` is a finite,
strictly positive parameter, `order` is nonnegative, `r > 1` is finite, and
`npoints` and `d` must be positive.

Return `(cset, sprinkling, chebyshev_coefs)`, with coordinates and coefficients
converted to `type`. The returned coefficients represent ``P``, not the volume
density. Keep `epsilon` alongside them and pass it to the regularized curvature
methods. Use `type=Float64` to retain the coefficients used for sampling.
In two dimensions, the metric is ``g = (P^2 + \\epsilon) \\eta``; in general it
is ``g = (P^2 + \\epsilon)^{2/d} \\eta``.
"""
function make_regularized_polynomial_manifold_cset(
    npoints::Int64,
    rng::Random.AbstractRNG,
    order::Int64,
    r::Float64,
    epsilon::Real;
    d::Int64 = 2,
    type::Type{T} = Float32,
)::Tuple{CausalSets.BitArrayCauset,Vector{NTuple{d,T}},Array{T,d}} where {T<:Number}
    epsilon = validate_regularization_epsilon(epsilon)
    if npoints <= 0
        throw(ArgumentError("npoints must be greater than 0, got $npoints"))
    end
    if order < 0
        throw(ArgumentError("order must be nonnegative, got $order"))
    end
    if !isfinite(r) || r <= 1
        throw(ArgumentError("r must be finite and greater than 1, got $r"))
    end
    if d < 1
        throw(ArgumentError("dimension d must be at least 1, got $d"))
    end

    chebyshev_coefs = zeros(Float64, ntuple(_ -> order + 1, d))
    for I in CartesianIndices(chebyshev_coefs)
        chebyshev_coefs[I] = r^(-sum(Tuple(I))) * randn(rng)
    end
    polym = regularized_polynomial_manifold(chebyshev_coefs, epsilon)
    boundary = CausalSets.BoxBoundary{d}((ntuple(_ -> -1.0, d), ntuple(_ -> 1.0, d)))
    sprinkling = CausalSets.generate_sprinkling(polym, boundary, npoints; rng = rng)
    cset = CausalSets.BitArrayCauset(polym, sprinkling)
    return cset, sprinkling, type.(chebyshev_coefs)
end
