function regularized_curvature_data_2D(
    coefs::Array{Float64,2},
    epsilon::Real;
    higher_derivatives::Bool = false,
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)
    epsilon = validate_regularization_epsilon(epsilon)
    if isempty(coefs) || !all(isfinite, coefs)
        throw(ArgumentError("Chebyshev coefficients must be nonempty and finite"))
    end
    D1, D2 = curvature_derivation_matrices(coefs, derivation_matrix1, derivation_matrix2)
    if size(D1) != size(coefs) || size(D2) != size(coefs)
        throw(
            ArgumentError(
                "Derivation matrices must have the same size as the coefficients",
            ),
        )
    end

    # Prepare polynomial derivatives once for an entire sprinkling or integral.
    derivatives = (
        p = coefs,
        p_t = D1 * coefs,
        p_x = coefs * D1',
        p_tt = D2 * coefs,
        p_xx = coefs * D2',
    )
    if higher_derivatives
        lp = derivatives.p_tt - derivatives.p_xx
        derivatives = merge(
            derivatives,
            (
                p_tx = D1 * coefs * D1',
                lp_t = D1 * lp,
                lp_x = lp * D1',
                llp = D2 * lp - lp * D2',
            ),
        )
    end
    return derivatives, epsilon, D1, D2
end

function regularized_curvature_evaluator_2D(
    invariant::Function,
    coefs::Array{Float64,2},
    epsilon::Real;
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)
    derivatives, epsilon, D1, D2 = regularized_curvature_data_2D(
        coefs,
        epsilon;
        higher_derivatives = invariant === Ricci_box_Ricci_2D,
        derivation_matrix1,
        derivation_matrix2,
    )
    if invariant in (
        Ricci_scalar_2D,
        Kretschmann_scalar_2D,
        Goroff_Sagnotti_scalar_2D,
        Ricci_box_Ricci_2D,
        Weyl_squared_2D,
        Weyl_box_Weyl_2D,
    )
        return point -> regularized_invariant_2D(invariant, derivatives, point, epsilon)
    end
    return point ->
        invariant(coefs, point, epsilon; derivation_matrix1 = D1, derivation_matrix2 = D2)
end

function regularized_density_derivatives_2D(derivatives, point, epsilon)
    evaluate(c) = chebyshev_evaluate_2D(c, point)
    p = evaluate(derivatives.p)
    p_t = evaluate(derivatives.p_t)
    p_x = evaluate(derivatives.p_x)
    p_tt = evaluate(derivatives.p_tt)
    p_xx = evaluate(derivatives.p_xx)
    return (
        f = p^2 + epsilon,
        f_t = 2 * p * p_t,
        f_x = 2 * p * p_x,
        f_tt = 2 * (p_t^2 + p * p_tt),
        f_xx = 2 * (p_x^2 + p * p_xx),
        p = p,
        p_t = p_t,
        p_x = p_x,
        p_tt = p_tt,
        p_xx = p_xx,
    )
end

function regularized_invariant_2D(::typeof(Ricci_scalar_2D), derivatives, point, epsilon)
    f = regularized_density_derivatives_2D(derivatives, point, epsilon)
    # g = F eta: R = [F (F_tt - F_xx) - F_t^2 + F_x^2] / F^3.
    return (f.f * (f.f_tt - f.f_xx) - f.f_t^2 + f.f_x^2) / f.f^3
end

function regularized_invariant_2D(
    ::typeof(Kretschmann_scalar_2D),
    derivatives,
    point,
    epsilon,
)
    return regularized_invariant_2D(Ricci_scalar_2D, derivatives, point, epsilon)^2
end

function regularized_invariant_2D(
    ::typeof(Goroff_Sagnotti_scalar_2D),
    derivatives,
    point,
    epsilon,
)
    return regularized_invariant_2D(Ricci_scalar_2D, derivatives, point, epsilon)^3
end

function regularized_ricci_derivatives_2D(derivatives, point, epsilon)
    f = regularized_density_derivatives_2D(derivatives, point, epsilon)
    evaluate(c) = chebyshev_evaluate_2D(c, point)
    p_tx = evaluate(derivatives.p_tx)
    lp = f.p_tt - f.p_xx
    lp_t = evaluate(derivatives.lp_t)
    lp_x = evaluate(derivatives.lp_x)
    llp = evaluate(derivatives.llp)

    f_tx = 2 * (f.p_t * f.p_x + f.p * p_tx)
    Lf = f.f_tt - f.f_xx
    Lf_t = 2 * (2 * f.p_t * f.p_tt - 2 * f.p_x * p_tx + f.p_t * lp + f.p * lp_t)
    Lf_x = 2 * (2 * f.p_t * p_tx - 2 * f.p_x * f.p_xx + f.p_x * lp + f.p * lp_x)
    LLf =
        4 * (f.p_tt^2 - 2 * p_tx^2 + f.p_xx^2) +
        8 * (f.p_t * lp_t - f.p_x * lp_x) +
        2 * lp^2 +
        2 * f.p * llp

    # R = N / F^3. Apply the product rule to box R = -L R / F,
    # where L = partial_t^2 - partial_x^2. No integration by parts is used.
    N = f.f * Lf - f.f_t^2 + f.f_x^2
    N_t = f.f_t * Lf + f.f * Lf_t - 2 * f.f_t * f.f_tt + 2 * f.f_x * f_tx
    N_x = f.f_x * Lf + f.f * Lf_x - 2 * f.f_t * f_tx + 2 * f.f_x * f.f_xx
    LN = 4 * f_tx^2 - (f.f_tt + f.f_xx)^2 + f.f * LLf
    box_R =
        -(
            f.f^2 * LN - 6 * f.f * (N_t * f.f_t - N_x * f.f_x) - 3 * f.f * N * Lf +
            12 * N * (f.f_t^2 - f.f_x^2)
        ) / f.f^6
    return (
        density = f,
        R = N / f.f^3,
        R_t = (f.f * N_t - 3 * N * f.f_t) / f.f^4,
        R_x = (f.f * N_x - 3 * N * f.f_x) / f.f^4,
        box_R = box_R,
    )
end

function regularized_invariant_2D(::typeof(Ricci_box_Ricci_2D), derivatives, point, epsilon)
    curvature = regularized_ricci_derivatives_2D(derivatives, point, epsilon)
    return curvature.R * curvature.box_R
end

regularized_invariant_2D(::typeof(Weyl_squared_2D), derivatives, point, epsilon) = 0.0
regularized_invariant_2D(::typeof(Weyl_box_Weyl_2D), derivatives, point, epsilon) = 0.0

"""
    Ricci_scalar_2D(coefs, position, epsilon; derivation_matrix1=nothing,
                    derivation_matrix2=nothing)

Compute the Ricci scalar for ``g = (P^2 + \\epsilon) \\mathrm{diag}(-1,1)``.
`coefs` contains the Chebyshev coefficients of ``P`` and `epsilon` must be finite
and strictly positive. This overload uses the volume density of
[`make_regularized_polynomial_manifold_cset`](@ref), including at zeros of ``P``.
Omitting `epsilon` retains the original metric ``g = P^2 \\mathrm{diag}(-1,1)``.
"""
function Ricci_scalar_2D(
    coefs::Array{Float64,2},
    position::CausalSets.Coordinates{2},
    epsilon::Real;
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Float64
    evaluate = regularized_curvature_evaluator_2D(
        Ricci_scalar_2D,
        coefs,
        epsilon;
        derivation_matrix1,
        derivation_matrix2,
    )
    return evaluate(position)
end

"""
    Kretschmann_scalar_2D(coefs, position, epsilon; kwargs...)

Compute ``R_{abcd}R^{abcd} = R^2`` for ``g = (P^2 + \\epsilon) \\eta``.
The coefficient and keyword conventions match [`Ricci_scalar_2D`](@ref).
"""
function Kretschmann_scalar_2D(
    coefs::Array{Float64,2},
    position::CausalSets.Coordinates{2},
    epsilon::Real;
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Float64
    return Ricci_scalar_2D(
        coefs,
        position,
        epsilon;
        derivation_matrix1,
        derivation_matrix2,
    )^2
end

"""
    Goroff_Sagnotti_scalar_2D(coefs, position, epsilon; kwargs...)

Compute ``R_{ab}{}^{cd}R_{cd}{}^{ef}R_{ef}{}^{ab} = R^3`` for
``g = (P^2 + \\epsilon) \\eta``. The coefficient and keyword conventions match
[`Ricci_scalar_2D`](@ref).
"""
function Goroff_Sagnotti_scalar_2D(
    coefs::Array{Float64,2},
    position::CausalSets.Coordinates{2},
    epsilon::Real;
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Float64
    return Ricci_scalar_2D(
        coefs,
        position,
        epsilon;
        derivation_matrix1,
        derivation_matrix2,
    )^3
end

"""
    Ricci_box_Ricci_2D(coefs, position, epsilon; kwargs...)

Compute ``R \\Box R`` for ``g = (P^2 + \\epsilon) \\eta``, with
``\\Box = (P^2 + \\epsilon)^{-1}(-\\partial_t^2 + \\partial_x^2)``.
Derivatives through fourth order are computed analytically from the Chebyshev
coefficients of ``P``. The keyword conventions match [`Ricci_scalar_2D`](@ref).
"""
function Ricci_box_Ricci_2D(
    coefs::Array{Float64,2},
    position::CausalSets.Coordinates{2},
    epsilon::Real;
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Float64
    evaluate = regularized_curvature_evaluator_2D(
        Ricci_box_Ricci_2D,
        coefs,
        epsilon;
        derivation_matrix1,
        derivation_matrix2,
    )
    return evaluate(position)
end

"""
    Weyl_squared_2D(coefs, position, epsilon; kwargs...)

Return zero for ``C_{abcd}C^{abcd}`` in the regularized 2D metric
``g = (P^2 + \\epsilon) \\eta``. Inputs are validated as for [`Ricci_scalar_2D`](@ref).
"""
function Weyl_squared_2D(
    coefs::Array{Float64,2},
    position::CausalSets.Coordinates{2},
    epsilon::Real;
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Float64
    evaluate = regularized_curvature_evaluator_2D(
        Weyl_squared_2D,
        coefs,
        epsilon;
        derivation_matrix1,
        derivation_matrix2,
    )
    return evaluate(position)
end

"""
    Weyl_box_Weyl_2D(coefs, position, epsilon; kwargs...)

Return zero for ``C_{abcd}\\Box C^{abcd}`` in the regularized 2D metric
``g = (P^2 + \\epsilon) \\eta``. Inputs are validated as for [`Ricci_scalar_2D`](@ref).
"""
function Weyl_box_Weyl_2D(
    coefs::Array{Float64,2},
    position::CausalSets.Coordinates{2},
    epsilon::Real;
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Float64
    evaluate = regularized_curvature_evaluator_2D(
        Weyl_box_Weyl_2D,
        coefs,
        epsilon;
        derivation_matrix1,
        derivation_matrix2,
    )
    return evaluate(position)
end

"""
    curvature_invariant_of_sprinkling(invariant, coefs, sprinkling, epsilon; kwargs...)

Evaluate an invariant for ``g = (P^2 + \\epsilon) \\eta`` at each sprinkled point.
The coefficients describe ``P``. The six built-in invariants reuse precomputed
polynomial derivatives across all points. A custom invariant must accept
`(coefs, point, epsilon; derivation_matrix1, derivation_matrix2)`.
"""
function curvature_invariant_of_sprinkling(
    invariant::Function,
    coefs::Array{Float64,2},
    sprinkling::Vector{CausalSets.Coordinates{2}},
    epsilon::Real;
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Vector{Float64}
    evaluate = regularized_curvature_evaluator_2D(
        invariant,
        coefs,
        epsilon;
        derivation_matrix1,
        derivation_matrix2,
    )
    return [evaluate(point) for point in sprinkling]
end

"""
    Ricci_scalar_2D_of_sprinkling(coefs, sprinkling, epsilon; kwargs...)

Evaluate the regularized Ricci scalar at each sprinkled point. This is shorthand
for `curvature_invariant_of_sprinkling(Ricci_scalar_2D, coefs, sprinkling, epsilon)`.
"""
function Ricci_scalar_2D_of_sprinkling(
    coefs::Array{Float64,2},
    sprinkling::Vector{CausalSets.Coordinates{2}},
    epsilon::Real;
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Vector{Float64}
    return curvature_invariant_of_sprinkling(
        Ricci_scalar_2D,
        coefs,
        sprinkling,
        epsilon;
        derivation_matrix1,
        derivation_matrix2,
    )
end

"""
    integrated_curvature_invariant_2D(invariant, coefs, epsilon;
        boundary=CausalSets.BoxBoundary{2}(((-1.0,-1.0),(1.0,1.0))),
        rtol=1e-6, atol=1e-10, maxevals=1_000_000,
        derivation_matrix1=nothing, derivation_matrix2=nothing)

Compute ``\\int_B I[g]\\,\\mathrm{d}V_g`` for the regularized 2D metric
``g = (P^2 + \\epsilon) \\eta`` by adaptive cubature. The integrand includes
the volume density ``P^2 + \\epsilon``. `coefs` represents ``P`` and `epsilon`
must be finite and strictly positive. The ordered, finite box bounds must lie
within ``[-1,1]^2``, where the Chebyshev evaluator is defined.

Return `(integral, error)`, with an estimated absolute quadrature error.
Cubature stops at the requested tolerance or evaluation limit; check
`error <= max(atol, rtol * abs(integral))` before using a result that needs that
accuracy. Error estimates are not rigorous bounds. In particular, small
`epsilon` can require many evaluations. `rtol` and `atol` must be finite and
nonnegative, with at least one positive; `maxevals` must be positive.

All six built-in invariants are supported. Custom invariants use the signature
documented in [`curvature_invariant_of_sprinkling`](@ref). ``R\\Box R`` is
integrated directly, without discarding boundary terms through integration by
parts. No sprinkling is needed for this continuum integral.
"""
function integrated_curvature_invariant_2D(
    invariant::Function,
    coefs::Array{Float64,2},
    epsilon::Real;
    boundary::CausalSets.BoxBoundary{2} = CausalSets.BoxBoundary{2}((
        (-1.0, -1.0),
        (1.0, 1.0),
    )),
    rtol::Real = 1e-6,
    atol::Real = 1e-10,
    maxevals::Int64 = 1_000_000,
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Tuple{Float64,Float64}
    epsilon = validate_regularization_epsilon(epsilon)
    lower, upper = validate_box_quadrature_2D(boundary, rtol, atol, maxevals)
    evaluate = regularized_curvature_evaluator_2D(
        invariant,
        coefs,
        epsilon;
        derivation_matrix1,
        derivation_matrix2,
    )
    function integrand(coords)
        point = CausalSets.Coordinates{2}(Tuple(coords))
        density = chebyshev_evaluate_2D(coefs, point)^2 + epsilon
        return evaluate(point) * density
    end
    return HCubature.hcubature(integrand, lower, upper; rtol, atol, maxevals)
end

function validate_box_quadrature_2D(boundary, rtol, atol, maxevals)
    lower, upper = boundary.edges
    if !all(isfinite, lower) ||
       !all(isfinite, upper) ||
       !all(-1.0 .<= lower .< upper .<= 1.0)
        throw(ArgumentError("Box bounds must be ordered, finite, and within [-1,1]^2"))
    end
    if !isfinite(rtol) ||
       !isfinite(atol) ||
       rtol < 0 ||
       atol < 0 ||
       (rtol == 0 && atol == 0) ||
       maxevals <= 0
        throw(ArgumentError("Invalid quadrature tolerances or evaluation limit"))
    end
    return lower, upper
end
