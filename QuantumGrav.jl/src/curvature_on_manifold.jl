""" 
    chebyshev_derivation_matrix(max_cheb_order::Int64, derivative_order::Int64) -> Array{Float64, 2}

Constructs the Chebyshev derivation matrix of a specified order and derivative order.

# Arguments
- `max_cheb_order::Int64`: The maximum order of the Chebyshev polynomial expansion (degree).
- `derivative_order::Int64`: The order of the derivative to compute (e.g., 1 for first derivative).

# Returns
- `Array{Float64, 2}`: A square matrix of size `(max_cheb_order+1, max_cheb_order+1)` representing the linear transformation from Chebyshev coefficients of a function to those of its derivative.

# Throws
- ArgumentError: if Chebyshev order < 0 or derivative order < 1.

# Details
The matrix acts on a coefficient vector of Chebyshev polynomials and returns the coefficients of the derivative of the function represented by those coefficients.

"""
function chebyshev_derivation_matrix(
    max_cheb_order::Int64,
    derivative_order::Int64,
)::Array{Float64,2}

    if max_cheb_order < 0
        throw(ArgumentError("max_cheb_order has to be at least 0, is $(max_cheb_order)."))
    end

    if derivative_order < 1
        throw(ArgumentError("max_cheb_order has to be at least 1, is $(derivative_order)."))
    end

    transformation_matrix_first_derivative = zeros(max_cheb_order + 1, max_cheb_order + 1)

    for n = 0:max_cheb_order

        for k = mod(n-1, 2):2:(n-1)
            prefac = k == 0 ? 0.5 : 1

            transformation_matrix_first_derivative[k+1, n+1] += 2 * n * prefac
        end
    end

    return transformation_matrix_first_derivative^derivative_order
end

"""
    chebyshev_derivative_2D(
        coefs::Array{Float64, 2}, 
        derivative_variable_index::Int64, 
        derivative_order::Int64; 
        derivation_matrix::Union{Nothing, Array{Float64, 2}}=nothing
    ) -> Array{Float64, 2}

Computes the derivative of a 2D Chebyshev expansion along one variable up to a specified derivative order.

# Arguments
- `coefs::Array{Float64, 2}`: A matrix of Chebyshev coefficients representing a function expanded in two variables.
- `derivative_variable_index::Int64`: The index of the variable along which to differentiate (1 for first variable, 2 for second).
- `derivative_order::Int64`: The order of the derivative to compute.

# Keyword Arguments
- `derivation_matrix::Union{Nothing, Array{Float64, 2}}=nothing`: Optional precomputed Chebyshev derivation matrix to speed up computation.

# Returns
- `Array{Float64, 2}`: The matrix of Chebyshev coefficients representing the derivative of the function.

# Throws
- ArgumentError: if `derivative_variable_index` is not 1 or 2.

"""
function chebyshev_derivative_2D(
    coefs::Array{Float64,2},
    derivative_variable_index::Int64,
    derivative_order::Int64;
    derivation_matrix::Union{Nothing,Array{Float64,2}} = nothing,
)::Array{Float64,2}

    if derivative_variable_index != 1 && derivative_variable_index != 2
        throw(
            ArgumentError(
                "derivative_variable_index has to be in (1,2) is $(derivative_variable_index)",
            ),
        )
    end

    if derivative_order < 1
        throw(ArgumentError("max_cheb_order has to be at least 1, is $(derivative_order)."))
    end

    order = size(coefs, derivative_variable_index)-1
    derivation_matrix =
        isnothing(derivation_matrix) ?
        chebyshev_derivation_matrix(order, derivative_order) : derivation_matrix

    return derivative_variable_index == 1 ? derivation_matrix * coefs :
           coefs * derivation_matrix'
end

"""
    chebyshev_evaluate_2D(
        coefs::Array{Float64, 2}, 
        position::CausalSets.Coordinates{2}
    ) -> Float64

Evaluates a 2D Chebyshev expansion at a given position.

# Arguments
- `coefs::Array{Float64, 2}`: A matrix of Chebyshev coefficients representing a function expanded in two variables.
- `position::CausalSets.Coordinates{2}`: A 2D coordinate at which to evaluate the function, with each component in the domain of the Chebyshev polynomials (typically [-1,1]).

# Returns
- `Float64`: The value of the function represented by the Chebyshev coefficients at the specified position.

"""
function chebyshev_evaluate_2D(
    coefs::Array{Float64,2},
    position::CausalSets.Coordinates{2},
)::Float64
    chebyshev_T(n::Int64, x::Float64) = cos(n * acos(x))
    val = 0.0
    for i = 1:size(coefs, 1)
        for j = 1:size(coefs, 2)
            val +=
                coefs[i, j] * chebyshev_T(i-1, position[1]) * chebyshev_T(j-1, position[2])
        end
    end
    return val
end

"""
    curvature_derivation_matrices(
        coefs::Array{Float64, 2},
        derivation_matrix1::Union{Nothing, Array{Float64, 2}},
        derivation_matrix2::Union{Nothing, Array{Float64, 2}},
    ) -> Tuple{Array{Float64, 2}, Array{Float64, 2}}

Validates a coefficient matrix for curvature computations and returns the Chebyshev derivation matrices for first and second derivatives, constructing those that are not given.

# Arguments
- `coefs::Array{Float64, 2}`: A matrix of Chebyshev coefficients representing the conformal factor expanded in two variables.
- `derivation_matrix1::Union{Nothing, Array{Float64, 2}}`: Precomputed Chebyshev derivation matrix for first derivatives or `nothing`.
- `derivation_matrix2::Union{Nothing, Array{Float64, 2}}`: Precomputed Chebyshev derivation matrix for second derivatives or `nothing`.

# Returns
- `Tuple{Array{Float64, 2}, Array{Float64, 2}}`: The derivation matrices for first and second derivatives.

# Throws
- ArgumentError: if the coefficient matrix is not a square matrix. Asymmetric orders are not supported (but can be emulated by adding 0s in the coefficient matrix).

"""
function curvature_derivation_matrices(
    coefs::Array{Float64,2},
    derivation_matrix1::Union{Nothing,Array{Float64,2}},
    derivation_matrix2::Union{Nothing,Array{Float64,2}},
)::Tuple{Array{Float64,2},Array{Float64,2}}
    if size(coefs, 1) != size(coefs, 2)
        throw(
            ArgumentError(
                "At the moment only square coefficient matrices are supported, i. e., Chebyshev expansions at equal order for both variables. Dimensions are $(size(coefs,1)) and $(size(coefs,2))",
            ),
        )
    end

    return (
        isnothing(derivation_matrix1) ? chebyshev_derivation_matrix(size(coefs, 1)-1, 1) :
        derivation_matrix1,
        isnothing(derivation_matrix2) ? chebyshev_derivation_matrix(size(coefs, 1)-1, 2) :
        derivation_matrix2,
    )
end

"""
    Ricci_scalar_2D(
        coefs::Array{Float64, 2},
        position::CausalSets.Coordinates{2};
        derivation_matrix::Union{Nothing, Array{Float64, 2}}=nothing
    ) -> Float64

Computes the Ricci scalar curvature at a given position for a 2D Lorentzian manifold in conformally flat slicing with conformal factor expressed as Chebyshev expansion.

# Arguments
- `coefs::Array{Float64, 2}`: A matrix of Chebyshev coefficients representing the conformal factor expanded in two variables.
- `position::CausalSets.Coordinates{2}`: A 2D coordinate at which to evaluate the Ricci scalar.

# Keyword Arguments
- `derivation_matrix1::Union{Nothing, Array{Float64, 2}}=nothing`: Optional precomputed Chebyshev derivation matrix for first derivatives to speed up computation.
- `derivation_matrix2::Union{Nothing, Array{Float64, 2}}=nothing`: Optional precomputed Chebyshev derivation matrix for second derivatives to speed up computation.

# Returns
- `Float64`: The Ricci scalar curvature evaluated at the given position.

# Throws
- ArgumentError: if the coefficient matrix is not a square matrix. Asymmetric orders are not supported (but can be emulated by adding 0s in the coefficient matrix).

"""
function Ricci_scalar_2D(
    coefs::Array{Float64,2},
    position::CausalSets.Coordinates{2};
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Float64
    derivation_matrix1, derivation_matrix2 =
        curvature_derivation_matrices(coefs, derivation_matrix1, derivation_matrix2)
    function_at_position = chebyshev_evaluate_2D(coefs, position)
    first_derivative_time = chebyshev_evaluate_2D(
        chebyshev_derivative_2D(coefs, 1, 1; derivation_matrix = derivation_matrix1),
        position,
    )
    first_derivative_space = chebyshev_evaluate_2D(
        chebyshev_derivative_2D(coefs, 2, 1; derivation_matrix = derivation_matrix1),
        position,
    )
    second_derivative_time = chebyshev_evaluate_2D(
        chebyshev_derivative_2D(coefs, 1, 2; derivation_matrix = derivation_matrix2),
        position,
    )
    second_derivative_space = chebyshev_evaluate_2D(
        chebyshev_derivative_2D(coefs, 2, 2; derivation_matrix = derivation_matrix2),
        position,
    )

    return 2 * (
        first_derivative_space^2 - first_derivative_time^2 +
        function_at_position * (second_derivative_time - second_derivative_space)
    ) / function_at_position^4
end

"""
    Kretschmann_scalar_2D(
        coefs::Array{Float64, 2},
        position::CausalSets.Coordinates{2};
        derivation_matrix1::Union{Nothing, Array{Float64, 2}}=nothing,
        derivation_matrix2::Union{Nothing, Array{Float64, 2}}=nothing,
    ) -> Float64

Computes the Kretschmann scalar ``R_{abcd} R^{abcd}`` at a given position for a 2D Lorentzian manifold in conformally flat slicing with conformal factor expressed as Chebyshev expansion.

# Arguments
- `coefs::Array{Float64, 2}`: A matrix of Chebyshev coefficients representing the conformal factor expanded in two variables.
- `position::CausalSets.Coordinates{2}`: A 2D coordinate at which to evaluate the Kretschmann scalar.

# Keyword Arguments
- `derivation_matrix1::Union{Nothing, Array{Float64, 2}}=nothing`: Optional precomputed Chebyshev derivation matrix for first derivatives to speed up computation.
- `derivation_matrix2::Union{Nothing, Array{Float64, 2}}=nothing`: Optional precomputed Chebyshev derivation matrix for second derivatives to speed up computation.

# Returns
- `Float64`: The Kretschmann scalar evaluated at the given position.

# Throws
- ArgumentError: if the coefficient matrix is not a square matrix.

# Details
In 2D the Riemann tensor is fixed by the Ricci scalar, ``R_{abcd} = \\frac{R}{2} (g_{ac} g_{bd} - g_{ad} g_{bc})``, so that the Kretschmann scalar equals ``R^2``.

"""
function Kretschmann_scalar_2D(
    coefs::Array{Float64,2},
    position::CausalSets.Coordinates{2};
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Float64
    return Ricci_scalar_2D(
        coefs,
        position;
        derivation_matrix1 = derivation_matrix1,
        derivation_matrix2 = derivation_matrix2,
    )^2
end

"""
    Goroff_Sagnotti_scalar_2D(
        coefs::Array{Float64, 2},
        position::CausalSets.Coordinates{2};
        derivation_matrix1::Union{Nothing, Array{Float64, 2}}=nothing,
        derivation_matrix2::Union{Nothing, Array{Float64, 2}}=nothing,
    ) -> Float64

Computes the Goroff-Sagnotti invariant ``R_{ab}{}^{cd} R_{cd}{}^{ef} R_{ef}{}^{ab}`` at a given position for a 2D Lorentzian manifold in conformally flat slicing with conformal factor expressed as Chebyshev expansion.

# Arguments
- `coefs::Array{Float64, 2}`: A matrix of Chebyshev coefficients representing the conformal factor expanded in two variables.
- `position::CausalSets.Coordinates{2}`: A 2D coordinate at which to evaluate the Goroff-Sagnotti invariant.

# Keyword Arguments
- `derivation_matrix1::Union{Nothing, Array{Float64, 2}}=nothing`: Optional precomputed Chebyshev derivation matrix for first derivatives to speed up computation.
- `derivation_matrix2::Union{Nothing, Array{Float64, 2}}=nothing`: Optional precomputed Chebyshev derivation matrix for second derivatives to speed up computation.

# Returns
- `Float64`: The Goroff-Sagnotti invariant evaluated at the given position.

# Throws
- ArgumentError: if the coefficient matrix is not a square matrix.

# Details
In 2D the Riemann tensor is fixed by the Ricci scalar, ``R_{ab}{}^{cd} = \\frac{R}{2} (\\delta_a^c \\delta_b^d - \\delta_a^d \\delta_b^c)``, so that the Goroff-Sagnotti invariant equals ``R^3``.

"""
function Goroff_Sagnotti_scalar_2D(
    coefs::Array{Float64,2},
    position::CausalSets.Coordinates{2};
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Float64
    return Ricci_scalar_2D(
        coefs,
        position;
        derivation_matrix1 = derivation_matrix1,
        derivation_matrix2 = derivation_matrix2,
    )^3
end

"""
    Ricci_box_Ricci_2D(
        coefs::Array{Float64, 2},
        position::CausalSets.Coordinates{2};
        derivation_matrix1::Union{Nothing, Array{Float64, 2}}=nothing,
        derivation_matrix2::Union{Nothing, Array{Float64, 2}}=nothing,
    ) -> Float64

Computes ``R \\Box R``, with ``R`` the Ricci scalar, at a given position for a 2D Lorentzian manifold in conformally flat slicing with conformal factor expressed as Chebyshev expansion.

# Arguments
- `coefs::Array{Float64, 2}`: A matrix of Chebyshev coefficients representing the conformal factor expanded in two variables.
- `position::CausalSets.Coordinates{2}`: A 2D coordinate at which to evaluate ``R \\Box R``.

# Keyword Arguments
- `derivation_matrix1::Union{Nothing, Array{Float64, 2}}=nothing`: Optional precomputed Chebyshev derivation matrix for first derivatives to speed up computation.
- `derivation_matrix2::Union{Nothing, Array{Float64, 2}}=nothing`: Optional precomputed Chebyshev derivation matrix for second derivatives to speed up computation.

# Returns
- `Float64`: ``R \\Box R`` evaluated at the given position.

# Throws
- ArgumentError: if the coefficient matrix is not a square matrix.

# Details
For the metric ``g = f^2 \\mathrm{diag}(-1, 1)`` the box operator is ``\\Box = -f^{-2} L`` with the flat wave operator ``L = \\partial_t^2 - \\partial_x^2``. This requires derivatives of the conformal factor up to fourth order.

"""
function Ricci_box_Ricci_2D(
    coefs::Array{Float64,2},
    position::CausalSets.Coordinates{2};
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Float64
    first_matrix, second_matrix =
        curvature_derivation_matrices(coefs, derivation_matrix1, derivation_matrix2)

    d_t(c) = chebyshev_derivative_2D(c, 1, 1; derivation_matrix = first_matrix)
    d_x(c) = chebyshev_derivative_2D(c, 2, 1; derivation_matrix = first_matrix)
    d_tt(c) = chebyshev_derivative_2D(c, 1, 2; derivation_matrix = second_matrix)
    d_xx(c) = chebyshev_derivative_2D(c, 2, 2; derivation_matrix = second_matrix)
    evaluate(c) = chebyshev_evaluate_2D(c, position)

    # L maps polynomials to polynomials, so Lf can be differentiated further in Chebyshev form
    Lf_coefs = d_tt(coefs) - d_xx(coefs)

    f = evaluate(coefs)
    f_t = evaluate(d_t(coefs))
    f_x = evaluate(d_x(coefs))
    f_tt = evaluate(d_tt(coefs))
    f_xx = evaluate(d_xx(coefs))
    f_tx = evaluate(d_x(d_t(coefs)))
    Lf = f_tt - f_xx
    Lf_t = evaluate(d_t(Lf_coefs))
    Lf_x = evaluate(d_x(Lf_coefs))
    LLf = evaluate(d_tt(Lf_coefs) - d_xx(Lf_coefs))

    # Ricci scalar R = 2 N / f^4 and the derivatives of its numerator N
    N = f_x^2 - f_t^2 + f * Lf
    N_t = 2 * f_x * f_tx - 2 * f_t * f_tt + f_t * Lf + f * Lf_t
    N_x = 2 * f_x * f_xx - 2 * f_t * f_tx + f_x * Lf + f * Lf_x
    LN = 4 * f_tx^2 - (f_tt + f_xx)^2 + f * LLf

    # box R = -L R / f^2 with the product rule applied to L(N f^-4)
    box_R =
        -2 * (
            f^2 * LN - 8 * f * (N_t * f_t - N_x * f_x) - 4 * f * N * Lf +
            20 * N * (f_t^2 - f_x^2)
        ) / f^8

    return 2 * N / f^4 * box_R
end

"""
    Weyl_squared_2D(
        coefs::Array{Float64, 2},
        position::CausalSets.Coordinates{2};
        derivation_matrix1::Union{Nothing, Array{Float64, 2}}=nothing,
        derivation_matrix2::Union{Nothing, Array{Float64, 2}}=nothing,
    ) -> Float64

Returns the Weyl tensor squared ``C_{abcd} C^{abcd}`` for a 2D Lorentzian manifold in conformally flat slicing, which vanishes identically. The signature matches the other curvature invariants so it can be used interchangeably with them.

# Arguments
- `coefs::Array{Float64, 2}`: A matrix of Chebyshev coefficients representing the conformal factor expanded in two variables.
- `position::CausalSets.Coordinates{2}`: A 2D coordinate at which to evaluate the invariant.

# Keyword Arguments
- `derivation_matrix1::Union{Nothing, Array{Float64, 2}}=nothing`: Unused.
- `derivation_matrix2::Union{Nothing, Array{Float64, 2}}=nothing`: Unused.

# Returns
- `Float64`: Always 0.

# Details
The Weyl tensor is the trace-free part of the Riemann tensor. In 2D the Riemann tensor is pure trace, so this part vanishes. The Weyl tensor also vanishes for conformally flat metrics in any dimension.

"""
function Weyl_squared_2D(
    coefs::Array{Float64,2},
    position::CausalSets.Coordinates{2};
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Float64
    return 0.0
end

"""
    Weyl_box_Weyl_2D(
        coefs::Array{Float64, 2},
        position::CausalSets.Coordinates{2};
        derivation_matrix1::Union{Nothing, Array{Float64, 2}}=nothing,
        derivation_matrix2::Union{Nothing, Array{Float64, 2}}=nothing,
    ) -> Float64

Returns ``C_{abcd} \\Box C^{abcd}`` for a 2D Lorentzian manifold in conformally flat slicing, which vanishes identically because the Weyl tensor does (see [`Weyl_squared_2D`](@ref)). The signature matches the other curvature invariants so it can be used interchangeably with them.

# Arguments
- `coefs::Array{Float64, 2}`: A matrix of Chebyshev coefficients representing the conformal factor expanded in two variables.
- `position::CausalSets.Coordinates{2}`: A 2D coordinate at which to evaluate the invariant.

# Keyword Arguments
- `derivation_matrix1::Union{Nothing, Array{Float64, 2}}=nothing`: Unused.
- `derivation_matrix2::Union{Nothing, Array{Float64, 2}}=nothing`: Unused.

# Returns
- `Float64`: Always 0.

"""
function Weyl_box_Weyl_2D(
    coefs::Array{Float64,2},
    position::CausalSets.Coordinates{2};
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Float64
    return 0.0
end

"""
    curvature_invariant_of_sprinkling(
        invariant::Function,
        coefs::Array{Float64, 2},
        sprinkling::Vector{CausalSets.Coordinates{2}};
        derivation_matrix1::Union{Nothing, Array{Float64, 2}}=nothing,
        derivation_matrix2::Union{Nothing, Array{Float64, 2}}=nothing,
    ) -> Vector{Float64}

Computes a curvature invariant at multiple points (sprinkling) for a 2D Lorentzian manifold in conformally flat slicing with conformal factor expressed as Chebyshev expansion.

# Arguments
- `invariant::Function`: The invariant to evaluate, called as `invariant(coefs, point; derivation_matrix1, derivation_matrix2)`, e.g. [`Kretschmann_scalar_2D`](@ref).
- `coefs::Array{Float64, 2}`: A matrix of Chebyshev coefficients representing the conformal factor expanded in two variables.
- `sprinkling::Vector{CausalSets.Coordinates{2}}`: A vector of 2D coordinates at which to evaluate the invariant.

# Keyword Arguments
- `derivation_matrix1::Union{Nothing, Array{Float64, 2}}=nothing`: Optional precomputed Chebyshev derivation matrix for first derivatives.
- `derivation_matrix2::Union{Nothing, Array{Float64, 2}}=nothing`: Optional precomputed Chebyshev derivation matrix for second derivatives.

# Returns
- `Vector{Float64}`: A vector of invariant values evaluated at the given points.

# Throws
- ArgumentError: if the coefficient matrix is not a square matrix.

"""
function curvature_invariant_of_sprinkling(
    invariant::Function,
    coefs::Array{Float64,2},
    sprinkling::Vector{CausalSets.Coordinates{2}};
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Vector{Float64}
    derivation_matrix1, derivation_matrix2 =
        curvature_derivation_matrices(coefs, derivation_matrix1, derivation_matrix2)
    return [
        invariant(
            coefs,
            point;
            derivation_matrix1 = derivation_matrix1,
            derivation_matrix2 = derivation_matrix2,
        ) for point in sprinkling
    ]
end

"""
    Ricci_scalar_2D_of_sprinkling(
        coefs::Array{Float64, 2},
        sprinkling::Vector{CausalSets.Coordinates{2}};
        derivation_matrix1::Union{Nothing, Array{Float64, 2}}=nothing,
        derivation_matrix2::Union{Nothing, Array{Float64, 2}}=nothing,
    ) -> Vector{Float64}

Computes the Ricci scalar curvature at multiple points (sprinkling) for a 2D Lorentzian manifold in conformally flat slicing with conformal factor expressed as Chebyshev expansion.

# Arguments
- `coefs::Array{Float64, 2}`: A matrix of Chebyshev coefficients representing the conformal factor expanded in two variables.
- `sprinkling::Vector{CausalSets.Coordinates{2}}`: A vector of 2D coordinates at which to evaluate the Ricci scalar.

# Keyword Arguments
- `derivation_matrix1::Union{Nothing, Array{Float64, 2}}=nothing`: Optional precomputed Chebyshev derivation matrix for first derivatives.
- `derivation_matrix2::Union{Nothing, Array{Float64, 2}}=nothing`: Optional precomputed Chebyshev derivation matrix for second derivatives.

# Returns
- `Vector{Float64}`: A vector of Ricci scalar curvature values evaluated at the given points.

# Throws
- ArgumentError: if the coefficient matrix is not a square matrix.

"""
function Ricci_scalar_2D_of_sprinkling(
    coefs::Array{Float64,2},
    sprinkling::Vector{CausalSets.Coordinates{2}};
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Vector{Float64}
    return curvature_invariant_of_sprinkling(
        Ricci_scalar_2D,
        coefs,
        sprinkling;
        derivation_matrix1 = derivation_matrix1,
        derivation_matrix2 = derivation_matrix2,
    )
end
