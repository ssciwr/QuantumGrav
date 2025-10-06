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
                                    derivative_order::Int64
)::Array{Float64, 2}

    if max_cheb_order < 0
        throw(ArgumentError("max_cheb_order has to be at least 0, is $(max_cheb_order)."))
    end

    if derivative_order < 1
        throw(ArgumentError("max_cheb_order has to be at least 1, is $(derivative_order)."))
    end

    transformation_matrix = zeros(max_cheb_order + 1, max_cheb_order + 1)

    for n in 0 : max_cheb_order

        for k in mod(n-derivative_order, 2) : 2 : n - derivative_order
            prefac = k == 0 ? 0.5 : 1

            transformation_matrix[k+1, n+1] += 2^derivative_order * n * prefac * 
                                                binomial(Int64((n + derivative_order - k) / 2 - 1), Int64((n - derivative_order - k) / 2)) * 
                                                factorial(Int64((n + derivative_order + k) / 2 - 1)) / 
                                                factorial(Int64((n - derivative_order + k) / 2))
        end
    end

    return transformation_matrix
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
                                                coefs::Array{Float64, 2}, 
                                                derivative_variable_index::Int64, 
                                                derivative_order::Int64; 
                                                derivation_matrix::Union{Nothing, Array{Float64, 2}}=nothing
                                                )::Array{Float64, 2}
    
    if derivative_variable_index != 1 && derivative_variable_index != 2
        throw(ArgumentError("derivative_variable_index has to be in (1,2) is $(derivative_variable_index)"))
    end      

    if derivative_order < 1
        throw(ArgumentError("max_cheb_order has to be at least 1, is $(derivative_order)."))
    end            
      
    order = size(coefs,derivative_variable_index)-1
    derivation_matrix = isnothing(derivation_matrix) ? chebyshev_derivation_matrix(order, derivative_order) : derivation_matrix
    
    return derivative_variable_index == 1 ? derivation_matrix * coefs : coefs * derivation_matrix'
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
function chebyshev_evaluate_2D(coefs::Array{Float64, 2}, position::CausalSets.Coordinates{2})::Float64
    chebyshev_T(n::Int64, x::Float64) = cos(n * acos(x))
    val = 0.0
    for i in 1:size(coefs, 1)
        for j in 1:size(coefs, 2)
            val += coefs[i,j] * chebyshev_T(i-1, position[1]) * chebyshev_T(j-1, position[2])
        end
    end
    return val
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
                coefs::Array{Float64, 2}, 
                position::CausalSets.Coordinates{2};
                derivation_matrix1::Union{Nothing, Array{Float64, 2}}=nothing,
                derivation_matrix2::Union{Nothing, Array{Float64, 2}}=nothing,
                )::Float64
    if size(coefs,1) != size(coefs,2)
        throw(ArgumentError("At the moment only square coefficient matrices are supported, i. e., Chebyshev expansions at equal order for both variables. Dimensions are $(size(coefs,1)) and $(size(coefs,2))"))
    end

    derivation_matrix1 = isnothing(derivation_matrix1) ? chebyshev_derivation_matrix(size(coefs,1)-1, 1) : derivation_matrix1
    derivation_matrix2 = isnothing(derivation_matrix2) ? chebyshev_derivation_matrix(size(coefs,1)-1, 2) : derivation_matrix2
    function_at_position    = chebyshev_evaluate_2D(coefs, position)
    first_derivative_time  = chebyshev_evaluate_2D(chebyshev_derivative_2D(coefs, 1, 1; derivation_matrix = derivation_matrix1), position)
    first_derivative_space   = chebyshev_evaluate_2D(chebyshev_derivative_2D(coefs, 2, 1; derivation_matrix = derivation_matrix1), position)
    second_derivative_time = chebyshev_evaluate_2D(chebyshev_derivative_2D(coefs, 1, 2; derivation_matrix = derivation_matrix2), position)
    second_derivative_space  = chebyshev_evaluate_2D(chebyshev_derivative_2D(coefs, 2, 2; derivation_matrix = derivation_matrix2), position)
    
    return 2 * (first_derivative_space^2 - first_derivative_time^2 + function_at_position * (second_derivative_time - second_derivative_space)) / function_at_position^4
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
                                        coefs::Array{Float64, 2},
                                        sprinkling::Vector{CausalSets.Coordinates{2}};
                                        derivation_matrix1::Union{Nothing, Array{Float64, 2}}=nothing,
                                        derivation_matrix2::Union{Nothing, Array{Float64, 2}}=nothing,
                                        )::Vector{Float64}
    order = size(coefs,1)
    if order != size(coefs,2)
        throw(ArgumentError("At the moment only square coefficient matrices are supported, i. e., Chebyshev expansions at equal order for both variables. Dimensions are $(order) and $(size(coefs,2))"))
    end
    derivation_matrix1 = isnothing(derivation_matrix1) ? chebyshev_derivation_matrix(size(coefs,1)-1, 1) : derivation_matrix1
    derivation_matrix2 = isnothing(derivation_matrix2) ? chebyshev_derivation_matrix(size(coefs,1)-1, 2) : derivation_matrix2
    return [Ricci_scalar_2D(coefs, point; derivation_matrix1=derivation_matrix1, derivation_matrix2=derivation_matrix2) for point in sprinkling]
end 