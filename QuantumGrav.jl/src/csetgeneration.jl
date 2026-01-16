"""
    make_polynomial_manifold_cset(npoints::Int64, rng::Random.AbstractRNG, order::Int64, r::Float64, d::Int64=2, type::Type{T}=Float32)::Tuple{CausalSets.BitArrayCauset,Vector{Tuple{T, Vargarg{T}}},Matrix{T}} where {T<:Number}

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
    - `chebyshev_coefs`: The matrix of Chebyshev coefficients used to construct the manifold.

# Throws
- `ArgumentError` if `npoints <= 0`
- `ArgumentError` if `order <= 0`
- `ArgumentError` if `r <= 1`
"""
function make_polynomial_manifold_cset(
    npoints::Int64,
    rng::Random.AbstractRNG,
    order::Int64,
    r::Float64;
    d::Int64 = 2,
    type::Type{T} = Float32,
)::Tuple{CausalSets.BitArrayCauset,Vector{Tuple{T,Vararg{T}}},Matrix{T}} where {T<:Number}

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

    # Generate a matrix of random Chebyshev coefficients that decay exponentially with base r
    # it has to be a (order x order)-matrix because we describe a function of two variables
    chebyshev_coefs = zeros(Float64, ntuple(_ -> order+1, d))
    for I in CartesianIndices(chebyshev_coefs)
        chebyshev_coefs[I] = r^(-sum(Tuple(I))) * randn(rng)
    end


    # Construct the Chebyshev-to-Taylor transformation matrix
    cheb_to_taylor_mat = CausalSets.chebyshev_coef_matrix(order)

    # Transform Chebyshev coefficients to Taylor coefficients
    taylorcoefs = CausalSets.transform_polynomial(chebyshev_coefs, cheb_to_taylor_mat)

    # Square the polynomial to ensure positivity
    squaretaylorcoefs = CausalSets.polynomial_pow(taylorcoefs, 2)

    # Create a polynomial manifold from the squared Taylor coefficients
    polym = CausalSets.PolynomialManifold{d}(squaretaylorcoefs)

    # Define the square box boundary in 2D -- this works only in 2D and with square box boundary at the moment
    boundary = CausalSets.BoxBoundary{d}((ntuple(_ -> -1.0, d), ntuple(_ -> 1.0, d)))

    # Generate a sprinkling of npoints in the manifold within the boundary
    sprinkling = CausalSets.generate_sprinkling(polym, boundary, npoints; rng = rng)

    # Construct the causal set from the manifold and sprinkling
    cset = CausalSets.BitArrayCauset(polym, sprinkling)

    return cset, sprinkling, type.(chebyshev_coefs)
end
