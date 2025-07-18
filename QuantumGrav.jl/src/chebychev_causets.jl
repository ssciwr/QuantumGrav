function generate_random_chebyshev_causet(npoints::Int, seed::Int, order::Int, r::Float64)
    """
    generate_random_chebyshev_causet(npoints::Int, seed::Int, order::Int, r::Float64)

    Generate a causal set by sampling from a positive polynomial constructed via a truncated 
    Chebyshev series with exponentially decaying coefficients.

    # Arguments
    - `npoints::Int`: Number of elements to sprinkle into the causal set. Must be > 0.
    - `seed::Int`: Seed for pseudo-random number generation to ensure reproducibility.
    - `order::Int`: Truncation order of the Chebyshev expansion (number of basis functions in each direction). Must be > 0.
    - `r::Float64`: Decay base for Chebyshev coefficients. Must be > 1 to ensure exponential convergence; defines the radius of analyticity in the complex plane.

    # Returns
    - A tuple `(cset, sprinkling, chebyshev_coefs)` where:
      - `cset`: The generated causal set.
      - `sprinkling`: The list of sprinkled points.
      - `chebyshev_coefs`: The matrix of Chebyshev coefficients used to construct the manifold.

    # Throws
    - `ArgumentError` if `npoints <= 0`, `order <= 0`, or `r <= 1`.
    """
    
    if npoints <= 0
        throw(ArgumentError("npoints must be greater than 0, got $npoints"))
    end
    if order <= 0
        throw(ArgumentError("order must be greater than 0, got $order"))
    end
    if r <= 1
        throw(ArgumentError("r must be greater than 1 for exponential convergence of the Chebyshev series, got $r"))
    end

    # Set the random seed for reproducibility
    Random.seed!(seed)

    # Generate a matrix of random Chebyshev coefficients that decay exponentially with base r
    # it has to be a (order x order)-matrix because we describe a function of two variables
    chebyshev_coefs = zeros(order, order)
    for i in 1:order
        for j in 1:order
            chebyshev_coefs[i, j] = r^(-i - j) * randn()
        end
    end

    # Construct the Chebyshev-to-Taylor transformation matrix
    cheb_to_taylor_mat = chebyshev_coef_matrix(order - 1)

    # Transform Chebyshev coefficients to Taylor coefficients
    taylorcoefs = transform_polynomial(chebyshev_coefs, cheb_to_taylor_mat)

    # Square the polynomial to ensure positivity
    squaretaylorcoefs = polynomial_pow(taylorcoefs, 2)

    # Create a polynomial manifold from the squared Taylor coefficients
    polym = PolynomialManifold{2}(squaretaylorcoefs)

    # Define the square box boundary in 2D -- this works only in 2D and with square box boundary at the moment
    boundary = BoxBoundary{2}(((-1., -1.), (1., 1.)))

    # Generate a sprinkling of npoints in the manifold within the boundary
    sprinkling = generate_sprinkling(polym, boundary, npoints)

    # Construct the causal set from the manifold and sprinkling
    cset = BitArrayCauset(polym, sprinkling)

    return cset, sprinkling, chebyshev_coefs
end