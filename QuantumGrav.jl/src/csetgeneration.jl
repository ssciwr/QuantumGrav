"""
    make_simple_cset(manifold, boundary, n, d, rng, type) -> (cset, coordinates)

Creates a causet and its coordinate representation from a manifold and boundary. This function only creates manifold-like csets for simple manifolds [HyperCylinder, deSitter, antiDeSitter, Minkowski, Torus] or non-manifold-like csets for PseudoManifold.

# Arguments
- `manifold::String`: The spacetime manifold
- `boundary::String`: The boundary type  
- `n::Int64`: Number of points in the causet
- `d::Int`: Dimension of the spacetime
- `rng::Random.AbstractRNG`: Random number generator
- `type::Type{T}`: Numeric type for coordinates
- rel_tol::Union{Float64,Nothing}`: Relative tolerance for MCMC breaking in random cset generation
- abs_tol::Union{Float64,Nothing}`: Absolute tolerance for MCMC breaking in random cset generation,
- acceptance::Float64`: Acceptance condition for MCMC

# Returns
- `Tuple`: (causet, converged, coordinates) where coordinates is a matrix of point positions, converged is a bool

# Notes
Special handling for PseudoManifold which generates random causets,
while other manifolds use CausalSets sprinkling generation.
"""
function make_simple_cset(
    manifold::String,
    boundary::String,
    n::Int64,
    d::Int,
    dist::Distributions.Distribution,
    markov_iter::Int,
    rng::Random.AbstractRNG;
    type::Type{T} = Float32,
    rel_tol::Union{Float64,Nothing}=nothing,
    abs_tol::Union{Float64,Nothing}=nothing,
    acceptance::Float64=5e5
) where {T<:Number}
    manifold = make_manifold(manifold, d)
    boundary = make_boundary(boundary, d)

    if manifold isa PseudoManifold
        # create a pseudosprinkling in a n-d euclidean space for a non-manifold like causalset
        return random_causet_by_connectivity_distribution(
            n,
            dist,
            markov_iter,
            rng;
            rel_tol = rel_tol,
            abs_tol = abs_tol,
            acceptance = acceptance,
        )...,
        stack(make_pseudosprinkling(n, d, -1.0, 1.0, type; rng = rng), dims = 1)
    else
        sprinkling = CausalSets.generate_sprinkling(manifold, boundary, n; rng = rng)
        cset = CausalSets.BitArrayCauset(manifold, sprinkling)
        return cset, true, type.(stack(collect.(sprinkling), dims = 1))
    end
end

"""
    make_manifold_cset(npoints::Int64, rng::Random.AbstractRNG, order::Int64, r::Float64, d::Int64=2, type::Type{T}=Float32)::Tuple{CausalSets.BitArrayCauset,Vector{Tuple{T, Vargarg{T}}},Matrix{T}} where {T<:Number}

Generate a causal set by sampling from a positive polynomial constructed via a truncated 
Chebyshev series with exponentially decaying coefficients.

# Arguments
- `npoints::Int`: Number of elements to sprinkle into the causal set. Must be > 0.
- `seed::Int`: Seed for pseudo-random number generation to ensure reproducibility.
- `order::Int`: Truncation order of the Chebyshev expansion (number of basis functions in each direction). Must be > 0.
- `r::Float64`: Decay base for Chebyshev coefficients. Must be > 1 to ensure exponential convergence; defines the radius of analyticity in the complex plane.
- `d::Int64`: Dimension of the manifold, defaults to 2. Currently, only 2D is supported.
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
- `ArgumentError` if `d != 2`. Currently, only 2D is supported.
"""
function make_manifold_cset(
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

    if order <= 0
        throw(ArgumentError("order must be greater than 0, got $order"))
    end

    if r <= 1
        throw(
            ArgumentError(
                "r must be greater than 1 for exponential convergence of the Chebyshev series, got $r",
            ),
        )
    end

    if d != 2
        throw(ArgumentError("Currently, only 2D is supported, got $d"))
    end

    # Generate a matrix of random Chebyshev coefficients that decay exponentially with base r
    # it has to be a (order x order)-matrix because we describe a function of two variables
    chebyshev_coefs = zeros(Float64, order, order)
    for i = 1:order
        for j = 1:order
            chebyshev_coefs[i, j] = r^(-i - j) * Random.randn(rng)
        end
    end

    # Construct the Chebyshev-to-Taylor transformation matrix
    cheb_to_taylor_mat = CausalSets.chebyshev_coef_matrix(order - 1)

    # Transform Chebyshev coefficients to Taylor coefficients
    taylorcoefs = CausalSets.transform_polynomial(chebyshev_coefs, cheb_to_taylor_mat)

    # Square the polynomial to ensure positivity
    squaretaylorcoefs = CausalSets.polynomial_pow(taylorcoefs, 2)

    # Create a polynomial manifold from the squared Taylor coefficients
    polym = CausalSets.PolynomialManifold{d}(squaretaylorcoefs)

    # Define the square box boundary in 2D -- this works only in 2D and with square box boundary at the moment
    boundary = CausalSets.BoxBoundary{d}(((-1.0, -1.0), (1.0, 1.0)))

    # Generate a sprinkling of npoints in the manifold within the boundary
    sprinkling = CausalSets.generate_sprinkling(polym, boundary, npoints)

    # Construct the causal set from the manifold and sprinkling
    cset = CausalSets.BitArrayCauset(polym, sprinkling)

    return cset, sprinkling, type.(chebyshev_coefs)
end

"""
    make_general_cset(rng::Random.AbstractRNG, npoints::Int64, order::Int64, r::Float64, d::Int64, type::Type{T})

Generates a causal set based on a random choice between a manifold-like distribution or a pseudo-manifold distribution. This currently only works in 2D.

# Arguments:
- `rng`: Random number generator for reproducibility.
- `npoints`: Number of points in the causal set.
- `order`: Order of the Chebyshev expansion for the manifold like csets.
- `r`: Decay base for Chebyshev coefficients.
- `d`: Dimension of the manifold (must be 2).
- `dist`: Distribution from which to draw the connectivity goal for pseudo-manifold causal sets.
- `markov_iter`: Number of iterations for the Markov chain when generating a random causet.
- `type`: Type to which the sprinkling coordinates will be converted.

# Returns:
- A tuple `(cset, sprinkling, chebyshev_coefs, manifold_like)` where:
  - `cset`: The generated causal set.
  - `sprinkling`: The list of sprinkled points.
  - `chebyshev_coefs`: The matrix of Chebyshev coefficients used to construct the manifold (or zeros for pseudo-manifold).
  - `manifold_like`: Indicator whether the generated causet came from a manifold-like distribution (1) or pseudo-manifold (0).

# Throws: 
   - `ArgumentError` if `d != 2`
"""
function make_general_cset(
    rng::Random.AbstractRNG,
    npoints::Int64,
    order::Int64,
    r::Float64,
    d::Int64,
    dist::Distributions.Distribution,
    markov_iter::Int,
    type::Type{T};
    rel_tol::Union{Float64,Nothing}=nothing,
    abs_tol::Union{Float64,Nothing}=nothing,
    acceptance::Float64=5e5,
)::Tuple{CausalSets.BitArrayCauset,Matrix{T},Matrix{T},Int64} where {T<:Number}

    if d != 2
        throw(ArgumentError("Currently, only 2D is supported, got $d"))
    end

    manifold_like_distr = Distributions.DiscreteUniform(0, 1)

    manifold_like = rand(rng, manifold_like_distr)

    if manifold_like == 1
        # create a simple causet with a Minkowski manifold and a causal diamond boundary
        cset, sprinkling, chebyshev_coefs =
            make_manifold_cset(npoints, rng, order, r; d = d, type = type)

    else
        cset = random_causet_by_connectivity_distribution(npoints,
                                                dist,
                                                markov_iter,
                                                rng;
                                                rel_tol = rel_tol,
                                                abs_tol = abs_tol,
                                                acceptance = acceptance,)[1]
        sprinkling = make_pseudosprinkling(npoints, d, -1.0, 1.0, type; rng = rng)
        chebyshev_coefs = zeros(order, order) # pseudo-coefficients, not used in this case
    end

    sprinkling = type.(stack(collect.(sprinkling), dims = 1))

    return cset, sprinkling, chebyshev_coefs, manifold_like
end