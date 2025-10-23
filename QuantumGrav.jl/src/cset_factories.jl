using CausalSets: CausalSets
using Random: Random
using Distributions: Distributions
import QuantumGrav as QG


function build_distr(cfg::Dict{String,Any}, name::String)::Distributions.Distribution

    distribution_type::Union{Nothing,Type} = nothing

    distr::Union{Nothing,Distributions.Distribution} = nothing

    try
        distribution_type = getfield(Distributions, Symbol(cfg[name]))
    catch e
        throw(ArgumentError("Distribution $(name) could not be retrieved $(e)"))
    end

    try
        distr = distribution_type(cfg[name*"_args"]...; cfg[name*"_kwargs"]...)
    catch e
        throw(ArgumentError("Distribution $(name) could not be build $(e)"))
    end

    return distr
end

"""
	PolynomialCsetMaker

	Causal set maker for a polynomial manifold.

# Fields:
- `order_distribution::Distributions.Distribution`: distribution of polynomial orders
- `r_distribution::Distributions.Distribution`: distribution of exponential decay exponents
"""
struct PolynomialCsetMaker
    order_distribution::Distributions.Distribution
    r_distribution::Distributions.Distribution
end

"""
	PolynomialCsetMaker(config)

	Creates a causal set maker for a polynomial manifold.

# Fields:
- config::Dict: configuration dictionary
"""
function PolynomialCsetMaker(config)
    order_distribution = build_distr(config, "order_distribution")
    r_distribution = build_distr(config, "r_distribution")

    return PolynomialCsetMaker(order_distribution, r_distribution)
end

"""
	m::PolynomialCsetMaker(n, config, rng)

	Creates a new polynomial causal set with the parameters stored in the calling `PolynomialCsetMaker` object m.

# Arguments:
- `n`: number of elements in the causal set
- `config`: configuration dictionary
- `rng`: random number generator
"""
function (m::PolynomialCsetMaker)(
    n,
    rng;
    config::Union{Dict,Nothing} = nothing,
)::CausalSets.BitArrayCauset
    o = rand(rng, m.order_distribution)
    r = rand(rng, m.r_distribution)
    cset, _, __ = QG.make_polynomial_manifold_cset(n, rng, o, r; d = 2, type = Float32)
    return cset
end

"""
	LayeredCsetMaker

	Causal set maker for a layered causal set.

# Fields:
- `connectivity_distribution::Distributions.Distribution`: distribution of connectivity goals
- `stddev_distribution::Distributions.Distribution`: distribution of standard deviations
- `layer_distribution::Distributions.Distribution`: distribution of layer counts
"""
struct LayeredCsetMaker
    connectivity_distribution::Distributions.Distribution
    stddev_distribution::Distributions.Distribution
    layer_distribution::Distributions.Distribution
end

"""
	LayeredCsetMaker(config::Dict)

	Creates a causal set maker for a layered causal set.

# Arguments:
	- config::Dict: configuration dictionary
"""
function LayeredCsetMaker(config::Dict)
    cdistr = build_distr(config, "connectivity_distribution")
    stddev_distr = build_distr(config, "stddev_distribution")
    ldistr = build_distr(config, "layer_distribution")
    return LayeredCsetMaker(cdistr, stddev_distr, ldistr)
end

"""
	lm::LayeredCsetMaker(n::Int64, config::Dict, rng::Random.AbstractRNG)

	Creates a new layered causal set with the parameters stored in the calling `LayeredCsetMaker` object lm.

# Arguments:
- `n`: number of elements in the causal set
- `config`: configuration dictionary
- `rng`: random number generator
"""
function (lm::LayeredCsetMaker)(
    n::Int64,
    rng::Random.AbstractRNG;
    config::Union{Dict,Nothing} = nothing,
)::CausalSets.BitArrayCauset

    connectivity_goal = rand(rng, lm.connectivity_distribution)
    while connectivity_goal < 1e-5
        connectivity_goal = rand(rng, lm.connectivity_distribution)
    end

    layers = rand(rng, lm.layer_distribution)
    while layers < 1e-5
        layers = rand(rng, lm.layer_distribution)
    end
    layers = Int(ceil(layers))

    s = rand(rng, lm.stddev_distribution)

    cset, _ = QG.create_random_layered_causet(
        n,
        layers;
        p = connectivity_goal,
        rng = rng,
        standard_deviation = s,
    )

    return cset
end

"""
	RandomCsetMaker

	Causal set maker for a random causal set.

# Fields:
- `cdistr::Distributions.Distribution`: distribution of connectivity goals
"""
struct RandomCsetMaker
    connectivity_distribution::Distributions.Distribution
    num_tries::Int64
    abs_tol::Float64
    rel_tol::Float64
end

"""
	RandomCsetMaker(config::Dict)

	Creates a causal set maker for a random causal set.

# Fields:
- config::Dict: configuration dictionary
"""
function RandomCsetMaker(config::Dict)
    cdistr = build_distr(config, "connectivity_distribution")

    if config["num_tries"] < 1
        throw(ArgumentError("Error, num_tries must be >= 1"))
    end

    return RandomCsetMaker(
        cdistr,
        config["num_tries"],
        config["abs_tol"],
        config["rel_tol"],
    )
end

"""
	rcm::RandomCsetMaker(n::Int64, rng::Random.AbstractRNG; config::Union{Dict, Nothing} = nothing)

	Creates a new random causal set with the parameters stored in the calling `RandomCsetMaker` object rcm.

# Arguments:
- `n`: number of elements in the causal set
- `rng`: random number generator

# Keyword arguments:
- `config`: configuration dictionary
"""
function (rcm::RandomCsetMaker)(
    n::Int64,
    rng::Random.AbstractRNG;
    config::Union{Dict,Nothing} = nothing,
)::CausalSets.BitArrayCauset

    connectivity_goal = rand(rng, rcm.connectivity_distribution)

    abs_tol = 1e-3

    converged = false

    cset = nothing

    tries = 0

    while converged == false
        cset_try, converged = QG.sample_bitarray_causet_by_connectivity(
            n,
            connectivity_goal,
            100,
            rng;
            abs_tol = abs_tol,
        )
        tries += 1

        if tries > rcm.num_tries
            cset = nothing
            break
        end
        cset = cset_try
    end

    if cset === nothing
        throw(
            ErrorException(
                "Failed to generate causet with n=$n and connectivity_goal=$connectivity_goal after $tries tries.",
            ),
        )
    end

    return cset
end

"""
	DestroyedCsetMaker

	Causal set maker for a destroyed causal set, which has a set of edges flipped in a polynomial causal set.

# Fields:
- `order_distribution::Distributions.Distribution`: distribution of order values
- `r_distribution::Distributions.Distribution`: distribution of r values
- `flip_distribution::Distributions.Distribution`: distribution of flip values
"""
struct DestroyedCsetMaker
    order_distribution::Distributions.Distribution
    r_distribution::Distributions.Distribution
    flip_distribution::Distributions.Distribution
end

"""
	DestroyedCsetMaker(config::Dict)

Create a new `destroyed` causal set maker object from the config dictionary.
"""
function DestroyedCsetMaker(config::Dict)

    order_distribution = build_distr(config, "order_distribution")

    r_distribution = build_distr(config, "r_distribution")

    flip_distribution = build_distr(config, "flip_distribution")

    return DestroyedCsetMaker(order_distribution, r_distribution, flip_distribution)
end

"""
	dcm::DestroyedCsetMaker(n::Int64, config::Dict, rng::Random.AbstractRNG)

Create a new `destroyed` causal set using a `DestroyedCsetMaker` object.

# Arguments:
- `n`: number of elements in the causal set
- `config`: configuration dictionary
- `rng`: random number generator
"""
function (dcm::DestroyedCsetMaker)(
    n::Int64,
    rng::Random.AbstractRNG;
    config::Union{Dict{String,Any},Nothing} = nothing,
)::CausalSets.BitArrayCauset

    o = rand(rng, dcm.order_distribution)

    r = rand(rng, dcm.r_distribution)

    f = convert(Int64, ceil(rand(rng, dcm.flip_distribution) * n))

    cset = QG.destroy_manifold_cset(n, f, rng, o, r; d = 2, type = Float32)[1]
    return cset
end


"""
	GridCsetMakerPolynomial

	Create a new `grid` causal set maker object from the config dictionary for polynomial spacetimes.

# Fields:
- `base::GridCsetMakerConstCurv`: base grid causal set maker
- `order_distribution::Distributions.Distribution`: distribution of polynomial order values
- `r_distribution::Distributions.Distribution`: distribution of radial values
"""
struct GridCsetMakerPolynomial
    grid_distribution::Distributions.Distribution
    rotate_distribution::Distributions.Distribution
    gamma_distribution::Distributions.Distribution
    order_distribution::Distributions.Distribution
    r_distribution::Distributions.Distribution
    grid_lookup::Dict
end

"""
	GridCsetMakerPolynomial(config)

	Create a new `grid` causal set maker object from the config dictionary for polynomial spacetimes.
"""
function GridCsetMakerPolynomial(config::Dict)

    grid_distribution = build_distr(config, "grid_distribution")

    rotate_distribution = build_distr(config, "rotate_distribution")

    gamma_distribution = build_distr(config, "gamma_distribution")

    order_distribution = build_distr(config, "order_distribution")

    r_distribution = build_distr(config, "r_distribution")

    grid_lookup = Dict(
        1 => "quadratic",
        2 => "rectangular",
        3 => "rhombic",
        4 => "hexagonal",
        5 => "triangular",
        6 => "oblique",
    )

    return GridCsetMakerPolynomial(
        grid_distribution,
        rotate_distribution,
        gamma_distribution,
        order_distribution,
        r_distribution,
        grid_lookup,
    )
end

"""
	gcm::GridCsetMakerPolynomial(n::Int64, config::Dict, rng::Random.AbstractRNG)

	Create a new `grid` causal set using a `GridCsetMakerPolynomial` object.

# Arguments:
- `n`: number of elements in the causal set
- `config`: configuration dictionary
- `rng`: random number generator
"""
function (gcm::GridCsetMakerPolynomial)(
    n::Int64,
    config::Dict,
    rng::Random.AbstractRNG;
    grid::Union{String,Nothing} = nothing,
)
    a_dist = build_distr(config[grid], "a_distribution")
    b_dist = build_distr(config[grid], "b_distribution")

    o = rand(rng, gcm.order_distribution)
    r = rand(rng, gcm.r_distribution)
    rotate_angle_deg = rand(rng, gcm.rotate_distribution)
    gamma_deg = rand(rng, gcm.gamma_distribution)

    if grid === nothing
        grid = gcm.grid_lookup[rand(rng, gcm.grid_distribution)]
    end

    a = rand(rng, a_dist)
    b = rand(rng, b_dist)
    cset, _, __ = QG.create_grid_causet_2D_polynomial_manifold(
        n,
        grid,
        rng,
        o,
        r;
        type = Float32,
        a = a,
        b = b,
        gamma_deg = gamma_deg,
        rotate_deg = rotate_angle_deg,
        origin = (0.0, 0.0),
    )

    return cset
end


"""
	ComplexTopCsetMaker

A callable struct to produce complex topology csets with various causality-cutting 'lines' in a 2D manifold

# Fields:
- `vertical_cut_distr::Distributions.Distribution`: Distribution to draw the number of vertical (time direction) cuts from
- `finite_cut_distr::Distributions.Distribution`: Distrbituion to draw the number of mixed direction cuts from
- `order_distribution::Distributions.Distribution`: Distribution to draw the number of orders for the polynomial expansion from
- `r_distribution::Distributions.Distribution`: Distribution to draw the decay exponent for the orders in the polynomial expansion from
- `tol::Float64`: Floating point comparison tolerance
"""
struct ComplexTopCsetMaker
    vertical_cut_distribution::Distributions.Distribution
    finite_cut_distribution::Distributions.Distribution
    order_distribution::Distributions.Distribution
    r_distribution::Distributions.Distribution
    tol::Float64
end

"""
	ComplexTopCsetMaker(config::Dict)

	Create a new `ComplexTopCsetMaker` object from the config dictionary.
"""
function ComplexTopCsetMaker(config::Dict)
    vertical_cut_distr = build_distr(config, "vertical_cut_distribution")
    finite_cut_distr = build_distr(config, "finite_cut_distribution")
    order_distr = build_distr(config, "order_distribution")
    r_distr = build_distr(config, "r_distribution")
    tol = config["tol"]

    return ComplexTopCsetMaker(
        vertical_cut_distr,
        finite_cut_distr,
        order_distr,
        r_distr,
        tol,
    )
end

"""
	ctm::ComplexTopCsetMaker(n::Int64, config::Dict, rng::Random.AbstractRNG)

	Create a new causal set using a `ComplexTopCsetMaker` object.

# Arguments:
- `n`: number of elements in the causal set
- `config`: configuration dictionary
- `rng`: random number generator
"""
function (ctm::ComplexTopCsetMaker)(
    n::Int64,
    rng::Random.AbstractRNG;
    config::Union{Dict{String,Any},Nothing} = nothing,
)::CausalSets.BitArrayCauset

    n_vertical_cuts = rand(rng, ctm.vertical_cut_distribution)
    if n_vertical_cuts isa Float64
        n_vertical_cuts = convert(Int64, round(n_vertical_cuts))
    end

    n_finite_cuts = rand(rng, ctm.finite_cut_distribution)
    if n_finite_cuts isa Float64
        n_finite_cuts = convert(Int64, round(n_finite_cuts))
    end

    order = rand(rng, ctm.order_distribution)
    r = rand(rng, ctm.r_distribution)

    cset, branched_sprinkling, branch_point_info, chebyshev_coefs =
        QG.make_branched_manifold_cset(
            n,
            n_vertical_cuts,
            n_finite_cuts,
            rng,
            order,
            r;
            d = 2,
            tolerance = ctm.tol,
        )

    return cset
end


"""
	MergedCsetMaker

	Causal set maker from a given configuration dictionary.

# Fields:
- `link_prob_distribution::Distributions.Distribution`: distribution of link probabilities
- `order_distribution::Distributions.Distribution`: distribution of order values
- `r_distribution::Distributions.Distribution`: distribution of r values
- `n2_rel_distribution::Distributions.Distribution`: distribution of n2 relative values
- `connectivity_distribution::Distributions.Distribution`: distribution of connectivity values
"""
struct MergedCsetMaker
    link_prob_distribution::Distributions.Distribution
    order_distribution::Distributions.Distribution
    r_distribution::Distributions.Distribution
    n2_rel_distribution::Distributions.Distribution
    connectivity_distribution::Distributions.Distribution
end

"""
	MergedCsetMaker(config::Dict)

Make a new merged causal set maker from a given configuration dictionary.
"""
function MergedCsetMaker(config::Dict)
    order_distr = build_distr(config, "order_distribution")
    r_distr = build_distr(config, "r_distribution")
    link_prob_distr = build_distr(config, "link_prob_distribution")
    n2_rel_distr = build_distr(config, "n2_rel_distribution")
    connectivity_distr = build_distr(config, "connectivity_distribution")

    return MergedCsetMaker(
        link_prob_distr,
        order_distr,
        r_distr,
        n2_rel_distr,
        connectivity_distr,
    )
end

"""
	mcm::MergedCsetMaker(n::Int64, config::Dict, rng::Random.AbstractRNG)

	Creates a merged causal set maker for a random causal set.

# Arguments:
- `n`: number of elements in the causal set
- `config`: configuration dictionary
- `rng`: random number generator
"""
function (mcm::MergedCsetMaker)(
    n::Int64,
    rng::Random.AbstractRNG;
    config::Union{Dict{String,Any},Nothing} = nothing,
)::CausalSets.BitArrayCauset

    o = rand(rng, mcm.order_distribution)
    r = rand(rng, mcm.r_distribution)
    n2rel = rand(rng, mcm.n2_rel_distribution)
    l = rand(rng, mcm.link_prob_distribution)
    p = rand(rng, mcm.connectivity_distribution)

    cset, success, sprinkling =
        QG.insert_KR_into_manifoldlike(n, o, r, l; rng = rng, n2_rel = n2rel, p = p)

    return cset
end
