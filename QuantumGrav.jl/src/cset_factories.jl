using CausalSets: CausalSets
using Random: Random
using Distributions: Distributions
import QuantumGrav as QG



"""
	ManifoldCsetMaker

	Causal set maker for a polynomial manifold.

# Fields:
- `order_distribution::Distributions.Distribution`: distribution of polynomial orders
- `r_distribution::Distributions.Distribution`: distribution of exponential decay exponents
"""
struct ManifoldCsetMaker
    order_distribution::Distributions.Distribution
    r_distribution::Distributions.Distribution
end

"""
	ManifoldCsetMaker(config)

	Creates a causal set maker for a polynomial manifold.

# Fields:
- config::Dict: configuration dictionary
"""
function ManifoldCsetMaker(config)
    order_distrname = getfield(Distributions, config["order_distribution"])
    r_distname = getfield(Distributions, config["r_distribution"])

    order_distribution = order_distrname(
        config["order_distribution_args"]...;
        config["order_distribution_kwargs"]...,
    )
    r_distribution =
        r_distname(config["r_distribution_args"]...; config["r_distribution_kwargs"]...)
    return ManifoldCsetMaker(order_distribution, r_distribution)
end

"""
	m::ManifoldCsetMaker(n, config, rng)

	Creates a new polynomial causal set with the parameters stored in the calling `ManifoldCsetMaker` object m.

# Arguments:
- `n`: number of elements in the causal set
- `config`: configuration dictionary
- `rng`: random number generator
"""
function (m::ManifoldCsetMaker)(n, config, rng)::CausalSets.BitArrayCauset
    o = rand(rng, m.order_distribution)
    r = rand(rng, m.r_distribution)
    cset, _, __ = QG.make_polynomial_manifold_cset(n, rng, o, r; d = 2, type = Float32)
    return cset
end

"""
	LayeredCausetMaker

	Causal set maker for a layered causal set.

# Fields:
- `cdistr::Distributions.Distribution`: distribution of connectivity goals
- `stddev_distr::Distributions.Distribution`: distribution of standard deviations
- `ldistr::Distributions.Distribution`: distribution of layer counts
"""
struct LayeredCausetMaker
    cdistr::Distributions.Distribution
    stddev_distr::Distributions.Distribution
    ldistr::Distributions.Distribution
end

"""
	LayeredCausetMaker(config::Dict)

	Creates a causal set maker for a layered causal set.

# Fields:
- config::Dict: configuration dictionary
"""
function LayeredCausetMaker(config::Dict)
    cdist_name = getfield(Distributions, config["connectivity_distribution"])
    stddev_name = getfield(Distributions, config["stddev_distribution"])
    ldist_name = getfield(Distributions, config["ldist_distributions"])

    cdistr = cdist_name(
        config["connectivity_distribution_args"]...;
        config["connectivity_distribution_kwargs"]...,
    )
    stddev_distr =
        stddev_name(config["stddev_distr_args"]...; config["stddev_distr_kwargs"]...)
    ldistr = ldist_name(config["layered_ldist_min"], config["layered_ldist_max"])

    return LayeredCausetMaker(cdistr, stddev_distr, ldistr)

end

"""
	lm::LayeredCausetMaker(n::Int64, config::Dict, rng::Random.AbstractRNG)

	Creates a new layered causal set with the parameters stored in the calling `LayeredCausetMaker` object lm.

# Arguments:
- `n`: number of elements in the causal set
- `config`: configuration dictionary
- `rng`: random number generator
"""
function (lm::LayeredCausetMaker)(
    n::Int64,
    config::Dict,
    rng::Random.AbstractRNG,
)::CausalSets.BitArrayCauset

    connectivity_goal = rand(rng, lm.cdistr)
    while connectivity_goal < 1e-5
        connectivity_goal = rand(rng, lm.cdistr)
    end

    layers = rand(rng, lm.ldistr)
    while layers < 1e-5
        layers = rand(rng, lm.ldistr)
    end
    layers = Int(ceil(layers))

    s = rand(rng, lm.stddev_distr)

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
    cdistr::Distributions.Distribution
end

"""
	RandomCsetMaker(config::Dict)

	Creates a causal set maker for a random causal set.

# Fields:
- config::Dict: configuration dictionary
"""
function RandomCsetMaker(config::Dict)
    cdistr_name = getfield(Distributions, config["connectivity_distribution"])
    cdistr = cdistr_name(
        config["connectivity_distribution_args"]...;
        config["connectivity_distribution_kwargs"]...,
    )
    return RandomCsetMaker(cdistr)
end

"""
	rcm::RandomCsetMaker(n::Int64, config::Dict, rng::Random.AbstractRNG)

	Creates a new random causal set with the parameters stored in the calling `RandomCsetMaker` object rcm.

# Arguments:
- `n`: number of elements in the causal set
- `config`: configuration dictionary
- `rng`: random number generator
"""
function (rcm::RandomCsetMaker)(
    n::Int64,
    config::Dict,
    rng::Random.AbstractRNG,
)::CausalSets.BitArrayCauset

    connectivity_goal = rand(rng, rcm.cdistr)

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

        if tries > 20
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
	DestroyedCausetMaker

	Causal set maker for a destroyed causal set, which has a set of edges flipped in a polynomial causal set.

# Fields:
- `order_distribution::Distributions.Distribution`: distribution of order values
- `r_distribution::Distributions.Distribution`: distribution of r values
- `flip_distribution::Distributions.Distribution`: distribution of flip values
"""
struct DestroyedCausetMaker
    order_distribution::Distributions.Distribution
    r_distribution::Distributions.Distribution
    flip_distribution::Distributions.Distribution
end

"""
	DestroyedCausetMaker(config::Dict)

Create a new `destroyed` causal set maker object from the config dictionary.
"""
function DestroyedCausetMaker(config::Dict)
    order_distrname = getfield(Distributions, config["order_distribution"])
    r_distname = getfield(Distributions, config["r_distribution"])
    flig_distname = getfield(Distributions, config["flip_distribution"])

    order_distribution = order_distrname(
        config["order_distribution_args"]...;
        config["order_distribution_kwargs"]...,
    )
    r_distribution =
        r_distname(config["r_distribution_args"]...; config["r_distribution_kwargs"]...)

    flip_distribution = flig_distname(
        config["flip_distribution_args"]...;
        config["flip_distribution_kwargs"]...,
    )

    return DestroyedCausetMaker(order_distribution, r_distribution, flip_distribution)
end

"""
	dcm::DestroyedCausetMaker(n::Int64, config::Dict, rng::Random.AbstractRNG)

Create a new `destroyed` causal set using a `DestroyedCausetMaker` object.

# Arguments:
- `n`: number of elements in the causal set
- `config`: configuration dictionary
- `rng`: random number generator
"""
function (dcm::DestroyedCausetMaker)(
    n::Int64,
    config::Dict,
    rng::Random.AbstractRNG,
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
    config::Dict
end

"""
	GridCsetMakerPolynomial(config)

	Create a new `grid` causal set maker object from the config dictionary for polynomial spacetimes.
"""
function GridCsetMakerPolynomial(config::Dict)
    grid_distrname = getfield(Distributions, config["grid_distribution"])
    rotate_distrname = getfield(Distributions, config["rotate_distribution"])
    gamma_distrname = getfield(Distributions, config["gamma_distribution"])
    order_distrname = getfield(Distributions, config["order_distribution"])
    r_distname = getfield(Distributions, config["r_distribution"])
    grid_distribution = grid_distrname(
        config["grid_distribution_args"]...;
        config["grid_distribution_kwargs"]...,
    )
    rotate_distribution = rotate_distrname(
        config["rotate_distribution_args"]...;
        config["rotate_distribution_kwargs"]...,
    )
    gamma_distribution = gamma_distrname(
        config["gamma_distribution_args"]...;
        config["gamma_distribution_kwargs"]...,
    )
    order_distribution = order_distrname(
        config["order_distribution_args"]...;
        config["order_distribution_kwargs"]...,
    )
    r_distribution =
        r_distname(config["r_distribution_args"]...; config["r_distribution_kwargs"]...)

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
        config,
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
function (gcm::GridCsetMakerPolynomial)(n::Int64, config::Dict, rng::Random.AbstractRNG)

    o = rand(rng, gcm.order_distribution)
    r = rand(rng, gcm.r_distribution)

    rotate_angle_deg = rand(rng, gcm.base.rotate_distribution)
    gamma_deg = rand(rng, gcm.base.gamma_distribution)
    grid = gcm.grid_lookup[rand(rng, gcm.base.grid_distribution)]

    a_dist = Distributions.Uniform(gcm.config[grid]["a_min"], gcm.config[grid]["a_max"])
    b_dist = Distributions.Uniform(gcm.config[grid]["b_min"], gcm.config[grid]["b_max"])
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

DOCSTRING

# Fields:
- `vertical_cut_distr::Distributions.Distribution`: DESCRIPTION
- `finite_cut_distr::Distributions.Distribution`: DESCRIPTION
- `order_distribution::Distributions.Distribution`: DESCRIPTION
- `r_distribution::Distributions.Distribution`: DESCRIPTION
- `tol::Float64`: DESCRIPTION
"""
struct ComplexTopCsetMaker
    vertical_cut_distr::Distributions.Distribution
    finite_cut_distr::Distributions.Distribution
    order_distribution::Distributions.Distribution
    r_distribution::Distributions.Distribution
    tol::Float64
end

"""
	ComplexTopCsetMaker(config::Dict)

	Create a new `ComplexTopCsetMaker` object from the config dictionary.
"""
function ComplexTopCsetMaker(config::Dict)
    vertical_cut_distrname = getfield(Distributions, config["vertical_cut_distribution"])
    finite_cut_distrname = getfield(Distributions, config["finite_cut_distribution"])
    order_distrname = getfield(Distributions, config["order_distribution"])
    r_distname = getfield(Distributions, config["r_distribution"])

    vertical_cut_distr = vertical_cut_distrname(
        config["vertical_cut_distribution_args"]...;
        config["vertical_cut_distribution_kwargs"]...,
    )
    finite_cut_distr = finite_cut_distrname(
        config["finite_cut_distribution_args"]...;
        config["finite_cut_distribution_kwargs"]...,
    )
    order_distr = order_distrname(
        config["order_distribution_args"]...;
        config["order_distribution_kwargs"]...,
    )
    r_distr =
        r_distname(config["r_distribution_args"]...; config["r_distribution_kwargs"]...)
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
    config::Dict,
    rng::Random.AbstractRNG,
)::CausalSets.BitArrayCauset

    n_vertical_cuts = rand(rng, ctm.vertical_cut_distr)
    n_finite_cuts = rand(rng, ctm.finite_cut_distr)
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
    order_distrname = getfield(Distributions, config["order_distribution"])
    r_distname = getfield(Distributions, config["r_distribution"])
    link_prob_distrname = getfield(Distributions, config["link_prob_distribution"])
    n2_rel_distrname = getfield(Distributions, config["n2_rel_distribution"])
    connectivity_distrname = getfield(Distributions, config["connectivity_distribution"])

    order_distr = order_distrname(
        config["order_distribution_args"]...;
        config["order_distribution_kwargs"]...,
    )
    r_distr =
        r_distname(config["r_distribution_args"]...; config["r_distribution_kwargs"]...)
    link_prob_distr =
        link_prob_distrname(config["link_prob_args"]...; config["link_prob_kwargs"]...)
    n2_rel_distr = n2_rel_distrname(config["n2_rel_args"]...; config["n2_rel_kwargs"]...)
    connectivity_distr = connectivity_distrname(
        config["connectivity_args"]...;
        config["connectivity_kwargs"]...,
    )

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
    config::Dict,
    rng::Random.AbstractRNG,
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
