"""
	build_distr(cfg::AbstractDict, name::String)

Build a new Distributions.jl univariate distribution from a config dictionary.

# Arguments:
- cfg::AbstractDict config dict
- name::String key in the dict referring to the type, args, kwargs needed

# Example:
```julia
config = Dict("connectivity_distribution" => "Cauchy",
			"connectivity_distribution_args" => [0.5, 0.2],
			"connectivity_distribution_kwargs" => Dict(),
		)

distributions = build_distr(config, "connectivity_distribution")

```

"""
function build_distr(cfg::AbstractDict, name::String)::Distributions.Distribution

    distribution_type::Union{Nothing,Type} = nothing

    distr::Union{Nothing,Distributions.Distribution} = nothing

    try
        distribution_type = getfield(Distributions, Symbol(cfg[name]))
    catch e
        throw(ArgumentError("Distribution $(name) could not be retrieved $(e)"))
    end

    kwargs = get(cfg, name*"_kwargs", Dict())

    if !(kwargs isa Dict{Symbol,Any})
        kwargs = Dict(Symbol(k) => v for (k, v) in kwargs)
    end

    try
        distr = distribution_type(cfg[name*"_args"]...; kwargs...)
    catch e
        throw(ArgumentError("Distribution $(name) could not be built $(e)"))
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

const PolynomialCsetMaker_schema = JSONSchema.Schema("""{
                "\$schema": "http://json-schema.org/draft-06/schema#",
                "title": "QuantumGrav Cset Factory Config",
                "type": "object",
                "additionalProperties": false,
                "properties": {
               "order_distribution": { "type": "string" },
               "order_distribution_args": {
                 "type": "array",
                 "items": { "type": "integer" }
               },
               "order_distribution_kwargs": {
                 "type": "object",
                 "additionalProperties": true
               },
               "r_distribution": { "type": "string" },
               "r_distribution_args": {
                 "type": "array",
                 "items": { "type": "number" }
               },
               "r_distribution_kwargs": {
                 "type": "object",
                 "additionalProperties": true
               }
                },
                "required": [
               "order_distribution",
               "order_distribution_args",
               "r_distribution",
               "r_distribution_args"
                ]
              }
              """)

"""
	PolynomialCsetMaker(config)

	Creates a causal set maker for a polynomial manifold.

# Arguments:
- config::AbstractDict: configuration dictionary
"""
function PolynomialCsetMaker(config)
    validate_config(PolynomialCsetMaker_schema, config)

    order_distribution = build_distr(config, "order_distribution")
    r_distribution = build_distr(config, "r_distribution")

    return PolynomialCsetMaker(order_distribution, r_distribution)
end

"""
	m::PolynomialCsetMaker(n, config, rng)

	Creates a new polynomial causal set with the parameters stored in the calling `PolynomialCsetMaker` object m.

# Arguments:
- `n`: number of elements in the causal set
- `rng`: random number generator

# Keyword arguments:
- `config`: configuration dictionary
- `derivation_matrix1`: optional derivation matrix transform characterising first derivative of chebyshev polynomials for curvature calculation
- `derivation_matrix2`: optional derivation matrix transform characterising second derivative of chebyshev polynomials for curvature calculation

# Returns
- causal set (BitArrayCauset)
- curvature at every point (Vector{Float64})
"""
function (m::PolynomialCsetMaker)(
    n,
    rng;
    config::Union{Dict,Nothing} = nothing,
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Tuple{CausalSets.BitArrayCauset,Vector{Float64}}
    o = rand(rng, m.order_distribution)
    r = rand(rng, m.r_distribution)
    cset, sprinkling, chebyshev_coefs =
        make_polynomial_manifold_cset(n, rng, o, r; d = 2, type = Float32)
    curvature_matrix = Ricci_scalar_2D_of_sprinkling(
        Float64.(chebyshev_coefs),
        Vector{CausalSets.Coordinates{2}}(sprinkling);
        derivation_matrix1 = derivation_matrix1,
        derivation_matrix2 = derivation_matrix2,
    )
    return cset, curvature_matrix
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

const LayeredCsetMaker_schema = JSONSchema.Schema(
    """{
      "\$schema": "http://json-schema.org/draft-06/schema#",
      "title": "Layered csetmaker config",
      "type": "object",
      "additionalProperties": false,
      "properties": {
    	"connectivity_distribution": { "type": "string" },
    	"connectivity_distribution_args": {
    	  "type": "array",
    	  "items": { "type": "number" }
    	},
    	"connectivity_distribution_kwargs": {
    	  "type": "object",
    	  "additionalProperties": true
    	},
    	"stddev_distribution": { "type": "string", "default": "Normal" },
    	"stddev_distribution_args": {
    	  "type": "array",
    	  "items": { "type": "number" }
    	},
    	"stddev_distribution_kwargs": {
    	  "type": "object",
    	  "additionalProperties": true
    	},
    	"layer_distribution": { "type": "string", "default": "DiscreteUniform" },
    	"layer_distribution_args": {
    	  "type": "array",
    	  "items": { "type": "integer" }
    	},
    	"layer_distribution_kwargs": {
    	  "type": "object",
    	  "additionalProperties": true
    	}
      },
      "required": [
    	"connectivity_distribution",
    	"connectivity_distribution_args",
    	"stddev_distribution",
    	"stddev_distribution_args",
    	"layer_distribution",
    	"layer_distribution_args"
      ]
      }
    """,
)

"""
	LayeredCsetMaker(config::AbstractDict)

	Creates a causal set maker for a layered causal set.

# Arguments:
	- config::AbstractDict: configuration dictionary
"""
function LayeredCsetMaker(config::AbstractDict)
    validate_config(LayeredCsetMaker_schema, config)

    cdistr = build_distr(config, "connectivity_distribution")
    stddev_distr = build_distr(config, "stddev_distribution")
    ldistr = build_distr(config, "layer_distribution")
    return LayeredCsetMaker(cdistr, stddev_distr, ldistr)
end

"""
	lm::LayeredCsetMaker(n::Int64, config::AbstractDict, rng::Random.AbstractRNG)

	Creates a new layered causal set with the parameters stored in the calling `LayeredCsetMaker` object lm.

# Arguments:
- `n`: number of elements in the causal set
- `rng`: random number generator

# Keyword arguments:
- `config`: configuration dictionary

# Returns
- causal set (BitArrayCauset)
- number of layers
"""
function (lm::LayeredCsetMaker)(
    n::Int64,
    rng::Random.AbstractRNG;
    config::Union{Dict,Nothing} = nothing,
)::Tuple{CausalSets.BitArrayCauset,Int64}
    connectivity_goal = rand(rng, lm.connectivity_distribution)
    layers = rand(rng, lm.layer_distribution)
    layers = Int(ceil(layers))

    s = rand(rng, lm.stddev_distribution)

    cset, _ = create_random_layered_causet(
        n,
        layers;
        p = connectivity_goal,
        rng = rng,
        standard_deviation = s,
    )

    return cset, layers
end

"""
	RandomCsetMaker

	Causal set maker for a random causal set.

# Fields:
- `cdistr::Distributions.Distribution`: distribution of connectivity goals
"""
struct RandomCsetMaker
    connectivity_distribution::Distributions.Distribution
    max_iter::Int64
    num_tries::Int64
    abs_tol::Union{Float64,Nothing}
    rel_tol::Union{Float64,Nothing}
end

const RandomCsetMaker_schema = JSONSchema.Schema("""{
               "\$schema": "http://json-schema.org/draft-06/schema#",
               "title": "Random csetmaker config",
               "type": "object",
               "additionalProperties": false,
               "properties": {
              "connectivity_distribution": { "type": "string" },
              "connectivity_distribution_args": {
                "type": "array",
                "items": { "type": "number" }
              },
              "connectivity_distribution_kwargs": {
                "type": "object",
                "additionalProperties": true
              },
              "max_iter": { "type": "integer", "minimum": 1 },
              "num_tries": { "type": "integer", "minimum": 1 },
              "abs_tol": { "type": ["number", "null"] },
              "rel_tol": { "type": ["number", "null"] }
               },
               "required": [
              "connectivity_distribution",
              "connectivity_distribution_args",
              "max_iter",
              "num_tries",
              "abs_tol",
              "rel_tol"
               ]
               }
             """)

"""
	RandomCsetMaker(config::AbstractDict)

	Creates a causal set maker for a random causal set.

# Fields:
- config::AbstractDict: configuration dictionary
"""
function RandomCsetMaker(config::AbstractDict)
    validate_config(RandomCsetMaker_schema, config)

    cdistr = build_distr(config, "connectivity_distribution")

    if config["max_iter"] < 1
        throw(ArgumentError("Error, max_iter must be >= 1, is $(config["max_iter"])."))
    end

    if config["num_tries"] < 1
        throw(ArgumentError("Error, num_tries must be >= 1, is $(config["num_tries"])."))
    end

    return RandomCsetMaker(
        cdistr,
        config["max_iter"],
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

    converged = false

    cset = nothing

    tries = 1

    while converged == false
        if tries > rcm.num_tries
            cset = nothing
            break
        end

        cset_try, converged = sample_bitarray_causet_by_connectivity(
            n,
            connectivity_goal,
            rcm.max_iter,
            rng;
            abs_tol = rcm.abs_tol,
            rel_tol = rcm.rel_tol,
        )
        tries += 1

        cset = cset_try
    end

    if cset === nothing
        throw(
            ErrorException(
                "Failed to generate causet with n=$n and connectivity_goal=$connectivity_goal after $(tries-1) tries.",
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

const DestroyedCsetMaker_schema = JSONSchema.Schema("""{
               "\$schema": "http://json-schema.org/draft-06/schema#",
               "title": "Destroyed csetmaker config",
               "type": "object",
               "additionalProperties": false,
               "properties": {
             	"order_distribution": { "type": "string" },
             	"order_distribution_args": {
             	  "type": "array",
             	  "items": { "type": "integer" }
             	},
             	"order_distribution_kwargs": {
             	  "type": "object",
             	  "additionalProperties": true
             	},
             	"r_distribution": { "type": "string" },
             	"r_distribution_args": {
             	  "type": "array",
             	  "items": { "type": "number" }
             	},
             	"r_distribution_kwargs": {
             	  "type": "object",
             	  "additionalProperties": true
             	},
             	"flip_distribution": { "type": "string" },
             	"flip_distribution_args": {
             	  "type": "array",
             	  "items": { "type": "number" }
             	},
             	"flip_distribution_kwargs": {
             	  "type": "object",
             	  "additionalProperties": true
             	}
               },
               "required": [
             	"order_distribution",
             	"order_distribution_args",
             	"r_distribution",
             	"r_distribution_args",
             	"flip_distribution",
             	"flip_distribution_args"
               ]
               }
             """)

"""
	DestroyedCsetMaker(config::AbstractDict)

Create a new `destroyed` causal set maker object from the config dictionary.
"""
function DestroyedCsetMaker(config::AbstractDict)
    validate_config(DestroyedCsetMaker_schema, config)

    order_distribution = build_distr(config, "order_distribution")

    r_distribution = build_distr(config, "r_distribution")

    flip_distribution = build_distr(config, "flip_distribution")

    return DestroyedCsetMaker(order_distribution, r_distribution, flip_distribution)
end

"""
	dcm::DestroyedCsetMaker(n::Int64, config::AbstractDict, rng::Random.AbstractRNG)

Create a new `destroyed` causal set using a `DestroyedCsetMaker` object.

# Arguments:
- `n`: number of elements in the causal set
- `rng`: random number generator

# Keyword arguments:
- `config`: configuration dictionary

# Returns
- causal set (BitArrayCauset)
- number of flipped edges relative to size of causal set

"""
function (dcm::DestroyedCsetMaker)(
    n::Int64,
    rng::Random.AbstractRNG;
    config::Union{AbstractDict,Nothing} = nothing,
)::Tuple{CausalSets.BitArrayCauset,Float64}

    o = rand(rng, dcm.order_distribution)

    r = rand(rng, dcm.r_distribution)

    f = convert(Int64, ceil(rand(rng, dcm.flip_distribution) * n * (n - 1) / 2))

    cset = destroy_manifold_cset(n, f, rng, o, r; d = 2, type = Float32)[1]
    return cset, f/(n * (n - 1) / 2)
end


"""
	GridCsetMakerPolynomial

	Create a new `grid` causal set maker object from the config dictionary for polynomial spacetimes.

# Fields:
	- grid_distribution::Distributions.Distribution: Random distribution to draw grid from
	- rotate_distribution::Distributions.Distribution: Random distribution to draw rotation angle from
	- order_distribution::Distributions.Distribution: Random distribution to draw order from
	- r_distribution::Distributions.Distribution: Random distribution to draw exponent for Chebyshev expansionfrom
	- grid_lookup::AbstractDict: grid lookup table by name
"""
struct GridCsetMakerPolynomial
    grid_distribution::Distributions.Distribution
    rotate_distribution::Distributions.Distribution
    order_distribution::Distributions.Distribution
    r_distribution::Distributions.Distribution
    grid_lookup::Dict
end

const GridCsetMakerPolynomial_schema = JSONSchema.Schema("""{
                 "\$schema": "http://json-schema.org/draft-06/schema#",
                 "title": "GridCsetMakerPolynomial config",
                 "type": "object",
                 "additionalProperties": false,
                 "properties": {
                "grid_distribution": { "type": "string" },
                "grid_distribution_args": {
                  "type": "array",
                  "items": { "type": "integer" }
                },
                "grid_distribution_kwargs": {
                  "type": "object",
                  "additionalProperties": true
                },
                "rotate_distribution": { "type": "string" },
                "rotate_distribution_args": {
                  "type": "array",
                  "items": { "type": "number" }
                },
                "rotate_distribution_kwargs": {
                  "type": "object",
                  "additionalProperties": true
                },
                "order_distribution": { "type": "string" },
                "order_distribution_args": {
                  "type": "array",
                  "items": { "type": "integer" }
                },
                "order_distribution_kwargs": {
                  "type": "object",
                  "additionalProperties": true
                },
                "r_distribution": { "type": "string" },
                "r_distribution_args": {
                  "type": "array",
                  "items": { "type": "number" }
                },
                "r_distribution_kwargs": {
                  "type": "object",
                  "additionalProperties": true
                },
                "quadratic": {
                  "type": "object",
                  "properties": {},
                  "additionalProperties": false
                },
                "rectangular": {
                  "type": "object",
                  "additionalProperties": false,
                  "properties": {
               	 "segment_ratio_distribution": { "type": "string" },
               	 "segment_ratio_distribution_args": {
               	   "type": "array",
               	   "items": { "type": "number" }
               	 },
               	 "segment_ratio_distribution_kwargs": {
               	   "type": "object",
               	   "additionalProperties": true
               	 }
                  },
                  "required": [
               	 "segment_ratio_distribution",
               	 "segment_ratio_distribution_args"
                  ]
                },
                "rhombic": {
                  "type": "object",
                  "additionalProperties": false,
                  "properties": {
               	 "segment_ratio_distribution": { "type": "string" },
               	 "segment_ratio_distribution_args": {
               	   "type": "array",
               	   "items": { "type": "number" }
               	 },
               	 "segment_ratio_distribution_kwargs": {
               	   "type": "object",
               	   "additionalProperties": true
               	 }
                  },
                  "required": [
               	 "segment_ratio_distribution",
               	 "segment_ratio_distribution_args"
                  ]
                },
                "hexagonal": {
                  "type": "object",
                  "additionalProperties": false,
                  "properties": {
               	 "segment_ratio_distribution": { "type": "string" },
               	 "segment_ratio_distribution_args": {
               	   "type": "array",
               	   "items": { "type": "number" }
               	 },
               	 "segment_ratio_distribution_kwargs": {
               	   "type": "object",
               	   "additionalProperties": true
               	 }
                  },
                  "required": [
               	 "segment_ratio_distribution",
               	 "segment_ratio_distribution_args"
                  ]
                },
                "triangular": {
                  "type": "object",
                  "additionalProperties": false,
                  "properties": {
               	 "segment_ratio_distribution": { "type": "string" },
               	 "segment_ratio_distribution_args": {
               	   "type": "array",
               	   "items": { "type": "number" }
               	 },
               	 "segment_ratio_distribution_kwargs": {
               	   "type": "object",
               	   "additionalProperties": true
               	 }
                  },
                  "required": [
               	 "segment_ratio_distribution",
               	 "segment_ratio_distribution_args"
                  ]
                },
                "oblique": {
                  "type": "object",
                  "additionalProperties": false,
                  "properties": {
               	 "segment_ratio_distribution": { "type": "string" },
               	 "segment_ratio_distribution_args": {
               	   "type": "array",
               	   "items": { "type": "number" }
               	 },
               	 "segment_ratio_distribution_kwargs": {
               	   "type": "object",
               	   "additionalProperties": true
               	 },
               	 "oblique_angle_distribution": { "type": "string" },
               	 "oblique_angle_distribution_args": {
               	   "type": "array",
               	   "items": { "type": "number" }
               	 },
               	 "oblique_angle_distribution_kwargs": {
               	   "type": "object",
               	   "additionalProperties": true
               	 }
                  },
                  "required": [
               	 "segment_ratio_distribution",
               	 "segment_ratio_distribution_args",
               	 "oblique_angle_distribution",
               	 "oblique_angle_distribution_args"
                  ]
                }
                 },
                 "required": [
                "grid_distribution",
                "grid_distribution_args",
                "rotate_distribution",
                "rotate_distribution_args",
                "order_distribution",
                "order_distribution_args",
                "r_distribution",
                "r_distribution_args"
                 ]
                }
               """)


"""
	GridCsetMakerPolynomial(config)

	Create a new `grid` causal set maker object from the config dictionary for polynomial spacetimes.
"""
function GridCsetMakerPolynomial(config::Dict)
    validate_config(GridCsetMakerPolynomial_schema, config)

    grid_distribution = build_distr(config, "grid_distribution")

    rotate_distribution = build_distr(config, "rotate_distribution")

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
        order_distribution,
        r_distribution,
        grid_lookup,
    )
end

"""
	gcm::GridCsetMakerPolynomial(n::Int64, config::AbstractDict, rng::Random.AbstractRNG)

	Create a new `grid` causal set using a `GridCsetMakerPolynomial` object.

# Arguments:
- `n`: number of elements in the causal set
- `rng`: random number generator
- `config`: configuration dictionary

# Keyword arguments:
- `grid`: name of the grid type to use
- `derivation_matrix1`: optional derivation matrix transform characterising first derivative of chebyshev polynomials for curvature calculation
- `derivation_matrix2`: optional derivation matrix transform characterising second derivative of chebyshev polynomials for curvature calculation

# Returns
- causal set (BitArrayCauset)
- grid type

"""
function (gcm::GridCsetMakerPolynomial)(
    n::Int64,
    rng::Random.AbstractRNG;
    config::Union{AbstractDict,Nothing} = nothing,
    grid::Union{String,Nothing} = nothing,
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Tuple{CausalSets.BitArrayCauset,Vector{Float64},String}

    if isnothing(config)
        throw(ArgumentError("Config cannot be None for GridCsetMakerPolynomial"))
    end

    if isnothing(grid)
        grid = gcm.grid_lookup[rand(rng, gcm.grid_distribution)]
    end

    o = rand(rng, gcm.order_distribution)
    r = rand(rng, gcm.r_distribution)
    rotate_angle_deg = rand(rng, gcm.rotate_distribution)

    gamma_deg =
        grid == "oblique" ?
        rand(rng, build_distr(config[grid], "oblique_angle_distribution")) : 60.0

    b =
        grid == "quadratic" ? 1.0 :
        rand(rng, build_distr(config[grid], "segment_ratio_distribution"))

    cset, _, pseudosprinkling, chebyshev_coefs = create_grid_causet_2D_polynomial_manifold(
        n,
        grid,
        rng,
        o,
        r;
        type = Float32,
        a = 1.0,
        b = b,
        gamma_deg = gamma_deg,
        rotate_deg = rotate_angle_deg,
        origin = (0.0, 0.0),
    )

    trans_pseudosprinkling =
        [(Float64(x), Float64(y)) for (x, y) in eachrow(Float64.(pseudosprinkling))]

    curvature_matrix = Ricci_scalar_2D_of_sprinkling(
        Float64.(chebyshev_coefs),
        trans_pseudosprinkling;
        derivation_matrix1 = derivation_matrix1,
        derivation_matrix2 = derivation_matrix2,
    )

    return cset, curvature_matrix, grid
end


"""
	ComplexTopCsetMaker

A callable struct to produce complex topology csets with various causality-cutting 'lines' in a 2D manifold

# Fields:
- `vertical_cut_distr::Distributions.Distribution`: Distribution to draw the number of vertical (time direction) cuts from
- `finite_cut_distr::Distributions.Distribution`: Distribution to draw the number of mixed direction cuts from
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

const ComplexTopCsetMaker_schema = JSONSchema.Schema("""{
                "\$schema": "http://json-schema.org/draft-06/schema#",
                "title": "Complex Topology csetmaker config",
                "type": "object",
                "additionalProperties": false,
                "properties": {
               "order_distribution": { "type": "string" },
               "order_distribution_args": {
                 "type": "array",
                 "items": { "type": "integer" }
               },
               "order_distribution_kwargs": {
                 "type": "object",
                 "additionalProperties": true
               },
               "r_distribution": { "type": "string" },
               "r_distribution_args": {
                 "type": "array",
                 "items": { "type": "number" }
               },
               "r_distribution_kwargs": {
                 "type": "object",
                 "additionalProperties": true
               },
               "vertical_cut_distribution": { "type": "string" },
               "vertical_cut_distribution_args": {
                 "type": "array",
                 "items": { "type": "number" }
               },
               "vertical_cut_distribution_kwargs": {
                 "type": "object",
                 "additionalProperties": true
               },
               "finite_cut_distribution": { "type": "string" },
               "finite_cut_distribution_args": {
                 "type": "array",
                 "items": { "type": "number" }
               },
               "finite_cut_distribution_kwargs": {
                 "type": "object",
                 "additionalProperties": true
               },
               "tol": { "type": "number" }
                },
                "required": [
               "order_distribution",
               "order_distribution_args",
               "r_distribution",
               "r_distribution_args",
               "vertical_cut_distribution",
               "vertical_cut_distribution_args",
               "finite_cut_distribution",
               "finite_cut_distribution_args",
               "tol"
                ]
              }
              """)

"""
	ComplexTopCsetMaker(config::AbstractDict)

	Create a new `ComplexTopCsetMaker` object from the config dictionary.
"""
function ComplexTopCsetMaker(config::AbstractDict)
    validate_config(ComplexTopCsetMaker_schema, config)

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
	ctm::ComplexTopCsetMaker(n::Int64, config::AbstractDict, rng::Random.AbstractRNG)

	Create a new causal set using a `ComplexTopCsetMaker` object.

# Arguments:
- `n`: number of elements in the causal set
- `config`: configuration dictionary
- `rng`: random number generator

# Keyword arguments:
- `config`: configuration dictionary

# Returns
- causal set (BitArrayCauset)
"""
function (ctm::ComplexTopCsetMaker)(
    n::Int64,
    rng::Random.AbstractRNG;
    config::Union{AbstractDict,Nothing} = nothing,
    derivation_matrix1::Union{Nothing,Array{Float64,2}} = nothing,
    derivation_matrix2::Union{Nothing,Array{Float64,2}} = nothing,
)::Tuple{CausalSets.BitArrayCauset,Vector{Float64}}

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
        make_branched_manifold_cset(
            n,
            n_vertical_cuts,
            n_finite_cuts,
            rng,
            order,
            r;
            d = 2,
            tolerance = ctm.tol,
        )

    curvature_matrix = Ricci_scalar_2D_of_sprinkling(
        Float64.(chebyshev_coefs),
        Vector{CausalSets.Coordinates{2}}(branched_sprinkling);
        derivation_matrix1 = derivation_matrix1,
        derivation_matrix2 = derivation_matrix2,
    )

    return cset, curvature_matrix
end


"""
	MergedCsetMaker

	Causal set maker from a given configuration dictionary.

# Fields:
- `link_prob_distribution::Distributions.Distribution`: distribution of link probabilities
- `order_distribution::Distributions.Distribution`: distribution of order values
- `r_distribution::Distributions.Distribution`: distribution of r values
- `n2_rel_distribution::Distributions.Distribution`: distribution of n2 relative values
"""
struct MergedCsetMaker
    link_prob_distribution::Distributions.Distribution
    order_distribution::Distributions.Distribution
    r_distribution::Distributions.Distribution
    n2_rel_distribution::Distributions.Distribution
end

const MergedCsetMaker_schema = JSONSchema.Schema("""{
               "\$schema": "http://json-schema.org/draft-06/schema#",
               "title": "merged csetmaker config",
              "type": "object",
               "additionalProperties": false,
               "properties": {
              "order_distribution": { "type": "string" },
              "order_distribution_args": {
                "type": "array",
                "items": { "type": "integer" }
              },
              "order_distribution_kwargs": {
                "type": "object",
                "additionalProperties": true
              },
              "r_distribution": { "type": "string" },
              "r_distribution_args": {
                "type": "array",
                "items": { "type": "number" }
              },
              "r_distribution_kwargs": {
                "type": "object",
                "additionalProperties": true
              },
              "n2_rel_distribution": { "type": "string" },
              "n2_rel_distribution_args": {
                "type": "array",
                "items": { "type": "number" }
              },
              "n2_rel_distribution_kwargs": {
                "type": "object",
                "additionalProperties": true
              },
              "link_prob_distribution": { "type": "string" },
              "link_prob_distribution_args": {
                "type": "array",
                "items": { "type": "number" }
              },
              "link_prob_distribution_kwargs": {
                "type": "object",
                "additionalProperties": true
              }
               },
               "required": [
              "order_distribution",
              "order_distribution_args",
              "r_distribution",
              "r_distribution_args",
              "n2_rel_distribution",
              "n2_rel_distribution_args",
              "link_prob_distribution",
              "link_prob_distribution_args"
               ]
             }
             """)

"""
	MergedCsetMaker(config::AbstractDict)

Make a new merged causal set maker from a given configuration dictionary.
"""
function MergedCsetMaker(config::AbstractDict)
    validate_config(MergedCsetMaker_schema, config)

    order_distr = build_distr(config, "order_distribution")
    r_distr = build_distr(config, "r_distribution")
    link_prob_distr = build_distr(config, "link_prob_distribution")
    n2_rel_distr = build_distr(config, "n2_rel_distribution")

    return MergedCsetMaker(
        link_prob_distr,
        order_distr,
        r_distr,
        n2_rel_distr,
    )
end

"""
	mcm::MergedCsetMaker(n::Int64, config::AbstractDict, rng::Random.AbstractRNG)

	Creates a merged causal set maker for a random causal set.

# Arguments:
- `n`: number of elements in the causal set
- `config`: configuration dictionary
- `rng`: random number generator

# Keyword arguments:
- `config`: configuration dictionary

# Returns
- causal set (BitArrayCauset)
- size of inserted KR-order relative to size of causal set
"""
function (mcm::MergedCsetMaker)(
    n::Int64,
    rng::Random.AbstractRNG;
    config::Union{AbstractDict,Nothing} = nothing,
)::Tuple{CausalSets.BitArrayCauset,Float64}

    o = rand(rng, mcm.order_distribution)
    r = rand(rng, mcm.r_distribution)
    n2rel = rand(rng, mcm.n2_rel_distribution)
    l = rand(rng, mcm.link_prob_distribution)

    cset, success, sprinkling =
        insert_KR_into_manifoldlike(n, o, r, l; rng = rng, n2_rel = n2rel)

    return cset, n2rel
end

csetfactory_schema = JSONSchema.Schema("""
        	{
        	  "\$schema": "http://json-schema.org/draft-06/schema#",
        	  "title": "QuantumGrav Cset Factory Config",
        	  "type": "object",
        	  "additionalProperties": true,
        	  "properties": {
        		"polynomial": {
        		  "type": "object",
        		  "properties": {},
        		  "additionalProperties": true
        		},
        		"random": {
        		  "type": "object",
        		  "properties": {},
        		  "additionalProperties": true
        		},
        		"layered": {
        		  "type": "object",
        		  "properties": {},
        		  "additionalProperties": true
        		},
        		"merged": {
        		  "type": "object",
        		  "properties": {},
        		  "additionalProperties": true
        		},
        		"complex_topology": {
        		  "type": "object",
        		  "properties": {},
        		  "additionalProperties": true
        		},
        		"destroyed": {
        		  "type": "object",
        		  "properties": {},
        		  "additionalProperties": true
        		},
        		"grid": {
        		  "type": "object",
        		  "properties": {},
        		  "additionalProperties": true
        		},
        		"seed": { "type": "integer" },
        		"num_datapoints": { "type": "integer", "minimum": 0 },
        		"csetsize_distr_args": {
        			"type": "array",
        			"items": { "type": "integer" }
        		},
        		"csetsize_distr_kwargs": {
        			"type": "object",
        			"additionalProperties": true
        		},
        		"csetsize_distr": {"type": "string"},
            "coarse_graining_distribution": {"type": "string"},
            "coarse_graining_distribution_args": {
        			"type": "array",
        			"items": { "type": "number" }
        		},
        		"coarse_graining_distribution_kwargs": {
        			"type": "object",
        			"additionalProperties": true
        		},
        		"output": { "type": "string" },
        		"cset_type": {
        		  "oneOf": [
        			{ "type": "string" },
        			{
        			  "type": "array",
        			  "items": { "type": "string" }
        			}
        		  ]
        		}
        	  },
        	  "required": [
        		"polynomial",
        		"random",
        		"layered",
        		"merged",
        		"complex_topology",
        		"destroyed",
        		"grid",
        		"seed",
        		"num_datapoints",
        		"csetsize_distr",
        		"csetsize_distr_args",
        		"cset_type",
        		"output"
        	  ]
        	}
        """)

"""
	CsetFactory

The `CsetFactory` struct serves as a container for generating causal sets (csets).
It holds the configuration, random number generator, and distribution information required to create csets,
and provides access to specialized factory functions for different cset types.

# Fields:
- `npoint_distribution::Distributions.Distribution`: Distribution object for drawing number of elements in a cset
- `conf::AbstractDict`: config dictionary
- `rng::Random.AbstractRNG`: random number generator to use
- `cset_makers::AbstractDict`: dict to hold all the different cset factory methods
"""
struct CsetFactory
    npoint_distribution::Distributions.Distribution
    conf::AbstractDict
    rng::Random.AbstractRNG
    cset_makers::AbstractDict
end

"""
	CsetFactory(config::AbstractDict)

Create a new CsetFactory instance that bundles all the different cset factories into one object
"""
function CsetFactory(config::AbstractDict)
    validate_config(csetfactory_schema, config)

    npoint_distribution = build_distr(config, "csetsize_distr")
    rng = Random.Xoshiro(config["seed"])
    cset_makers = Dict(
        "random" => RandomCsetMaker(config["random"]),
        "complex_topology" => ComplexTopCsetMaker(config["complex_topology"]),
        "merged" => MergedCsetMaker(config["merged"]),
        "merged_ambiguous" =>
            MergedCsetMaker(get(config, "merged_ambiguous", config["merged"])),
        "polynomial" => PolynomialCsetMaker(config["polynomial"]),
        "layered" => LayeredCsetMaker(config["layered"]),
        "grid" => GridCsetMakerPolynomial(config["grid"]),
        "destroyed" => DestroyedCsetMaker(config["destroyed"]),
        "destroyed_ambiguous" =>
            DestroyedCsetMaker(get(config, "destroyed_ambiguous", config["destroyed"])),
    )
    return CsetFactory(npoint_distribution, config, rng, cset_makers)
end


"""
	cf::CsetFactory(csetname::String, n::Int64, rng::Random.AbstractRNG; config::Union{Dict, Nothing} = nothing)

Create a new cset, accessing the specialized factory functors held by the caller.

# Arguments:
- `csetname`: name of cset to create.
- `n`: number of events in the cset to create
- `rng`: rng to use for any stochastic part of the cset creation
- `config`: config that defines cset parameters (optional). Defaults to nothing
"""
function (cf::CsetFactory)(csetname::String, n::Int64, rng::Random.AbstractRNG;)
    cset_return = cf.cset_makers[csetname](n, rng; config = cf.conf[csetname])

    # make the csetfactory return a cset and additional args or a dummy thereof always.
    if cset_return isa Tuple
        return cset_return
    else
        return cset_return, nothing
    end
end

"""
	encode_csettype(config)

Encode known cset types into numeric scheme.
"""
encode_csettype = Dict(
    "polynomial" => 1,
    "layered" => 2,
    "random" => 3,
    "grid" => 4,
    "destroyed" => 5,
    "destroyed_ambiguous" => 6,
    "merged" => 7,
    "merged_ambiguous" => 8,
    "complex_topology" => 9,
)
