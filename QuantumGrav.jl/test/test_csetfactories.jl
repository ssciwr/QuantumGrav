
using TestItems

@testsnippet importModules begin
    using QuantumGrav
    using TestItemRunner
    using CausalSets
    using SparseArrays
    using Random
    using Distributions
end

@testsnippet config begin
    cfg = Dict(
        "polynomial" => Dict(
            "order_distribution" => "DiscreteUniform",
            "order_distribution_args" => [2, 8],
            "order_distribution_kwargs" => Dict(),
            "r_distribution" => "Normal",
            "r_distribution_args" => [4.0, 2.0],
            "r_distribution_kwargs" => Dict(),
        ),
        "random" => Dict(
            "connectivity_distribution" => "Cauchy",
            "connectivity_distribution_args" => [0.5, 0.2],
            "connectivity_distribution_kwargs" => Dict(),
            "num_tries" => 100,
            "abs_tol" => 1e-2,
            "rel_tol" => nothing,
        ),
        "layered" => Dict(
            "connectivity_distribution" => "Uniform",
            "connectivity_distribution_args" => [0.0, 1.0],
            "connectivity_distribution_kwargs" => Dict(),
            "stddev_distribution" => "Normal",
            "stddev_distribution_args" => [0.3, 0.1],
            "stddev_distribution_kwargs" => Dict(),
            "layer_distribution" => "DiscreteUniform",
            "layer_distribution_args" => [2, 20],
            "layer_distribution_kwargs" => Dict(),
        ),
        "merged" => Dict(
            "order_distribution" => "DiscreteUniform",
            "order_distribution_args" => [2, 8],
            "order_distribution_kwargs" => Dict(),
            "r_distribution" => "Normal",
            "r_distribution_args" => [4.0, 2.0],
            "r_distribution_kwargs" => Dict(),
            "n2_rel_distribution" => "Cauchy",
            "n2_rel_distribution_args" => [2.0, 1.0],
            "n2_rel_distribution_kwargs" => Dict(),
            "connectivity_distribution" => "Beta",
            "connectivity_distribution_args" => [0.5, 0.1],
            "connectivity_distribution_kwargs" => Dict(),
            "link_prob_distribution" => "Normal",
            "link_prob_distribution_args" => [2.0, 1.5],
            "link_prob_distribution_kwargs" => Dict(),
        ),
        "complex_topology" => Dict(
            "order_distribution" => "DiscreteUniform",
            "order_distribution_args" => [2, 8],
            "order_distribution_kwargs" => Dict(),
            "r_distribution" => "Normal",
            "r_distribution_args" => [4.0, 2.0],
            "r_distribution_kwargs" => Dict(),
            "vertical_cut_distribution" => "Cauchy",
            "vertical_cut_distribution_args" => [2.0, 0.8],
            "vertical_cut_distribution_kwargs" => Dict(),
            "finite_cut_distribution" => "Normal",
            "finite_cut_distribution_args" => [4.0, 2.0],
            "finite_cut_distribution_kwargs" => Dict(),
            "tol" => 1e-2,
        ),
        "destroyed" => Dict(
            "order_distribution" => "DiscreteUniform",
            "order_distribution_args" => [2, 8],
            "order_distribution_kwargs" => Dict(),
            "r_distribution" => "Uniform",
            "r_distribution_args" => [4.0, 8.0],
            "r_distribution_kwargs" => Dict(),
            "flip_distribution" => "Normal",
            "flip_distribution_args" => [4.0, 2.0],
            "flip_distribution_kwargs" => Dict(),
        ),
        "grid" => Dict(
            "grid_distribution" => "DiscreteUniform",
            "grid_distribution_args" => [1, 6],
            "grid_distribution_kwargs" => Dict(),
            "rotate_distribution" => "Uniform",
            "rotate_distribution_args" => [0.0, 180.0],
            "rotate_distribution_kwargs" => Dict(),
            "gamma_distribution" => "Normal",
            "gamma_distribution_args" => [45.0, 15.0],
            "gamma_distribution_kwargs" => Dict(),
            "order_distribution" => "DiscreteUniform",
            "order_distribution_args" => [2, 8],
            "order_distribution_kwargs" => Dict(),
            "r_distribution" => "Uniform",
            "r_distribution_args" => [4.0, 8.0],
            "r_distribution_kwargs" => Dict(),
            "origin" => [0.5, 2.0],
            "quadratic" => Dict(
                "a_distribution" => "Uniform",
                "a_distribution_args" => [0.5, 5.5],
                "a_distribution_kwargs" => Dict(),
                "b_distribution" => "Normal",
                "b_distribution_args" => [1.0, 2.0],
                "b_distribution_kwargs" => Dict(),
            ),
            "rectangular" => Dict(
                "a_distribution" => "Normal",
                "a_distribution_args" => [3.0, 0.4],
                "a_distribution_kwargs" => Dict(),
                "b_distribution" => "Beta",
                "b_distribution_args" => [2.0, 1.2],
                "b_distribution_kwargs" => Dict(),
            ),
            "rhombic" => Dict(
                "a_distribution" => "Cauchy",
                "a_distribution_args" => [5.0, 1.0],
                "a_distribution_kwargs" => Dict(),
                "b_distribution" => "Uniform",
                "b_distribution_args" => [0.5, 5.5],
                "b_distribution_kwargs" => Dict(),
            ),
            "hexagonal" => Dict(
                "a_distribution" => "Uniform",
                "a_distribution_args" => [0.5, 5.5],
                "a_distribution_kwargs" => Dict(),
                "b_distribution" => "Normal",
                "b_distribution_args" => [2.0, 0.5],
                "b_distribution_kwargs" => Dict(),
            ),
            "triangular" => Dict(
                "a_distribution" => "Cauchy",
                "a_distribution_args" => [2.5, 0.7],
                "a_distribution_kwargs" => Dict(),
                "b_distribution" => "Normal",
                "b_distribution_args" => [3.3, 1.2],
                "b_distribution_kwargs" => Dict(),
            ),
            "oblique" => Dict(
                "a_distribution" => "Normal",
                "a_distribution_args" => [5.5, 0.7],
                "a_distribution_kwargs" => Dict(),
                "b_distribution" => "Logistic",
                "b_distribution_args" => [2.0, 1.0],
                "b_distribution_kwargs" => Dict(),
            ),
        ),
        "seed" => 42,
    )

    return cfg
end

@testitem "test_polynomial_factory_construction" tags = [:csetfactories] setup =
    [importModules, config] begin
    csetmaker = PolynomialCsetMaker(cfg["polynomial"])
    @test csetmaker.order_distribution isa Distributions.DiscreteUniform
    @test params(csetmaker.order_distribution) ==
          tuple(cfg["polynomial"]["order_distribution_args"]...)
    @test csetmaker.r_distribution isa Distributions.Normal
    @test params(csetmaker.r_distribution) ==
          tuple(cfg["polynomial"]["r_distribution_args"]...)
end

@testitem "test_polynomial_factory_broken_config" tags = [:csetfactories] setup =
    [importModules, config] begin
    broken_cfg = deepcopy(cfg)
    broken_cfg["polynomial"]["order_distribution"] = nothing
    @test_throws ArgumentError PolynomialCsetMaker(broken_cfg["polynomial"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["polynomial"]["r_distribution"] = nothing
    @test_throws ArgumentError PolynomialCsetMaker(broken_cfg["polynomial"])

end

@testitem "test_polynomial_factory_produce_csets" tags = [:csetfactories] setup =
    [importModules, config] begin
    csetmaker = PolynomialCsetMaker(cfg["polynomial"])
    rng = Random.Xoshiro(cfg["seed"])
    cset = csetmaker(25, rng)
    @test isnothing(cset) === false
    @test cset.atom_count == 25
end

@testitem "test_random_factory_construction" tags = [:csetfactories] setup =
    [importModules, config] begin
    csetmaker = RandomCsetMaker(cfg["random"])
    @test csetmaker.connectivity_distribution isa Distributions.Cauchy
    @test params(csetmaker.connectivity_distribution) ==
          tuple(cfg["random"]["connectivity_distribution_args"]...)
    @test csetmaker.num_tries == 100
end

@testitem "test_random_factory_broken_config" tags = [:csetfactories] setup =
    [importModules, config] begin
    broken_cfg = deepcopy(cfg)
    broken_cfg["random"]["connectivity_distribution"] = nothing
    @test_throws ArgumentError RandomCsetMaker(broken_cfg["random"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["random"]["num_tries"] = 0
    @test_throws ArgumentError RandomCsetMaker(broken_cfg["random"])
end

@testitem "test_random_factory_produce_csets" tags = [:csetfactories] setup =
    [importModules, config] begin
    csetmaker = RandomCsetMaker(cfg["random"])
    rng = Random.Xoshiro(cfg["seed"])
    cset = csetmaker(25, rng)
    @test isnothing(cset) === false
    @test cset.atom_count == 25
end

@testitem "test_layered_factory_construction" tags = [:csetfactories] setup =
    [importModules, config] begin
    csetmaker = LayeredCsetMaker(cfg["layered"])

    @test csetmaker.connectivity_distribution isa Distributions.Uniform
    @test params(csetmaker.connectivity_distribution) ==
          tuple(cfg["layered"]["connectivity_distribution_args"]...)

    @test csetmaker.stddev_distribution isa Distributions.Normal
    @test params(csetmaker.stddev_distribution) ==
          tuple(cfg["layered"]["stddev_distribution_args"]...)

    @test csetmaker.layer_distribution isa Distributions.DiscreteUniform
    @test params(csetmaker.layer_distribution) ==
          tuple(cfg["layered"]["layer_distribution_args"]...)
end

@testitem "test_layered_factory_broken_config" tags = [:csetfactories] setup =
    [importModules, config] begin
    broken_cfg = deepcopy(cfg)
    broken_cfg["layered"]["connectivity_distribution"] = nothing
    @test_throws ArgumentError LayeredCsetMaker(broken_cfg["layered"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["layered"]["stddev_distribution"] = nothing
    @test_throws ArgumentError LayeredCsetMaker(broken_cfg["layered"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["layered"]["layer_distribution"] = nothing
    @test_throws ArgumentError LayeredCsetMaker(broken_cfg["layered"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["layered"]["layer_distribution_args"] = nothing
    @test_throws ArgumentError LayeredCsetMaker(broken_cfg["layered"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["layered"]["stddev_distribution_args"] = nothing
    @test_throws ArgumentError LayeredCsetMaker(broken_cfg["layered"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["layered"]["connectivity_distribution_args"] = nothing
    @test_throws ArgumentError LayeredCsetMaker(broken_cfg["layered"])
end

@testitem "test_layered_factory_produce_csets" tags = [:csetfactories] setup =
    [importModules, config] begin
    csetmaker = LayeredCsetMaker(cfg["layered"])
    rng = Random.Xoshiro(cfg["seed"])
    cset = csetmaker(25, rng)
    @test isnothing(cset) === false
    @test cset.atom_count == 25
end

@testitem "test_destroyed_factory_construction" tags = [:csetfactories] setup =
    [importModules, config] begin
    csetmaker = DestroyedCsetMaker(cfg["destroyed"])

    @test csetmaker.order_distribution isa Distributions.DiscreteUniform
    @test params(csetmaker.order_distribution) ==
          tuple(cfg["destroyed"]["order_distribution_args"]...)

    @test csetmaker.r_distribution isa Distributions.Uniform
    @test params(csetmaker.r_distribution) ==
          tuple(cfg["destroyed"]["r_distribution_args"]...)

    @test csetmaker.flip_distribution isa Distributions.Normal
    @test params(csetmaker.flip_distribution) ==
          tuple(cfg["destroyed"]["flip_distribution_args"]...)

end

@testitem "test_destroyed_factory_broken_config" tags = [:csetfactories] setup =
    [importModules, config] begin
    broken_cfg = deepcopy(cfg)
    broken_cfg["destroyed"]["order_distribution"] = nothing
    @test_throws ArgumentError DestroyedCsetMaker(broken_cfg["destroyed"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["destroyed"]["r_distribution"] = nothing
    @test_throws ArgumentError DestroyedCsetMaker(broken_cfg["destroyed"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["destroyed"]["flip_distribution"] = nothing
    @test_throws ArgumentError DestroyedCsetMaker(broken_cfg["destroyed"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["destroyed"]["order_distribution_args"] = nothing
    @test_throws ArgumentError DestroyedCsetMaker(broken_cfg["destroyed"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["destroyed"]["r_distribution_args"] = nothing
    @test_throws ArgumentError DestroyedCsetMaker(broken_cfg["destroyed"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["destroyed"]["flip_distribution_args"] = nothing
    @test_throws ArgumentError DestroyedCsetMaker(broken_cfg["destroyed"])

end

@testitem "test_destroyed_factory_produce_csets" tags = [:csetfactories] setup =
    [importModules, config] begin
    csetmaker = DestroyedCsetMaker(cfg["destroyed"])
    rng = Random.Xoshiro(cfg["seed"])
    cset = csetmaker(25, rng)
    @test isnothing(cset) === false
    @test cset.atom_count == 25
end

@testitem "test_merged_factory_construction" tags = [:csetfactories] setup =
    [importModules, config] begin
    csetmaker = MergedCsetMaker(cfg["merged"])
    @test csetmaker.order_distribution isa Distributions.DiscreteUniform
    @test params(csetmaker.order_distribution) ==
          tuple(cfg["merged"]["order_distribution_args"]...)

    @test csetmaker.r_distribution isa Distributions.Normal
    @test params(csetmaker.r_distribution) == tuple(cfg["merged"]["r_distribution_args"]...)

    @test csetmaker.n2_rel_distribution isa Distributions.Cauchy
    @test params(csetmaker.n2_rel_distribution) ==
          tuple(cfg["merged"]["n2_rel_distribution_args"]...)

    @test csetmaker.connectivity_distribution isa Distributions.Beta
    @test params(csetmaker.connectivity_distribution) ==
          tuple(cfg["merged"]["connectivity_distribution_args"]...)

    @test csetmaker.link_prob_distribution isa Distributions.Normal
    @test params(csetmaker.link_prob_distribution) ==
          tuple(cfg["merged"]["link_prob_distribution_args"]...)
end

@testitem "test_merged_factory_broken_config" tags = [:csetfactories] setup =
    [importModules, config] begin
    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["order_distribution"] = nothing
    @test_throws ArgumentError MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["r_distribution"] = nothing
    @test_throws ArgumentError MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["n2_rel_distribution"] = nothing
    @test_throws ArgumentError MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["link_prob_distribution"] = nothing
    @test_throws ArgumentError MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["connectivity_distribution"] = nothing
    @test_throws ArgumentError MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["order_distribution_args"] = nothing
    @test_throws ArgumentError MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["r_distribution_args"] = nothing
    @test_throws ArgumentError MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["n2_rel_distribution_args"] = nothing
    @test_throws ArgumentError MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["link_prob_distribution_args"] = nothing
    @test_throws ArgumentError MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["connectivity_distribution_args"] = nothing
    @test_throws ArgumentError MergedCsetMaker(broken_cfg["merged"])

end

@testitem "test_merged_factory_produce_csets" tags = [:csetfactories] setup =
    [importModules, config] begin
    csetmaker = MergedCsetMaker(cfg["merged"])
    rng = Random.Xoshiro(cfg["seed"])
    cset = csetmaker(25, rng)
    @test isnothing(cset) === false
    @test cset.atom_count > 25
end

@testitem "test_complex_topology_factory_construction" tags = [:csetfactories] setup =
    [importModules, config] begin
    csetmaker = ComplexTopCsetMaker(cfg["complex_topology"])

    @test csetmaker.order_distribution isa Distributions.DiscreteUniform
    @test params(csetmaker.order_distribution) ==
          tuple(cfg["complex_topology"]["order_distribution_args"]...)

    @test csetmaker.r_distribution isa Distributions.Normal
    @test params(csetmaker.r_distribution) ==
          tuple(cfg["complex_topology"]["r_distribution_args"]...)

    @test csetmaker.vertical_cut_distribution isa Distributions.Cauchy
    @test params(csetmaker.vertical_cut_distribution) ==
          tuple(cfg["complex_topology"]["vertical_cut_distribution_args"]...)

    @test csetmaker.finite_cut_distribution isa Distributions.Normal
    @test params(csetmaker.finite_cut_distribution) ==
          tuple(cfg["complex_topology"]["finite_cut_distribution_args"]...)

end

@testitem "test_complex_topology_factory_broken_config" tags = [:csetfactories] setup =
    [importModules, config] begin
    broken_cfg = deepcopy(cfg)
    broken_cfg["complex_topology"]["order_distribution"] = nothing
    @test_throws ArgumentError ComplexTopCsetMaker(broken_cfg["complex_topology"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["complex_topology"]["r_distribution"] = nothing
    @test_throws ArgumentError ComplexTopCsetMaker(broken_cfg["complex_topology"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["complex_topology"]["vertical_cut_distribution"] = nothing
    @test_throws ArgumentError ComplexTopCsetMaker(broken_cfg["complex_topology"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["complex_topology"]["finite_cut_distribution"] = nothing
    @test_throws ArgumentError ComplexTopCsetMaker(broken_cfg["complex_topology"])
end

@testitem "test_complex_topology_factory_produce_csets" tags = [:csetfactories] setup =
    [importModules, config] begin
    csetmaker = ComplexTopCsetMaker(cfg["complex_topology"])
    rng = Random.Xoshiro(cfg["seed"])
    cset = csetmaker(25, rng)
    @test isnothing(cset) === false
    @test cset.atom_count <= 25
end

@testitem "test_grid_factory_construction" tags = [:csetfactories] setup =
    [importModules, config] begin
    csetmaker = GridCsetMakerPolynomial(cfg["grid"])

    @test csetmaker.grid_distribution isa Distributions.DiscreteUniform
    @test params(csetmaker.grid_distribution) ==
          tuple(cfg["grid"]["grid_distribution_args"]...)

    @test csetmaker.rotate_distribution isa Distributions.Uniform
    @test params(csetmaker.rotate_distribution) ==
          tuple(cfg["grid"]["rotate_distribution_args"]...)

    @test csetmaker.gamma_distribution isa Distributions.Normal
    @test params(csetmaker.gamma_distribution) ==
          tuple(cfg["grid"]["gamma_distribution_args"]...)

    @test csetmaker.order_distribution isa Distributions.DiscreteUniform
    @test params(csetmaker.order_distribution) ==
          tuple(cfg["grid"]["order_distribution_args"]...)

    @test csetmaker.r_distribution isa Distributions.Uniform
    @test params(csetmaker.r_distribution) == tuple(cfg["grid"]["r_distribution_args"]...)

    @test csetmaker.origin == [0.5, 2.0]

    @test haskey(csetmaker.grid_lookup, 1)
    @test haskey(csetmaker.grid_lookup, 2)
    @test haskey(csetmaker.grid_lookup, 3)
    @test haskey(csetmaker.grid_lookup, 4)
    @test haskey(csetmaker.grid_lookup, 5)
    @test haskey(csetmaker.grid_lookup, 6)
end

@testitem "test_grid_factory_broken_config" tags = [:csetfactories] setup =
    [importModules, config] begin
    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["grid_distribution"] = nothing
    @test_throws ArgumentError GridCsetMakerPolynomial(broken_cfg["grid"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["rotate_distribution"] = nothing
    @test_throws ArgumentError GridCsetMakerPolynomial(broken_cfg["grid"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["gamma_distribution"] = nothing
    @test_throws ArgumentError GridCsetMakerPolynomial(broken_cfg["grid"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["order_distribution"] = nothing
    @test_throws ArgumentError GridCsetMakerPolynomial(broken_cfg["grid"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["r_distribution"] = nothing
    @test_throws ArgumentError GridCsetMakerPolynomial(broken_cfg["grid"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["grid_distribution_args"] = nothing
    @test_throws ArgumentError GridCsetMakerPolynomial(broken_cfg["grid"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["rotate_distribution_args"] = nothing
    @test_throws ArgumentError GridCsetMakerPolynomial(broken_cfg["grid"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["gamma_distribution_args"] = nothing
    @test_throws ArgumentError GridCsetMakerPolynomial(broken_cfg["grid"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["order_distribution_args"] = nothing
    @test_throws ArgumentError GridCsetMakerPolynomial(broken_cfg["grid"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["r_distribution_args"] = nothing
    @test_throws ArgumentError GridCsetMakerPolynomial(broken_cfg["grid"])
end

@testitem "test_grid_factory_produce_csets" tags = [:csetfactories] setup =
    [importModules, config] begin
    # Test all grid types: quadratic, rectangular, rhombic, hexagonal, triangular, oblique
    grid_types =
        ["quadratic", "rectangular", "rhombic", "hexagonal", "triangular", "oblique"]

    rng = Random.Xoshiro(cfg["seed"])

    for grid_type in grid_types
        csetmaker = GridCsetMakerPolynomial(cfg["grid"])

        # Produce a cset for this grid type
        cset = csetmaker(25, rng, cfg["grid"]; grid = grid_type)
        @test isnothing(cset) === false
        @test cset.atom_count == 25
    end

    csetmaker = GridCsetMakerPolynomial(cfg["grid"])
    cset = csetmaker(25, rng, cfg["grid"]) # randomly chosen grid type
    @test isnothing(cset) === false
    @test cset.atom_count == 25

end
