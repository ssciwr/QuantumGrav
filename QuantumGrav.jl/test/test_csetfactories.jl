



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
            "max_iter" => 30,
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
            "n2_rel_distribution" => "Uniform",
            "n2_rel_distribution_args" => [0., 1.0],
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
            "flip_distribution" => "Uniform",
            "flip_distribution_args" => [0., 1.0],
            "flip_distribution_kwargs" => Dict(),
        ),
        "grid" => Dict(
            "grid_distribution" => "DiscreteUniform",
            "grid_distribution_args" => [1, 6],
            "grid_distribution_kwargs" => Dict(),
            "rotate_distribution" => "Uniform",
            "rotate_distribution_args" => [0.0, 180.0],
            "rotate_distribution_kwargs" => Dict(),
            "order_distribution" => "DiscreteUniform",
            "order_distribution_args" => [2, 8],
            "order_distribution_kwargs" => Dict(),
            "r_distribution" => "Uniform",
            "r_distribution_args" => [4.0, 8.0],
            "r_distribution_kwargs" => Dict(),
            "quadratic" => Dict(),
            "rectangular" => Dict(
                "segment_ratio_distribution" => "Beta",
                "segment_ratio_distribution_args" => [2.0, 1.2],
                "segment_ratio_distribution_kwargs" => Dict(),
            ),
            "rhombic" => Dict(
                "segment_ratio_distribution" => "Uniform",
                "segment_ratio_distribution_args" => [0.5, 5.5],
                "segment_ratio_distribution_kwargs" => Dict(),
            ),
            "hexagonal" => Dict(
                "segment_ratio_distribution" => "Normal",
                "segment_ratio_distribution_args" => [2.0, 0.5],
                "segment_ratio_distribution_kwargs" => Dict(),
            ),
            "triangular" => Dict(
                "segment_ratio_distribution" => "Normal",
                "segment_ratio_distribution_args" => [3.3, 1.2],
                "segment_ratio_distribution_kwargs" => Dict(),
            ),
            "oblique" => Dict(
                "segment_ratio_distribution" => "Logistic",
                "segment_ratio_distribution_args" => [2.0, 1.0],
                "segment_ratio_distribution_kwargs" => Dict(),
                "oblique_angle_distribution" => "Normal",
                "oblique_angle_distribution_args" => [45.0, 15.0],
                "oblique_angle_distribution_kwargs" => Dict(),
            ),
        ),
        "seed" => 42,
        "num_datapoints" => 5,
        "csetsize_distr_args" => [10, 20],
        "csetsize_distr" => "DiscreteUniform",
        "cset_type" => "polynomial",
        "output" => "./",
    )

    return cfg
end

@testitem "test_Csetfactory_works" tags=[:csetfactories] setup = [config] begin
    using Random: Random
    using Distributions: Distributions

    csetfactory = QuantumGrav.CsetFactory(cfg)
    @test csetfactory.rng isa Random.Xoshiro
    @test csetfactory.npoint_distribution isa Distributions.DiscreteUniform
    @test Distributions.params(csetfactory.npoint_distribution) ==
          tuple(cfg["csetsize_distr_args"]...)
    @test csetfactory.conf == cfg
    for key in [
        "random",
        "complex_topology",
        "merged",
        "polynomial",
        "layered",
        "grid",
        "destroyed",
    ]
        @test key in keys(csetfactory.cset_makers)
    end

    csetfactory.cset_makers["foo"] = (n, rng; conf = nothing) -> 42

    @test "foo" in keys(csetfactory.cset_makers)
end

@testitem "test_Csetfactory_broken_config" tags=[:csetfactories] setup = [config] begin

    broken_cfg = deepcopy(cfg)
    broken_cfg["output"] = nothing
    @test_throws ArgumentError QuantumGrav.CsetFactory(broken_cfg)

    broken_cfg = deepcopy(cfg)
    broken_cfg["unallowed_key"] = "blah"
    @test_throws ArgumentError QuantumGrav.CsetFactory(broken_cfg)

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"] = Dict()
    @test_throws ArgumentError QuantumGrav.CsetFactory(broken_cfg)
end


@testitem "test_Csetfactory_call" tags=[:csetfactories] setup = [config] begin
    using Random: Random
    using Distributions: Distributions
    using CausalSets: CausalSets

    rng = Random.Xoshiro(1234)
    csetfactory = QuantumGrav.CsetFactory(cfg)

    cset = csetfactory("random", 12, rng)

    @test cset isa CausalSets.BitArrayCauset
    @test cset.atom_count == 12
end

@testitem "test_polynomial_factory_construction" tags = [:csetfactories] setup = [config] begin
    using Distributions: Distributions

    csetmaker = QuantumGrav.PolynomialCsetMaker(cfg["polynomial"])
    @test csetmaker.order_distribution isa Distributions.DiscreteUniform
    @test Distributions.params(csetmaker.order_distribution) ==
          tuple(cfg["polynomial"]["order_distribution_args"]...)
    @test csetmaker.r_distribution isa Distributions.Normal
    @test Distributions.params(csetmaker.r_distribution) ==
          tuple(cfg["polynomial"]["r_distribution_args"]...)
end

@testitem "test_polynomial_factory_broken_config" tags = [:csetfactories] setup = [config] begin
    broken_cfg = deepcopy(cfg)
    broken_cfg["polynomial"]["order_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.PolynomialCsetMaker(broken_cfg["polynomial"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["polynomial"]["r_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.PolynomialCsetMaker(broken_cfg["polynomial"])

end

@testitem "test_polynomial_factory_produce_csets" tags = [:csetfactories] setup = [config] begin
    using Random: Random

    csetmaker = QuantumGrav.PolynomialCsetMaker(cfg["polynomial"])
    rng = Random.Xoshiro(cfg["seed"])
    cset, curvature_matrix = csetmaker(25, rng)
    @test isnothing(cset) === false
    @test cset.atom_count == 25
    @test length(curvature_matrix) == 25
end

@testitem "test_random_factory_construction" tags = [:csetfactories] setup = [config] begin
    using Distributions: Distributions

    csetmaker = QuantumGrav.RandomCsetMaker(cfg["random"])
    @test csetmaker.connectivity_distribution isa Distributions.Cauchy
    @test Distributions.params(csetmaker.connectivity_distribution) ==
          tuple(cfg["random"]["connectivity_distribution_args"]...)
    @test csetmaker.num_tries == 100
end

@testitem "test_random_factory_broken_config" tags = [:csetfactories] setup = [config] begin

    broken_cfg = deepcopy(cfg)
    broken_cfg["random"]["connectivity_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.RandomCsetMaker(broken_cfg["random"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["random"]["max_iter"] = 0
    @test_throws ArgumentError QuantumGrav.RandomCsetMaker(broken_cfg["random"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["random"]["num_tries"] = 0
    @test_throws ArgumentError QuantumGrav.RandomCsetMaker(broken_cfg["random"])
end

@testitem "test_random_factory_produce_csets" tags = [:csetfactories] setup = [config] begin
    using Random: Random

    csetmaker = QuantumGrav.RandomCsetMaker(cfg["random"])
    rng = Random.Xoshiro(cfg["seed"])
    cset = csetmaker(25, rng)
    @test isnothing(cset) === false
    @test cset.atom_count == 25
end

@testitem "test_layered_factory_construction" tags = [:csetfactories] setup = [config] begin
    using Distributions: Distributions

    csetmaker = QuantumGrav.LayeredCsetMaker(cfg["layered"])

    @test csetmaker.connectivity_distribution isa Distributions.Uniform
    @test Distributions.params(csetmaker.connectivity_distribution) ==
          tuple(cfg["layered"]["connectivity_distribution_args"]...)

    @test csetmaker.stddev_distribution isa Distributions.Normal
    @test Distributions.params(csetmaker.stddev_distribution) ==
          tuple(cfg["layered"]["stddev_distribution_args"]...)

    @test csetmaker.layer_distribution isa Distributions.DiscreteUniform
    @test Distributions.params(csetmaker.layer_distribution) ==
          tuple(cfg["layered"]["layer_distribution_args"]...)
end

@testitem "test_layered_factory_broken_config" tags = [:csetfactories] setup = [config] begin

    broken_cfg = deepcopy(cfg)
    broken_cfg["layered"]["connectivity_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.LayeredCsetMaker(broken_cfg["layered"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["layered"]["stddev_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.LayeredCsetMaker(broken_cfg["layered"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["layered"]["layer_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.LayeredCsetMaker(broken_cfg["layered"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["layered"]["layer_distribution_args"] = nothing
    @test_throws ArgumentError QuantumGrav.LayeredCsetMaker(broken_cfg["layered"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["layered"]["stddev_distribution_args"] = nothing
    @test_throws ArgumentError LayeredCsetMaker(broken_cfg["layered"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["layered"]["connectivity_distribution_args"] = nothing
    @test_throws ArgumentError QuantumGrav.LayeredCsetMaker(broken_cfg["layered"])
end

@testitem "test_layered_factory_produce_csets" tags = [:csetfactories] setup = [config] begin
    using Random: Random
    csetmaker = QuantumGrav.LayeredCsetMaker(cfg["layered"])
    rng = Random.Xoshiro(cfg["seed"])
    cset, layers = csetmaker(25, rng)
    @test isnothing(cset) === false
    @test cset.atom_count == 25
    @test layers >= 2 
end

@testitem "test_destroyed_factory_construction" tags = [:csetfactories] setup = [config] begin
    using Distributions: Distributions

    csetmaker = QuantumGrav.DestroyedCsetMaker(cfg["destroyed"])

    @test csetmaker.order_distribution isa Distributions.DiscreteUniform
    @test Distributions.params(csetmaker.order_distribution) ==
          tuple(cfg["destroyed"]["order_distribution_args"]...)

    @test csetmaker.r_distribution isa Distributions.Uniform
    @test Distributions.params(csetmaker.r_distribution) ==
          tuple(cfg["destroyed"]["r_distribution_args"]...)

    @test csetmaker.flip_distribution isa Distributions.Uniform
    @test Distributions.params(csetmaker.flip_distribution) ==
          tuple(cfg["destroyed"]["flip_distribution_args"]...)

end

@testitem "test_destroyed_factory_broken_config" tags = [:csetfactories] setup = [config] begin

    broken_cfg = deepcopy(cfg)
    broken_cfg["destroyed"]["order_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.DestroyedCsetMaker(broken_cfg["destroyed"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["destroyed"]["r_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.DestroyedCsetMaker(broken_cfg["destroyed"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["destroyed"]["flip_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.DestroyedCsetMaker(broken_cfg["destroyed"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["destroyed"]["order_distribution_args"] = nothing
    @test_throws ArgumentError QuantumGrav.DestroyedCsetMaker(broken_cfg["destroyed"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["destroyed"]["r_distribution_args"] = nothing
    @test_throws ArgumentError QuantumGrav.DestroyedCsetMaker(broken_cfg["destroyed"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["destroyed"]["flip_distribution_args"] = nothing
    @test_throws ArgumentError QuantumGrav.DestroyedCsetMaker(broken_cfg["destroyed"])

end

@testitem "test_destroyed_factory_produce_csets" tags = [:csetfactories] setup = [config] begin
    using Random: Random

    csetmaker = QuantumGrav.DestroyedCsetMaker(cfg["destroyed"])
    rng = Random.Xoshiro(cfg["seed"])
    cset, rel_num_flips = csetmaker(25, rng)
    @test isnothing(cset) === false
    @test cset.atom_count == 25
    @test rel_num_flips >= 0.0 && rel_num_flips <= 1.0
end

@testitem "test_merged_factory_construction" tags = [:csetfactories] setup = [config] begin
    using Distributions: Distributions

    csetmaker = QuantumGrav.MergedCsetMaker(cfg["merged"])
    @test csetmaker.order_distribution isa Distributions.DiscreteUniform
    @test Distributions.params(csetmaker.order_distribution) ==
          tuple(cfg["merged"]["order_distribution_args"]...)

    @test csetmaker.r_distribution isa Distributions.Normal
    @test Distributions.params(csetmaker.r_distribution) ==
          tuple(cfg["merged"]["r_distribution_args"]...)

    @test csetmaker.n2_rel_distribution isa Distributions.Uniform
    @test Distributions.params(csetmaker.n2_rel_distribution) ==
          tuple(cfg["merged"]["n2_rel_distribution_args"]...)

    @test csetmaker.connectivity_distribution isa Distributions.Beta
    @test Distributions.params(csetmaker.connectivity_distribution) ==
          tuple(cfg["merged"]["connectivity_distribution_args"]...)

    @test csetmaker.link_prob_distribution isa Distributions.Normal
    @test Distributions.params(csetmaker.link_prob_distribution) ==
          tuple(cfg["merged"]["link_prob_distribution_args"]...)
end

@testitem "test_merged_factory_broken_config" tags = [:csetfactories] setup = [config] begin

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["order_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["r_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["n2_rel_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["link_prob_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["connectivity_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["order_distribution_args"] = nothing
    @test_throws ArgumentError QuantumGrav.MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["r_distribution_args"] = nothing
    @test_throws ArgumentError QuantumGrav.MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["n2_rel_distribution_args"] = nothing
    @test_throws ArgumentError QuantumGrav.MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["link_prob_distribution_args"] = nothing
    @test_throws ArgumentError MergedCsetMaker(broken_cfg["merged"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["merged"]["connectivity_distribution_args"] = nothing
    @test_throws ArgumentError MergedCsetMaker(broken_cfg["merged"])

end

@testitem "test_merged_factory_produce_csets" tags = [:csetfactories] setup = [config] begin
    using Random: Random
    csetmaker = QuantumGrav.MergedCsetMaker(cfg["merged"])
    rng = Random.Xoshiro(cfg["seed"])
    cset, n2_rel = csetmaker(25, rng)
    @test isnothing(cset) === false
    @test cset.atom_count == 25
    @test n2_rel >= 0.0 && n2_rel <= 1.0
end

@testitem "test_complex_topology_factory_construction" tags = [:csetfactories] setup =
    [config] begin
    using Distributions: Distributions

    csetmaker = QuantumGrav.ComplexTopCsetMaker(cfg["complex_topology"])

    @test csetmaker.order_distribution isa Distributions.DiscreteUniform
    @test Distributions.params(csetmaker.order_distribution) ==
          tuple(cfg["complex_topology"]["order_distribution_args"]...)

    @test csetmaker.r_distribution isa Distributions.Normal
    @test Distributions.params(csetmaker.r_distribution) ==
          tuple(cfg["complex_topology"]["r_distribution_args"]...)

    @test csetmaker.vertical_cut_distribution isa Distributions.Cauchy
    @test Distributions.params(csetmaker.vertical_cut_distribution) ==
          tuple(cfg["complex_topology"]["vertical_cut_distribution_args"]...)

    @test csetmaker.finite_cut_distribution isa Distributions.Normal
    @test Distributions.params(csetmaker.finite_cut_distribution) ==
          tuple(cfg["complex_topology"]["finite_cut_distribution_args"]...)

end

@testitem "test_complex_topology_factory_broken_config" tags = [:csetfactories] setup =
    [config] begin

    broken_cfg = deepcopy(cfg)
    broken_cfg["complex_topology"]["order_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.ComplexTopCsetMaker(
        broken_cfg["complex_topology"],
    )

    broken_cfg = deepcopy(cfg)
    broken_cfg["complex_topology"]["r_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.ComplexTopCsetMaker(
        broken_cfg["complex_topology"],
    )

    broken_cfg = deepcopy(cfg)
    broken_cfg["complex_topology"]["vertical_cut_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.ComplexTopCsetMaker(
        broken_cfg["complex_topology"],
    )

    broken_cfg = deepcopy(cfg)
    broken_cfg["complex_topology"]["finite_cut_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.ComplexTopCsetMaker(
        broken_cfg["complex_topology"],
    )
end

@testitem "test_complex_topology_factory_produce_csets" tags = [:csetfactories] setup =
    [config] begin
    using Random: Random

    csetmaker = QuantumGrav.ComplexTopCsetMaker(cfg["complex_topology"])
    rng = Random.Xoshiro(cfg["seed"])
    cset, curvature_matrix = csetmaker(25, rng)
    @test isnothing(cset) === false
    @test cset.atom_count <= 25
    @test length(curvature_matrix) <= 25
end

@testitem "test_grid_factory_construction" tags = [:csetfactories] setup = [config] begin
    using Distributions: Distributions

    csetmaker = QuantumGrav.GridCsetMakerPolynomial(cfg["grid"])

    @test csetmaker.grid_distribution isa Distributions.DiscreteUniform
    @test Distributions.params(csetmaker.grid_distribution) ==
          tuple(cfg["grid"]["grid_distribution_args"]...)

    @test csetmaker.rotate_distribution isa Distributions.Uniform
    @test Distributions.params(csetmaker.rotate_distribution) ==
          tuple(cfg["grid"]["rotate_distribution_args"]...)


    @test csetmaker.order_distribution isa Distributions.DiscreteUniform
    @test Distributions.params(csetmaker.order_distribution) ==
          tuple(cfg["grid"]["order_distribution_args"]...)

    @test csetmaker.r_distribution isa Distributions.Uniform
    @test Distributions.params(csetmaker.r_distribution) ==
          tuple(cfg["grid"]["r_distribution_args"]...)

    @test haskey(csetmaker.grid_lookup, 1)
    @test haskey(csetmaker.grid_lookup, 2)
    @test haskey(csetmaker.grid_lookup, 3)
    @test haskey(csetmaker.grid_lookup, 4)
    @test haskey(csetmaker.grid_lookup, 5)
    @test haskey(csetmaker.grid_lookup, 6)
end

@testitem "test_grid_factory_broken_config" tags = [:csetfactories] setup = [config] begin

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["grid_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.GridCsetMakerPolynomial(broken_cfg["grid"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["rotate_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.GridCsetMakerPolynomial(broken_cfg["grid"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["order_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.GridCsetMakerPolynomial(broken_cfg["grid"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["r_distribution"] = nothing
    @test_throws ArgumentError QuantumGrav.GridCsetMakerPolynomial(broken_cfg["grid"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["grid_distribution_args"] = nothing
    @test_throws ArgumentError QuantumGrav.GridCsetMakerPolynomial(broken_cfg["grid"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["rotate_distribution_args"] = nothing
    @test_throws ArgumentError QuantumGrav.GridCsetMakerPolynomial(broken_cfg["grid"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["order_distribution_args"] = nothing
    @test_throws ArgumentError GridCsetMakerPolynomial(broken_cfg["grid"])

    broken_cfg = deepcopy(cfg)
    broken_cfg["grid"]["r_distribution_args"] = nothing
    @test_throws ArgumentError QuantumGrav.GridCsetMakerPolynomial(broken_cfg["grid"])
end

@testitem "test_grid_factory_produce_csets" tags = [:csetfactories] setup = [config] begin
    using Random: Random
    # Test all grid types: quadratic, rectangular, rhombic, hexagonal, triangular, oblique
    grid_types =
        ["quadratic", "rectangular", "rhombic", "hexagonal", "triangular", "oblique"]

    rng = Random.Xoshiro(cfg["seed"])

    for grid_type in grid_types
        csetmaker = QuantumGrav.GridCsetMakerPolynomial(cfg["grid"])

        # Produce a cset for this grid type
        cset, curvature_matrix, grid_type_out = csetmaker(25, rng, cfg["grid"]; grid = grid_type)
        @test isnothing(cset) === false
        @test cset.atom_count == 25
        @test grid_type == grid_type_out
    end

    csetmaker = QuantumGrav.GridCsetMakerPolynomial(cfg["grid"])
    cset, curvature_matrix, grid_type_out = csetmaker(25, rng, cfg["grid"]) # randomly chosen grid type
    @test isnothing(cset) === false
    @test cset.atom_count == 25
    @test length(curvature_matrix) == 25
    @test grid_type_out in grid_types

end
