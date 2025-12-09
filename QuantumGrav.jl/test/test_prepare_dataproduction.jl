@testsnippet config begin
    cfg = Dict(
        "polynomial" => Dict(
            "order_distribution" => "DiscreteUniform",
            "order_distribution_args" => [2, 8],
            "order_distribution_kwargs" => Dict(),
            "r_distribution" => "Uniform",
            "r_distribution_args" => [2.1, 3.1],
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
            "r_distribution_args" => [4.0, 0.1],
            "r_distribution_kwargs" => Dict(),
            "n2_rel_distribution" => "Uniform",
            "n2_rel_distribution_args" => [0.0, 1.0],
            "n2_rel_distribution_kwargs" => Dict(),
            "connectivity_distribution" => "Beta",
            "connectivity_distribution_args" => [0.5, 0.1],
            "connectivity_distribution_kwargs" => Dict(),
            "link_prob_distribution" => "Normal",
            "link_prob_distribution_args" => [0.1, 0.05],
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
            "flip_distribution_args" => [0.0, 1.0],
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

@testitem "check_copy_sourcecode" tags = [:preparation] begin
    import CausalSets
    mktempdir() do targetpath
        funcs = [CausalSets.cardinality_of, QuantumGrav.make_adj]
        @test length(filter(x -> occursin(".jl", x), readdir(targetpath))) == 0

        QuantumGrav.copy_sourcecode(funcs, targetpath)

        @test length(filter(x -> occursin(".jl", x), readdir(targetpath))) == 2
    end
end


@testitem "get_git_info" tags = [:preparation] begin
    import CausalSets
    config = Dict()

    QuantumGrav.get_git_info!(config)

    @test haskey(config, "QuantumGrav")
    @test haskey(config["QuantumGrav"], "git_source")
    @test haskey(config["QuantumGrav"], "git_branch")
    @test haskey(config["QuantumGrav"], "git_tree_hash")
end

@testitem "prepare_dataproduction" tags = [:preparation] begin
    import CausalSets
    import Zarr
    mktempdir() do targetpath
        config = Dict(
            "num_datapoints" => 5,
            "seed" => 42,
            "output_format" => "zarr",
            "output" => targetpath,
        )
        @test length(filter(x -> occursin(".jl", x), readdir(targetpath))) == 0
        @test length(filter(x -> occursin(".yaml", x), readdir(targetpath))) == 0
        @test length(filter(x -> occursin(".zarr", x), readdir(targetpath))) == 0

        funcs = [CausalSets.cardinality_of, QuantumGrav.make_adj]
        QuantumGrav.prepare_dataproduction(config, funcs; name = "testdata")

        @test haskey(config, "QuantumGrav")
        zarr_files = filter(x -> occursin(".zarr", x), readdir(targetpath))
        @test length(filter(x -> occursin(".jl", x), readdir(targetpath))) == 2
        @test length(filter(x -> occursin(".yaml", x), readdir(targetpath))) == 1
        @test length(zarr_files) == 1
        @test occursin("testdata", zarr_files[1])

        # # test
        store = zarr_files[1]
        group = Zarr.zopen(joinpath(targetpath, store), "r"; path = "") # open root
        @test group isa Zarr.ZGroup

    end
end


@testitem "prepare_dataproduction_throws" tags = [:preparation] begin
    import CausalSets
    mktempdir() do targetpath
        config =
            Dict("num_datapoints" => 5, "output_format" => "zarr", "output" => targetpath)

        funcs = [CausalSets.cardinality_of, QuantumGrav.make_adj]
        @test_throws ArgumentError QuantumGrav.prepare_dataproduction(config, funcs)

    end
end

@testitem "setup_multiprocessing_test" tags = [:preparation] setup = [config] begin
    import Distributed
    @test length(Distributed.workers()) == 0
    Distributed.addprocs(2)
    QuantumGrav.setup_mp(cfg)

    @test 3 == 6 # check here that each worker has a csetfactory

end

@testitem "setup_config_test" tags = [:preparation] begin

end

@testitem "produce_data" tags=[:preparation] begin
    import CausalSets
    import Zarr

    mktempdir() do configbasepath
        cfgpath = joinpath(configbasepath, "config.yml")

        QuantumGrav.produce_data(2, 2, 2, 5)
    end
end
