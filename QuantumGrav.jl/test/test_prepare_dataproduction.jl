@testsnippet prepare_config begin
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

@testsnippet run_dataproduction begin
    using Distributed

    # add processes
    addprocs(4; exeflags = ["--threads=2", "--optimize=3"], enable_threaded_blas = true)

    # use @everywhere to include necessary modules on all workers
    @everywhere using QuantumGrav
    @everywhere using Random
    @everywhere using YAML
    @everywhere using Zarr
    @everywhere using LinearAlgebra
    @everywhere using Dates
    @everywhere using CausalSets


    @everywhere @eval Main function make_data(factory::CsetFactory)
        n = rand(factory.rng, factory.npoint_distribution)
        cset, _ = factory("random", n, factory.rng)
        return Dict("n" => cset.atom_count)
    end
    # make a temporary output path
    targetpath = mktempdir()

    try
        # read the default config and modify it to use the temporary output path
        # and produce more data. This serves as a dummy user config
        defaultconfigpath =
            joinpath(dirname(dirname(@__DIR__)), "configs", "createdata_default.yaml")
        cfg = YAML.load_file(defaultconfigpath)
        cfg["output"] = targetpath
        cfg["num_datapoints"] = 9

        configpath = joinpath(targetpath, "config.yaml")

        # .. then write back again
        open(configpath, "w") do io
            YAML.write(io, cfg)
        end

        # produce 20 data points using multiprocessing
        # hole max. 10 datapoints in the writing queue at once
        # and use the make data function that we had defined above
        QuantumGrav.produce_data(3, configpath, Main.make_data)

    finally
        println("done with produce_data_mp_testitem.jl")
        rmprocs(workers()...)
    end
    return targetpath
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


@testitem "setup_config_test" tags = [:preparation] begin
    import YAML

    # success: load default when configpath is nothing
    let cfg = QuantumGrav.setup_config(nothing)
        @test isa(cfg, Dict)
        @test haskey(cfg, "seed")
        @test haskey(cfg, "num_datapoints")
        @test haskey(cfg, "cset_type")
        @test cfg["num_datapoints"] == 5
        @test cfg["seed"] == 42
        @test cfg["cset_type"] == "polynomial"
    end
end

@testitem "setup_config_test_override" tags = [:preparation] begin
    using YAML
    # success: merge user config overriding a default value
    mktempdir() do tmp
        cfgpath = joinpath(tmp, "override.yaml")
        # override a couple of keys from default
        user_cfg = Dict(
            "seed" => 1234,
            "num_datapoints" => 7,
            "cset_type" => "random",
            "output" => tmp,
        )
        YAML.write_file(cfgpath, user_cfg)

        cfg = QuantumGrav.setup_config(cfgpath)
        @test cfg["seed"] == 1234
        @test cfg["num_datapoints"] == 7
        @test cfg["cset_type"] == "random"
        # Ensure other defaults still present
        @test haskey(cfg, "output")
        @test cfg["output"] == tmp
    end
end

@testitem "setup_config_test_path_normalization" tags = [:preparation] begin
    using YAML
    # success: path normalization works with ~ and relative paths
    mktempdir() do tmp
        # create in tmp and pass relative path from within tmp
        cfgname = "rel_override.yaml"
        cfgpath = joinpath(tmp, cfgname)
        YAML.write_file(cfgpath, Dict("seed" => 999))
        cd(tmp) do
            cfg = QuantumGrav.setup_config(cfgname)
            @test cfg["seed"] == 999
        end
    end
end

@testitem "setup_config_test_failure" tags = [:preparation] begin
    using YAML: YAML
    # failure: missing file throws ArgumentError
    @test_throws ArgumentError QuantumGrav.setup_config("/path/that/does/not/exist.yaml")
end

@testitem "setup_multiprocessing_test" tags = [:multiprocessing] setup=[prepare_config] begin
    using Distributed
    addprocs(4; exeflags = ["--threads=2", "--optimize=3"], enable_threaded_blas = true)

    @everywhere using QuantumGrav
    @everywhere using Random

    try
        factories = setup_multiprocessing(cfg)

        # global RNGs on each worker
        rng_global_res = Dict(p => [] for p in workers())
        @sync for p in workers()
            # we have to use remotecall_eval here to avoid closures which would pull in local scope and with
            # that things that cannot be serialized
            rng_global_res[p] = remotecall_eval(Main, p, :(rand(1:100, 10)))
        end

        for (k1, v1) in rng_global_res
            for (k2, v2) in rng_global_res
                if k1 != k2
                    @test Set(v1) != Set(v2)
                end
            end
        end

        # local RNGs from factories
        rng_local_results = Dict(p => [] for p in workers())
        for p in workers()
            rng_local_results[p] = remotecall_eval(Main, p, :(
                begin

                    factory = take!($factories[$p])
                    local_rng = factory.rng
                    x = rand(local_rng, 1:100, 10)
                    put!($factories[$p], factory)
                    return x
                end
            ))
        end

        for (k1, v1) in rng_local_results
            for (k2, v2) in rng_global_res
                if k1 != k2
                    @test Set(v1) != Set(v2)
                end
            end
        end
    finally
        rmprocs(workers()...)
    end
end

@testitem "test_mp_dataproduction" tags = [:dataproduction] setup=[run_dataproduction] begin
    import Zarr
    # check that data was produced
    zarr_files = filter(x -> occursin(".zarr", x), readdir(targetpath))
    @test length(zarr_files) == 1

    # test data content
    store = zarr_files[1]
    group = Zarr.zopen(joinpath(targetpath, store), "r"; path = "") # open root
    @test group isa Zarr.ZGroup
    @test length(keys(group.groups)) == 9 # 9 datapoints produced

    ns = []
    for i = 1:9
        @test "cset_$i" in keys(group.groups)
        @test "n" in keys(group.groups["cset_$i"].arrays)
        n = group.groups["cset_$i"].arrays["n"][1]
        push!(ns, n)
    end
    @test 7 < length(unique(ns)) <= 9 # generally expected to have high uniqueness
end


@testitem "test_mp_dataproduction_throws" tags=[:dataproduction] setup=[run_dataproduction] begin
    @eval Main function make_data(factory::CsetFactory)
        cset, _ = factory("random", 32, factory.rng)
        return Dict("n" => cset.atom_count)
    end

    @test_throws ErrorException QuantumGrav.produce_data(3, nothing, Main.make_data)
end
