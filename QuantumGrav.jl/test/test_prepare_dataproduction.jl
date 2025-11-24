
@testitem "check_copy_sourcecode" tags = [:graph_utils] begin
    import CausalSets
    mktempdir() do targetpath
        funcs = [CausalSets.cardinality_of, QuantumGrav.make_adj]
        @test length(filter(x -> occursin(".jl", x), readdir(targetpath))) == 0

        QuantumGrav.copy_sourcecode(funcs, targetpath)

        @test length(filter(x -> occursin(".jl", x), readdir(targetpath))) == 2
    end
end


@testitem "get_git_info" tags = [:graph_utils] begin
    import CausalSets
    config = Dict{String,Any}()

    QuantumGrav.get_git_info!(config)

    @test haskey(config, "QuantumGrav")
    @test haskey(config["QuantumGrav"], "git_source")
    @test haskey(config["QuantumGrav"], "git_branch")
    @test haskey(config["QuantumGrav"], "git_tree_hash")
end

@testitem "prepare_dataproduction" tags = [:graph_utils] begin
    import CausalSets
    mktempdir() do targetpath
        config = Dict{String,Any}(
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


    end
end


@testitem "prepare_dataproduction_throws" tags = [:graph_utils] begin
    import CausalSets
    mktempdir() do targetpath
        config = Dict{String,Any}(
            "num_datapoints" => 5,
            "output_format" => "zarr",
            "output" => targetpath,
        )

        funcs = [CausalSets.cardinality_of, QuantumGrav.make_adj]
        @test_throws ArgumentError QuantumGrav.prepare_dataproduction(config, funcs)

    end
end
