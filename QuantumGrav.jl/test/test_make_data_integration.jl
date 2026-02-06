@testsnippet config_roundtrip begin
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

@testsnippet create_data begin

    function make_data(cset)::Dict
        adj = transpose(hcat(cset.future_relations...)) |> Matrix{Int64}
        in_deg = sum(adj, dims = 1)[1, :]
        out_deg = sum(adj, dims = 2)[:, 1]
        return Dict("adj" => adj, "in_degree" => in_deg, "out_degree" => out_deg)
    end

    return make_data
end


@testitem "test_datacreation_and_saving_roundtrip" tags = [:cset_creation] setup =
    [config_roundtrip, create_data] begin
    using Random
    using Distributions
    using QuantumGrav
    using Zarr


    rng = Random.Xoshiro(1234)
    csetfactory = QuantumGrav.CsetFactory(cfg)

    if isdir(joinpath(tempdir(), "data.zarr"))
        rm(joinpath(tempdir(), "data.zarr"), recursive = true)
    end

    file = DirectoryStore(joinpath(tempdir(), "data.zarr"))
    root = zgroup(file, "")

    for i in [1, 2, 3]
        cset, _ = csetfactory("polynomial", 64, rng)
        data = make_data(cset)
        dict_to_zarr(root, Dict("data_$(i)"=>data))
    end

    file = nothing
    root = nothing
    file = DirectoryStore(joinpath(tempdir(), "data.zarr"))

    for i in [1, 2, 3]
        adj = Zarr.zopen(file, "r"; path = "data_$(i)/adj")
        indeg = Zarr.zopen(file, "r"; path = "data_$(i)/in_degree")
        outdeg = Zarr.zopen(file, "r"; path = "data_$(i)/out_degree")

        @test adj[:, :] isa Matrix{Int64}
        @test adj.metadata.shape[] == (64, 64)
        @test size(adj) == (64, 64)

        @test indeg[:] isa Vector
        @test indeg.metadata.shape[] == (64,)
        @test size(indeg) == (64,)

        @test outdeg[:] isa Vector
        @test outdeg.metadata.shape[] == (64,)
        @test size(outdeg) == (64,)
    end

    if isdir(joinpath(tempdir(), "data.zarr"))
        rm(joinpath(tempdir(), "data.zarr"), recursive = true)
    end
end
