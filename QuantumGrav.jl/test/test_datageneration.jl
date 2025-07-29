using TestItems

@testsnippet importModules begin
    import QuantumGrav
    import CausalSets
    import SparseArrays
    import Distributions
    import Random
    import Graphs
    import HDF5
    import YAML
end

@testsnippet makeData begin
    import CausalSets
    import QuantumGrav
    import SparseArrays
    import Distributions
    import Graphs
    import HDF5
    import YAML

    function MockData(n)
        manifold = CausalSets.MinkowskiManifold{2}()
        boundary = CausalSets.CausalDiamondBoundary{2}(1.0)
        sprinkling = CausalSets.generate_sprinkling(manifold, boundary, Int(n))
        cset = CausalSets.BitArrayCauset(manifold, sprinkling)
        return cset, sprinkling
    end

    cset_empty, sprinkling_empty = MockData(0)

    cset_links, sprinkling_links = MockData(100)
end


# Test makelink_matrix
@testitem "test_make_link_matrix" tags = [:featuregeneration] setup = [makeData] begin
    link_matrix_empty = QuantumGrav.make_link_matrix(cset_empty)

    @test link_matrix_empty == SparseArrays.spzeros(Float32, 0, 0)

    link_matrix = QuantumGrav.make_link_matrix(cset_links)
    @test all(link_matrix .== 0.0) == false
    @test all(link_matrix .== 1.0) == false
    @test size(link_matrix) == (100, 100)

    for i = 1:100
        for j = 1:100
            @test link_matrix[i, j] == Float32(CausalSets.is_link(cset_links, i, j))
        end
    end
end


@testitem "test_make_cardinality_matrix" tags = [:featuregeneration] setup = [makeData] begin
    @test_throws "The causal set must not be empty." QuantumGrav.make_cardinality_matrix(
        cset_empty,
    )

    # Test case 2: Causal set with some cardinalities
    cardinality_matrix_links = QuantumGrav.make_cardinality_matrix(cset_links)
    expected_matrix = SparseArrays.spzeros(Float32, 100, 100)
    @test 0 < SparseArrays.nnz(cardinality_matrix_links) <= 100 * 100


    cardinality_matrix_links_mt =
        QuantumGrav.make_cardinality_matrix(cset_links, multithreading = true)
    @test cardinality_matrix_links_mt == cardinality_matrix_links
end


@testitem "make_adj" tags = [:featuregeneration] setup = [makeData] begin
    adj = QuantumGrav.make_adj(cset_links; type = Float32)
    @test size(adj) == (100, 100)
    @test all(adj .== 0.0) == false
    @test all(adj .== 1.0) == false

    # alternative approach to make the adjacency matrix that uses Graphs.jl's 
    # capability
    test_adj =
        cset_links |>
        QuantumGrav.make_link_matrix |>
        Graphs.SimpleDiGraph |>
        Graphs.transitiveclosure |>
        Graphs.adjacency_matrix |>
        SparseArrays.SparseMatrixCSC{Float32}

    @test all(adj .== test_adj) == true

    @test_throws ArgumentError QuantumGrav.make_adj(cset_empty; type = Float32)
end

@testitem "test_max_pathlen" tags = [:featuregeneration] setup = [makeData] begin

    adj =
        cset_links |>
        QuantumGrav.make_link_matrix |>
        Graphs.SimpleDiGraph |>
        Graphs.transitiveclosure |>
        Graphs.adjacency_matrix |>
        SparseArrays.SparseMatrixCSC{Float32}

    max_path = QuantumGrav.max_pathlen(adj, collect(1:cset_links.atom_count), 1)

    sdg = Graphs.SimpleDiGraph(adj)
    max_path_expected =
        Graphs.dag_longest_path(sdg, topological_order = collect(1:cset_links.atom_count))

    @test max_path > 1
    @test max_path <= cset_links.atom_count
    @test max_path <= length(max_path_expected)
end

@testitem "test_calculate_angles" tags = [:featuregeneration] setup = [makeData] begin

    sprinkling_links = Float32.(stack(collect.(sprinkling_links), dims = 1))

    angles = QuantumGrav.calculate_angles(
        sprinkling_links,
        1,
        cset_links.future_relations[1],
        type = Float32,
    )
    n_neighbors = length(cset_links.future_relations[1])
    @test length(angles) == n_neighbors^2 - 3 * n_neighbors + 2

    # Check that the angles are calculated correctly
    for angle in angles
        @test 0.0 <= angle <= π
    end

    # use multithreading
    angles_mt = QuantumGrav.calculate_angles(
        sprinkling_links,
        1,
        cset_links.future_relations[1],
        type = Float32,
        multithreading = true,
    )

    @test length(angles_mt) == n_neighbors^2 - 3 * n_neighbors + 2


    # Check that the angles are calculated correctly
    for angle in angles_mt
        @test 0.0 <= angle <= π
    end

end


@testitem "test_calculate_distances" tags = [:featuregeneration] setup = [makeData] begin
    sprinkling_links = Float32.(stack(collect.(sprinkling_links), dims = 1))

    # sprinkling with 100 points
    distances = QuantumGrav.calculate_distances(
        sprinkling_links,
        1,
        cset_links.future_relations[1];
        type = Float32,
        multithreading = false,
    )

    @test length(distances) == length(cset_links.future_relations[1]) - 1
    @test all(distances .>= 0.0)

    distances = QuantumGrav.calculate_distances(
        sprinkling_links,
        1,
        cset_links.future_relations[1];
        type = Float32,
        multithreading = true,
    )

    @test length(distances) == length(cset_links.future_relations[1]) - 1
    @test all(distances .>= 0.0)
end

@testitem "test_make_data" tags = [:featuregeneration] setup = [importModules] begin

    bad_config = Dict(
        "num_datapoints" => 10,
        "file_mode" => "w",
        "num_threads" => Threads.nthreads(),
        "seed" => 42,
    )

    wrong_config = Dict(
        "num_datapoints" => 10,
        "output" => joinpath(tempdir(), "test_data"),
        "file_mode" => "w",
        "num_threads" => 2 * Threads.nthreads(),
        "seed" => 42,
    )

    config = Dict(
        "num_datapoints" => 10,
        "output" => joinpath(tempdir(), "test_data"),
        "file_mode" => "w",
        "num_threads" => Threads.nthreads(),
        "seed" => 42,
    )

    function transform(config, rng::Random.AbstractRNG)

        cset, sprinkling = QuantumGrav.make_simple_cset(
            "Minkowski",
            "CausalDiamond",
            100,
            2,
            rng;
            type = Float32,
        )
        adj = QuantumGrav.make_adj(cset; type = Float32)

        return Dict("adjacency_matrices" => Matrix(adj), "sprinkling" => sprinkling)
    end

    function prepare_output(file, config::Dict)
        dset = QuantumGrav.HDF5.create_dataset(
            file,
            "/adjacency_matrices",
            Float32,
            QuantumGrav.HDF5.dataspace((100, 100, 0), (100, 100, -1)),
            chunk = (100, 100, 1),
            deflate = 8,
        )
        close(dset)

        dset = QuantumGrav.HDF5.create_dataset(
            file,
            "/sprinkling",
            Float32,
            QuantumGrav.HDF5.dataspace((100, 2, 0), (100, 2, -1)),
            chunk = (100, 2, 1),
            deflate = 8,
        )
        close(dset)
    end

    function write_data(file, config::Dict, data::Dict)

        dset = QuantumGrav.HDF5.open_dataset(file, "/adjacency_matrices")
        old_size = size(dset)
        new_size =
            (old_size[1], old_size[2], old_size[3] + size(data["adjacency_matrices"], 3))

        QuantumGrav.HDF5.set_extent_dims(dset, new_size)

        dset[1:new_size[1], 1:new_size[2], (old_size[3]+1):new_size[3]] .=
            data["adjacency_matrices"]

        close(dset)
    end

    @test_throws ArgumentError QuantumGrav.make_data(
        transform,
        prepare_output,
        write_data;
        config = bad_config,
    )

    QuantumGrav.make_data(transform, prepare_output, write_data; config = config)
    outputcontent = readdir(config["output"])

    @test [occursin(".h5", file) for file in outputcontent] == [true]
    @test [occursin(".yaml", file) for file in outputcontent] == [true]
    @test [occursin(".jl", file) for file in outputcontent] == [true]

    file = [f for f in outputcontent if occursin(".h5", f)][1]

    QuantumGrav.HDF5.h5open(file, "r") do file
        @test haskey(file, "/adjacency_matrices")
        @test size(file["/adjacency_matrices"]) == (100, 100, 10)
    end

end
