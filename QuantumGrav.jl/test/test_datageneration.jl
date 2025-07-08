using TestItems

@testsnippet importModules begin
    import QuantumGrav
    import CausalSets
    import SparseArrays
    import Distributions
    import Random
    import Graphs
end

@testsnippet makeData begin
    import CausalSets
    import QuantumGrav
    import SparseArrays
    import Distributions
    import Graphs

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

@testitem "test_make_csets" tags = [:datageneration] setup = [importModules] begin
    rng = Random.MersenneTwister(42)
    type = Float32
    cset, sprinkling =
        QuantumGrav.make_cset("Minkowski", "CausalDiamond", 100, 2, rng, type)

    @test cset.atom_count == 100
    @test length(cset.future_relations) == 100
    @test length(cset.past_relations) == 100
    @test size(sprinkling) == (100, 2)

    cset, sprinkling = QuantumGrav.make_cset("Random", "BoxBoundary", 100, 3, rng, type)

    @test cset.atom_count == 100
    @test length(cset.future_relations) == 100
    @test length(cset.past_relations) == 100
    @test size(sprinkling) == (100, 3)

    @test_throws ArgumentError cset, sprinkling =
        QuantumGrav.make_cset("Minkowski", "CausalDiamond", 0, 4, rng, type)
end

# Test makelink_matrix
@testitem "make_link_matrix" tags = [:datageneration] setup = [makeData] begin
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


@testitem "make_cardinality_matrix" tags = [:datageneration] setup = [makeData] begin
    @test_throws "The causal set must not be empty." QuantumGrav.DataGeneration.make_cardinality_matrix(
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


@testitem "make_adj" tags = [:datageneration] setup = [makeData] begin
    adj = make_adj(cset_links, Float32)
    @test size(adj) == (100, 100)
    @test all(adj .== 0.0) == false
    @test all(adj .== 1.0) == false

    # alternative approach to make the adjacency matrix that uses Graphs.jl's 
    # capability
    test_adj =
        cset_links |>
        make_link_matrix |>
        Graphs.SimpleDiGraph |>
        Graphs.transitiveclosure |>
        Graphs.adjacency_matrix |>
        SparseArrays.SparseMatrixCSC{Float32}()

    @test all(adj .== test_adj) == true

    @test_throws ArgumentError make_adj(cset_empty, Float32)
end

@testitem "test_max_pathlen" tags = [:datageneration] setup = [makeData] begin

    adj =
        cset_links |>
        make_link_matrix |>
        Graphs.SimpleDiGraph |>
        Graphs.transitiveclosure |>
        Graphs.adjacency_matrix |>
        SparseArrays.SparseMatrixCSC{Float32}()

    max_path = QuantumGrav.max_pathlen(adj, 1:cset_links.atom_count, 1)

    max_path_expected = CausalSets.extremal_path_dijkstra(
        cset_links,
        1,
        cset_links.atom_count,
        false,
        false,
    )

    @test max_path == max_path_expected
    @test max_path > 1
    @test max_path <= cset_links.atom_count
end

@testitem "test_calculate_angles" tags = [:datageneration] setup = [makeData] begin

    # empty sprinkling
    angles = QuantumGrav.calculate_angles(sprinkling_empty, 1, Int[], 100, Float32)

    @test angles == SparseArrays.spzeros(Float32, 100, 100)

    # sprinkling with 100 points
    angles = QuantumGrav.calculate_angles(
        sprinkling_links,
        1,
        cset_links.future_relations[1],
        100,
        Float32,
    )

    @test size(angles) < (100, 100)
    @test SparseArrays.nnz(angles) == length(cset_links.future_relations[1])

    # Check that the angles are calculated correctly
    for i = 1:length(cset_links.future_relations[1])
        for j = 1:length(cset_links.future_relations[1])
            if i != j && angles[i, j] > 0
                @test 0.0 <= angles[i, j] <= π
            end
        end
    end

    # use multithreading
    angles_mt = QuantumGrav.calculate_angles(
        sprinkling_links,
        1,
        cset_links.future_relations[1],
        100,
        Float32,
        multithreading = true,
    )
end


@testitem "test_calculate_distances" tags = [:datageneration] setup = [makeData] begin
    # empty sprinkling
    distances = QuantumGrav.calculate_distances(sprinkling_empty, 1, Int[], 100, Float32)

    @test distances == SparseArrays.spzeros(Float32, 100)

    # sprinkling with 100 points
    distances = QuantumGrav.calculate_distances(
        sprinkling_links,
        1,
        cset_links.future_relations[1],
        100,
        Float32,
    )

    @test size(distances) < (100, 100)
    @test SparseArrays.nnz(distances) > 0
    @test SparseArrays.nnz(distances) == length(cset_links.future_relations[1])

end

@testitem "test_make_data" tags = [:datageneration] setup = [importModules] begin
    @test 3 == 6
end
