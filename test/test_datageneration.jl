using TestItems

@testsnippet importModules begin
    import QuantumGrav
    import CausalSets
    import SparseArrays
    import Distributions
end

@testsnippet makeData begin
    import CausalSets
    import QuantumGrav
    import SparseArrays
    import Distributions

    function MockData(n)
        manifold = CausalSets.MinkowskiManifold{2}()
        boundary = CausalSets.CausalDiamondBoundary{2}(1.0)
        sprinkling = CausalSets.generate_sprinkling(
            manifold, boundary, Int(n))
        cset = CausalSets.BitArrayCauset(manifold, sprinkling)
        return cset
    end

    cset_empty = MockData(0)

    cset_links = MockData(100)
end

@testitem "get_manifolds_for_dim" tags=[:datageneration] setup=[importModules] begin

    manifolds = QuantumGrav.DataGeneration.get_manifolds_of_dim(2)

    expected_manifolds = ["minkowski", "hypercylinder", "deSitter", "antiDeSitter", "torus"]

    @test Set(keys(manifolds)) == Set(expected_manifolds)

    manifolds = QuantumGrav.DataGeneration.get_manifolds_of_dim(21)

    @test Set(keys(manifolds)) == Set(expected_manifolds)
end

# Test makelink_matrix
@testitem "make_link_matrix" tags=[:datageneration] setup=[makeData] begin

    link_matrix_empty = QuantumGrav.DataGeneration.make_link_matrix(cset_empty)

    @test link_matrix_empty == SparseArrays.spzeros(Float32, 0, 0)

    link_matrix = QuantumGrav.DataGeneration.make_link_matrix(cset_links)
    @test all(link_matrix .== 0.0) == false
    @test all(link_matrix .== 1.0) == false
    @test size(link_matrix) == (100, 100)

    for i in 1:100
        for j in 1:100
            if CausalSets.is_link(cset_links, i, j)
                @test link_matrix[i, j] == 1.0
            else
                @test link_matrix[i, j] == 0.0
            end
        end
    end
end

@testitem "make_Bd_matrix" tags=[:datageneration] setup=[makeData] begin

    # test that it works
    ds_multiple = [3, 4, 5]
    mat_multiple = QuantumGrav.DataGeneration.make_Bd_matrix(ds_multiple, 4)
    @test size(mat_multiple) == (4, 3)
    @test 0 < SparseArrays.nnz(mat_multiple) <= 3 * 4

    # test that bad arguments are caught
    @test_throws "The dimensions must not be empty." QuantumGrav.DataGeneration.make_Bd_matrix(Int[])
    @test_throws "maxCardinality must be a positive integer." QuantumGrav.DataGeneration.make_Bd_matrix(
        [1, 2], 0)
end

@testitem "make_cardinality_matrix" tags=[:datageneration] setup=[makeData] begin

    @test_throws "The causal set must not be empty." QuantumGrav.DataGeneration.make_cardinality_matrix(cset_empty)

    # Test case 2: Causal set with some cardinalities
    cardinality_matrix_links = QuantumGrav.DataGeneration.make_cardinality_matrix(cset_links)
    expected_matrix = SparseArrays.spzeros(Float32, 100, 100)
    @test 0 < SparseArrays.nnz(cardinality_matrix_links) <= 100 * 100
end

@testitem "generate_data_for_manifold" tags=[:datageneration] setup=[importModules] begin

    # Test with small parameters for quick testing
    fne = d -> Distributions.Uniform(0.7 * 10^d, 1.3 * 10^d)

    data = QuantumGrav.DataGeneration.generate_data_for_manifold(
        dimension = 2,
        num_datapoints = 100,
        seed = 12345,
        choose_num_events = fne
    )

    # Check if the returned value is a dictionary
    @test isa(data, Dict)

    # Check if all expected keys exist in the dictionary
    expected_keys = ["idx", "n", "dimension", "manifold", "coords", "future_relations",
        "past_relations", "link_matrix", "relation_count", "chains_3",
        "chains_4", "chains_10", "cardinality_abundances", "relation_dimension",
        "chain_dimension_3", "chain_dimension_4"]

    for key in expected_keys
        @test haskey(data, Symbol(key))
    end

    for m in QuantumGrav.DataGeneration.valid_manifolds
        @test any(data[:manifold] .== m)
    end

    # Check if all arrays have the correct length
    @test sort(data[:idx]) == collect(1:100)
    @test all(data[:nmax] .== 130)
    @test minimum(data[:n]) >= 70
    @test maximum(data[:n]) <= 130

    # Check if the dimension and manifold are correct
    @test all(data[:dimension] .== 2)
end
