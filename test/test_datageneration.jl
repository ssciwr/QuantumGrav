using TestItems

@testsnippet importModules begin
    import QuantumGrav
    import CausalSets
    import SparseArrays
end

@testsnippet makeData begin
    import CausalSets
    import QuantumGrav
    import SparseArrays

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
    manifolds=QuantumGrav.DataGeneration.get_manifolds_of_dim(2)
    expected_manifolds=["minkowski", "hypercylinder", "deSitter", "antiDeSitter", "torus"]
    @test Set(keys(manifolds)) == Set(expected_manifolds)

    manifolds=QuantumGrav.DataGeneration.get_manifolds_of_dim(21)
    @test Set(keys(manifolds)) == Set(expected_manifolds)
end

# Test makeLinkMatrix
@testitem "makeLinkMatrix" tags=[:datageneration] setup=[makeData] begin
    link_matrix_empty=QuantumGrav.DataGeneration.makeLinkMatrix(cset_empty)
    @test link_matrix_empty == SparseArrays.spzeros(Float32, 0, 0)

    link_matrix=QuantumGrav.DataGeneration.makeLinkMatrix(cset_links)
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

@testitem "makeBdMatrix" tags=[:datageneration] setup=[makeData] begin

    # test that it works
    ds_multiple=[3, 4, 5]
    mat_multiple=QuantumGrav.DataGeneration.makeBdMatrix(ds_multiple, 4)
    @test size(mat_multiple) == (4, 3)
    @test 0 < SparseArrays.nnz(mat_multiple) <= 3*4

    # test that bad arguments are caught
    @test_throws "The dimensions must not be empty." QuantumGrav.DataGeneration.makeBdMatrix(Int[])
    @test_throws "maxCardinality must be a positive integer." QuantumGrav.DataGeneration.makeBdMatrix(
        [1, 2], 0)
end

@testitem "makeCardinalityMatrix" tags=[:datageneration] setup=[makeData] begin
    @test_throws "The causal set must not be empty." QuantumGrav.DataGeneration.makeCardinalityMatrix(cset_empty)

    # Test case 2: Causal set with some cardinalities
    cardinality_matrix_links=QuantumGrav.DataGeneration.makeCardinalityMatrix(cset_links)
    expected_matrix=SparseArrays.spzeros(Float32, 100, 100)
    @test 0 < SparseArrays.nnz(cardinality_matrix_links) <= 100*100
end

@testitem "generateDataForManifold" tags=[:datageneration] begin

    # Test with small parameters for quick testing
    data = QuantumGrav.DataGeneration.generateDataForManifold(
        dimension = 2,
        manifoldname = "minkowski",
        num_datapoints = 3,
        seed = 12345
    )

    # Check if the returned value is a dictionary
    @test isa(data, Dict)

    # Check if all expected keys exist in the dictionary
    expected_keys = ["idx", "n", "dimension", "manifold", "coords", "future_relations",
        "past_relations", "linkMatrix", "relation_count", "chains_3",
        "chains_4", "chains_10", "cardinality_abundances", "relation_dimension",
        "chain_dimension_3", "chain_dimension_4"]
    for key in expected_keys
        @test haskey(data, key)
    end

    # Check if all arrays have the correct length
    @test length(data["idx"]) == 3
    @test length(data["dimension"]) == 3
    @test length(data["manifold"]) == 3

    # Check if the dimension and manifold are correct
    @test all(x -> x == 2, data["dimension"])
    @test all(x -> x == "minkowski", data["manifold"])

    # Test with equal_size = true
    data_equal = QuantumGrav.DataGeneration.generateDataForManifold(
        dimension = 2,
        manifoldname = "minkowski",
        num_datapoints = 2,
        equal_size = true,
        seed = 12345
    )

    # All n values should be the same
    @test all(x -> x == data_equal["n"][1], data_equal["n"])

    # Test with different manifold
    data_torus = QuantumGrav.DataGeneration.generateDataForManifold(
        dimension = 2,
        manifoldname = "torus",
        num_datapoints = 2,
        seed = 12345
    )

    @test all(x -> x == "torus", data_torus["manifold"])
end
