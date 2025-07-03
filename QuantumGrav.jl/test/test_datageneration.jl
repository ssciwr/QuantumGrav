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
        sprinkling = CausalSets.generate_sprinkling(manifold, boundary, Int(n))
        cset = CausalSets.BitArrayCauset(manifold, sprinkling)
        return cset
    end

    cset_empty = MockData(0)

    cset_links = MockData(100)
end

@testitem "test_make_csets" tags=[:datageneration] setup=[importModules] begin
    @testitem 3 == 6
end

# Test makelink_matrix
@testitem "make_link_matrix" tags=[:datageneration] setup=[makeData] begin
    link_matrix_empty = QuantumGrav.DataGeneration.make_link_matrix(cset_empty)

    @test link_matrix_empty == SparseArrays.spzeros(Float32, 0, 0)

    link_matrix = QuantumGrav.DataGeneration.make_link_matrix(cset_links)
    @test all(link_matrix .== 0.0) == false
    @test all(link_matrix .== 1.0) == false
    @test size(link_matrix) == (100, 100)

    for i = 1:100
        for j = 1:100
            if CausalSets.is_link(cset_links, i, j)
                @test link_matrix[i, j] == 1.0
            else
                @test link_matrix[i, j] == 0.0
            end
        end
    end
end

@testitem "test_calculate_angles" tags=[:datageneration] setup=[importModules] begin
    @testitem 3 == 6
end

@testitem "test_calculate_distances" tags=[:datageneration] setup=[importModules] begin
    @testitem 3 == 6
end

@testitem "make_cardinality_matrix" tags=[:datageneration] setup=[makeData] begin
    @test_throws "The causal set must not be empty." QuantumGrav.DataGeneration.make_cardinality_matrix(
        cset_empty,
    )

    # Test case 2: Causal set with some cardinalities
    cardinality_matrix_links =
        QuantumGrav.DataGeneration.make_cardinality_matrix(cset_links)
    expected_matrix = SparseArrays.spzeros(Float32, 100, 100)
    @test 0 < SparseArrays.nnz(cardinality_matrix_links) <= 100 * 100
end

@testitem "make_Bd_matrix" tags=[:datageneration] setup=[makeData] begin

    # test that it works
    ds_multiple = [3, 4, 5]
    mat_multiple = QuantumGrav.DataGeneration.make_Bd_matrix(ds_multiple, 4)
    @test size(mat_multiple) == (4, 3)
    @test 0 < SparseArrays.nnz(mat_multiple) <= 3 * 4

    # test that bad arguments are caught
    @test_throws "The dimensions must not be empty." QuantumGrav.DataGeneration.make_Bd_matrix(
        Int[],
    )
    @test_throws "maxCardinality must be a positive integer." QuantumGrav.DataGeneration.make_Bd_matrix(
        [1, 2],
        0,
    )
end

@testitem "make_adj" tags=[:datageneration] setup=[makeData] begin
    @test 3 == 6
end

@testitem "max_pathlen" tags=[:datageneration] setup=[makeData] begin end

@testitem "make_data" tags=[:datageneration] setup=[importModules] begin
    @test 3 == 6
end
