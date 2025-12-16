


@testsnippet makeData begin
    using Random: Random

    function MockData(n)
        rng = Random.Xoshiro(12345)
        manifold = CausalSets.MinkowskiManifold{2}()
        boundary = CausalSets.CausalDiamondBoundary{2}(1.0)
        sprinkling = CausalSets.generate_sprinkling(manifold, boundary, Int(n); rng = rng)
        cset = CausalSets.BitArrayCauset(manifold, sprinkling)
        return cset, sprinkling
    end
end

@testitem "test_make_adj" tags = [:graph_utils] setup = [makeData] begin
    using CausalSets
    using SparseArrays
    using Graphs
    cset, _ = MockData(10)
    adj = QuantumGrav.make_adj(cset; type = SparseArrays.SparseMatrixCSC, eltype = Float32)

    g = Graphs.SimpleDiGraph(transpose(hcat(cset.future_relations...)))
    test_adj = SparseArrays.SparseMatrixCSC{Float32}(Graphs.adjacency_matrix(g))
    @test all(adj .== test_adj)

    adj_dense = QuantumGrav.make_adj(cset; type = Matrix, eltype = Bool)
    test_adj_dense = Graphs.adjacency_matrix(g)
    @test all(adj_dense .== test_adj_dense)

    adj_bit = QuantumGrav.make_adj(cset; type = BitMatrix)
    test_adj_bit = test_adj_dense |> BitMatrix
    @test all(adj_bit .== test_adj_bit)

end

@testitem "test_max_pathlen" tags = [:graph_utils_pathlen] begin
    using CausalSets
    using SparseArrays
    using Graphs

    # build a DAG and test explicitly
    adj = zeros(10,10)
    adj[1, 3] = 1
    adj[2, 3] = 1
    adj[3, 4] = 1
    adj[3, 6] = 1
    adj[4, 5] = 1
    adj[6, 7] = 1
    adj[6, 8] = 1
    adj[8, 9] = 1

    g = Graphs.SimpleDiGraph(adj)

    max_path1 = QuantumGrav.max_pathlen(adj, 1:10, 1)
    max_path2 = QuantumGrav.max_pathlen(adj, 1:10, 2)
    max_path6 = QuantumGrav.max_pathlen(adj, 1:10, 6)

    @test max_path1 == 4
    @test max_path2 == 4
    @test max_path6 == 2

end

@testitem "make_transitive_reduction" tags = [:graph_utils] setup = [makeData] begin
    using CausalSets
    using Graphs
    cset, _ = MockData(25)

    g = Graphs.SimpleDiGraph(transpose(hcat(cset.future_relations...)))

    test_adj = Graphs.adjacency_matrix(g)
    links = deepcopy(test_adj)
    QuantumGrav.transitive_reduction!(links)

    for i ∈ 1:cset.atom_count
        for j ∈ 1:cset.atom_count
            if links[i, j] > 0
                @test CausalSets.is_link(cset, i, j) == true
            end

            if test_adj[i, j] > 0 && links[i, j] == 0
                @test CausalSets.is_link(cset, i, j) == false
            end
        end
    end
end
