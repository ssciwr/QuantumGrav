


@testsnippet makeData begin
    import Random

    function MockData(n)
        rng = Random.Xoshiro(12345)
        manifold = CausalSets.MinkowskiManifold{2}()
        boundary = CausalSets.CausalDiamondBoundary{2}(1.0)
        sprinkling = CausalSets.generate_sprinkling(manifold, boundary, Int(n); rng=rng)
        cset = CausalSets.BitArrayCauset(manifold, sprinkling)
        return cset, sprinkling
    end
end

@testitem "test_make_adj" tags = [:graph_utils] setup = [makeData] begin
    import CausalSets
    using SparseArrays
    using Graphs
    cset, _ = MockData(10)
    adj = QuantumGrav.make_adj(cset; type = Float32)

    g = Graphs.SimpleDiGraph(transpose(hcat(cset.future_relations...)))

    test_adj = SparseArrays.SparseMatrixCSC{Float32}(Graphs.adjacency_matrix(g))

    @test all(adj .== test_adj) == true

end

@testitem "test_max_pathlen" tags = [:graph_utils] setup = [makeData] begin
    import CausalSets
    using SparseArrays
    using Graphs
    cset, _ = MockData(10)

    g = Graphs.SimpleDiGraph(transpose(hcat(cset.future_relations...)))

    adj = SparseArrays.SparseMatrixCSC{Float32}(Graphs.adjacency_matrix(g))

    max_path = QuantumGrav.max_pathlen(adj, collect(1:(cset.atom_count)), 1)

    sdg = Graphs.SimpleDiGraph(adj)
    max_path_expected =
        Graphs.dag_longest_path(sdg; topological_order = collect(1:(cset.atom_count)))
    print("max_path_expected: ", max_path_expected, "\n")
    print("max_path: ", max_path, "\n")
    @test max_path > 1
    @test max_path <= cset.atom_count
    @test max_path == (length(max_path_expected) - 1) # dag_longest_path counts differently
end

@testitem "make_transitive_reduction" tags = [:graph_utils] setup = [makeData] begin
    import CausalSets
    using Graphs
    cset, _ = MockData(25)

    g = Graphs.SimpleDiGraph(transpose(hcat(cset.future_relations...)))

    test_adj = Graphs.adjacency_matrix(g)
    links = deepcopy(test_adj)
    QuantumGrav.transitive_reduction!(links)

    for i = 1:cset.atom_count
        for j = 1:cset.atom_count
            if links[i, j] > 0
                @test CausalSets.is_link(cset, i, j) == true
            end

            if test_adj[i, j] > 0 && links[i, j] == 0
                @test CausalSets.is_link(cset, i, j) == false
            end
        end
    end
end
