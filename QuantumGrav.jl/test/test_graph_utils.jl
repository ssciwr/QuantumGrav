using TestItems

@testsnippet importModules begin
    using QuantumGrav
    using TestItemRunner
    using CausalSets
    using SparseArrays
    using Random
    using Distributions
end

@testsnippet makeData begin
    using CausalSets
    using SparseArrays
    using QuantumGrav
    using TestItemRunner
    using Graphs

    function MockData(n)
        manifold = CausalSets.MinkowskiManifold{2}()
        boundary = CausalSets.CausalDiamondBoundary{2}(1.0)
        sprinkling = CausalSets.generate_sprinkling(manifold, boundary, Int(n))
        cset = CausalSets.BitArrayCauset(manifold, sprinkling)
        return cset, sprinkling
    end

@test_item "test_make_adj" tags = [:graph_utils] setup = [makeData] begin
    adj = QuantumGrav.make_adj(cset; type=Float32)

    g = Graphs.SimpleDiGraph(hcat(cset.future_relations...))

    test_adj = SparseArrays.SparseMatrixCSC{Float32}(
        Graphs.adjacency_matrix(g)
    )

    @test all(adj .== test_adj) == true

end

@testitem "test_max_pathlen" tags = [:featuregeneration] setup = [makeData] begin

    g = Graphs.SimpleDiGraph(hcat(cset.future_relations...))

    adj = SparseArrays.SparseMatrixCSC{Float32}(
        Graphs.adjacency_matrix(g)
    )

    max_path = QuantumGrav.max_pathlen(adj, collect(1:(cset.atom_count)), 1)

    sdg = Graphs.SimpleDiGraph(adj)
    max_path_expected =
        Graphs.dag_longest_path(sdg; topological_order = collect(1:(cset.atom_count)))

    @test max_path > 1
    @test max_path <= cset.atom_count
    @test max_path == length(max_path_expected)
end

@test_item "make_transitive_reduction" tags = [:graph_utils] setup = [makeData] begin
    g = Graphs.SimpleDiGraph(hcat(cset.future_relations...))

    test_adj = Graphs.adjacency_matrix(g)
    
    QuantumGrav.transitive_reduction!(test_adj)

    for i in 1:cset.atom_count
        for j in 1:cset.atom_count 
            if test_adj[i,j]
                @test CausalSets.is_link(i,j) == true
            end
        end
    end
end