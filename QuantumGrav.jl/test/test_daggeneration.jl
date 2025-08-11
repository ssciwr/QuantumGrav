using TestItems

@testsnippet importModules begin
    using QuantumGrav
    using CausalSets
    using SparseArrays
    using Distributions
    using Random
end

@testsnippet make_testgraph begin

    # use bitvectors like in the main code
    adj = [falses(10) for _ = 1:10]
    # for this graph, 1,2,3,4,5,6,7,8,9,10 is a topological ordering
    adj[1][3] = true
    adj[1][4] = true
    adj[2][5] = true
    adj[2][6] = true
    adj[2][7] = true
    adj[4][7] = true
    adj[4][10] = true
    adj[5][10] = true
    adj[6][7] = true
    adj[7][8] = true
    adj[7][9] = true
    adj[1][10] = true
end


@testitem "transitive_closure" tags = [:dag_processing] setup =
    [importModules, make_testgraph] begin
    adjcp = deepcopy(adj)
    QuantumGrav.transitive_closure!(adjcp)

    # old parts unchanged
    for i in eachindex(adj)
        for j in eachindex(adj[i])
            if adj[i][j]
                # what was an edge before should stay an edge
                @test adjcp[i][j]
            end
        end
    end

    # newly added edges
    @test adjcp[1][7]
    @test adjcp[2][8]
    @test adjcp[2][9]
    @test adjcp[2][10]
    @test adjcp[4][8]
    @test adjcp[4][9]
    @test adjcp[6][8]
    @test adjcp[6][9]
end

@testitem "transitive_reduction" tags = [:dag_processing] setup=[
    importModules,
    make_testgraph,
] begin
    adjcp = deepcopy(adj)

    QuantumGrav.transitive_reduction!(adjcp)

    # old parts unchanged
    for i in eachindex(adj)
        for j in eachindex(adj[i])
            if !adj[i][j]
                # what was *not* an edge before should stay an edge
                @test !adjcp[i][j]
            end
        end
    end

    # removed edges 
    @test !adjcp[1][10]
end
