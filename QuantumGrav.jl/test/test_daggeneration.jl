using TestItems

@testsnippet importModules begin
    using QuantumGrav
    using CausalSets
    using SparseArrays
    using Distributions
    using Random
end

@testsnippet dag_params begin
    atom_count = 12

    rng = Random.Xoshiro(42)

    function future_deg(rng, i, range, n)
        return 2 # constant degree
    end

    function link_prob(rng, i, j, fdeg)
        return 0.2
    end

end

@testsnippet make_testgraph begin

    # use bitvectors like in the main code
    adj = [falses(10) for _ = 1:10]
    reversed = [falses(10) for _ = 1:10]
    # for this graph, 1,2,3,4,5,6,7,8,9,10 is a topological ordering
    adj[1][3] = true
    adj[1][4] = true
    adj[1][10] = true
    adj[2][5] = true
    adj[2][6] = true
    adj[2][7] = true
    adj[4][7] = true
    adj[4][10] = true
    adj[5][10] = true
    adj[6][7] = true
    adj[7][8] = true
    adj[7][9] = true

    reversed[3][1] = true
    reversed[4][1] = true
    reversed[10][1] = true
    reversed[5][2] = true
    reversed[6][2] = true
    reversed[7][2] = true
    reversed[7][4] = true
    reversed[10][4] = true
    reversed[10][5] = true
    reversed[7][6] = true
    reversed[8][7] = true
    reversed[9][7] = true
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

@testitem "transitive_closure_empty" tags = [:dag_processing] setup = [importModules] begin
    adj = Vector{BitVector}()

    QuantumGrav.transitive_closure!(adj)

    @test length(adj) == 0
end

@testitem "transitive_reduction" tags = [:dag_processing] setup =
    [importModules, make_testgraph] begin
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

@testitem "transitive_reduction_sparse" tags = [:dag_processing] setup =
    [importModules, make_testgraph] begin

    adjcp = [SparseArrays.sparse(convert.(Int, v)) for v in deepcopy(adj)]

    @test adjcp isa Vector{SparseArrays.SparseVector{Int64,Int64}}

    QuantumGrav.transitive_reduction!(adjcp)

    # old parts unchanged
    for i in eachindex(adj)
        for j in eachindex(adj[i])
            if adj[i][j] == 0
                # what was *not* an edge before should stay an edge
                @test adjcp[i][j] == 0
            end
        end
    end

    # removed edges 
    @test adjcp[1][10] == 0
    @test adjcp isa Vector{SparseArrays.SparseVector{Int64,Int64}}
end

@testitem "transitive_reduction_empty" tags = [:dag_processing] setup = [importModules] begin
    adj = Vector{BitVector}()

    QuantumGrav.transitive_reduction!(adj)

    @test length(adj) == 0
end

@testitem "make_csets" tags = [:dag_processing] setup = [importModules, make_testgraph] begin

    cset_bits = QuantumGrav.mat_to_bit_cs(adj)

    @test cset_bits isa CausalSets.BitArrayCauset
    @test cset_bits.future_relations == adj
    @test cset_bits.past_relations == reversed
    @test cset_bits.atom_count == 10

    cset_sparse = QuantumGrav.mat_to_sparse_cs(adj)

    sparse_futures = [SparseArrays.sparse(convert.(Int, v)) for v in deepcopy(adj)]
    sparse_past = [SparseArrays.sparse(convert.(Int, v)) for v in deepcopy(reversed)]

    @test cset_sparse isa CausalSets.SparseArrayCauset
    @test cset_sparse.future_relations == sparse_futures
    @test cset_sparse.past_relations == sparse_past
    @test cset_sparse.atom_count == 10
end


@testitem "create_dag_works" tags = [:dag_creation] setup = [importModules, dag_params] begin

    cset = QuantumGrav.create_random_cset(
        atom_count,
        future_deg,
        link_prob,
        rng;
        type = CausalSets.BitArrayCauset,
        parallel = false,
    )

    @test cset isa CausalSets.BitArrayCauset
    @test cset.atom_count == atom_count

    in_deg = sum.([convert.(Int, b) for b in cset.future_relations])

    for i = 1:atom_count
        @test in_deg[i] <= max(length((i+1):atom_count), 0)
    end

    # make sure the atoms are in topological order-> only upper triangle 
    # make sure there are no loops -> no back connections

    for i = 1:atom_count
        for j = 1:atom_count
            if cset.future_relations[i][j]
                @test j > i
                @test cset.future_relations[j][i] == false
            end

            if j < i
                @test cset.future_relations[i][j] == false
            end
        end
    end
end
