using TestItems

@testsnippet TestsCSetMerging begin
    using QuantumGrav
    using CausalSets
    using Random
    using Distributions

    rng = Random.Xoshiro(42)
    rng2 = Random.Xoshiro(24)

    # check that for each edge (u,v), u < v in the index order => topologically sorted
    function is_toposorted(cset::CausalSets.AbstractCauset)::Bool
        consistent = true
        for i = 1:length(cset.future_relations)
            for j = 1:length(cset.future_relations[i])
                if cset.future_relations[i][j]
                    consistent = consistent && i < j
                end
            end
        end
        return consistent
    end
end



@testitem "test_merge_csets" tags = [:csetmerging] setup = [TestsCSetMerging] begin
    cset1, _ = QuantumGrav.sample_bitarray_causet_by_connectivity(
        2^10,
        0.3,
        20,
        rng;
        abs_tol = 0.01,
    )
    cset2, _ = QuantumGrav.sample_bitarray_causet_by_connectivity(
        2^9,
        0.5,
        20,
        rng2;
        abs_tol = 0.01,
    )
    cset_merged = QuantumGrav.merge_csets(cset1, cset2, 0.2)
    @test is_toposorted(cset_merged)
    @test cset_merged.atom_count == 2^10 + 2^9
    @test length(cset_merged.future_relations) == 2^10 + 2^9
    @test length(cset_merged.past_relations) == 2^10 + 2^9
    @test [BitVector(cset_merged.future_relations[i][1:(2^10)]) for i = 1:(2^10)] == cset1.future_relations # cset is 1st subset of cset_merged
    @test [
        BitVector(cset_merged.future_relations[i][(2^10+1):(2^10+2^9)]) for
        i = (2^10+1):(2^10+2^9)
    ] == cset2.future_relations # cset is 2nd subset of cset_merged
end

@testitem "test_merge_cset_throws" tags = [:csetmergingthrows] setup = [TestsCSetMerging] begin
    cset1, _ = QuantumGrav.sample_bitarray_causet_by_connectivity(
        2^10,
        0.3,
        20,
        rng;
        abs_tol = 0.01,
    )
    cset2, _ = QuantumGrav.sample_bitarray_causet_by_connectivity(
        2^9,
        0.5,
        20,
        rng2;
        abs_tol = 0.01,
    )
    @test_throws ArgumentError QuantumGrav.merge_csets(cset1, cset2, 1.1)
end

@testitem "test_insert_cset" tags = [:csetmerging] setup = [TestsCSetMerging] begin
    cset1, _ = QuantumGrav.sample_bitarray_causet_by_connectivity(
        100,
        0.4,
        10,
        rng;
        abs_tol = 0.01,
    )
    cset2, _ = QuantumGrav.sample_bitarray_causet_by_connectivity(
        30,
        0.5,
        10,
        rng2;
        abs_tol = 0.01,
    )

    cset_inserted = QuantumGrav.insert_cset(cset1, cset2, 0.1; rng = rng, position = 25)

    @test is_toposorted(cset_inserted)

    @test cset_inserted.atom_count == 130
    @test length(cset_inserted.future_relations) == 130
    @test length(cset_inserted.past_relations) == 130
end

@testitem "test_insert_cset_throws" tags = [:csetmergingthrows] setup = [TestsCSetMerging] begin
    cset1, _ =
        QuantumGrav.sample_bitarray_causet_by_connectivity(50, 0.4, 10, rng; abs_tol = 0.01)
    cset2, _ = QuantumGrav.sample_bitarray_causet_by_connectivity(
        20,
        0.5,
        10,
        rng2;
        abs_tol = 0.01,
    )

    @test_throws ArgumentError QuantumGrav.insert_cset(cset1, cset2, 1.1; rng = rng)
    @test_throws ArgumentError QuantumGrav.insert_cset(cset1, cset2, -0.1; rng = rng)
    @test_throws ArgumentError QuantumGrav.insert_cset(
        cset1,
        cset2,
        0.2;
        rng = rng,
        position = -1,
    )
    @test_throws ArgumentError QuantumGrav.insert_cset(
        cset1,
        cset2,
        0.2;
        rng = rng,
        position = 51,
    )
end

@testitem "test_insert_KR_into_manifoldlike" tags = [:csetmerging] setup =
    [TestsCSetMerging] begin
    n2_rel = 0.05
    total_size = 200 + round(Int, 200 * n2_rel)
    cset, flag, coords = QuantumGrav.insert_KR_into_manifoldlike(
        200,
        10,
        1.5,
        0.2;
        rng = rng,
        n2_rel = n2_rel,
    )

    @test is_toposorted(cset)
    @test flag === true
    @test isa(cset, BitArrayCauset)
    @test cset.atom_count == total_size
    @test size(coords, 1) == total_size
end

@testitem "test_insert_KR_into_manifoldlike_throws" tags = [:csetmergingthrows] setup =
    [TestsCSetMerging] begin
    @test_throws ArgumentError QuantumGrav.insert_KR_into_manifoldlike(
        100,
        10,
        1.0,
        -0.1;
        rng = rng,
    )
    @test_throws ArgumentError QuantumGrav.insert_KR_into_manifoldlike(
        100,
        10,
        1.0,
        1.1;
        rng = rng,
    )
    @test_throws ArgumentError QuantumGrav.insert_KR_into_manifoldlike(
        100,
        10,
        1.0,
        0.5;
        rng = rng,
        position = -1,
    )
    @test_throws ArgumentError QuantumGrav.insert_KR_into_manifoldlike(
        100,
        10,
        1.0,
        0.5;
        rng = rng,
        position = 101,
    )
    @test_throws ArgumentError QuantumGrav.insert_KR_into_manifoldlike(
        100,
        10,
        1.0,
        0.5;
        rng = rng,
        n2_rel = 0.0,
    )
end
