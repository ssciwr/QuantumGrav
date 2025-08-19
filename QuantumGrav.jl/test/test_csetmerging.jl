using TestItems

@testsnippet TestsCSetMerging begin
    using QuantumGrav
    using CausalSets
    using Random

    rng = Random.Xoshiro(42)
    rng2 = Random.Xoshiro(24)
end

@testitem "test_merge_csets" tags = [:csetmerging] setup =
    [TestsCSetMerging] begin
    cset1,_ = QuantumGrav.sample_bitarray_causet_by_connectivity(
        2^10,
        0.3,
        20,
        rng;
        abs_tol = 0.01,
        )
    cset2,_ = QuantumGrav.sample_bitarray_causet_by_connectivity(
        2^9,
        0.5,
        20,
        rng2;
        abs_tol = 0.01,
        )
    cset_merged = QuantumGrav.merge_csets(
        cset1,
        cset2,
        0.2
        )
    @test cset_merged.atom_count == 2^10 + 2^9
    @test length(cset_merged.future_relations) == 2^10 + 2^9
    @test length(cset_merged.past_relations) == 2^10 + 2^9
    @test [BitVector(cset_merged.future_relations[i][1:2^10]) for i in 1:2^10] == cset1.future_relations # cset is 1st subset of cset_merged
    @test [BitVector(cset_merged.future_relations[i][2^10+1:2^10 + 2^9]) for i in 2^10+1:2^10 + 2^9] == cset2.future_relations # cset is 2nd subset of cset_merged
end

@testitem "test_merge_cset_throws" tags =
    [:csetmergingthrows] setup = [TestsCSetMerging] begin
    cset1,_ = sample_bitarray_causet_by_connectivity(
        2^10,
        0.3,
        20,
        rng;
        abs_tol = 0.01,
        )
    cset2,_ = sample_bitarray_causet_by_connectivity(
        2^9,
        0.5,
        20,
        rng2;
        abs_tol = 0.01,
        )
    @test_throws ArgumentError QuantumGrav.merge_csets(
        cset1,
        cset2,
        1.1
    )
end
