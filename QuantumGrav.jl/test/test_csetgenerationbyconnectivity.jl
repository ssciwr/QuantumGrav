using TestItems

@testsnippet TestsCSetByConnectivity begin
    using QuantumGrav
    using CausalSets
    using Random
    using CSV
    using DataFrames
    using Dierckx

    Random.seed!(42)  # Set a seed for reproducibility
    rng = Random.Xoshiro(42)
end

@testitem "test_make_simple_cset_by_connectivity" tags = [:csetgenerationbyconnectivity] setup = [TestsCSetByConnectivity] begin
    connectivity_goal = 0.5
    cset = QuantumGrav.sample_bitarray_causet_by_connectivity(2^10, connectivity_goal, 20, rng; abs_tol = 0.01)
    @test cset.atom_count == 2^10
    @test length(cset.future_relations) == 2^10
    @test length(cset.past_relations) == 2^10
    @test abs(CausalSets.count_relations(cset) / (cset.atom_count * (cset.atom_count - 1) / 2) - connectivity_goal) < 0.01 # connectivity has indeed reached connectivity_goal to withn abs_tol
end

@testitem "test_simple_cset_by_connectivity_throws" tags = [:csetgenerationbyconnectivitythrows] setup = [TestsCSetByConnectivity] begin
    @test_throws ArgumentError QuantumGrav.sample_bitarray_causet_by_connectivity(-1, .5, 20, rng; abs_tol = 0.01)
    @test_throws ArgumentError QuantumGrav.sample_bitarray_causet_by_connectivity(2^7, -.5, 20, rng; abs_tol = 0.01)
    @test_throws ArgumentError QuantumGrav.sample_bitarray_causet_by_connectivity(2^7, 1.1, 20, rng; abs_tol = 0.01)
    @test_throws ArgumentError QuantumGrav.sample_bitarray_causet_by_connectivity(2^7, .6, -3, rng; abs_tol = 0.01)
    @test_throws ArgumentError QuantumGrav.sample_bitarray_causet_by_connectivity(2^7, .5, 20, rng; abs_tol = -0.01)
end