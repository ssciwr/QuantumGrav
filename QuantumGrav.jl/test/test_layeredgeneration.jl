using TestItems

@testsnippet LayeredTests begin
    using QuantumGrav: QuantumGrav
    using CausalSets: CausalSets
    using Random: Random

    Random.seed!(42)  # Set a seed for reproducibility
    rng = Random.Xoshiro(42)
end

@testitem "test_make_simple_layered_cset" tags = [:layeredgeneration] setup = [LayeredTests] begin
    cset, cuts = QuantumGrav.create_random_layered_causet(100, 3; rng=rng)

    @test cset.atom_count == 100
    @test length(cset.future_relations) == 100
    @test length(cset.past_relations) == 100
    @test CausalSets.count_chains(cset, 3) != 0
    @test CausalSets.count_chains(cset, 4) == 0
    @test CausalSets.count_chains(cset, 5) == 0
    @test CausalSets.count_chains(cset, 6) == 0
    @test CausalSets.count_chains(cset, 7) == 0

    cset, cuts = QuantumGrav.create_random_layered_causet(100, 5; rng=rng)

    @test cset.atom_count == 100
    @test length(cset.future_relations) == 100
    @test length(cset.past_relations) == 100
    @test CausalSets.count_chains(cset, 5) != 0
    @test CausalSets.count_chains(cset, 6) == 0
    @test CausalSets.count_chains(cset, 7) == 0
    @test CausalSets.count_chains(cset, 8) == 0
    @test CausalSets.count_chains(cset, 9) == 0

    @test_throws AssertionError QuantumGrav.create_random_layered_causet(2, 3; rng=rng)
    @test_throws AssertionError QuantumGrav.create_random_layered_causet(100, 3; p=-.1, rng=rng)
end