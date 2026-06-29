


@testsnippet LayeredTests begin
    import Random
    Random.seed!(42)  # Set a seed for reproducibility
    rng = Random.Xoshiro(42)
end

@testitem "test_make_simple_layered_cset" tags = [:layeredgeneration] setup = [LayeredTests] begin

    import CausalSets

    cset, cuts = QuantumGrav.create_random_layered_causet(100, 3; rng = rng)

    @test cset.atom_count == 100
    @test length(cset.future_relations) == 100
    @test length(cset.past_relations) == 100
    @test CausalSets.count_chains(cset, 3) != 0
    @test CausalSets.count_chains(cset, 4) == 0
    @test CausalSets.count_chains(cset, 5) == 0
    @test CausalSets.count_chains(cset, 6) == 0
    @test CausalSets.count_chains(cset, 7) == 0

    cset, cuts = QuantumGrav.create_random_layered_causet(100, 5; rng = rng)

    @test cset.atom_count == 100
    @test length(cset.future_relations) == 100
    @test length(cset.past_relations) == 100
    @test CausalSets.count_chains(cset, 5) != 0
    @test CausalSets.count_chains(cset, 6) == 0
    @test CausalSets.count_chains(cset, 7) == 0
    @test CausalSets.count_chains(cset, 8) == 0
    @test CausalSets.count_chains(cset, 9) == 0

    @test_throws ArgumentError QuantumGrav.create_random_layered_causet(2, 3; rng = rng)
    @test_throws ArgumentError QuantumGrav.create_random_layered_causet(
        100,
        3;
        p = -0.1,
        rng = rng,
    )
end


@testitem "test_create_KR_order" tags = [:layeredgeneration] setup = [LayeredTests] begin

    import CausalSets

    cset, cuts = QuantumGrav.create_KR_order(1000; rng = rng)

    @test cset.atom_count == 1000
    @test length(cset.future_relations) == 1000
    @test length(cset.past_relations) == 1000
    @test CausalSets.count_chains(cset, 3) != 0
    @test CausalSets.count_chains(cset, 4) == 0
    @test CausalSets.count_chains(cset, 5) == 0
    @test CausalSets.count_chains(cset, 6) == 0
    @test CausalSets.count_chains(cset, 7) == 0

    n1, n2, n3 = cuts

    @test abs(n1/cset.atom_count - 0.25) < 0.02
    @test abs(n2/cset.atom_count - 0.50) < 0.02
    @test abs(n3/cset.atom_count - 0.25) < 0.02

    A = cset.future_relations
    
    idx1 = 1:n1
    idx2 = (n1+1):(n1+n2)
    idx3 = (n1+n2+1):(n1+n2+n3)

    links =
        sum(A[a][b] for a in idx1 for b in idx2) +
        sum(A[a][b] for a in idx2 for b in idx3)

    possible = n1*n2 + n2*n3
    p̂ = links / possible

    @test isapprox(p̂, 0.5; atol = 0.005)
end

@testitem "test_create_KR_order_warns_for_too_small_KR_order" tags = [:layeredgeneration] setup = [LayeredTests] begin

    import CausalSets

    cset, atoms_per_layer = @test_logs (:warn, r"KR orders conventionally need at least 3 elements") QuantumGrav.create_KR_order(
        2;
        rng = rng,
    )

    @test cset isa CausalSets.BitArrayCauset
    @test cset.atom_count == 2
    @test sum(atoms_per_layer) == 2
end
