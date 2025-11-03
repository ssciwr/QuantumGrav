

@testsnippet TestsCSetByConnectivity begin
    import Random

    Random.seed!(42)  # Set a seed for reproducibility
    rng = Random.Xoshiro(42)
end

@testsnippet TestsCSetByConnectivityDistribution begin
    import Random
    import Distributions

    Random.seed!(42)  # Set a seed for reproducibility
    rng = Random.Xoshiro(42)
end

@testitem "test_make_simple_cset_by_connectivity" tags = [:csetgenerationbyconnectivity] setup =
    [TestsCSetByConnectivity] begin

    import CausalSets
    connectivity_goal = 0.5
    cset, converged = QuantumGrav.sample_bitarray_causet_by_connectivity(
        2^10,
        connectivity_goal,
        20,
        rng;
        abs_tol = 0.01,
    )
    @test converged == true
    @test cset.atom_count == 2^10
    @test length(cset.future_relations) == 2^10
    @test length(cset.past_relations) == 2^10
    @test abs(
        CausalSets.count_relations(cset) / (cset.atom_count * (cset.atom_count - 1) / 2) -
        connectivity_goal,
    ) < 0.01 # connectivity has indeed reached connectivity_goal to withn abs_tol

    cset, converged =
        QuantumGrav.sample_bitarray_causet_by_connectivity(2^10, connectivity_goal, 20, rng) # no abs_tol set, so markov_steps is used as stopping criterion
    @test converged == false
    @test cset.atom_count == 2^10
    @test length(cset.future_relations) == 2^10
    @test length(cset.past_relations) == 2^10
    # no connectivity check here, as the stopping criterion is markov_steps, which is not guaranteed to reach the connectivity goal

end

@testitem "test_simple_cset_by_connectivity_throws" tags =
    [:csetgenerationbyconnectivitythrows] setup = [TestsCSetByConnectivity] begin


    @test_throws ArgumentError QuantumGrav.sample_bitarray_causet_by_connectivity(
        -1,
        0.5,
        20,
        rng;
        abs_tol = 0.01,
    )

    @test_throws ArgumentError QuantumGrav.sample_bitarray_causet_by_connectivity(
        2^7,
        -0.5,
        20,
        rng;
        abs_tol = 0.01,
    )

    @test_throws ArgumentError QuantumGrav.sample_bitarray_causet_by_connectivity(
        2^7,
        1.1,
        20,
        rng;
        abs_tol = 0.01,
    )

    @test_throws ArgumentError QuantumGrav.sample_bitarray_causet_by_connectivity(
        2^7,
        0.6,
        -3,
        rng;
        abs_tol = 0.01,
    )

    @test_throws ArgumentError QuantumGrav.sample_bitarray_causet_by_connectivity(
        2^7,
        0.6,
        20,
        rng;
        abs_tol = 0.01,
        rel_tol = 0.01,
    )

    @test_throws ArgumentError QuantumGrav.sample_bitarray_causet_by_connectivity(
        2^7,
        0.5,
        20,
        rng;
        abs_tol = -0.01,
    )
end

@testitem "test_make_simple_cset_by_connectivity_distribution" tags =
    [:csetgenerationbyconnectivitydist] setup = [TestsCSetByConnectivityDistribution] begin

    import Distributions

    dist = Distributions.Beta(2, 2)
    cset, converged = QuantumGrav.random_causet_by_connectivity_distribution(
        2^10,
        dist,
        20,
        rng;
        abs_tol = 0.01,
    )
    @test converged == true
    @test cset.atom_count == 2^10
    @test length(cset.future_relations) == 2^10
    @test length(cset.past_relations) == 2^10

    cset, converged = QuantumGrav.random_causet_by_connectivity_distribution(
        2^10,
        Distributions.Beta(2, 2),
        20,
        rng,
    ) # no abs_tol set, so markov_steps is used as stopping criterion
    @test converged == false
    @test cset.atom_count == 2^10
    @test length(cset.future_relations) == 2^10
    @test length(cset.past_relations) == 2^10
    # no connectivity check here, as the stopping criterion is markov_steps, which is not guaranteed to reach the connectivity goal

end


@testitem "test_simple_cset_by_connectivity_distribution_throws" tags =
    [:csetgenerationbyconnectivitythrows] setup = [TestsCSetByConnectivityDistribution] begin

    import Distributions

    @test_throws ArgumentError QuantumGrav.random_causet_by_connectivity_distribution(
        -1,
        Distributions.Beta(2, 2),
        20,
        rng;
        abs_tol = 0.01,
    )

    @test_throws ArgumentError QuantumGrav.random_causet_by_connectivity_distribution(
        2^7,
        Distributions.Normal(2, 2), # distribution with support outside [0,1]
        20,
        rng;
        abs_tol = 0.01,
    )

    @test_throws ArgumentError QuantumGrav.random_causet_by_connectivity_distribution(
        2^7,
        Distributions.Beta(2, 2),
        -3,
        rng;
        abs_tol = 0.01,
    )

    @test_throws ArgumentError QuantumGrav.random_causet_by_connectivity_distribution(
        2^7,
        Distributions.Beta(2, 2),
        20,
        rng;
        abs_tol = 0.01,
        rel_tol = 0.01,
    )

    @test_throws ArgumentError QuantumGrav.random_causet_by_connectivity_distribution(
        2^7,
        Distributions.Beta(2, 2),
        20,
        rng;
        abs_tol = -0.01,
    )
end
