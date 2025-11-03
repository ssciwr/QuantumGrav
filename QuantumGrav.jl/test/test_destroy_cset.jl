

@testsnippet setupTests begin
    using Distributions
    using Random

    Random.seed!(42)  # Set a seed for reproducibility
    rng = Random.Xoshiro(42)
    npoint_distribution = Distributions.DiscreteUniform(2, 1000)
    order_distribution = Distributions.DiscreteUniform(2, 9)
    r_distribution = Distributions.Uniform(1.0, 2.0)
end

@testitem "test_destroy_manifold_cset" tags = [:destroy_causets] setup = [setupTests] begin
    using QuantumGrav

    r = 1.0 + rand(rng, r_distribution)
    npoints = rand(rng, npoint_distribution)
    order = rand(rng, order_distribution)
    num_flips = 1

    cset, sprinkling, chebyshev_coefs = QuantumGrav.destroy_manifold_cset(
        npoints,
        num_flips,
        rng,
        order,
        r;
        d = 2,
        type = Float32,
    )

    @test length(sprinkling) == npoints
    @test size(chebyshev_coefs) == (order, order)
    @test typeof(cset) != Nothing
    @test cset.atom_count == npoints
    @test length(cset.future_relations) == npoints
    @test length(cset.past_relations) == npoints

    # Additional tests to verify support across the domain
    xs = [p[1] for p in sprinkling]
    ys = [p[2] for p in sprinkling]

    x_spread = maximum(xs) - minimum(xs)
    y_spread = maximum(ys) - minimum(ys)

    @test x_spread > 0.9
    @test y_spread > 0.9

    envelope = [r^(-i - j) for i = 1:order, j = 1:order]
    for i = 1:order, j = 1:order
        @test abs(chebyshev_coefs[i, j]) ≤ 10 * envelope[i, j]
    end
end
