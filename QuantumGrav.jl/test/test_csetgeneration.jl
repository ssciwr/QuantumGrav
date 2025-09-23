using TestItems

@testsnippet setupTests begin
    using QuantumGrav: QuantumGrav
    using CausalSets: CausalSets
    using SparseArrays: SparseArrays
    using Distributions: Distributions
    using Random: Random
    using Graphs: Graphs
    using HDF5: HDF5
    using YAML: YAML

    Random.seed!(42)  # Set a seed for reproducibility
    rng = Random.Xoshiro(42)
    npoint_distribution = Distributions.DiscreteUniform(2, 1000)
    order_distribution = Distributions.DiscreteUniform(2, 9)
    r_distribution = Distributions.Uniform(1.0, 2.0)
end

@testitem "test_make_polynomial_manifold_cset" tags = [:chebyshev_causets] setup =
    [setupTests] begin
    r = 1.0 + rand(rng, r_distribution)
    npoints = rand(rng, npoint_distribution)
    order = rand(rng, order_distribution)

    cset, sprinkling, chebyshev_coefs = QuantumGrav.make_polynomial_manifold_cset(
        npoints,
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


@testitem "test_make_polynomial_manifold_cset_throws" tags = [:chebyshev_causets] setup =
    [setupTests] begin

    @test_throws ArgumentError QuantumGrav.make_polynomial_manifold_cset(
        100,
        rng,
        0,
        2.0;
        d = 2,
        type = Float32,
    )
    @test_throws ArgumentError QuantumGrav.make_polynomial_manifold_cset(
        100,
        rng,
        1,
        0.0;
        d = 2,
        type = Float32,
    )
    @test_throws ArgumentError QuantumGrav.make_polynomial_manifold_cset(
        100,
        rng,
        2,
        2.0;
        d = -1,
        type = Float32,
    )

end

@testitem "test_make_manifold_cset_positivity_of_squared_polynomial" tags =
    [:chebyshev_causets, :positivity] setup = [setupTests] begin
    r = 1.0 + rand(rng, r_distribution)
    npoints = rand(rng, npoint_distribution)
    order = rand(rng, order_distribution)

    chebyshev_coefs = [r^(-i - j) * randn(rng) for i = 1:order, j = 1:order]

    cheb_to_taylor_mat = CausalSets.chebyshev_coef_matrix(order - 1)
    taylor = CausalSets.transform_polynomial(chebyshev_coefs, cheb_to_taylor_mat)
    squared = CausalSets.polynomial_pow(taylor, 2)

    for _ = 1:400
        x = 2rand(rng) - 1
        y = 2rand(rng) - 1
        val = sum(
            squared[i, j] * x^(i - 1) * y^(j - 1) for
            i = 1:size(squared, 1), j = 1:size(squared, 2)
        )
        @test val ≥ -1e-10  # allow minor negative values due to roundoff
    end
end

@testitem "test_make_manifold_cset_squared_polynomial_symmetry_from_symmetric_chebyshev" tags =
    [:chebyshev_causets, :symmetry] setup = [setupTests] begin
    r = 1.0 + rand(rng, r_distribution)
    npoints = rand(rng, npoint_distribution)
    order = rand(rng, order_distribution)

    chebyshev_coefs = [0.0 for i = 1:order, j = 1:order]
    for i = 1:order, j = 1:i
        val = r^(-i - j) * randn(rng)
        chebyshev_coefs[i, j] = val
        chebyshev_coefs[j, i] = val
    end

    cheb_to_taylor_mat = CausalSets.chebyshev_coef_matrix(order - 1)
    taylor = CausalSets.transform_polynomial(chebyshev_coefs, cheb_to_taylor_mat)
    squared = CausalSets.polynomial_pow(taylor, 2)

    for i = 1:size(squared, 1), j = 1:i
        @test isapprox(squared[i, j], squared[j, i]; atol = 1e-10)
    end
end
