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

@testitem "test_make_simple_cset" tags = [:csetgeneration] setup = [setupTests] begin
    type = Float32
    cset, converged, sprinkling = QuantumGrav.make_simple_cset(
        "Minkowski",
        "CausalDiamond",
        100,
        2,
        Distributions.Beta(2,2),
        300,
        rng;
        type = type,
        abs_tol = 0.01
    )

    @test cset.atom_count == 100
    @test length(cset.future_relations) == 100
    @test length(cset.past_relations) == 100
    @test size(sprinkling) == (100, 2)

    cset, converged, sprinkling =
        QuantumGrav.make_simple_cset("Random", "BoxBoundary", 100, 3, Distributions.Beta(2,2), 300, rng; type = type, abs_tol = 0.01)

    @test cset.atom_count == 100
    @test length(cset.future_relations) == 100
    @test length(cset.past_relations) == 100
    @test size(sprinkling) == (100, 3)

    @test_throws ArgumentError QuantumGrav.make_simple_cset(
        "Minkowski",
        "CausalDiamond",
        0,
        4,
        Distributions.Beta(2,2),
        300,
        rng;
        type = type,
        abs_tol = 0.01,
    )
end

@testitem "test_make_manifold_cset" tags = [:chebyshev_causets] setup = [setupTests] begin
    r = 1.0 + rand(rng, r_distribution)
    npoints = rand(rng, npoint_distribution)
    order = rand(rng, order_distribution)

    cset, sprinkling, chebyshev_coefs =
        QuantumGrav.make_manifold_cset(npoints, rng, order, r; d = 2, type = Float32)

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

@testitem "make_general_cset_works" tags = [:csetgeneration] setup = [setupTests] begin
    rng = Random.Xoshiro(42)
    npoints = 100
    order = 3
    r = 1.5
    d = 2
    markov_iter = 200
    dist = Distributions.Beta(2,2)
    abs_tol = 0.01

    cset, sprinkling, chebyshev_coefs, manifold_like, converged =
        QuantumGrav.make_general_cset(rng, npoints, order, r, d, dist, markov_iter, Float32; abs_tol = abs_tol)

    @test cset.atom_count == npoints
    @test size(sprinkling) == (npoints, d)
    @test size(chebyshev_coefs) == (order, order)
    @test manifold_like in 0:1

    @test typeof(cset) == CausalSets.BitArrayCauset
    @test length(cset.future_relations) == npoints
    @test length(cset.past_relations) == npoints
    @test all(size(sprinkling) .== (npoints, d))

    if manifold_like == 1
        @test any(chebyshev_coefs .!= 0)
    else
        @test all(chebyshev_coefs .== 0)
    end
end
