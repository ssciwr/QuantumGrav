@testitem "test_regularized_polynomial_manifold density" tags = [:regularized] begin
    import CausalSets

    # P(t,x) = t: the old zero line now has density epsilon.
    coefs = zeros(2, 2)
    coefs[2, 1] = 1.0
    original = copy(coefs)
    epsilon = 0.3
    manifold = QuantumGrav.regularized_polynomial_manifold(coefs, epsilon)
    for t in (-1.0, -0.2, 0.0, 0.4, 1.0), x in (-0.7, 0.0, 0.8)

        point = CausalSets.Coordinates{2}((t, x))
        density = CausalSets.eval_polynomial(manifold.coefs, point, (false, false))
        @test density ≈ t^2 + epsilon
        @test density >= epsilon
    end
    @test coefs == original

    # A constant (including zero) polynomial works at order zero in any dimension.
    for d in (1, 2, 4), constant in (0.0, -2.0)

        constant_manifold = QuantumGrav.regularized_polynomial_manifold(
            fill(constant, ntuple(_ -> 1, d)),
            epsilon,
        )
        @test constant_manifold.coefs.c[ntuple(_->1, d)...] ≈ constant^2 + epsilon
    end
end

@testitem "test_make_regularized_polynomial_manifold_cset" tags = [:regularized] begin
    import Random
    import CausalSets

    for d in (2, 4), type in (Float32, Float64)

        cset, sprinkling, coefs = QuantumGrav.make_regularized_polynomial_manifold_cset(
            32,
            Random.Xoshiro(42),
            1,
            2.0,
            0.1;
            d = d,
            type = type,
        )
        @test cset.atom_count == 32
        @test length(sprinkling) == 32
        @test size(coefs) == ntuple(_ -> 2, d)
        @test eltype(coefs) == type
        @test all(point -> all(x -> -1 < x < 1, point), sprinkling)
        @test all(point -> all(x -> x isa type, point), sprinkling)

        cset2, sprinkling2, coefs2 = QuantumGrav.make_regularized_polynomial_manifold_cset(
            32,
            Random.Xoshiro(42),
            1,
            2.0,
            0.1;
            d = d,
            type = type,
        )
        @test coefs == coefs2
        @test sprinkling == sprinkling2
        @test QuantumGrav.make_adj(cset) == QuantumGrav.make_adj(cset2)
    end

    # The random P is unchanged; epsilon changes the sampling measure.
    _, oldpoints, oldcoefs = QuantumGrav.make_polynomial_manifold_cset(
        32,
        Random.Xoshiro(42),
        2,
        2.0;
        type = Float64,
    )
    _, points, coefs = QuantumGrav.make_regularized_polynomial_manifold_cset(
        32,
        Random.Xoshiro(42),
        2,
        2.0,
        0.1;
        type = Float64,
    )
    @test coefs == oldcoefs
    @test points != oldpoints

    cset, points, coefs = QuantumGrav.make_regularized_polynomial_manifold_cset(
        16,
        Random.Xoshiro(42),
        0,
        2.0,
        0.1;
        type = Float64,
    )
    @test cset.atom_count == length(points) == 16
    @test size(coefs) == (1, 1)
end

@testitem "test_regularized_polynomial_manifold sampling measure" tags = [:regularized] begin
    import Random
    import CausalSets

    coefs = zeros(2, 2)
    coefs[2, 1] = 1.0
    epsilon = 0.5
    manifold = QuantumGrav.regularized_polynomial_manifold(coefs, epsilon)
    boundary = CausalSets.BoxBoundary{2}(((-1.0, -1.0), (1.0, 1.0)))
    points =
        CausalSets.generate_sprinkling(manifold, boundary, 6000; rng = Random.Xoshiro(42))
    # Exact moments for density t^2 + epsilon, independent of x.
    expected_t2 = (1 / 5 + epsilon / 3) / (1 / 3 + epsilon)
    @test isapprox(sum(p[1]^2 for p in points) / length(points), expected_t2; atol = 0.015)
    @test isapprox(sum(p[2]^2 for p in points) / length(points), 1 / 3; atol = 0.015)
end

@testitem "test_regularized_polynomial_manifold throws" tags = [:regularized, :throws] begin
    import Random

    for epsilon in (0.0, -0.1, Inf, -Inf, NaN)
        @test_throws ArgumentError QuantumGrav.regularized_polynomial_manifold(
            zeros(2, 2),
            epsilon,
        )
        @test_throws ArgumentError QuantumGrav.make_regularized_polynomial_manifold_cset(
            16,
            Random.Xoshiro(42),
            1,
            2.0,
            epsilon,
        )
    end
    for coefs in (zeros(0, 0), zeros(2, 3), fill(NaN, 2, 2), fill(Inf, 2, 2))
        @test_throws ArgumentError QuantumGrav.regularized_polynomial_manifold(coefs, 0.1)
    end
    for (npoints, order, r, d) in (
        (0, 1, 2.0, 2),
        (16, -1, 2.0, 2),
        (16, 1, 1.0, 2),
        (16, 1, Inf, 2),
        (16, 1, NaN, 2),
        (16, 1, 2.0, 0),
    )
        @test_throws ArgumentError QuantumGrav.make_regularized_polynomial_manifold_cset(
            npoints,
            Random.Xoshiro(42),
            order,
            r,
            0.1;
            d = d,
        )
    end
end
