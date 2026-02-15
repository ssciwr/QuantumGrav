


@testsnippet setupTests begin
    import Distributions
    import Random

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
    @test size(chebyshev_coefs) == (order + 1, order + 1)
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

    envelope = [r^(-i - j + 2) for i = 1:(order+1), j = 1:(order+1)]
    for i = 1:(order+1), j = 1:(order+1)
        @test abs(chebyshev_coefs[i, j]) ≤ 10 * envelope[i, j]
    end
end

@testitem "test_make_polynomial_manifold_cset_d4" tags = [:chebyshev_causets] setup =
    [setupTests] begin
    r = 1.0 + rand(rng, r_distribution)
    npoints = rand(rng, npoint_distribution)
    order = rand(rng, order_distribution)
    d = 4

    cset, sprinkling, chebyshev_coefs = QuantumGrav.make_polynomial_manifold_cset(
        npoints,
        rng,
        order,
        r;
        d = d,
        type = Float32,
    )

    @test length(sprinkling) == npoints
    @test size(chebyshev_coefs) == ntuple(_ -> order+1, d)
    @test typeof(cset) != Nothing
    @test cset.atom_count == npoints
    @test length(cset.future_relations) == npoints
    @test length(cset.past_relations) == npoints

    # Additional tests to verify support across the domain
    for k = 1:d
        coords = [p[k] for p in sprinkling]
        spread = maximum(coords) - minimum(coords)
        @test spread > 0.9
    end

    for I in CartesianIndices(chebyshev_coefs)
        envelope = r^(-sum(Tuple(I)) + d)
        @test abs(chebyshev_coefs[I]) ≤ 10 * envelope
    end
end


@testitem "test_make_polynomial_manifold_cset_throws" tags = [:chebyshev_causets] setup =
    [setupTests] begin

    @test_throws ArgumentError QuantumGrav.make_polynomial_manifold_cset(
        100,
        rng,
        -1,
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
        1,
        2.0;
        d = 0,
        type = Float32,
    )

end

@testitem "test_random_chebyshev_coefficients_shape_and_decay" tags = [:chebyshev_causets] begin
    using Random

    rng = Random.Xoshiro(1234)
    order = 4
    d = 3
    r = 2.0

    coefs = QuantumGrav.random_chebyshev_coefficients(rng, order, r, d)
    @test size(coefs) == (order + 1, order + 1, order + 1)

    for I in CartesianIndices(coefs)
        envelope = r^(-sum(Tuple(I)) + d)
        @test abs(coefs[I]) <= 20 * envelope
    end
end

@testitem "test_transform_polynomial_cached_matches_bruteforce" tags = [:chebyshev_causets] begin
    using CausalSets
    using Random

    rng = Random.Xoshiro(17)
    αmax = (4, 4)
    logr = (log(2.0), log(3.0))
    Λ = 2.7

    order_inds = NTuple{2,Int}[]
    for a = 0:αmax[1], b = 0:αmax[2]
        if a * logr[1] + b * logr[2] <= Λ
            push!(order_inds, (a, b))
        end
    end

    cheb = randn(rng, αmax[1] + 1, αmax[2] + 1)
    cheb_to_taylor = CausalSets.chebyshev_coef_matrix(maximum(αmax))

    got = CausalSets.transform_polynomial(cheb, cheb_to_taylor, order_inds; αmax = collect(αmax))
    ref = zeros(Float64, αmax[1] + 1, αmax[2] + 1)
    for α in order_inds
        cα = cheb[α[1] + 1, α[2] + 1]
        for β1 = 0:α[1], β2 = 0:α[2]
            ref[β1 + 1, β2 + 1] += cα * cheb_to_taylor[β1 + 1, α[1] + 1] * cheb_to_taylor[β2 + 1, α[2] + 1]
        end
    end
    @test got ≈ ref atol = 1e-12
end

@testitem "test_polynomial_pow_cached_matches_bruteforce" tags = [:chebyshev_causets] begin
    using CausalSets
    using Random

    rng = Random.Xoshiro(19)
    αmax = (3, 3)
    logr = (log(2.0), log(2.5))
    Λ = 2.3

    order_inds = NTuple{2,Int}[]
    for a = 0:αmax[1], b = 0:αmax[2]
        if a * logr[1] + b * logr[2] <= Λ
            push!(order_inds, (a, b))
        end
    end

    coefs = randn(rng, αmax[1] + 1, αmax[2] + 1)
    got = CausalSets.polynomial_pow(coefs, 2, order_inds; αmax = collect(αmax))

    ref = zeros(Float64, αmax[1] + 1, αmax[2] + 1)
    for a1 = 0:αmax[1], a2 = 0:αmax[2]
        wa = a1 * logr[1] + a2 * logr[2]
        wa <= Λ || continue
        ca = coefs[a1 + 1, a2 + 1]
        for b1 = 0:αmax[1], b2 = 0:αmax[2]
            wb = b1 * logr[1] + b2 * logr[2]
            wb <= Λ || continue
            g1, g2 = a1 + b1, a2 + b2
            g1 <= αmax[1] || continue
            g2 <= αmax[2] || continue
            wg = g1 * logr[1] + g2 * logr[2]
            wg <= Λ || continue
            ref[g1 + 1, g2 + 1] += ca * coefs[b1 + 1, b2 + 1]
        end
    end

    @test got ≈ ref atol = 1e-12
end

@testitem "test_eval_polynomial_cached_matches_bruteforce" tags = [:chebyshev_causets] begin
    using CausalSets

    αmax = (3, 3)
    logr = (log(2.0), log(3.0))
    Λ = 2.4
    x = (0.2, -0.4)
    integrate = (false, false)

    c = zeros(Float64, αmax[1] + 1, αmax[2] + 1)
    for i = 1:size(c, 1), j = 1:size(c, 2)
        c[i, j] = 0.1 * i - 0.05 * j
    end
    coefs = CausalSets.PolynomialCoefs(c)

    order_inds = NTuple{2,Int}[]
    for a = 0:αmax[1], b = 0:αmax[2]
        if a * logr[1] + b * logr[2] <= Λ
            push!(order_inds, (a, b))
        end
    end
    got = CausalSets.eval_polynomial(coefs, x, integrate, order_inds)

    ref = Ref(0.0)
    for a = 0:αmax[1], b = 0:αmax[2]
        if a * logr[1] + b * logr[2] <= Λ
            ref[] += c[a + 1, b + 1] * x[1]^a * x[2]^b
        end
    end

    @test isapprox(got, ref[]; atol = 1e-12)
end

@testitem "test_make_anisotropically_weighted_polynomial_manifold_cset" tags = [:chebyshev_causets] begin
    using Random

    rng = Random.Xoshiro(123)
    npoints = 32
    r_vec = (2.1, 2.8)

    cset, sprinkling, cheb = QuantumGrav.make_anisotropically_weighted_polynomial_manifold_cset(
        npoints,
        rng,
        r_vec;
        d = 2,
        type = Float32,
    )

    @test cset.atom_count == npoints
    @test length(sprinkling) == npoints
    @test ndims(cheb) == 2
    @test size(cheb, 1) >= 1
    @test size(cheb, 2) >= 1
end

@testitem "test_make_anisotropically_weighted_polynomial_manifold_cset_throws" tags =
    [:chebyshev_causets, :throws] begin
    using Random
    rng = Random.Xoshiro(1)
    @test_throws ArgumentError QuantumGrav.make_anisotropically_weighted_polynomial_manifold_cset(
        10,
        rng,
        (1.0, 2.0);
        d = 2,
    )
end

@testitem "test_weighted_simplex_indices_matches_bruteforce" tags = [:chebyshev_causets] begin
    r_vec = (2.0, 3.0, 5.0)
    npoints = 64
    order_inds, αmax = QuantumGrav.weighted_simplex_indices(r_vec, npoints)

    logr = Tuple(log.(r_vec))
    Λ = 2 * log(npoints) + 1
    brute = NTuple{3,Int}[]
    for a = 0:αmax[1], b = 0:αmax[2], c = 0:αmax[3]
        if a * logr[1] + b * logr[2] + c * logr[3] <= Λ
            push!(brute, (a, b, c))
        end
    end

    @test Set(order_inds) == Set(brute)
    @test αmax == [floor(Int, Λ / logr[i]) for i in 1:3]
end

@testitem "test_eval_polynomial_order_inds_matches_bruteforce_integrated" tags = [:chebyshev_causets] begin
    using CausalSets

    αmax = (4, 3)
    logr = (log(2.0), log(3.0))
    Λ = 2.6
    x = (0.31, -0.44)
    integrate = (true, false)

    c = zeros(Float64, αmax[1] + 1, αmax[2] + 1)
    for i = 1:size(c, 1), j = 1:size(c, 2)
        c[i, j] = 0.07 * i - 0.03 * j
    end
    coefs = CausalSets.PolynomialCoefs(c)

    order_inds = NTuple{2,Int}[]
    for a = 0:αmax[1], b = 0:αmax[2]
        if a * logr[1] + b * logr[2] <= Λ
            push!(order_inds, (a, b))
        end
    end

    got_order_inds = CausalSets.eval_polynomial(coefs, x, integrate, order_inds)

    ref = 0.0
    for a = 0:αmax[1], b = 0:αmax[2]
        if a * logr[1] + b * logr[2] <= Λ
            p1 = a + 1
            p2 = b
            ref += c[a + 1, b + 1] * x[1]^p1 * x[2]^p2 / (a + 1)
        end
    end
    @test isapprox(got_order_inds, ref; atol = 1e-12)
end

@testitem "test_truncated_polynomial_primitives" tags = [:chebyshev_causets] begin
    using CausalSets
    using Random

    order_inds = [(0, 0)]
    c = zeros(Float64, 1, 1)
    c[1, 1] = 1.0

    manifold = QuantumGrav.TruncatedPolynomialManifold{2}(c, order_inds)
    boundary = CausalSets.BoxBoundary{2}(((-1.0, -1.0), (1.0, 1.0)))

    @test QuantumGrav.sampling_primary_coord(manifold) == 1
    @test QuantumGrav.is_in_boundary(manifold, boundary, (0.0, 0.0))
    @test !QuantumGrav.is_in_boundary(manifold, boundary, (1.0, 0.0))

    seg = QuantumGrav.TruncatedPolynomialRectangularSegment(boundary.edges, manifold.coefs, order_inds)
    @test CausalSets.min_max_primary_coords(seg) == (-1.0, 1.0)
    @test isapprox(CausalSets.partial_volume(seg, 0.0), 2.0; atol = 1e-12)
    @test isapprox(CausalSets.inverse_partial_volume(seg, 3.0), 0.5; atol = 1e-8)

    rng = Random.Xoshiro(9)
    coords = CausalSets.sample_at_primary_coord(rng, seg, 0.2)
    @test isapprox(coords[1], 0.2; atol = 1e-12)
    @test -1.0 <= coords[2] <= 1.0

    seq = CausalSets.sprinkling_segment_sequence(manifold, boundary)
    @test length(seq.segments) == 1
    @test isapprox(seq.integrated_partial_volumes[1], 4.0; atol = 1e-10)
end

@testitem "test_definite_integral_and_sample_step_with_order_inds" tags = [:chebyshev_causets] begin
    using CausalSets
    using Random

    order_inds = [(0, 0)]
    c = CausalSets.PolynomialCoefs(reshape([1.0], 1, 1))

    box = CausalSets.DimLimits(((-1.0, 1.0), (-1.0, 1.0)))
    @test isapprox(CausalSets.definite_integral(c, box, (true, true), order_inds), 4.0; atol = 1e-12)

    rng = Random.Xoshiro(11)
    params = CausalSets.DimLimits((0.0, (-1.0, 1.0)))
    sampled = CausalSets.sample_step(rng, c, params, 2, order_inds)
    @test sampled.l[1] == 0.0
    @test -1.0 <= sampled.l[2] <= 1.0
end

@testitem "test_weighted_truncation_equals_dense_cset_at_resolved_scale" tags = [:chebyshev_causets] begin
    using CausalSets
    using Random

    d = 2
    r_vec = (8.0, 8.0)
    npoints = 64
    order_full = 12
    logr = Tuple(log.(r_vec))

    # Same dense Chebyshev tensor for both pipelines.
    rng_coef = Random.Xoshiro(123)
    full_inds = [(a, b) for a = 0:order_full for b = 0:order_full]
    cheb_full = QuantumGrav.random_weighted_chebyshev_coefficients(
        rng_coef,
        (order_full, order_full),
        logr,
        full_inds,
    )

    # Dense/original manifold from full tensor.
    dense_mat = CausalSets.chebyshev_coef_matrix(order_full)
    t_dense = CausalSets.transform_polynomial(cheb_full, dense_mat)
    sq_dense = CausalSets.polynomial_pow(t_dense, 2)
    poly_dense = CausalSets.PolynomialManifold{d}(sq_dense)

    # Weighted-truncated manifold: keep only cached admissible weighted indices.
    order_inds, αmax = QuantumGrav.weighted_simplex_indices(r_vec, npoints)
    cheb_trunc = zeros(Float64, αmax[1] + 1, αmax[2] + 1)
    for α in order_inds
        cheb_trunc[α[1] + 1, α[2] + 1] = cheb_full[α[1] + 1, α[2] + 1]
    end
    trunc_mat = CausalSets.chebyshev_coef_matrix(maximum(αmax))
    t_trunc = CausalSets.transform_polynomial(cheb_trunc, trunc_mat, order_inds; αmax = αmax)
    sq_trunc = CausalSets.polynomial_pow(t_trunc, 2, order_inds; αmax = αmax)
    poly_trunc = QuantumGrav.TruncatedPolynomialManifold{d}(sq_trunc, order_inds)

    boundary = CausalSets.BoxBoundary{d}(((-1.0, -1.0), (1.0, 1.0)))
    rng_dense = Random.Xoshiro(777)
    rng_trunc = Random.Xoshiro(777)

    spr_dense = CausalSets.generate_sprinkling(poly_dense, boundary, npoints; rng = rng_dense)
    spr_trunc = CausalSets.generate_sprinkling(poly_trunc, boundary, npoints; rng = rng_trunc)

    cset_dense = CausalSets.BitArrayCauset(poly_dense, spr_dense)
    cset_trunc = CausalSets.BitArrayCauset(poly_trunc, spr_trunc)
    @test cset_trunc.atom_count == cset_dense.atom_count
    @test cset_trunc.future_relations == cset_dense.future_relations
    @test cset_trunc.past_relations == cset_dense.past_relations

    # Coordinates can differ slightly while causal relations remain unchanged.
    max_dx = maximum(abs(spr_dense[i][1] - spr_trunc[i][1]) for i in 1:npoints)
    max_dy = maximum(abs(spr_dense[i][2] - spr_trunc[i][2]) for i in 1:npoints)
    @test max_dx < 1e-3
    @test max_dy < 1e-3
end

@testitem "test_production_speed_measurement_truncated_vs_dense" tags = [:chebyshev_causets, :performance] begin
    using CausalSets
    using Random

    d = 2
    r_vec = (2.1, 2.9)
    npoints = 64
    order_inds, αmax_vec = QuantumGrav.weighted_simplex_indices(r_vec, npoints)
    αmax = Tuple(αmax_vec)
    logr = Tuple(log.(r_vec))
    mat = CausalSets.chebyshev_coef_matrix(maximum(αmax))

    # Dense CausalSets polynomial utilities expect equal extent per axis.
    nmax = maximum(αmax)
    rng = Random.Xoshiro(2028)
    cheb = QuantumGrav.random_weighted_chebyshev_coefficients(
        rng,
        (nmax, nmax),
        logr,
        order_inds,
    )

    # Warmup to exclude compilation from timings.
    begin
        t_dense = CausalSets.transform_polynomial(cheb, mat)
        sq_dense = CausalSets.polynomial_pow(t_dense, 2)
        CausalSets.PolynomialManifold{d}(sq_dense)

        t_trunc = CausalSets.transform_polynomial(cheb, mat, order_inds; αmax = αmax_vec)
        sq_trunc = CausalSets.polynomial_pow(t_trunc, 2, order_inds; αmax = αmax_vec)
        QuantumGrav.TruncatedPolynomialManifold{d}(sq_trunc, order_inds)
    end

    reps = 8
    dense_time = @elapsed begin
        for _ = 1:reps
            t_dense = CausalSets.transform_polynomial(cheb, mat)
            sq_dense = CausalSets.polynomial_pow(t_dense, 2)
            CausalSets.PolynomialManifold{d}(sq_dense)
        end
    end

    trunc_time = @elapsed begin
        for _ = 1:reps
            t_trunc = CausalSets.transform_polynomial(cheb, mat, order_inds; αmax = αmax_vec)
            sq_trunc = CausalSets.polynomial_pow(t_trunc, 2, order_inds; αmax = αmax_vec)
            QuantumGrav.TruncatedPolynomialManifold{d}(sq_trunc, order_inds)
        end
    end

    speedup = dense_time / trunc_time
    @info "polynomial_manifold_production_speed_measurement" dense_time = dense_time trunc_time = trunc_time speedup = speedup reps = reps terms = length(order_inds)

    @test dense_time > 0
    @test trunc_time > 0
    @test isfinite(speedup)
end

@testitem "test_make_manifold_cset_positivity_of_squared_polynomial" tags =
    [:chebyshev_causets, :positivity] setup = [setupTests] begin
    import CausalSets


    r = 1.0 + rand(rng, r_distribution)
    npoints = rand(rng, npoint_distribution)
    order = rand(rng, order_distribution)

    chebyshev_coefs = QuantumGrav.random_chebyshev_coefficients(rng, order, r, 2)

    cheb_to_taylor_mat = CausalSets.chebyshev_coef_matrix(order)
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

    import CausalSets

    r = 1.0 + rand(rng, r_distribution)
    npoints = rand(rng, npoint_distribution)
    order = rand(rng, order_distribution)

    base_chebyshev_coefs = QuantumGrav.random_chebyshev_coefficients(rng, order, r, 2)
    chebyshev_coefs = zeros(Float64, order + 1, order + 1)
    for i = 1:(order+1), j = 1:i
        val = base_chebyshev_coefs[i, j]
        chebyshev_coefs[i, j] = val
        chebyshev_coefs[j, i] = val
    end

    cheb_to_taylor_mat = CausalSets.chebyshev_coef_matrix(order)
    taylor = CausalSets.transform_polynomial(chebyshev_coefs, cheb_to_taylor_mat)
    squared = CausalSets.polynomial_pow(taylor, 2)

    for i = 1:size(squared, 1), j = 1:i
        @test isapprox(squared[i, j], squared[j, i]; atol = 1e-10)
    end
end
