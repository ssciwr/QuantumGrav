using TestItems

@testsnippet chebyshev_causets_full_function begin
    import QuantumGrav
    import CausalSets
    import Random
    const test_seed = rand(1:10^6)
    @info "Using test seed" seed=test_seed
    Random.seed!(test_seed)
end

@testsnippet chebyshev_polynomial_positivity begin
    import QuantumGrav
    import CausalSets
    import Random
    const test_seed = rand(1:10^6)
    @info "Using test seed for positivity test" seed=test_seed
    Random.seed!(test_seed)
end

@testitem "test_create_cset_from_chebyshevs" tags = [:chebyshev_causets] setup = [chebyshev_causets_full_function] begin
    seed = rand(Random.GLOBAL_RNG, 1:10^6)
    r = 1.0 + rand(Random.GLOBAL_RNG)
    npoints = rand(Random.GLOBAL_RNG, 2:1000)
    order = rand(Random.GLOBAL_RNG, 2:9)

    cset, sprinkling, chebyshev_coefs = QuantumGrav.generate_random_chebyshev_causet(npoints, seed, order, r)

    @test length(sprinkling) == npoints
    @test size(chebyshev_coefs) == (order, order)
    @test typeof(cset) != Nothing
    @test cset.atom_count == npoints
    @test length(cset.future_relations) == npoints
    @test length(cset.past_relations) == npoints
    @test length(sprinkling) == npoints
    @test length(sprinkling[rand(Random.GLOBAL_RNG, 1:length(sprinkling))]) == 2


    # Additional tests to verify support across the domain
    xs = [p[1] for p in test_sprinkling]
    ys = [p[2] for p in test_sprinkling]

    x_spread = maximum(xs) - minimum(xs)
    y_spread = maximum(ys) - minimum(ys)

    @test x_spread > 0.9
    @test y_spread > 0.9
    
    envelope = [r^(-i - j) for i in 1:order, j in 1:order]
    for i in 1:order, j in 1:order
        @test abs(chebyshev_coefs[i, j]) ≤ 10 * envelope[i, j]
    end

    @test_throws ArgumentError QuantumGrav.generate_random_chebyshev_causet(10, 42, 3, 0.9)
    @test_throws ArgumentError QuantumGrav.generate_random_chebyshev_causet(0, 42, 3, 1.1)
    @test_throws ArgumentError QuantumGrav.generate_random_chebyshev_causet(10, 42, 0, 1.1)
end

@testitem "positivity_of_squared_polynomial" tags = [:chebyshev_causets, :positivity] setup = [chebyshev_polynomial_positivity] begin
    r = 1.0 + rand(Random.GLOBAL_RNG)
    order = rand(Random.GLOBAL_RNG, 2:9)
    chebyshev_coefs = [r^(-i - j) * randn(Random.GLOBAL_RNG) for i in 1:order, j in 1:order]

    cheb_to_taylor_mat =  chebyshev_coef_matrix(order-1)
    taylor = transform_polynomial(chebyshev_coefs, cheb_to_taylor_mat)
    squared = polynomial_pow(taylor, 2)

    for _ in 1:400
        x = 2rand(Random.GLOBAL_RNG) - 1
        y = 2rand(Random.GLOBAL_RNG) - 1
        val = sum(squared[i, j] * x^(i-1) * y^(j-1) for i in 1:size(squared, 1), j in 1:size(squared, 2))
        @test val ≥ -1e-10  # allow minor negative values due to roundoff
    end
end


@testitem "squared_polynomial_symmetry_from_symmetric_chebyshev" tags = [:chebyshev_causets, :symmetry] setup = [chebyshev_polynomial_positivity] begin
    r = 1.0 + rand(Random.GLOBAL_RNG)
    order = rand(Random.GLOBAL_RNG, 2:9)

    chebyshev_coefs = [0.0 for i in 1:order, j in 1:order]
    for i in 1:order, j in 1:i
        val = r^(-i - j) * randn(Random.GLOBAL_RNG)
        chebyshev_coefs[i, j] = val
        chebyshev_coefs[j, i] = val
    end

    cheb_to_taylor_mat = chebyshev_coef_matrix(order - 1)
    taylor = transform_polynomial(chebyshev_coefs, cheb_to_taylor_mat)
    squared = polynomial_pow(taylor, 2)

    for i in 1:size(squared, 1), j in 1:i
        @test isapprox(squared[i, j], squared[j, i]; atol=1e-10)
    end
end