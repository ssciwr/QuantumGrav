using TestItems

@testsnippet CurvatureOnManifold begin
    using QuantumGrav
    using CausalSets
    using Random

    Random.seed!(1234)
    rng = Random.Xoshiro(1234)
end

@testitem "test_chebyshev_derivation_matrix" tags = [:curvatureonmanifold] setup = [CurvatureOnManifold] begin
    # Test size of the derivation matrix for order 5
    order = 10
    D = QuantumGrav.chebyshev_derivation_matrix(order, 1)
    @test size(D) == (order+1, order+1) # order starts counting at 0

    # Test second-derivative matrix equals square of first-derivative matrix
    @test QuantumGrav.chebyshev_derivation_matrix(order, 1) * QuantumGrav.chebyshev_derivation_matrix(order, 1) == QuantumGrav.chebyshev_derivation_matrix(order, 2)

    # Test fourth-derivative matrix equals square of second-derivative matrix
    @test QuantumGrav.chebyshev_derivation_matrix(order, 2) * QuantumGrav.chebyshev_derivation_matrix(order, 2) == QuantumGrav.chebyshev_derivation_matrix(order, 4)
end

@testitem "test_chebyshev_derivation_matrix throws" tags = [:curvatureonmanifold, :throws] setup = [CurvatureOnManifold] begin
    # Test error thrown for invalid input order < 0
    @test_throws ArgumentError QuantumGrav.chebyshev_derivation_matrix(-1, 2)
    # Test error thrown for invalid input derivative_order < 1
    @test_throws ArgumentError QuantumGrav.chebyshev_derivation_matrix(1, 0)
end

@testitem "test_chebyshev_derivative_2D" tags = [:curvatureonmanifold] setup = [CurvatureOnManifold] begin

    N = 4
    # Define coefficients for f(x,y) = x - y in Chebyshev basis
    # Since f(t,x) = t + x, in Chebyshev basis:
    # f(x,y) = T1(x)*T0(y) - T0(x)*T1(y)
    coefs = zeros(N, N)
    coefs[2, 1] = 1.0   # T1(x)*T0(y)
    coefs[1, 2] = -1.0  # T0(x)*T1(y)

    # Compute derivative with respect to t (variable index 1), first order
    dfdt = QuantumGrav.chebyshev_derivative_2D(coefs, 1, 1)
    @test size(dfdt) == (N, N)

    # Compute derivative with respect to x (variable index 2), first order
    dfdx = QuantumGrav.chebyshev_derivative_2D(coefs, 2, 1)
    @test size(dfdx) == (N, N)

    point = CausalSets.Coordinates{2}((1., 1.))


    # Evaluate derivatives on grid
    vals_dfdt = QuantumGrav.chebyshev_evaluate_2D(dfdt, point)
    vals_dfdx = QuantumGrav.chebyshev_evaluate_2D(dfdx, point)

    @test all(abs.(vals_dfdt .- 1) .< 1e-6)
    @test all(abs.(vals_dfdx .- -1) .< 1e-6)
end

@testitem "test_chebyshev_derivative_2D throw" tags = [:curvatureonmanifold, :throw] setup = [CurvatureOnManifold] begin
    
    coefs = zeros(5, 5)
    # Test error handling: invalid derivative_variable_index
    @test_throws ArgumentError QuantumGrav.chebyshev_derivative_2D(coefs, 0, 1)
    @test_throws ArgumentError QuantumGrav.chebyshev_derivative_2D(coefs, 3, 1)

    # Test error handling: invalid derivative_order
    @test_throws ArgumentError QuantumGrav.chebyshev_derivative_2D(coefs, 1, 0)
    @test_throws ArgumentError QuantumGrav.chebyshev_derivative_2D(coefs, 1, -1)
end

@testitem "test_chebyshev_evaluate_2D" tags = [:curvatureonmanifold] setup = [CurvatureOnManifold] begin
    N = 3
    coefs = [1. 2. 3. 4.;
            5. 6. 7. 8.;
            9. 10. 11. 12.;
            13. 14. 15. 16.]

    point = CausalSets.Coordinates{2}((1., 1.))

    QuantumGrav.chebyshev_evaluate_2D(coefs, point) == 136 # number computed independently in mathematica
end

@testitem "test_Ricci_scalar_2D" tags = [:curvatureonmanifold] setup = [CurvatureOnManifold] begin
    N = 5
    # Define a flat manifold metric (identity matrix) on the grid
    # We use a simple function with zero curvature: f(x,y) = constant
    f = zeros(N, N)
    f[1,1] = 1
    point = CausalSets.Coordinates{2}((1., 1.))
    R = QuantumGrav.Ricci_scalar_2D(f, point)

    # For a constant function, Ricci scalar should be zero or very close to zero
    @test maximum(abs(R)) < 1e-6

    # Test on a simple curved manifold: f(x,y) = t^2 + x^2, expecting nonzero curvature
    coefs = [1. 2. 3. 4.;
        5. 6. 7. 8.;
        9. 10. 11. 12.;
        13. 14. 15. 16.]
    R2 = QuantumGrav.Ricci_scalar_2D(coefs, point)
    @test isapprox(R2, -0.0009549; atol=1e-6) # computed value alternatively with mathematica
end

@testitem "test_Ricci_scalar_2D throws" tags = [:curvatureonmanifold, :throws] setup = [CurvatureOnManifold] begin
    # differing orders in space and time
    coefs = [1. 2. 3. 4.;
        5. 6. 7. 8.;
        9. 10. 11. 12.]
    point = CausalSets.Coordinates{2}((1., 1.))
    @test_throws ArgumentError QuantumGrav.Ricci_scalar_2D(coefs, point)
end

@testitem "test_Ricci_scalar_2D_of_sprinkling" tags = [:curvatureonmanifold] setup = [CurvatureOnManifold] begin
    npoints = 100
    # Generate a sprinkling of points on 2D curved geometry
    cset, sprinkling, coefs = QuantumGrav.make_polynomial_manifold_cset(npoints, MersenneTwister(1234), 5, 3.; type = Float64)
    sprinkling_correct_type = [CausalSets.Coordinates{2}((s)) for s in sprinkling]

    # Compute Ricci scalar of the sprinkling
    R = QuantumGrav.Ricci_scalar_2D_of_sprinkling(coefs, sprinkling_correct_type)
    @test typeof(R) <: AbstractArray
    @test length(R) == npoints

    # Values should be finite numbers
    @test all(isfinite, R)
end
