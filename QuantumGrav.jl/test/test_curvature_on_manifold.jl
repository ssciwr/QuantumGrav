


@testsnippet CurvatureOnManifold begin
    import Random
    Random.seed!(1234)
    rng = Random.Xoshiro(1234)
end

@testitem "test_chebyshev_derivation_matrix" tags = [:curvatureonmanifold] setup =
    [CurvatureOnManifold] begin
    # Test size of the derivation matrix for order 5
    order = 10
    D = QuantumGrav.chebyshev_derivation_matrix(order, 1)
    @test size(D) == (order+1, order+1) # order starts counting at 0

    # Test second-derivative matrix equals square of first-derivative matrix -- test trivial now
    @test QuantumGrav.chebyshev_derivation_matrix(order, 1) *
          QuantumGrav.chebyshev_derivation_matrix(order, 1) ==
          QuantumGrav.chebyshev_derivation_matrix(order, 2)

    # Test fourth-derivative matrix equals square of second-derivative matrix -- test trivial now
    @test QuantumGrav.chebyshev_derivation_matrix(order, 2) *
          QuantumGrav.chebyshev_derivation_matrix(order, 2) ==
          QuantumGrav.chebyshev_derivation_matrix(order, 4)

    # Test new implementation without factorials equals earlier implementation for 1st to 5th derivative
    function old_chebyshev_derivation_matrix(
        max_cheb_order::Int64,
        derivative_order::Int64,
    )::Array{Float64,2}

        transformation_matrix = zeros(max_cheb_order + 1, max_cheb_order + 1)

        for n = 0:max_cheb_order

            for k = mod(n-derivative_order, 2):2:(n-derivative_order)
                prefac = k == 0 ? 0.5 : 1

                transformation_matrix[k+1, n+1] +=
                    2^derivative_order *
                    n *
                    prefac *
                    binomial(
                        Int64((n + derivative_order - k) / 2 - 1),
                        Int64((n - derivative_order - k) / 2),
                    ) *
                    factorial(Int64((n + derivative_order + k) / 2 - 1)) /
                    factorial(Int64((n - derivative_order + k) / 2))
            end
        end

        return transformation_matrix
    end

    @test QuantumGrav.chebyshev_derivation_matrix(order, 1) ==
          old_chebyshev_derivation_matrix(order, 1)
    @test QuantumGrav.chebyshev_derivation_matrix(order, 2) ==
          old_chebyshev_derivation_matrix(order, 2)
    @test QuantumGrav.chebyshev_derivation_matrix(order, 3) ==
          old_chebyshev_derivation_matrix(order, 3)
    @test QuantumGrav.chebyshev_derivation_matrix(order, 4) ==
          old_chebyshev_derivation_matrix(order, 4)
    @test QuantumGrav.chebyshev_derivation_matrix(order, 5) ==
          old_chebyshev_derivation_matrix(order, 5)

end

@testitem "test_chebyshev_derivation_matrix throws" tags = [:curvatureonmanifold, :throws] setup =
    [CurvatureOnManifold] begin
    # Test error thrown for invalid input order < 0
    @test_throws ArgumentError QuantumGrav.chebyshev_derivation_matrix(-1, 2)
    # Test error thrown for invalid input derivative_order < 1
    @test_throws ArgumentError QuantumGrav.chebyshev_derivation_matrix(1, 0)
end

@testitem "test_chebyshev_derivative_2D" tags = [:curvatureonmanifold] setup =
    [CurvatureOnManifold] begin

    import CausalSets

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

    point = CausalSets.Coordinates{2}((1.0, 1.0))


    # Evaluate derivatives on grid
    vals_dfdt = QuantumGrav.chebyshev_evaluate_2D(dfdt, point)
    vals_dfdx = QuantumGrav.chebyshev_evaluate_2D(dfdx, point)

    @test all(abs.(vals_dfdt .- 1) .< 1e-6)
    @test all(abs.(vals_dfdx .- -1) .< 1e-6)
end

@testitem "test_chebyshev_derivative_2D throw" tags = [:curvatureonmanifold, :throw] setup =
    [CurvatureOnManifold] begin

    coefs = zeros(5, 5)
    # Test error handling: invalid derivative_variable_index
    @test_throws ArgumentError QuantumGrav.chebyshev_derivative_2D(coefs, 0, 1)
    @test_throws ArgumentError QuantumGrav.chebyshev_derivative_2D(coefs, 3, 1)

    # Test error handling: invalid derivative_order
    @test_throws ArgumentError QuantumGrav.chebyshev_derivative_2D(coefs, 1, 0)
    @test_throws ArgumentError QuantumGrav.chebyshev_derivative_2D(coefs, 1, -1)
end

@testitem "test_chebyshev_evaluate_2D" tags = [:curvatureonmanifold] setup =
    [CurvatureOnManifold] begin
    import CausalSets
    N = 3
    coefs = [
        1.0 2.0 3.0 4.0;
        5.0 6.0 7.0 8.0;
        9.0 10.0 11.0 12.0;
        13.0 14.0 15.0 16.0
    ]

    point = CausalSets.Coordinates{2}((1.0, 1.0))

    QuantumGrav.chebyshev_evaluate_2D(coefs, point) == 136 # number computed independently in mathematica
end

@testitem "test_Ricci_scalar_2D" tags = [:curvatureonmanifold] setup = [CurvatureOnManifold] begin
    import CausalSets

    N = 5
    # Define a flat manifold metric (identity matrix) on the grid
    # We use a simple function with zero curvature: f(x,y) = constant
    f = zeros(N, N)
    f[1, 1] = 1
    point = CausalSets.Coordinates{2}((1.0, 1.0))
    R = QuantumGrav.Ricci_scalar_2D(f, point)

    # For a constant function, Ricci scalar should be zero or very close to zero
    @test maximum(abs(R)) < 1e-6

    # Test on a simple curved manifold: f(x,y) = t^2 + x^2, expecting nonzero curvature
    coefs = [
        1.0 2.0 3.0 4.0;
        5.0 6.0 7.0 8.0;
        9.0 10.0 11.0 12.0;
        13.0 14.0 15.0 16.0
    ]
    R2 = QuantumGrav.Ricci_scalar_2D(coefs, point)
    @test isapprox(R2, -0.0009549; atol = 1e-6) # computed value alternatively with mathematica
end

@testitem "test_Ricci_scalar_2D throws" tags = [:curvatureonmanifold, :throws] setup =
    [CurvatureOnManifold] begin
    import CausalSets

    # differing orders in space and time
    coefs = [
        1.0 2.0 3.0 4.0;
        5.0 6.0 7.0 8.0;
        9.0 10.0 11.0 12.0
    ]
    point = CausalSets.Coordinates{2}((1.0, 1.0))
    @test_throws ArgumentError QuantumGrav.Ricci_scalar_2D(coefs, point)
end

@testitem "test_Ricci_scalar_2D_of_sprinkling" tags = [:curvatureonmanifold] setup =
    [CurvatureOnManifold] begin
    import CausalSets

    npoints = 100
    # Generate a sprinkling of points on 2D curved geometry
    cset, sprinkling, coefs = QuantumGrav.make_polynomial_manifold_cset(
        npoints,
        Random.MersenneTwister(1234),
        5,
        3.0;
        type = Float64,
    )
    sprinkling_correct_type = [CausalSets.Coordinates{2}((s)) for s in sprinkling]

    # Compute Ricci scalar of the sprinkling
    R = QuantumGrav.Ricci_scalar_2D_of_sprinkling(coefs, sprinkling_correct_type)
    @test typeof(R) <: AbstractArray
    @test length(R) == npoints

    # Values should be finite numbers
    @test all(isfinite, R)
end

@testitem "test_curvature_invariants_2D flat" tags = [:curvatureonmanifold] setup =
    [CurvatureOnManifold] begin
    import CausalSets

    # constant conformal factor: flat space, all invariants vanish
    coefs = zeros(5, 5)
    coefs[1, 1] = 1.0
    point = CausalSets.Coordinates{2}((0.3, -0.4))

    @test abs(QuantumGrav.Kretschmann_scalar_2D(coefs, point)) < 1e-12
    @test abs(QuantumGrav.Goroff_Sagnotti_scalar_2D(coefs, point)) < 1e-12
    @test abs(QuantumGrav.Ricci_box_Ricci_2D(coefs, point)) < 1e-12
end

@testitem "test_curvature_invariants_2D analytic" tags = [:curvatureonmanifold] setup =
    [CurvatureOnManifold] begin
    import CausalSets

    # g = t^2 (-dt^2 + dx^2): R = -2 / t^4, box R = 40 / t^8
    t = 0.5
    coefs_time = zeros(3, 3)
    coefs_time[2, 1] = 1.0 # T1(t)*T0(x)
    point_time = CausalSets.Coordinates{2}((t, 0.3))

    @test QuantumGrav.Kretschmann_scalar_2D(coefs_time, point_time) ≈ 4 / t^8
    @test QuantumGrav.Goroff_Sagnotti_scalar_2D(coefs_time, point_time) ≈ -8 / t^12
    @test QuantumGrav.Ricci_box_Ricci_2D(coefs_time, point_time) ≈ -80 / t^12

    # g = x^2 (-dt^2 + dx^2): R = 2 / x^4, box R = 40 / x^8
    x = 0.7
    coefs_space = zeros(3, 3)
    coefs_space[1, 2] = 1.0 # T0(t)*T1(x)
    point_space = CausalSets.Coordinates{2}((0.3, x))

    @test QuantumGrav.Kretschmann_scalar_2D(coefs_space, point_space) ≈ 4 / x^8
    @test QuantumGrav.Goroff_Sagnotti_scalar_2D(coefs_space, point_space) ≈ 8 / x^12
    @test QuantumGrav.Ricci_box_Ricci_2D(coefs_space, point_space) ≈ 80 / x^12
end

@testitem "test_curvature_invariants_2D relation to Ricci scalar" tags =
    [:curvatureonmanifold] setup = [CurvatureOnManifold] begin
    import CausalSets

    coefs = [
        1.0 0.2 0.1;
        0.3 0.1 0.05;
        0.1 0.02 0.01
    ]
    point = CausalSets.Coordinates{2}((0.2, -0.3))
    R = QuantumGrav.Ricci_scalar_2D(coefs, point)

    @test QuantumGrav.Kretschmann_scalar_2D(coefs, point) ≈ R^2
    @test QuantumGrav.Goroff_Sagnotti_scalar_2D(coefs, point) ≈ R^3
    @test QuantumGrav.Weyl_squared_2D(coefs, point) == 0.0
    @test QuantumGrav.Weyl_box_Weyl_2D(coefs, point) == 0.0
end

@testitem "test_Ricci_box_Ricci_2D finite differences" tags = [:curvatureonmanifold] setup =
    [CurvatureOnManifold] begin
    import CausalSets

    coefs = [
        1.0 0.2 0.1;
        0.3 0.1 0.05;
        0.1 0.02 0.01
    ]
    t, x = 0.2, -0.3
    h = 1e-3
    R(t, x) = QuantumGrav.Ricci_scalar_2D(coefs, CausalSets.Coordinates{2}((t, x)))

    # box R = (-R_tt + R_xx) / f^2, with the derivatives of R taken by central differences
    f = QuantumGrav.chebyshev_evaluate_2D(coefs, CausalSets.Coordinates{2}((t, x)))
    R_tt = (R(t + h, x) - 2 * R(t, x) + R(t - h, x)) / h^2
    R_xx = (R(t, x + h) - 2 * R(t, x) + R(t, x - h)) / h^2
    expected = R(t, x) * (-R_tt + R_xx) / f^2

    @test isapprox(
        QuantumGrav.Ricci_box_Ricci_2D(coefs, CausalSets.Coordinates{2}((t, x))),
        expected;
        rtol = 1e-5,
    )
end

@testitem "test_curvature_invariant_of_sprinkling" tags = [:curvatureonmanifold] setup =
    [CurvatureOnManifold] begin
    import CausalSets

    npoints = 100
    cset, sprinkling, coefs = QuantumGrav.make_polynomial_manifold_cset(
        npoints,
        Random.MersenneTwister(1234),
        5,
        3.0;
        type = Float64,
    )
    sprinkling_correct_type = [CausalSets.Coordinates{2}((s)) for s in sprinkling]
    derivation_matrix1 = QuantumGrav.chebyshev_derivation_matrix(size(coefs, 1) - 1, 1)
    derivation_matrix2 = QuantumGrav.chebyshev_derivation_matrix(size(coefs, 1) - 1, 2)

    for invariant in (
        QuantumGrav.Ricci_scalar_2D,
        QuantumGrav.Kretschmann_scalar_2D,
        QuantumGrav.Goroff_Sagnotti_scalar_2D,
        QuantumGrav.Ricci_box_Ricci_2D,
        QuantumGrav.Weyl_squared_2D,
        QuantumGrav.Weyl_box_Weyl_2D,
    )
        values = QuantumGrav.curvature_invariant_of_sprinkling(
            invariant,
            coefs,
            sprinkling_correct_type,
        )
        @test length(values) == npoints
        @test all(isfinite, values)
        @test values == [invariant(coefs, point) for point in sprinkling_correct_type]

        # precomputed derivation matrices must not change the result
        @test values == QuantumGrav.curvature_invariant_of_sprinkling(
            invariant,
            coefs,
            sprinkling_correct_type;
            derivation_matrix1 = derivation_matrix1,
            derivation_matrix2 = derivation_matrix2,
        )
    end

    @test QuantumGrav.Ricci_scalar_2D_of_sprinkling(coefs, sprinkling_correct_type) ==
          QuantumGrav.curvature_invariant_of_sprinkling(
        QuantumGrav.Ricci_scalar_2D,
        coefs,
        sprinkling_correct_type,
    )
end

@testitem "test_curvature_invariants_2D throws" tags = [:curvatureonmanifold, :throws] setup =
    [CurvatureOnManifold] begin
    import CausalSets

    # differing orders in space and time
    coefs = [
        1.0 2.0 3.0 4.0;
        5.0 6.0 7.0 8.0;
        9.0 10.0 11.0 12.0
    ]
    point = CausalSets.Coordinates{2}((1.0, 1.0))

    @test_throws ArgumentError QuantumGrav.Kretschmann_scalar_2D(coefs, point)
    @test_throws ArgumentError QuantumGrav.Goroff_Sagnotti_scalar_2D(coefs, point)
    @test_throws ArgumentError QuantumGrav.Ricci_box_Ricci_2D(coefs, point)
    @test_throws ArgumentError QuantumGrav.curvature_invariant_of_sprinkling(
        QuantumGrav.Weyl_squared_2D,
        coefs,
        [point],
    )
end
