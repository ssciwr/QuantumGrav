@testsnippet RegularizedCurvature begin
    import CausalSets
    import Random
    invariants = (
        QuantumGrav.Ricci_scalar_2D,
        QuantumGrav.Kretschmann_scalar_2D,
        QuantumGrav.Goroff_Sagnotti_scalar_2D,
        QuantumGrav.Ricci_box_Ricci_2D,
        QuantumGrav.Weyl_squared_2D,
        QuantumGrav.Weyl_box_Weyl_2D,
    )
end

@testitem "test_regularized_curvature flat" tags=[:regularized] setup=[RegularizedCurvature] begin
    point=CausalSets.Coordinates{2}((0.0, 0.0))
    for constant in (0.0, -2.0, 1.0), epsilon in (0.01, 1.0), invariant in invariants
        @test invariant(fill(constant, 1, 1), point, epsilon) == 0.0
    end
end

@testitem "test_regularized_curvature analytic" tags=[:regularized] setup=[
    RegularizedCurvature,
] begin
    epsilon=0.3
    for axis in (1, 2), u in (0.0, -0.4, 0.7)

        coefs=zeros(2, 2)
        coefs[axis==1 ? 2 : 1, axis==1 ? 1 : 2]=1.0
        point=CausalSets.Coordinates{2}(axis==1 ? (u, 0.2) : (0.2, u))
        density=u^2+epsilon
        R=(axis==1 ? 1 : -1)*2*(epsilon-u^2)/density^3
        RboxR=(axis==1 ? 1 : -1) * 16 * (epsilon-u^2) * (5*u^4-17*epsilon*u^2+2*epsilon^2)/density^9
        @test isapprox(QuantumGrav.Ricci_scalar_2D(coefs, point, epsilon), R; rtol = 1e-12)
        @test QuantumGrav.Kretschmann_scalar_2D(coefs, point, epsilon) ≈ R^2
        @test QuantumGrav.Goroff_Sagnotti_scalar_2D(coefs, point, epsilon) ≈ R^3
        @test QuantumGrav.Ricci_box_Ricci_2D(coefs, point, epsilon) ≈ RboxR
        @test QuantumGrav.Weyl_squared_2D(coefs, point, epsilon) == 0.0
        @test QuantumGrav.Weyl_box_Weyl_2D(coefs, point, epsilon) == 0.0
    end
end

@testitem "test_regularized_curvature mixed fourth derivatives" tags=[:regularized] setup=[
    RegularizedCurvature,
] begin
    coefs=zeros(5, 5)
    coefs[1, 1]=0.8
    coefs[2, 1]=0.2
    coefs[1, 2]=0.1
    coefs[2, 2]=0.05
    coefs[5, 1]=0.04
    coefs[3, 4]=0.03
    t, x, epsilon=0.2, -0.3, 0.4
    point=CausalSets.Coordinates{2}((t, x))
    h=1e-4
    R(t, x) = QuantumGrav.Ricci_scalar_2D(coefs, CausalSets.Coordinates{2}((t, x)), epsilon)
    F=QuantumGrav.chebyshev_evaluate_2D(coefs, point)^2+epsilon
    R_tt=(R(t+h, x)-2*R(t, x)+R(t-h, x))/h^2
    R_xx=(R(t, x+h)-2*R(t, x)+R(t, x-h))/h^2
    @test isapprox(
        QuantumGrav.Ricci_box_Ricci_2D(coefs, point, epsilon),
        R(t, x) * (-R_tt + R_xx) / F;
        rtol = 1e-5,
    )

    for invariant in invariants
        @test invariant(coefs, point, epsilon) ≈ invariant(-coefs, point, epsilon)
        @test isapprox(invariant(coefs, point, 1e-10), invariant(coefs, point); rtol = 1e-7)
    end
end

@testitem "test_regularized_curvature_of_sprinkling" tags=[:regularized] setup=[
    RegularizedCurvature,
] begin
    _, points, coefs=QuantumGrav.make_regularized_polynomial_manifold_cset(
        48,
        Random.Xoshiro(42),
        3,
        2.0,
        0.2;
        type = Float64,
    )
    D1=QuantumGrav.chebyshev_derivation_matrix(3, 1)
    D2=QuantumGrav.chebyshev_derivation_matrix(3, 2)
    for invariant in invariants
        values=QuantumGrav.curvature_invariant_of_sprinkling(invariant, coefs, points, 0.2)
        @test length(values) == 48
        @test all(isfinite, values)
        @test values ≈ [invariant(coefs, point, 0.2) for point in points]
        @test values == QuantumGrav.curvature_invariant_of_sprinkling(
            invariant,
            coefs,
            points,
            0.2;
            derivation_matrix1 = D1,
            derivation_matrix2 = D2,
        )
        @test isempty(
            QuantumGrav.curvature_invariant_of_sprinkling(
                invariant,
                coefs,
                CausalSets.Coordinates{2}[],
                0.2,
            ),
        )
    end
    @test QuantumGrav.Ricci_scalar_2D_of_sprinkling(coefs, points, 0.2) ==
          QuantumGrav.curvature_invariant_of_sprinkling(
        QuantumGrav.Ricci_scalar_2D,
        coefs,
        points,
        0.2,
    )
end

@testitem "test_integrated_curvature_invariant_2D analytic" tags=[:regularized] setup=[
    RegularizedCurvature,
] begin
    # P=t, epsilon=1. Independent antiderivatives of the rational integrands
    # give these values on [-1,1]^2, including the volume factor t^2+1.
    coefs=zeros(2, 2)
    coefs[2, 1]=1.0
    expected=(4.0, 8/3+7*pi/8, 454/105+11*pi/8, 1031/105+101*pi/32, 0.0, 0.0)
    for (invariant, exact) in zip(invariants, expected)
        value, error=QuantumGrav.integrated_curvature_invariant_2D(
            invariant,
            coefs,
            1.0;
            rtol = 1e-8,
            atol = 1e-10,
        )
        @test isapprox(value, exact; rtol = 2e-8, atol = 1e-10)
        @test 0 <= error <= max(1e-10, 1e-8 * abs(value))
    end

    # A custom constant invariant integrates to the proper volume, not coordinate area.
    constant(coefs, point, epsilon; kwargs...) = 1.0
    boundary=CausalSets.BoxBoundary{2}(((-0.5, -0.3), (0.8, 0.7)))
    volume, error=QuantumGrav.integrated_curvature_invariant_2D(
        constant,
        coefs,
        0.4;
        boundary = boundary,
    )
    @test volume ≈ (0.8^3 - (-0.5)^3) / 3 + 0.4 * 1.3
    value, error=QuantumGrav.integrated_curvature_invariant_2D(
        QuantumGrav.Ricci_scalar_2D,
        coefs,
        0.4;
        boundary = boundary,
    )
    @test isapprox(value, 2 * (0.8 / (0.8^2 + 0.4) + 0.5 / (0.5^2 + 0.4)); rtol = 1e-6)

    # Every invariant also integrates to zero for a flat regularized geometry.
    for invariant in invariants
        @test QuantumGrav.integrated_curvature_invariant_2D(invariant, zeros(1, 1), 0.2) ==
              (0.0, 0.0)
    end
end

@testitem "test_regularized_curvature throws" tags=[:regularized, :throws] setup=[
    RegularizedCurvature,
] begin
    point=CausalSets.Coordinates{2}((0.2, 0.3))
    for invariant in invariants
        for epsilon in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError invariant(zeros(2, 2), point, epsilon)
            @test_throws ArgumentError QuantumGrav.curvature_invariant_of_sprinkling(
                invariant,
                zeros(2, 2),
                [point],
                epsilon,
            )
            @test_throws ArgumentError QuantumGrav.integrated_curvature_invariant_2D(
                invariant,
                zeros(2, 2),
                epsilon,
            )
        end
        for coefs in (zeros(2, 3), zeros(0, 0), fill(NaN, 2, 2))
            @test_throws ArgumentError invariant(coefs, point, 0.1)
            @test_throws ArgumentError QuantumGrav.integrated_curvature_invariant_2D(
                invariant,
                coefs,
                0.1,
            )
        end
        @test_throws ArgumentError invariant(
            zeros(2, 2),
            point,
            0.1;
            derivation_matrix1 = zeros(3, 3),
        )
    end
    for edges in (((-2.0, -1.0), (1.0, 1.0)), ((-1.0, -1.0), (Inf, 1.0)))
        @test_throws ArgumentError QuantumGrav.integrated_curvature_invariant_2D(
            QuantumGrav.Ricci_scalar_2D,
            zeros(2, 2),
            0.1;
            boundary = CausalSets.BoxBoundary{2}(edges),
        )
    end
    for kwargs in (
        (rtol = -1.0,),
        (atol = -1.0,),
        (rtol = NaN,),
        (atol = Inf,),
        (rtol = 0.0, atol = 0.0),
        (maxevals = 0,),
    )
        @test_throws ArgumentError QuantumGrav.integrated_curvature_invariant_2D(
            QuantumGrav.Ricci_scalar_2D,
            zeros(2, 2),
            0.1;
            kwargs...,
        )
    end
end
