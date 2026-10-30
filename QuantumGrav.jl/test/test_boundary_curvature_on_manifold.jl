@testitem "test_boundary_curvature_invariant_2D analytic faces" tags=[:regularized] setup=[
    RegularizedCurvature,
] begin
    coefs=zeros(2, 2)
    coefs[2, 1]=1.0 # P=t
    epsilon=0.4
    lo, hi, width=-0.5, 0.7, 1.1
    boundary=CausalSets.BoxBoundary{2}(((lo, -0.3), (hi, 0.8)))
    F(t) = t^2+epsilon
    R(t) = 2*(epsilon-t^2)/F(t)^3
    R_t(t) = 8*t*(t^2-2*epsilon)/F(t)^4
    boxR(t) = 8*(5*t^4-17*epsilon*t^2+2*epsilon^2)/F(t)^6
    expected_lower=(
        2*lo/F(lo),
        4*R(lo)*lo/F(lo),
        6*R(lo)^2*lo/F(lo),
        4*boxR(lo)*lo/F(lo)-R(lo)*R_t(lo),
        0.0,
        0.0,
    )
    expected_upper=(
        -2*hi/F(hi),
        -4*R(hi)*hi/F(hi),
        -6*R(hi)^2*hi/F(hi),
        -4*boxR(hi)*hi/F(hi)+R(hi)*R_t(hi),
        0.0,
        0.0,
    )
    for (invariant, lower, upper) in zip(invariants, expected_lower, expected_upper)
        result=QuantumGrav.boundary_curvature_invariant_2D(
            invariant,
            coefs,
            epsilon;
            boundary,
        )
        @test isapprox(
            result.faces.t_lower.integral,
            width * lower;
            rtol = 1e-12,
            atol = 1e-12,
        )
        @test isapprox(
            result.faces.t_upper.integral,
            width * upper;
            rtol = 1e-12,
            atol = 1e-12,
        )
        @test result.faces.x_lower.integral == 0.0
        @test result.faces.x_upper.integral == 0.0
        @test result.joint == 0.0
        @test result.integral ≈ width * (lower + upper)
        @test result.error >= 0
        @test result.error == sum(face.error for face in result.faces)
    end

    flux=QuantumGrav.ricci_box_ricci_boundary_flux_2D(coefs, epsilon; boundary)
    @test flux.integral ≈ width * (R(lo) * R_t(lo) - R(hi) * R_t(hi))
    @test flux.faces.t_lower.integral ≈ width * R(lo) * R_t(lo)
    @test flux.faces.t_upper.integral ≈ -width * R(hi) * R_t(hi)
end

@testitem "test_einstein_hilbert_action_2D cancellation" tags=[:regularized] setup=[
    RegularizedCurvature,
] begin
    # Dependence on both coordinates exercises all four face orientations.
    coefs=[0.8 0.2 0.03; 0.3 0.07 0.02; 0.04 0.01 0.005]
    boundary=CausalSets.BoxBoundary{2}(((-0.7, -0.8), (0.9, 0.6)))
    action=QuantumGrav.einstein_hilbert_action_2D(
        coefs,
        0.2;
        boundary,
        rtol = 1e-8,
        atol = 1e-10,
    )
    @test abs(action.bulk) > 1e-3
    @test all(face -> abs(face.integral) > 1e-3, action.faces)
    @test abs(action.total) <= action.error + 1e-10
    @test isapprox(action.bulk, -action.boundary; rtol = 1e-8)
    @test action.total == action.bulk + action.boundary + action.joint
    @test action.error == action.bulk_error + action.boundary_error

    # A spacelike normal changes the sign relative to time-dependent P.
    spacecoefs=zeros(2, 2)
    spacecoefs[1, 2]=1.0
    spaceaction=QuantumGrav.einstein_hilbert_action_2D(spacecoefs, 1.0)
    @test spaceaction.bulk ≈ -4.0
    @test spaceaction.boundary ≈ 4.0
    @test abs(spaceaction.total) <= spaceaction.error + 1e-10
end

@testitem "test_Ricci_box_Ricci_2D Green identity" tags=[:regularized] setup=[
    RegularizedCurvature,
] begin
    coefs=zeros(2, 2)
    coefs[2, 1]=1.0
    epsilon=0.6
    boundary=CausalSets.BoxBoundary{2}(((-0.6, -0.8), (0.9, 0.5)))
    function gradient_squared(coefs, point, epsilon; kwargs...)
        t=point[1]
        F=t^2+epsilon
        R_t=8*t*(t^2-2*epsilon)/F^4
        return -R_t^2/F
    end
    bulk, bulk_error=QuantumGrav.integrated_curvature_invariant_2D(
        QuantumGrav.Ricci_box_Ricci_2D,
        coefs,
        epsilon;
        boundary,
        rtol = 1e-8,
    )
    gradient, gradient_error=QuantumGrav.integrated_curvature_invariant_2D(
        gradient_squared,
        coefs,
        epsilon;
        boundary,
        rtol = 1e-8,
    )
    flux=QuantumGrav.ricci_box_ricci_boundary_flux_2D(coefs, epsilon; boundary, rtol = 1e-8)
    @test abs(bulk + gradient - flux.integral) <=
          bulk_error + gradient_error + flux.error + 1e-8

    # A point with R=0 still has a finite, nonzero box R. Boundary terms
    # must compute box R directly, rather than divide R box R by R.
    root_boundary=CausalSets.BoxBoundary{2}(((-sqrt(epsilon), -0.4), (0.8, 0.6)))
    result=QuantumGrav.boundary_curvature_invariant_2D(
        QuantumGrav.Ricci_box_Ricci_2D,
        coefs,
        epsilon;
        boundary = root_boundary,
    )
    @test isfinite(result.integral)
    @test abs(result.faces.t_lower.integral) > 1.0
end

@testitem "test_curvature_action_2D flat and consistency" tags=[:regularized] setup=[
    RegularizedCurvature,
] begin
    for invariant in invariants
        flat=QuantumGrav.curvature_action_2D(invariant, zeros(1, 1), 0.2)
        @test flat.bulk == flat.boundary == flat.joint == flat.total == flat.error == 0.0
    end
    coefs=[0.7 0.1; 0.2 0.05]
    D1=QuantumGrav.chebyshev_derivation_matrix(1, 1)
    D2=QuantumGrav.chebyshev_derivation_matrix(1, 2)
    for invariant in invariants
        result=QuantumGrav.curvature_action_2D(invariant, coefs, 0.2)
        bulk, _=QuantumGrav.integrated_curvature_invariant_2D(invariant, coefs, 0.2)
        boundary=QuantumGrav.boundary_curvature_invariant_2D(invariant, coefs, 0.2)
        @test result.bulk == bulk
        @test result.boundary == boundary.integral
        @test result.total == bulk + boundary.integral
        @test result == QuantumGrav.curvature_action_2D(
            invariant,
            coefs,
            0.2;
            derivation_matrix1 = D1,
            derivation_matrix2 = D2,
        )
    end
    flux=QuantumGrav.ricci_box_ricci_boundary_flux_2D(zeros(1, 1), 0.2)
    @test flux.integral == flux.error == 0.0
end

@testitem "test_boundary_curvature_2D throws" tags=[:regularized, :throws] setup=[
    RegularizedCurvature,
] begin
    for invariant in invariants, epsilon in (0.0, -1.0, Inf, NaN)

        @test_throws ArgumentError QuantumGrav.boundary_curvature_invariant_2D(
            invariant,
            zeros(2, 2),
            epsilon,
        )
        @test_throws ArgumentError QuantumGrav.curvature_action_2D(
            invariant,
            zeros(2, 2),
            epsilon,
        )
        @test_throws ArgumentError QuantumGrav.ricci_box_ricci_boundary_flux_2D(
            zeros(2, 2),
            epsilon,
        )
    end
    constant(coefs, point, epsilon; kwargs...) = 1.0
    @test_throws ArgumentError QuantumGrav.curvature_action_2D(constant, zeros(2, 2), 0.1)
    for coefs in (zeros(2, 3), zeros(0, 0), fill(NaN, 2, 2))
        @test_throws ArgumentError QuantumGrav.boundary_curvature_invariant_2D(
            QuantumGrav.Ricci_scalar_2D,
            coefs,
            0.1,
        )
        @test_throws ArgumentError QuantumGrav.ricci_box_ricci_boundary_flux_2D(coefs, 0.1)
    end
    for kwargs in (
        (rtol = -1.0,),
        (atol = Inf,),
        (maxevals = 0,),
        (boundary = CausalSets.BoxBoundary{2}(((-2.0, -1.0), (1.0, 1.0))),),
    )
        @test_throws ArgumentError QuantumGrav.boundary_curvature_invariant_2D(
            QuantumGrav.Ricci_scalar_2D,
            zeros(2, 2),
            0.1;
            kwargs...,
        )
        @test_throws ArgumentError QuantumGrav.ricci_box_ricci_boundary_flux_2D(
            zeros(2, 2),
            0.1;
            kwargs...,
        )
    end
end
