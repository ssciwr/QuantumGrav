


@testsnippet setupTestsQuasicrystal begin
    import Distributions
    import Random
    import CausalSets
    import LinearAlgebra

    Random.seed!(42)  # Set a seed for reproducibility
    rng = Random.Xoshiro(42)
end

@testitem "test_quasicrystal_geometry_and_sorting" tags = [:quasicrystal] setup = [setupTestsQuasicrystal] begin
    αin, αout = QuantumGrav.quasicrystal(2.0)

    @test αin isa Vector{Float64}
    @test αout isa Vector{Float64}
    @test length(αin) == length(αout)
    @test length(αin) > 0
    @test issorted(αin)
    @test all(isfinite, αin)
    @test all(isfinite, αout)
    @test all(0.0 .<= αin .<= 1.0)
    @test all(0.0 .<= αout .<= 1.0)
end

@testitem "test_quasicrystal_exhaustive_enumeration_finds_all_points" tags = [:quasicrystal] setup =
    [setupTestsQuasicrystal] begin
    ρ = 2.0
    xmax = 15
    √ = sqrt

    vout = ComplexF64[
        -√(4 + √17) + (5 + √17)/2,
        (5 + √17 - 2*√(53/2 + (13*√17)/2)) / 4,
        (5 + √17 - 2*√(13/2 + (5*√17)/2)) / 4,
        1.0,
    ]

    vin = ComplexF64[
        √(4 + √17) + (5 + √17)/2,
        (5 + √17 + 2*√(53/2 + (13*√17)/2)) / 4,
        (5 + √17 + 2*√(13/2 + (5*√17)/2)) / 4,
        1.0,
    ]

    v1 = ComplexF64[
        (5 - √17 + 2im * √(-4 + √17)) / 2,
        (5 - √17 - 1im * √(2 * (-53 + 13 * √17))) / 4,
        (5 - √17 + 1im * √(2 * (-13 + 5 * √17))) / 4,
        1.0,
    ]

    η = LinearAlgebra.Diagonal(ComplexF64[-1.0, 1.0, 1.0, 1.0])
    MinkSp(x, y) = LinearAlgebra.dot(conj.(x), η * y)
    den_causal = MinkSp(vin, vout)
    den_window = MinkSp(v1, conj.(v1))

    brute_set = Set{Tuple{Float64,Float64}}()
    found_on_box_boundary = false

    for x0 in -xmax:xmax, x1 in -xmax:xmax, x2 in -xmax:xmax, x3 in -xmax:xmax
        xvec = ComplexF64[x0, x1, x2, x3]

        αout_raw = MinkSp(xvec, vin) / den_causal
        αin_raw = MinkSp(xvec, vout) / den_causal
        window_raw =
            (MinkSp(xvec, v1) / den_window) * (MinkSp(conj.(v1), xvec) / conj(den_window))

        # Small numerical imaginary parts are expected from floating-point arithmetic.
        if abs(imag(αin_raw)) > 1e-9 || abs(imag(αout_raw)) > 1e-9 || abs(imag(window_raw)) > 1e-9
            continue
        end

        αin = real(αin_raw)
        αout = real(αout_raw)
        window_val = real(window_raw)

        if 0.0 <= αin <= 1.0 && 0.0 <= αout <= 1.0 && window_val <= ρ^2
            push!(brute_set, (round(αin, digits = 12), round(αout, digits = 12)))
            if abs(x0) == xmax || abs(x1) == xmax || abs(x2) == xmax || abs(x3) == xmax
                found_on_box_boundary = true
            end
        end
    end

    αin, αout = QuantumGrav.quasicrystal(ρ)
    qc_set = Set((round(a, digits = 12), round(b, digits = 12)) for (a, b) in zip(αin, αout))

    @test found_on_box_boundary == false
    @test brute_set == qc_set
end

@testitem "test_translate_sub_spacetime_crystal_exact_size_from_precomputed_crystal" tags =
    [:quasicrystal] setup = [setupTestsQuasicrystal] begin
    npoints = 80
    center = CausalSets.Coordinates{2}((0.5, 0.5))
    crystal = QuantumGrav.quasicrystal(2.0)

    coords = QuantumGrav.translate_sub_spacetime_crystal(
        npoints,
        center;
        crystal = crystal,
        exact_size = true,
        deviation_from_mean_size = 0.3,
        max_iter = 200,
    )

    @test coords isa Vector{CausalSets.Coordinates{2}}
    @test length(coords) == npoints
    @test all(p -> 0.0 <= p[1] <= 1.0, coords)
    @test all(p -> 0.0 <= p[2] <= 1.0, coords)
end

@testitem "test_translate_sub_spacetime_crystal_non_exact_matches_filter_definition" tags =
    [:quasicrystal] setup = [setupTestsQuasicrystal] begin
    npoints = 120
    center = CausalSets.Coordinates{2}((0.5, 0.5))
    crystal = QuantumGrav.quasicrystal(2.0)
    αin, αout = crystal
    halfℓ = sqrt(npoints / length(αin)) / 2
    αin_lo = center[1] - halfℓ
    αin_hi = center[1] + halfℓ
    αout_lo = center[2] - halfℓ
    αout_hi = center[2] + halfℓ

    expected = NTuple{2,Float64}[]
    i_lo = searchsortedfirst(αin, αin_lo)
    i_hi = searchsortedlast(αin, αin_hi)
    if i_lo <= i_hi
        for i in i_lo:i_hi
            if αout_lo <= αout[i] <= αout_hi
                push!(expected, (αin[i], αout[i]))
            end
        end
    end

    coords = QuantumGrav.translate_sub_spacetime_crystal(
        npoints,
        center;
        crystal = crystal,
        exact_size = false,
    )

    @test coords == expected
end

@testitem "test_translate_sub_spacetime_crystal_throws" tags = [:quasicrystal,:throws] setup =
    [setupTestsQuasicrystal] begin
    @test_throws ErrorException QuantumGrav.translate_sub_spacetime_crystal(
        100,
        CausalSets.Coordinates{2}((0.5, 0.5)),
    )

    @test_throws ErrorException QuantumGrav.translate_sub_spacetime_crystal(
        100,
        CausalSets.Coordinates{2}((0.5, 0.5));
        crystal = (Float64[], Float64[]),
    )

    @test_throws ErrorException QuantumGrav.translate_sub_spacetime_crystal(
        100,
        CausalSets.Coordinates{2}((0.01, 0.01));
        ρ = 2.0,
    )
end

@testitem "test_create_Minkowski_quasicrystal_cset_from_precomputed_crystal" tags =
    [:quasicrystal] setup = [setupTestsQuasicrystal] begin
    npoints = 90
    crystal = QuantumGrav.quasicrystal(2.0)

    cset = QuantumGrav.create_Minkowski_quasicrystal_cset(
        npoints,
        CausalSets.Coordinates{2}((0.5, 0.5));
        crystal = crystal,
        exact_size = true,
        deviation_from_mean_size = 0.3,
        max_iter = 200,
    )

    @test typeof(cset) === CausalSets.BitArrayCauset
    @test cset.atom_count == npoints
end

@testitem "test_create_Minkowski_quasicrystal_cset" tags = [:quasicrystal] setup =
    [setupTestsQuasicrystal] begin
    npoints = 100

    cset = QuantumGrav.create_Minkowski_quasicrystal_cset(
        npoints,
        CausalSets.Coordinates{2}((0.5,0.5));
        ρ = 2.,
        exact_size = true,
        deviation_from_mean_size = .1,
        max_iter = 100,
    )

    @test typeof(cset) === CausalSets.BitArrayCauset
    @test cset.atom_count == npoints
    @test length(cset.future_relations) == npoints
    @test length(cset.past_relations) == npoints
end


@testitem "test_create_Minkowski_quasicrystal_cset_throws" tags = [:quasicrystal,:throws] setup =
    [setupTestsQuasicrystal] begin

    # either ρ or crystal must be provided
    @test_throws ErrorException     cset = QuantumGrav.create_Minkowski_quasicrystal_cset(
        100,
        CausalSets.Coordinates{2}((0.,0.))
    )

end
