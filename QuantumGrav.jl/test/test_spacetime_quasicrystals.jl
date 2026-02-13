


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
