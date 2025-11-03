
@testitem "test_make_pseudosprinkling" tags = [:utils] begin

    import Random
    n = 10
    d = 3
    box_min = -1.0
    box_max = 1.0
    type = Float32
    rng = Random.MersenneTwister(1234)

    sprinkling = QuantumGrav.make_pseudosprinkling(n, d, box_min, box_max, type; rng = rng)

    @test length(sprinkling) == n
    @test all(length(s) == d for s in sprinkling)
    @test all(all(x -> x >= box_min && x <= box_max, s) for s in sprinkling)
    @test all(eltype(s) == type for s in sprinkling)

    @test_throws ArgumentError QuantumGrav.make_pseudosprinkling(
        n,
        d,
        box_max,
        box_min,
        type;
        rng = rng,
    ) # box_min must be less than box_max
end
