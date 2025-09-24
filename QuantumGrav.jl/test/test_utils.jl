using TestItems

@testsnippet importModules begin
    using QuantumGrav
    using TestItemRunner
    using CausalSets
    using SparseArrays
    using Random
    using Distributions
end

@testitem "test_make_boundary" tags = [:utils] setup = [importModules] begin
    @test isa(
        QuantumGrav.make_boundary("CausalDiamond", 2),
        CausalSets.CausalDiamondBoundary{2},
    )
    @test isa(
        QuantumGrav.make_boundary("CausalDiamond", 3),
        CausalSets.CausalDiamondBoundary{3},
    )
    @test isa(
        QuantumGrav.make_boundary("CausalDiamond", 4),
        CausalSets.CausalDiamondBoundary{4},
    )
    @test_throws ArgumentError QuantumGrav.make_boundary("UnknownBoundary", 2)
    @test_throws ArgumentError QuantumGrav.make_boundary("CausalDiamond", 1)
end

@testitem "test_make_manifold" tags = [:utils] setup = [importModules] begin
    @test isa(QuantumGrav.make_manifold("Minkowski", 2), CausalSets.MinkowskiManifold{2})
    @test isa(QuantumGrav.make_manifold("DeSitter", 2), CausalSets.DeSitterManifold{2})
    @test isa(
        QuantumGrav.make_manifold("AntiDeSitter", 2),
        CausalSets.AntiDeSitterManifold{2},
    )
    @test isa(
        QuantumGrav.make_manifold("HyperCylinder", 2),
        CausalSets.HypercylinderManifold{2},
    )
    @test isa(QuantumGrav.make_manifold("Torus", 2), CausalSets.TorusManifold{2})

    @test isa(QuantumGrav.make_manifold("Minkowski", 3), CausalSets.MinkowskiManifold{3})
    @test isa(QuantumGrav.make_manifold("DeSitter", 3), CausalSets.DeSitterManifold{3})
    @test isa(
        QuantumGrav.make_manifold("AntiDeSitter", 3),
        CausalSets.AntiDeSitterManifold{3},
    )
    @test isa(
        QuantumGrav.make_manifold("HyperCylinder", 3),
        CausalSets.HypercylinderManifold{3},
    )
    @test isa(QuantumGrav.make_manifold("Torus", 3), CausalSets.TorusManifold{3})

    @test isa(QuantumGrav.make_manifold("Minkowski", 4), CausalSets.MinkowskiManifold{4})
    @test isa(QuantumGrav.make_manifold("DeSitter", 4), CausalSets.DeSitterManifold{4})
    @test isa(
        QuantumGrav.make_manifold("AntiDeSitter", 4),
        CausalSets.AntiDeSitterManifold{4},
    )
    @test isa(
        QuantumGrav.make_manifold("HyperCylinder", 4),
        CausalSets.HypercylinderManifold{4},
    )
    @test isa(QuantumGrav.make_manifold("Torus", 4), CausalSets.TorusManifold{4})

    @test_throws ArgumentError QuantumGrav.make_manifold("Minkowski", 0)

    @test_throws ArgumentError QuantumGrav.make_manifold("UnknownManifold", 2)
end

@testitem "test_make_pseudosprinkling" tags = [:utils] setup = [importModules] begin
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
