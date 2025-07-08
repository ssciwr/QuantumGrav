@testsnippet importModules begin
    using QuantumGrav
    using TestItemRunner
    using CausalSets
    using SparseArrays
    using Random
    using Distributions
end

@testitem "test_get_manifold_name" tags=[:utils] setup=[importModules] begin
    @test QuantumGrav.get_manifold_name(CausalSets.MinkowskiManifold{2}) == "Minkowski"
    @test QuantumGrav.get_manifold_name(CausalSets.DeSitterManifold{2}) == "DeSitter"
    @test QuantumGrav.get_manifold_name(CausalSets.AntiDeSitterManifold{2}) ==
          "AntiDeSitter"
    @test QuantumGrav.get_manifold_name(CausalSets.HypercylinderManifold{2}) ==
          "HyperCylinder"
    @test QuantumGrav.get_manifold_name(CausalSets.TorusManifold{2}) == "Torus"
    @test QuantumGrav.get_manifold_name(QuantumGrav.PseudoManifold{2}) == "Random"

    @test QuantumGrav.get_manifold_name(CausalSets.MinkowskiManifold{3}) == "Minkowski"
    @test QuantumGrav.get_manifold_name(CausalSets.DeSitterManifold{3}) == "DeSitter"
    @test QuantumGrav.get_manifold_name(CausalSets.AntiDeSitterManifold{3}) ==
          "AntiDeSitter"
    @test QuantumGrav.get_manifold_name(CausalSets.HypercylinderManifold{3}) ==
          "HyperCylinder"
    @test QuantumGrav.get_manifold_name(CausalSets.TorusManifold{3}) == "Torus"
    @test QuantumGrav.get_manifold_name(QuantumGrav.PseudoManifold{3}) == "Random"

    @test QuantumGrav.get_manifold_name(CausalSets.MinkowskiManifold{4}) == "Minkowski"
    @test QuantumGrav.get_manifold_name(CausalSets.DeSitterManifold{4}) == "DeSitter"
    @test QuantumGrav.get_manifold_name(CausalSets.AntiDeSitterManifold{4}) ==
          "AntiDeSitter"
    @test QuantumGrav.get_manifold_name(CausalSets.HypercylinderManifold{4}) ==
          "HyperCylinder"
    @test QuantumGrav.get_manifold_name(CausalSets.TorusManifold{4}) == "Torus"
    @test QuantumGrav.get_manifold_name(QuantumGrav.PseudoManifold{4}) == "Random"

    @test_throws KeyError QuantumGrav.get_manifold_name(CausalSets.MinkowskiManifold{1})
    @test_throws KeyError QuantumGrav.get_manifold_name(CausalSets.MinkowskiManifold{5})
end

@testitem "test_get_manifold_encoding" tags=[:utils] setup=[importModules] begin
    @test QuantumGrav.get_manifold_encoding["Minkowski"] == 1
    @test QuantumGrav.get_manifold_encoding["DeSitter"] == 3
    @test QuantumGrav.get_manifold_encoding["AntiDeSitter"] == 4
    @test QuantumGrav.get_manifold_encoding["HyperCylinder"] == 2
    @test QuantumGrav.get_manifold_encoding["Torus"] == 5
    @test QuantumGrav.get_manifold_encoding["Random"] == 6
    @test_throws KeyError QuantumGrav.get_manifold_encoding["UnknownManifold"]
end


@testitem "test_make_boundary" tags=[:utils] setup=[importModules] begin
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
    @test_throws KeyError QuantumGrav.make_boundary("UnknownBoundary", 2)
    @test_throws ArgumentError QuantumGrav.make_boundary("CausalDiamond", 1)
end
@test SparseArrays.nnz(angles) > 0

@testitem "test_make_manifold_from_name" tags=[:utils] setup=[importModules] begin
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
    @test isa(QuantumGrav.make_manifold("Random", 2), QuantumGrav.PseudoManifold{2})

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
    @test isa(QuantumGrav.make_manifold("Random", 3), QuantumGrav.PseudoManifold{3})

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
    @test isa(QuantumGrav.make_manifold("Random", 4), QuantumGrav.PseudoManifold{4})
end

@testitem "test_make_manifold_index" tags=[:utils] setup=[importModules] begin

    @test QuantumGrav.make_manifold(1, 2) isa CausalSets.MinkowskiManifold{2}

    @test isa(QuantumGrav.make_manifold(2, 2), CausalSets.HypercylinderManifold{2})
    @test isa(QuantumGrav.make_manifold(3, 2), CausalSets.DeSitterManifold{2})
    @test isa(QuantumGrav.make_manifold(4, 2), CausalSets.AntiDeSitterManifold{2})
    @test isa(QuantumGrav.make_manifold(5, 2), CausalSets.TorusManifold{2})
    @test isa(QuantumGrav.make_manifold(6, 2), QuantumGrav.PseudoManifold{2})

    @test isa(QuantumGrav.make_manifold(1, 3), CausalSets.MinkowskiManifold{3})
    @test isa(QuantumGrav.make_manifold(2, 3), CausalSets.HypercylinderManifold{3})
    @test isa(QuantumGrav.make_manifold(3, 3), CausalSets.DeSitterManifold{3})
    @test isa(QuantumGrav.make_manifold(4, 3), CausalSets.AntiDeSitterManifold{3})
    @test isa(QuantumGrav.make_manifold(5, 3), CausalSets.TorusManifold{3})
    @test isa(QuantumGrav.make_manifold(6, 3), QuantumGrav.PseudoManifold{3})

    @test isa(QuantumGrav.make_manifold(1, 4), CausalSets.MinkowskiManifold{4})
    @test isa(QuantumGrav.make_manifold(2, 4), CausalSets.HypercylinderManifold{4})
    @test isa(QuantumGrav.make_manifold(3, 4), CausalSets.DeSitterManifold{4})
    @test isa(QuantumGrav.make_manifold(4, 4), CausalSets.AntiDeSitterManifold{4})
    @test isa(QuantumGrav.make_manifold(5, 4), CausalSets.TorusManifold{4})
    @test isa(QuantumGrav.make_manifold(6, 4), QuantumGrav.PseudoManifold{4})
    @test_throws KeyError QuantumGrav.make_manifold(0, 2)
    @test_throws KeyError QuantumGrav.make_manifold(7, 2)
    @test_throws ArgumentError QuantumGrav.make_manifold(1, 1)
    @test_throws ArgumentError QuantumGrav.make_manifold(1, 5)
end

@testitem "test_resize" tags=[:utils] setup=[importModules] begin
    A = SparseArrays.spzeros(Float32, 5, 5)
    A[1, 1] = 1.0
    A[2, 2] = 2.0
    A[3, 3] = 3.0
    A[4, 4] = 4.0
    A[5, 5] = 5.0

    B = QuantumGrav.resize(A, (3, 4))

    @test size(B) == (3, 4)
    @test B[1, 1] == 1.0
    @test B[2, 2] == 2.0
    @test B[3, 3] == 3.0

    C = QuantumGrav.resize(A, (6, 6))
    @test size(C) == (6, 6)
    @test C[1, 1] == 1.0
    @test C[2, 2] == 2.0
    @test C[3, 3] == 3.0
    @test C[4, 4] == 4.0
    @test C[5, 5] == 5.0
    @test C[6, 6] == 0.0

    D = rand(Float32, 5, 5)
    D[1, 1] = 1.0
    D[2, 2] = 2.0
    D[3, 3] = 3.0
    D[4, 4] = 4.0
    D[5, 5] = 5.0

    E = QuantumGrav.resize(D, (3, 4))
    @test size(E) == (3, 4)
    @test E[1, 1] == 1.0
    @test E[2, 2] == 2.0
    @test E[3, 3] == 3.0

    F = QuantumGrav.resize(D, (6, 6))
    @test size(F) == (6, 6)
    @test F[1, 1] == 1.0
    @test F[2, 2] == 2.0
    @test F[3, 3] == 3.0
    @test F[4, 4] == 4.0
    @test F[5, 5] == 5.0
    @test F[6, 6] == 0.0
end

@testitem "test_make_pseudosprinkling" tags=[:utils] setup=[importModules] begin
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
