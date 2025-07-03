@testsnippet imporModules begin
    using QuantumGrav
    using TestItemRunner
    using CausalSets
    using SparseArrays
end



@testitem "test_get_manifold_name" tags=[:utils] setup=[importModules] begin
    @test get_manifold_name(CausalSets.MinkowskiManifold, 2) == "Minkowski"
    @test get_manifold_name(CausalSets.DeSitterManifold, 2) == "DeSitter"
    @test get_manifold_name(CausalSets.AntiDeSitterManifold, 2) == "AntiDeSitter"
    @test get_manifold_name(CausalSets.HypercylinderManifold, 2) == "HyperCylinder"
    @test get_manifold_name(CausalSets.TorusManifold, 2) == "Torus"
    @test get_manifold_name(PseudoManifold, 2) == "Random"

    @test get_manifold_name(CausalSets.MinkowskiManifold, 3) == "Minkowski"
    @test get_manifold_name(CausalSets.DeSitterManifold, 3) == "DeSitter"
    @test get_manifold_name(CausalSets.AntiDeSitterManifold, 3) == "AntiDeSitter"
    @test get_manifold_name(CausalSets.HypercylinderManifold, 3) == "HyperCylinder"
    @test get_manifold_name(CausalSets.TorusManifold, 3) == "Torus"
    @test get_manifold_name(PseudoManifold, 3) == "Random"

    @test get_manifold_name(CausalSets.MinkowskiManifold, 4) == "Minkowski"
    @test get_manifold_name(CausalSets.DeSitterManifold, 4) == "DeSitter"
    @test get_manifold_name(CausalSets.AntiDeSitterManifold, 4) == "AntiDeSitter"
    @test get_manifold_name(CausalSets.HypercylinderManifold, 4) == "HyperCylinder"
    @test get_manifold_name(CausalSets.TorusManifold, 4) == "Torus"
    @test get_manifold_name(PseudoManifold, 4) == "Random"

    @test_throws ArgumentError get_manifold_name(CausalSets.MinkowskiManifold, 1)
    @test_throws ArgumentError get_manifold_name(CausalSets.MinkowskiManifold, 5)
end

@testitem "test_get_manifold_encoding" tags=[:utils] setup=[importModules] begin
    @test get_manifold_encoding["Minkowski"] == 1
    @test get_manifold_encoding["DeSitter"] == 3
    @test get_manifold_encoding["AntiDeSitter"] == 4
    @test get_manifold_encoding["HyperCylinder"] == 2
    @test get_manifold_encoding["Torus"] == 5
    @test get_manifold_encoding["Random"] == 6

    @test_throws KeyError get_manifold_encoding["UnknownManifold"]
end

@testitem "test_make_manifold" tags=[:utils] setup=[importModules] begin
    @test isa(make_manifold(1, 2), CausalSets.MinkowskiManifold{2})
    @test isa(make_manifold(2, 2), CausalSets.HypercylinderManifold{2})
    @test isa(make_manifold(3, 2), CausalSets.DeSitterManifold{2})
    @test isa(make_manifold(4, 2), CausalSets.AntiDeSitterManifold{2})
    @test isa(make_manifold(5, 2), CausalSets.TorusManifold{2})
    @test isa(make_manifold(6, 2), PseudoManifold{2})

    @test isa(make_manifold(1, 3), CausalSets.MinkowskiManifold{3})
    @test isa(make_manifold(2, 3), CausalSets.HypercylinderManifold{3})
    @test isa(make_manifold(3, 3), CausalSets.DeSitterManifold{3})
    @test isa(make_manifold(4, 3), CausalSets.AntiDeSitterManifold{3})
    @test isa(make_manifold(5, 3), CausalSets.TorusManifold{3})
    @test isa(make_manifold(6, 3), PseudoManifold{3})

    @test isa(make_manifold(1, 4), CausalSets.MinkowskiManifold{4})
    @test isa(make_manifold(2, 4), CausalSets.HypercylinderManifold{4})
    @test isa(make_manifold(3, 4), CausalSets.DeSitterManifold{4})
    @test isa(make_manifold(4, 4), CausalSets.AntiDeSitterManifold{4})
    @test isa(make_manifold(5, 4), CausalSets.TorusManifold{4})
    @test isa(make_manifold(6, 4), PseudoManifold{4})

    @test_throws ArgumentError make_manifold(0, 2)
    @test_throws ArgumentError make_manifold(7, 2)
    @test_throws ArgumentError make_manifold(1, 1)
    @test_throws ArgumentError make_manifold(1, 5)
end

@testitem "test_resize" tags=[:utils] setup=[importModules] begin
    A = spzeros(Float32, 5, 5)
    A[1, 1] = 1.0
    A[2, 2] = 2.0
    A[3, 3] = 3.0
    A[4, 4] = 4.0
    A[5, 5] = 5.0

    B = resize(A, 3, 4)

    @test size(B) == (3, 4)
    @test B[1, 1] == 1.0
    @test B[2, 2] == 2.0
    @test B[3, 3] == 3.0

    C = resize(A, 6, 6)
    @test size(C) == (6, 6)
    @test C[1, 1] == 1.0
    @test C[2, 2] == 2.0
    @test C[3, 3] == 3.0
    @test C[4, 4] == 4.0
    @test C[5, 5] == 5.0
    @test C[6, 6] == 0.0

    @test_throws Error D = resize(A, 2, 2, 2) # no N-D sparse arrays in Julia

    D = rand(Float32, 5, 5)
    D[1, 1] = 1.0
    D[2, 2] = 2.0
    D[3, 3] = 3.0
    D[4, 4] = 4.0
    D[5, 5] = 5.0

    E = resize(D, 3, 4)
    @test size(E) == (3, 4)
    @test E[1, 1] == 1.0
    @test E[2, 2] == 2.0
    @test E[3, 3] == 3.0

    F = resize(D, 6, 6)
    @test size(F) == (6, 6)
    @test F[1, 1] == 1.0
    @test F[2, 2] == 2.0
    @test F[3, 3] == 3.0
    @test F[4, 4] == 4.0
    @test F[5, 5] == 5.0
    @test F[6, 6] == 0.0

    G = resize(D, 2, 2, 2)
    @test size(G) == (2, 2, 2)
    @test G[1, 1, 0] == 1.0
    @test G[2, 2, 0] == 2.0
    @test G[1, 1, 2] == 0.0

end

@testitem "test_make_pseudosprinkling" tags=[:utils] setup=[importModules] begin
    @testitem 3 == 6
end

@testitem "test_topsort" tags=[:utils] setup=[importModules] begin
    @testitem 3 == 6
end
