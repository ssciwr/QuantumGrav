
@testitem "test_get_manifolds_for_dim" tags=[:utils] setup=[importModules] begin

    manifolds = QuantumGrav.DataGeneration.get_manifolds_of_dim(2)

    expected_manifolds = ["minkowski", "hypercylinder", "deSitter", "antiDeSitter", "torus"]

    @test Set(keys(manifolds)) == Set(expected_manifolds)

    manifolds = QuantumGrav.DataGeneration.get_manifolds_of_dim(21)

    @test Set(keys(manifolds)) == Set(expected_manifolds)
end

@testitem "test_get_manifold_name" tags=[:utils] setup=[importModules] begin 
    @testitem 3 == 6
end

@testitem "test_get_manifold_encoding" tags=[:utils] setup=[importModules] begin
    @testitem 3 == 6
end

@testitem "test_make_manifold" tags=[:utils] setup=[importModules] begin
    @testitem 3 == 6
end

@testitem "test_resize" tags=[:utils] setup=[importModules] begin
    @testitem 3 == 6
end

@testitem "test_make_pseudosprinkling" tags=[:utils] setup=[importModules] begin
    @testitem 3 == 6
end

@testitem "test_topsort" tags=[:utils] setup=[importModules] begin
    @testitem 3 == 6
end