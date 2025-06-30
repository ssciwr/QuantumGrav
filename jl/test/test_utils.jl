using TestItems

@testsnippet importModules begin
    import CausalSets
    import SparseArrays
    import Distributions
    import Random
    import Graphs
end

@testitem "get_manifold_names" tags=[:utils] setup=[importModules] begin
    @test 3 == 6
end

@testitem "make_manifold" tags=[:utils] setup=[importModules] begin
    @test 3 == 6
end

@testitem "resize" tags=[:utils] setup=[importModules] begin
    @test 3 == 6
end

@testitem "make_pseudosprinkling" tags=[:utils] setup=[importModules] begin
    @test 3 == 6
end

@testitem "topsort" tags=[:utils] setup=[importModules] begin
    @test 3 == 6
end
