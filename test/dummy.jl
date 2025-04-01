using TestItems
using Test

@testitem "dummy test" begin
    x = foo("bar")  # Now foo should be accessible

    @test length(x) == 3
    @test x == "bar"
    @test x != "baz"
end