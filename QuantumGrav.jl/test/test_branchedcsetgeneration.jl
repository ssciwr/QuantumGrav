using TestItems

@testsnippet branchedtests begin
    using QuantumGrav: QuantumGrav
    using CausalSets: CausalSets
    using Distributions: Distributions
    using Random: Random

    Random.seed!(42)  # Set a seed for reproducibility
    rng = Random.Xoshiro(42)
    npoint_distribution = Distributions.DiscreteUniform(2, 1000)
    order_distribution = Distributions.DiscreteUniform(2, 9)
    r_distribution = Distributions.Uniform(1.0, 2.0)
end

@testitem "generate_random_branch_points" tags = [:branchedcsetgeneration, :branch_points] setup = [branchedtests] begin
    sprinkling = [(i / 10, i / 20) for i in 1:100]

    # correct number and time-sorted
    branch_points = QuantumGrav.generate_random_branch_points(sprinkling, 5; rng = rng)
    @test length(branch_points) == 5
    @test all(branch_points[i][1] ≤ branch_points[i+1][1] for i in 1:4)

    # edge case: n = 0
    empty = QuantumGrav.generate_random_branch_points(sprinkling, 0)
    @test isempty(empty)
end

@testitem "generate_random_branch_points_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    sprinkling = [(i / 10, i / 20) for i in 1:100]

    # throw: too many branch points
    @test_throws ArgumentError QuantumGrav.generate_random_branch_points(sprinkling, 101)

    # throw: negative input
    @test_throws ArgumentError QuantumGrav.generate_random_branch_points(sprinkling, -1)
end

@testitem "in_past_of" tags = [:branchedcsetgeneration, :branch_points] setup = [branchedtests] begin
    polym = CausalSets.PolynomialManifold{2}(randn(rng, 3, 3))
    x = (0.0, 0.0)
    y = (0.0, 0.1)
    z = (0.2, 0.2)

    no_branch_points = Vector{CausalSets.Coordinates{2}}()
    branch_point = [y]

    

    # valid causal past
    @test CausalSets.in_past_of(polym, x, z, no_branch_points)

    # not in past via branch
    @test !CausalSets.in_past_of(polym, x, z, branch_point)
end

@testitem "BranchedManifoldCauset constructor" tags = [:branchedcsetgeneration, :branch_points] setup = [branchedtests] begin
    polym = CausalSets.PolynomialManifold{2}(randn(rng, 3, 3))
    coords = [(i / 10, 0.0) for i in 1:10]
    branch_points = [coords[3], coords[4]] # dummy relation matrix

    causet = QuantumGrav.BranchedManifoldCauset(polym, coords, branch_points)
    @test causet isa QuantumGrav.BranchedManifoldCauset
    @test causet.atom_count == length(coords)
    @test causet.sprinkling[1] isa CausalSets.Coordinates{2}
end

@testitem "in_past_of_unchecked" tags = [:branchedcsetgeneration, :branch_points] setup = [branchedtests] begin
    polym = CausalSets.PolynomialManifold{2}(randn(rng, 3, 3))
    x = (0.0, 0.0)
    y = (0.0, 0.1)
    z = (0.3, 0.2)
    sprinkling = [x, y, z]
    branch_points = [y]

    causet = QuantumGrav.BranchedManifoldCauset(polym, sprinkling, branch_points)
    @test !CausalSets.in_past_of_unchecked(causet, 1, 3)
    @test !CausalSets.in_past_of_unchecked(causet, 1, 2)
end

@testitem "convert to BitArrayCauset" tags = [:branchedcsetgeneration, :branch_points] setup = [branchedtests] begin
    polym = CausalSets.PolynomialManifold{2}(randn(rng, 3, 3))
    x = (0.0, 0.0)
    y = (0.0, 0.1)
    z = (0.1, 0.2)
    sprinkling = [x, y, z]
    branch_points = [y]
    causet = QuantumGrav.BranchedManifoldCauset(polym, sprinkling, branch_points)

    bitcset = CausalSets.BitArrayCauset(causet)
    @test bitcset isa CausalSets.BitArrayCauset
    @test bitcset.atom_count == 3
    @test bitcset.future_relations[1][3] == false
    @test bitcset.past_relations[3][1] == false
end

@testitem "make_branched_manifold_cset" tags = [:branchedcsetgeneration, :generation] setup = [branchedtests] begin
    npoints = 100
    nbranchpoints = 5
    order = 4
    r = 1.5

    cset, sprinkling, branch_points, coefs = QuantumGrav.make_branched_manifold_cset(npoints, nbranchpoints, rng, order, r)

    @test cset isa CausalSets.BitArrayCauset
    @test length(sprinkling) == npoints
    @test length(branch_points) == nbranchpoints
    @test size(coefs) == (order, order)
    @test cset.atom_count == npoints
end

@testitem "make_branched_manifold_cset_branches_implemented_correctly" tags = [:branchedcsetgeneration, :branch_points] setup = [branchedtests] begin
    polym = CausalSets.PolynomialManifold{2}(randn(rng, 3, 3))
    x = (0.0, 0.0)
    y = (0.0, 0.1)
    z = (0.3, 0.2)
    sprinkling = [x, y, z]
    branch_points = [y]

    manifold_causet = convert(CausalSets.BitArrayCauset, CausalSets.ManifoldCauset(polym, sprinkling))
    branched_causet = convert(CausalSets.BitArrayCauset, QuantumGrav.BranchedManifoldCauset(polym, sprinkling, branch_points))
    @test manifold_causet.future_relations[1][3]  # x and z are connected in ManifoldCauset
    @test !branched_causet.future_relations[1][3] # x and z are disconnected in BranchedManifoldCauset because y cuts through
end

@testitem "make_branched_manifold_cset throws" tags = [:branchedcsetgeneration, :generation, :throws] setup = [branchedtests] begin
    order = 4
    r = 1.5

    @test_throws ArgumentError QuantumGrav.make_branched_manifold_cset(0, 0, rng, order, r)
    @test_throws ArgumentError QuantumGrav.make_branched_manifold_cset(10, 11, rng, order, r)
    @test_throws ArgumentError QuantumGrav.make_branched_manifold_cset(10, -1, rng, order, r)
    @test_throws ArgumentError QuantumGrav.make_branched_manifold_cset(10, 5, rng, 0, r)
    @test_throws ArgumentError QuantumGrav.make_branched_manifold_cset(10, 5, rng, order, 0.5)
    @test_throws ArgumentError QuantumGrav.make_branched_manifold_cset(10, 5, rng, order, r; d=3)
end