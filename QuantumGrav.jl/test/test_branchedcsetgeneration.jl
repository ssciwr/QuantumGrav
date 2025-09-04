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

@testitem "assign_branch" tags = [:branchedcsetgeneration] setup = [branchedtests] begin
    polym = CausalSets.PolynomialManifold{2}(randn(rng, 3, 3))
    sprinkling = [(x, -0.5) for x in range(-0.9, stop=0.9, length=9)]
    branch_points = QuantumGrav.generate_random_branch_points([(x, 0.0) for x in range(-0.9, stop=0.9, length=5)], 5)

    # Each point below the branch should be assigned correctly
    for x in sprinkling
        b = QuantumGrav.assign_branch(x, branch_points, polym)
        @test 1 ≤ b ≤ 2 * length(branch_points) + 1
    end
end

@testitem "assign_branch_throws" tags = [:branchedcsetgeneration, :throws] setup = [branchedtests] begin
    polym = CausalSets.PolynomialManifold{2}(randn(rng, 3, 3))
    sprinkling = [(x, -0.5) for x in range(-0.9, stop=0.9, length=9)]
    branch_points = QuantumGrav.generate_random_branch_points([(x, 0.0) for x in range(-0.9, stop=0.9, length=5)], 5)

    # throw on unsorted input
    unsorted = reverse(branch_points)
    @test_throws ArgumentError QuantumGrav.assign_branch((0.0, 0.0), unsorted, polym)
end

@testitem "compute_branch_relations" tags = [:branchedcsetgeneration, :throws] setup = [branchedtests] begin
    polym = CausalSets.PolynomialManifold{2}(randn(rng, 3, 3))
    branch_points = [(i / 10, i / 20) for i in 1:5]  # already time-sorted

    rel = QuantumGrav.compute_branch_relations(polym, branch_points)
    n = length(branch_points)

    # correct size
    @test length(rel) == 2n + 1
    @test all(length(r) == 2n + 1 for r in rel)

    # reflexivity
    @test all(rel[i][i] for i in 1:2n+1)

    # branch 1 in past of all
    @test all(rel[1][j] for j in 1:2n+1)
end

@testitem "compute_branch_relations_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    polym = CausalSets.PolynomialManifold{2}(randn(rng, 3, 3))
    branch_points = [(i / 10, i / 20) for i in 1:5]
    # throw on unsorted input
    reversed = reverse(branch_points)
    @test_throws ArgumentError QuantumGrav.compute_branch_relations(polym, reversed)
end

@testitem "in_past_of" tags = [:branchedcsetgeneration, :branch_points] setup = [branchedtests] begin
    polym = CausalSets.PolynomialManifold{2}(randn(rng, 3, 3))
    x = QuantumGrav.BranchedCoordinates{2}((0.0, 0.0), 1)
    y = QuantumGrav.BranchedCoordinates{2}((0.1, 0.0), 2)

    rel = [BitVector([i ≤ j for j in 1:3]) for i in 1:3]

    # valid causal past
    @test CausalSets.in_past_of(polym, x, y, rel)

    # not in past via branch
    x2 = QuantumGrav.BranchedCoordinates{2}((0.0, 0.0), 3)
    y2 = QuantumGrav.BranchedCoordinates{2}((0.1, 0.0), 1)
    @test !CausalSets.in_past_of(polym, x2, y2, rel)
end

@testitem "in_past_of_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    polym = CausalSets.PolynomialManifold{2}(randn(rng, 3, 3))
    x = QuantumGrav.BranchedCoordinates{2}((0.0, 0.0), 1)
    y = QuantumGrav.BranchedCoordinates{2}((0.1, 0.0), 2)

    rel = [BitVector([i ≤ j for j in 1:3]) for i in 1:3]

    # throw on invalid branch indices
    xbad = QuantumGrav.BranchedCoordinates{2}((0.0, 0.0), 0)
    ybad = QuantumGrav.BranchedCoordinates{2}((0.0, 0.0), 4)
    @test_throws ArgumentError CausalSets.in_past_of(polym, xbad, y, rel)
    @test_throws ArgumentError CausalSets.in_past_of(polym, x, ybad, rel)
end

@testitem "BranchedManifoldCauset constructor" tags = [:branchedcsetgeneration, :branch_points] setup = [branchedtests] begin
    polym = CausalSets.PolynomialManifold{2}(randn(rng, 3, 3))
    coords = [QuantumGrav.BranchedCoordinates{2}((i / 10, 0.0), 1) for i in 1:10]
    rel = [falses(2n + 1) for n in 1:10]  # dummy relation matrix

    causet = QuantumGrav.BranchedManifoldCauset(polym, coords, rel)
    @test causet isa QuantumGrav.BranchedManifoldCauset
    @test causet.atom_count == length(coords)
    @test causet.sprinkling[1] isa QuantumGrav.BranchedCoordinates{2}
end

@testitem "in_past_of_unchecked" tags = [:branchedcsetgeneration, :branch_points] setup = [branchedtests] begin
    polym = CausalSets.PolynomialManifold{2}(randn(rng, 3, 3))
    x = QuantumGrav.BranchedCoordinates{2}((0.0, 0.0), 1)
    y = QuantumGrav.BranchedCoordinates{2}((0.1, 0.0), 2)
    sprinkling = [x, y]
    rel = [BitVector([true, true]), BitVector([false, true])]

    causet = QuantumGrav.BranchedManifoldCauset(polym, sprinkling, rel)
    @test CausalSets.in_past_of_unchecked(causet, 1, 2)
    @test !CausalSets.in_past_of_unchecked(causet, 2, 1)
end

@testitem "convert to BitArrayCauset" tags = [:branchedcsetgeneration, :branch_points] setup = [branchedtests] begin
    polym = CausalSets.PolynomialManifold{2}(randn(rng, 3, 3))
    x = QuantumGrav.BranchedCoordinates{2}((0.0, 0.0), 1)
    y = QuantumGrav.BranchedCoordinates{2}((0.1, 0.0), 2)
    sprinkling = [x, y]
    rel = [BitVector([true, true]), BitVector([false, true])]
    causet = QuantumGrav.BranchedManifoldCauset(polym, sprinkling, rel)

    bitcset = CausalSets.BitArrayCauset(causet)
    @test bitcset isa CausalSets.BitArrayCauset
    @test bitcset.atom_count == 2
    @test bitcset.future_relations[1][2] == true
    @test bitcset.past_relations[2][1] == true
end