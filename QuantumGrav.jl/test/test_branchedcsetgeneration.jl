using TestItems

@testsnippet branchedtests begin
    using QuantumGrav
    using CausalSets
    using Distributions
    using Random

    Random.seed!(42)  # Set a seed for reproducibility
    rng = Random.Xoshiro(42)
    npoint_distribution = Distributions.DiscreteUniform(2, 1000)
    order_distribution = Distributions.DiscreteUniform(2, 9)
    r_distribution = Distributions.Uniform(1.0, 2.0)
end

@testitem "are_colinear_overlapping" tags = [:branchedcsetgeneration, :branch_points] setup = [branchedtests] begin
    seg1 = (CausalSets.Coordinates{2}((0., 0.)), CausalSets.Coordinates{2}((1., 0.)))
    seg2 = (CausalSets.Coordinates{2}((.5, 0.)), CausalSets.Coordinates{2}((1.5, 0.))) # colinear and overlapping with seg1
    seg3 = (CausalSets.Coordinates{2}((1.5, 0.)), CausalSets.Coordinates{2}((2., 0.))) # colinear but not overlapping with seg1
    seg4 = (CausalSets.Coordinates{2}((0.5, -.5)), CausalSets.Coordinates{2}((0.5, 0.5))) # overlapping but not colinear with seg1

    @test QuantumGrav.are_colinear_overlapping(seg1, seg2)
    @test !QuantumGrav.are_colinear_overlapping(seg1, seg3)
    @test !QuantumGrav.are_colinear_overlapping(seg1, seg4)
end

@testitem "are_colinear_overlapping_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    seg1 = (CausalSets.Coordinates{2}((0., 0.)), CausalSets.Coordinates{2}((1., 0.)))
    seg2 = (CausalSets.Coordinates{2}((.5, 0.)), CausalSets.Coordinates{2}((1.5, 0.)))
    tolerance = -1. # negative tolerance

    @test_throws ArgumentError QuantumGrav.are_colinear_overlapping(seg1, seg2; tolerance = tolerance)
end

@testitem "is_colinear_overlapping_with_cuts" tags=[:branchedcsetgeneration, :branch_points] setup=[branchedtests] begin
    seg = (CausalSets.Coordinates{2}((0.0, 1.0)), CausalSets.Coordinates{2}((1.0, 1.0)))
    single = [CausalSets.Coordinates{2}((0.5, 1.0))]
    tuples = [(CausalSets.Coordinates{2}((0.2, 0.0)), CausalSets.Coordinates{2}((0.8, 0.0)))]
    # Should not overlap with tuples, but does with vertical cut at x=1.0
    @test QuantumGrav.is_colinear_overlapping_with_cuts(seg, single, tuples; tmax=1.0)
    # Negative result for non-overlapping segment
    seg2 = (CausalSets.Coordinates{2}((0.0, 0.2)), CausalSets.Coordinates{2}((1.0, 0.2)))
    @test !QuantumGrav.is_colinear_overlapping_with_cuts(seg2, single, tuples; tmax=1.0)
    
    seg3 = (CausalSets.Coordinates{2}((0.0, 0.0)), CausalSets.Coordinates{2}((0.5, 0.0)))
end

@testitem "is_colinear_overlapping_with_cuts_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    seg = (CausalSets.Coordinates{2}((0.0, 1.0)), CausalSets.Coordinates{2}((1.0, 1.0)))
    single = [CausalSets.Coordinates{2}((0.5, 1.0))]
    tuples = [(CausalSets.Coordinates{2}((0.2, 0.0)), CausalSets.Coordinates{2}((0.8, 0.0)))]
    tolerance = -1. # negative tolerance

    @test_throws ArgumentError QuantumGrav.is_colinear_overlapping_with_cuts(seg, single, tuples; tolerance = tolerance)
end

@testitem "interpolate_point" tags=[:branchedcsetgeneration] setup=[branchedtests] begin
    x = CausalSets.Coordinates{2}((0.0, 0.0))
    y = CausalSets.Coordinates{2}((1.0, 2.0))
    # Interpolate t=0.5 along segment
    p = QuantumGrav.interpolate_point(x, y, 0.5, 1)
    @test p == CausalSets.Coordinates{2}((0.5, 1.0))
    # Interpolate x at t=0.5
    xval = QuantumGrav.interpolate_point(x, y, 0.5, 1; idx_out=2)
    @test xval ≈ 1.0
end

@testitem "interpolate_point_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    x = CausalSets.Coordinates{2}((0.0, 1.0))
    y = CausalSets.Coordinates{2}((0.0, 2.0))
    tolerance = -1. # negative tolerance
    @test_throws ArgumentError QuantumGrav.interpolate_point(x, y, 0.5, 2; tolerance = tolerance)

    # Throws for segment parallel to requested axis
    @test_throws ArgumentError QuantumGrav.interpolate_point(x, y, 0.5, 1)
end

@testitem "segments_intersect" tags = [:branchedcsetgeneration, :branch_points] setup = [branchedtests] begin
    # Simple intersection (diagonal crossing)
    seg1 = (CausalSets.Coordinates{2}((0.0, 0.0)), CausalSets.Coordinates{2}((1.0, 1.0)))
    seg2 = (CausalSets.Coordinates{2}((0.0, 1.0)), CausalSets.Coordinates{2}((1.0, 0.0)))
    ok, pt = QuantumGrav.segments_intersect(seg1, seg2)
    @test ok
    @test pt === CausalSets.Coordinates{2}((0.5, 0.5))

    # Non-intersection
    seg3 = (CausalSets.Coordinates{2}((0.0, 0.0)), CausalSets.Coordinates{2}((1.0, 0.0)))
    seg4 = (CausalSets.Coordinates{2}((0.0, 1.0)), CausalSets.Coordinates{2}((1.0, 1.0)))
    ok2, pt2 = QuantumGrav.segments_intersect(seg3, seg4)
    @test !ok2
    @test pt2 === nothing

    # Colinear overlap returns (true, nothing)
    seg5 = (CausalSets.Coordinates{2}((0.0, 0.0)), CausalSets.Coordinates{2}((1.0, 0.0)))
    seg6 = (CausalSets.Coordinates{2}((0.5, 0.0)), CausalSets.Coordinates{2}((1.5, 0.0)))
    ok3, pt3 = QuantumGrav.segments_intersect(seg5, seg6)
    @test ok3
    @test pt3 === nothing
end

@testitem "segments_intersect_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    seg1 = (CausalSets.Coordinates{2}((0.0, 0.0)), CausalSets.Coordinates{2}((1.0, 1.0)))
    seg2 = (CausalSets.Coordinates{2}((0.0, 1.0)), CausalSets.Coordinates{2}((1.0, 0.0)))
    # Throws if tolerance <= 0
    @test_throws ArgumentError QuantumGrav.segments_intersect(seg1, seg2; tolerance=0.0)
    @test_throws ArgumentError QuantumGrav.segments_intersect(seg1, seg2; tolerance=-1.0)
end

@testitem "generate_random_branch_points" tags=[:branchedcsetgeneration] setup=[branchedtests] begin
    pts, cuts = QuantumGrav.generate_random_branch_points(8, 10; rng=rng, consecutive_intersections = false)
    @test length(pts) == 8
    @test length(cuts) == 10
    @test all(x -> x isa CausalSets.Coordinates{2}, pts)
    @test all(t -> t[1][1] <= t[2][1], cuts)
    @test issorted(pts, by=p->p[1])

    # Check intersection condition: no cut intersects more than once
    intersections = QuantumGrav.cut_intersections(cuts) # cut_intersections tested below
    counts = Dict(i => 0 for i in 1:length(cuts))
    for ((i,j),_) in intersections
        counts[i] += 1
        counts[j] += 1
    end
    @test all(v -> v <= 1, values(counts))
end

@testitem "generate_random_branch_points_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    @test_throws ArgumentError QuantumGrav.generate_random_branch_points(-1, 1)
    @test_throws ArgumentError QuantumGrav.generate_random_branch_points(1, -2)
end

@testitem "point_segment_distance" tags=[:branchedcsetgeneration] setup=[branchedtests] begin
    p = CausalSets.Coordinates{2}((0.5, 0.5)) # point whose distance is to be computed
    a = CausalSets.Coordinates{2}((0.0, 0.0)) # endpoint segment
    b = CausalSets.Coordinates{2}((1.0, 0.0)) # endpoint segment
    # Closest point in (a,b) to p is (0.5, 0.0): distance = 0.5
    d = QuantumGrav.point_segment_distance(p, (a, b))
    @test abs(d - 0.5) < 1e-12
    # Point on segment
    p2 = CausalSets.Coordinates{2}((0.5, 0.0))
    @test QuantumGrav.point_segment_distance(p2, (a, b)) < 1e-12
    # Degenerate segment
    @test QuantumGrav.point_segment_distance(p, (a, a)) ≈ sqrt(0.5^2 + 0.5^2)
    end

@testitem "point_segment_distance_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    p = CausalSets.Coordinates{2}((0.5, 0.5)) # point whose distance is to be computed
    a = CausalSets.Coordinates{2}((0.0, 0.0)) # endpoint segment
    b = CausalSets.Coordinates{2}((1.0, 0.0)) # endpoint segment
    # Throws for bad tolerance
    @test_throws ArgumentError QuantumGrav.point_segment_distance(p, (a, b); tolerance=0.)
end

@testitem "filter_sprinkling_near_cuts" tags=[:branchedcsetgeneration] setup=[branchedtests] begin
    spr = [
        CausalSets.Coordinates{2}((0.1, 0.0)),
        CausalSets.Coordinates{2}((0.5, 0.5)),
        CausalSets.Coordinates{2}((0.9, 1.0))
    ]
    single = [CausalSets.Coordinates{2}((0.3, 0.5))]
    tuples = [(CausalSets.Coordinates{2}((0.6, 0.0)), CausalSets.Coordinates{2}((0.6, 1.0)))]
    # Point (0.5, 0.5) is near vertical cut at x=0.5 if tolerance > 0.0
    filtered = QuantumGrav.filter_sprinkling_near_cuts(spr, (single, tuples); tolerance=0.11)
    @test length(filtered) == 2
    @test spr[2] ∉ filtered
end

@testitem "filter_sprinkling_near_cuts_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    # Throws for unsorted
    spr = [
        CausalSets.Coordinates{2}((0.1, 0.0)),
        CausalSets.Coordinates{2}((0.5, 0.5)),
        CausalSets.Coordinates{2}((0.9, 1.0))
    ]
    badspr = [spr[2], spr[1], spr[3]]
    single = [CausalSets.Coordinates{2}((0.3, 0.5))]
    tuples = [(CausalSets.Coordinates{2}((0.6, 0.0)), CausalSets.Coordinates{2}((0.6, 1.0)))]
    @test_throws ArgumentError QuantumGrav.filter_sprinkling_near_cuts(badspr, (single, tuples))
    # Throws for bad tolerance
    @test_throws ArgumentError QuantumGrav.filter_sprinkling_near_cuts(spr, (single, tuples); tolerance=0.)
end

@testitem "next_intersection" tags=[:branchedcsetgeneration] setup=[branchedtests] begin
    x = CausalSets.Coordinates{2}((0.0, 0.0))
    y = CausalSets.Coordinates{2}((1.0, 1.0))
    slope = 1.0
    cut = (CausalSets.Coordinates{2}((0.5, -0.5)), CausalSets.Coordinates{2}((0.5, 0.5)))
    manifold = CausalSets.MinkowskiManifold{2}()
    res = QuantumGrav.next_intersection(manifold, [cut], x, y, slope)
    @test res !== nothing
    pt, intersecting_cut = res
    @test intersecting_cut == cut
    @test pt == CausalSets.Coordinates{2}((0.5, 0.5))
    # No intersection if ray misses
    cut2 = (CausalSets.Coordinates{2}((2.0, 2.0)), CausalSets.Coordinates{2}((2.0, 3.0)))
    @test isnothing(QuantumGrav.next_intersection(manifold, [cut2], x, y, slope))
end

@testitem "next_intersection_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    x = CausalSets.Coordinates{2}((0.0, 0.0))
    y = CausalSets.Coordinates{2}((1.0, 1.0))
    slope = 1.0
    cut = (CausalSets.Coordinates{2}((0.5, -0.5)), CausalSets.Coordinates{2}((0.5, 0.5)))
    manifold = CausalSets.MinkowskiManifold{2}()
    tolerance = 0.
    # Throws for bad tolerance
    @test_throws ArgumentError  QuantumGrav.next_intersection(manifold, [cut], x, y, slope; tolerance = tolerance)
end


@testitem "diamond_corners" tags=[:branchedcsetgeneration] setup=[branchedtests] begin
    x = CausalSets.Coordinates{2}((0.0, 0.0))
    y = CausalSets.Coordinates{2}((2.0, 0.0))
    manifold = CausalSets.MinkowskiManifold{2}()
    l, r = QuantumGrav.diamond_corners(manifold, x, y)
    @test l == CausalSets.Coordinates{2}((1.0, -1.0))
    @test r == CausalSets.Coordinates{2}((1.0, 1.0))
end

@testitem "diamond_corners_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    x = CausalSets.Coordinates{2}((0.0, 0.0))
    y = CausalSets.Coordinates{2}((2.0, 0.0))
    manifold = CausalSets.MinkowskiManifold{2}()
    # Throws if not in past
    @test_throws ArgumentError QuantumGrav.diamond_corners(manifold, y, x)
end

@testitem "point_in_diamond" tags=[:branchedcsetgeneration] setup=[branchedtests] begin
    x = CausalSets.Coordinates{2}((0.0, 0.0))
    y = CausalSets.Coordinates{2}((2.0, 0.0))
    p1 = CausalSets.Coordinates{2}((1.0, 0.5))
    p2 = CausalSets.Coordinates{2}((3.0, 0.0))
    manifold = CausalSets.MinkowskiManifold{2}()
    @test QuantumGrav.point_in_diamond(manifold, p1, x, y)
    @test !QuantumGrav.point_in_diamond(manifold, p2, x, y)
end

@testitem "point_in_diamond_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    x = CausalSets.Coordinates{2}((0.0, 0.0))
    y = CausalSets.Coordinates{2}((2.0, 0.0))
    p = CausalSets.Coordinates{2}((1.0, 0.5))
    manifold = CausalSets.MinkowskiManifold{2}()
    # Throws if not in past
    @test_throws ArgumentError QuantumGrav.point_in_diamond(manifold, p, y, x)
end

@testitem "cut_crosses_diamond" tags=[:branchedcsetgeneration] setup=[branchedtests] begin
    x = CausalSets.Coordinates{2}((0.0, 0.0))
    y = CausalSets.Coordinates{2}((2.0, 0.0))
    cut1 = (CausalSets.Coordinates{2}((1.0, -1.5)), CausalSets.Coordinates{2}((1.0, 1.5))) # cuts diamond
    cut2 = (CausalSets.Coordinates{2}((0.5, -0.5)), CausalSets.Coordinates{2}((0.5, 0.3))) # cuts only one edge of the diamond
    manifold = CausalSets.MinkowskiManifold{2}()
    @test QuantumGrav.cut_crosses_diamond(manifold, x, y, cut1)
    @test !QuantumGrav.cut_crosses_diamond(manifold, x, y, cut2)
end

@testitem "cut_crosses_diamond_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    x = CausalSets.Coordinates{2}((0.0, 0.0))
    y = CausalSets.Coordinates{2}((2.0, 0.0))
    cut = (CausalSets.Coordinates{2}((1.0, -1.5)), CausalSets.Coordinates{2}((1.0, 1.5))) # cuts diamond
    manifold = CausalSets.MinkowskiManifold{2}()
    # Throws if not in past
    @test_throws ArgumentError QuantumGrav.cut_crosses_diamond(manifold, y, x, cut)    
    # Throws for bad tolerance
    tolerance = 0.
    @test_throws ArgumentError QuantumGrav.cut_crosses_diamond(manifold, x, y, cut; tolerance = tolerance)
end

@testitem "intersected_cut_crosses_diamond" tags=[:branchedcsetgeneration] setup=[branchedtests] begin
    x = CausalSets.Coordinates{2}((0.5, 0.1))
    y = CausalSets.Coordinates{2}((2.0, 0.0))
    cut1 = (CausalSets.Coordinates{2}((0.0, 0.0)), CausalSets.Coordinates{2}((2.0, 0.0)))
    cut2 = (CausalSets.Coordinates{2}((1.0, -0.1)), CausalSets.Coordinates{2}((1.0, 1.0))) # large cross with cut1 which intersects diamond
    cut3 = (CausalSets.Coordinates{2}((1.0, -0.1)), CausalSets.Coordinates{2}((1.0, 0.1))) # small cross with cut1 which only intersects one edge of diamond
    intersection = CausalSets.Coordinates{2}((1.0, 0.0))
    manifold = CausalSets.MinkowskiManifold{2}()
    @test QuantumGrav.intersected_cut_crosses_diamond(manifold, x, y, cut1, cut2, intersection)
    @test !QuantumGrav.intersected_cut_crosses_diamond(manifold, x, y, cut1, cut3, intersection)
end

@testitem "intersected_cut_crosses_diamond_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    x = CausalSets.Coordinates{2}((0.5, 0.1))
    y = CausalSets.Coordinates{2}((2.0, 0.0))
    cut1 = (CausalSets.Coordinates{2}((0.0, 0.0)), CausalSets.Coordinates{2}((2.0, 0.0)))
    cut2 = (CausalSets.Coordinates{2}((1.0, -0.1)), CausalSets.Coordinates{2}((1.0, 1.0))) 
    intersection = CausalSets.Coordinates{2}((1.0, 0.0))
    manifold = CausalSets.MinkowskiManifold{2}()
    
    # Throws for bad tolerance
    tolerance = 0.
    @test_throws ArgumentError QuantumGrav.intersected_cut_crosses_diamond(manifold, x, y, cut1, cut2, intersection; tolerance=tolerance)
end

@testitem "propagate_ray" tags=[:branchedcsetgeneration] setup=[branchedtests] begin
    x = CausalSets.Coordinates{2}((0.0, 0.0))
    y = CausalSets.Coordinates{2}((2.0, 0.0))
    manifold = CausalSets.MinkowskiManifold{2}()
    # No cuts: straight without obstructions
    path = QuantumGrav.propagate_ray(manifold, x, y, +1.0, Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}[])
    @test path[1] === x
    @test path[end][1] === y[1]
    @test length(path) === 2

    # backwards
    path_b = QuantumGrav.propagate_ray(manifold, y, x, +1.0, Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}[])
    @test path_b[1] === y
    @test path_b[end][1] === x[1]
    @test length(path_b) === 2


    # Completely intersected (spacelike)
    cut1 = (CausalSets.Coordinates{2}((1.0, -1.1)), CausalSets.Coordinates{2}((1.0, 1.1)))
    path1 = QuantumGrav.propagate_ray(manifold, x, y, 1.0, [cut1])
    @test path1[1] == x
    @test path1[end] == CausalSets.Coordinates{2}((1., -1.))
    @test length(path1) == 3

    # backwards
    path1_b = QuantumGrav.propagate_ray(manifold, y, x, 1.0, [cut1])
    @test path1_b[1] === y
    @test path1_b[end] === CausalSets.Coordinates{2}((1.0, 1.0))
    @test length(path1_b) === 3

    # Completely intersected but not in the way of ray (timelike)
    cut2 = (CausalSets.Coordinates{2}((0., -0.1)), CausalSets.Coordinates{2}((2.0, 0.1)))
    path2 = QuantumGrav.propagate_ray(manifold, x, y, 1.0, [cut2])
    @test path2[1] == x
    @test path2[end][1] == y[1]
    @test length(path2) == 2

    # backwards
    path2_b = QuantumGrav.propagate_ray(manifold, y, x, 1.0, [cut2])
    @test path2_b[1] === y
    @test path2_b[end][1] == x[1]
    @test length(path2_b) === 2

    # Halfway intersected such that y can still be reached (spacelike)
    cut3 = (CausalSets.Coordinates{2}((1.0, -0.9)), CausalSets.Coordinates{2}((1.0, 1.1)))
    path3 = QuantumGrav.propagate_ray(manifold, x, y, 1.0, [cut3])
    @test path3[1] == x
    @test path3[end][1] == y[1]
    @test length(path3) == 4

    # backwards
    path3_b = QuantumGrav.propagate_ray(manifold, y, x, 1.0, [cut3])
    @test path3_b[1] === y
    @test path3_b[end][1] == x[1]
    @test length(path3_b) === 2

    # Halfway intersected such that y can still be reached (timelike, one intersection with diamond edges)
    cut4 = (CausalSets.Coordinates{2}((0.0, 0.1)), CausalSets.Coordinates{2}((1., 0.2)))
    path4 = QuantumGrav.propagate_ray(manifold, x, y, 1.0, [cut4])
    @test path4[1] == x
    @test path4[end][1] == y[1]
    @test length(path4) == 4

    # backwards
    path4_b = QuantumGrav.propagate_ray(manifold, y, x, 1.0, [cut4])
    @test path4_b[1] === y
    @test path4_b[end][1] == x[1]
    @test length(path4_b) === 2

    # Halfway intersected such that y can still be reached (timelike, one intersection with diamond edges)
    cut5 = (CausalSets.Coordinates{2}((0.0, 0.5)), CausalSets.Coordinates{2}((2., 0.5)))
    path5 = QuantumGrav.propagate_ray(manifold, x, y, 1.0, [cut5])
    @test path5[1] == x
    @test path5[end] == CausalSets.Coordinates{2}((2., 0.5))
    @test length(path5) == 3

    # backwards
    path5_b = QuantumGrav.propagate_ray(manifold, y, x, 1.0, [cut5])
    @test path5_b[1] === y
    @test path5_b[end][1] == x[1]
    @test length(path5_b) === 2

    # Intersected by two cuts which don't cross diamond alone but conspire to inhibit reaching y[1]
    cuts6 = [(CausalSets.Coordinates{2}((0.0, 0.5)), CausalSets.Coordinates{2}((0.9, 0.5))), (CausalSets.Coordinates{2}((1.0, 0.8)), CausalSets.Coordinates{2}((1.0, -2.0)))]
    path6 = QuantumGrav.propagate_ray(manifold, x, y, 1.0, cuts6)
    @test path6[1] == x
    @test path6[end] == CausalSets.Coordinates{2}((1., -1.))
    @test length(path6) == 5

    # backwards
    path6_b = QuantumGrav.propagate_ray(manifold, y, x, 1.0, cuts6)
    @test path6_b[1] === y
    @test path6_b[end][1] != x[1]
    @test length(path6_b) === 5

    # intersecting cuts block whole diamond
    cuts7 = [(CausalSets.Coordinates{2}((0.0, 0.5)), CausalSets.Coordinates{2}((2.0, 0.5))), 
            (CausalSets.Coordinates{2}((1.0, -2.)), CausalSets.Coordinates{2}((1.0, 0.6)))]
    path7 = QuantumGrav.propagate_ray(manifold, x, y, 1.0, cuts7)
    @test path7[1] == x
    @test path7[end][1] != y[1]
    @test length(path7) == 4

    # backwards
    path7b = QuantumGrav.propagate_ray(manifold, y, x, 1.0, cuts7)
    @test path7b[1] == y
    @test path7b[end][1] != x[1]
    @test length(path7b) == 4

    # intersecting cuts conspire with third cut to block whole diamond
    cuts8 = [(CausalSets.Coordinates{2}((0.0, 0.5)), CausalSets.Coordinates{2}((2.0, 0.5))), 
            (CausalSets.Coordinates{2}((1.0, -.5)), CausalSets.Coordinates{2}((1.0, 0.6))), 
            (CausalSets.Coordinates{2}((1.1, -.3)), CausalSets.Coordinates{2}((1.1, -2.0)))]
    path8 = QuantumGrav.propagate_ray(manifold, x, y, 1.0, cuts8)
    @test path8[1] == x
    @test path8[end][1] != y[1]
    @test length(path8) == 6

    # intersecting cuts conspire with third cut to block whole diamond
    path8b = QuantumGrav.propagate_ray(manifold, y, x, 1.0, cuts8)
    @test path8b[1] == y
    @test path8b[end][1] != x[1]
    @test length(path8b) == 6
end

@testitem "propagate_ray_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    x = CausalSets.Coordinates{2}((0.0, 0.0))
    y = CausalSets.Coordinates{2}((2.0, 0.0))
    manifold = CausalSets.MinkowskiManifold{2}()
    
    # Throws for bad tolerance
    tolerance = 0.
    @test_throws ArgumentError QuantumGrav.propagate_ray(manifold, x, y, +1.0, Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}[]; tolerance=tolerance)
end


@testitem "cut_intersections" tags=[:branchedcsetgeneration] setup=[branchedtests] begin
    cuts = [
        (CausalSets.Coordinates{2}((0.0, 0.0)), CausalSets.Coordinates{2}((1.0, 1.0))),
        (CausalSets.Coordinates{2}((0.0, 1.0)), CausalSets.Coordinates{2}((1.0, 0.0))),
        (CausalSets.Coordinates{2}((0.5, 0.5)), CausalSets.Coordinates{2}((1.5, 1.5)))
    ]
    intersections = QuantumGrav.cut_intersections(cuts)
    # Only one intersection (1,2)
    @test any(x -> x[1] == (1,2) && x[2] == CausalSets.Coordinates{2}((0.5,0.5)), intersections)
end

@testitem "cut_intersections_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    cuts = [
        (CausalSets.Coordinates{2}((0.0, 0.0)), CausalSets.Coordinates{2}((1.0, 1.0))),
        (CausalSets.Coordinates{2}((0.0, 1.0)), CausalSets.Coordinates{2}((1.0, 0.0))),
        (CausalSets.Coordinates{2}((0.5, 0.5)), CausalSets.Coordinates{2}((1.5, 1.5)))
    ]
    # Throws for bad tolerance
    tolerance = 0.
    @test_throws ArgumentError QuantumGrav.cut_intersections(cuts; tolerance=tolerance)
end

@testitem "in_wedge_of" tags=[:branchedcsetgeneration] setup=[branchedtests] begin
    x = CausalSets.Coordinates{2}((0.0, 0.0))
    y = CausalSets.Coordinates{2}((2.0, 0.0))
    manifold = CausalSets.MinkowskiManifold{2}()
    # No cuts: straight without obstructions
    no_cuts = Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}[]
    @test QuantumGrav.in_wedge_of(manifold, no_cuts, x, y)

    # Completely intersected (spacelike)
    cuts1 = [(CausalSets.Coordinates{2}((1.0, -1.1)), CausalSets.Coordinates{2}((1.0, 1.1)))]
    @test !QuantumGrav.in_wedge_of(manifold, cuts1, x, y)

    # Completely intersected (timelike)
    cuts2 = [(CausalSets.Coordinates{2}((0., -0.1)), CausalSets.Coordinates{2}((2.0, 0.1)))]
    @test !QuantumGrav.in_wedge_of(manifold, cuts2, x, y)

    # Halfway intersected such that y can still be reached (spacelike)
    cuts3 = [(CausalSets.Coordinates{2}((1.0, -0.9)), CausalSets.Coordinates{2}((1.0, 1.1)))]
    @test QuantumGrav.in_wedge_of(manifold, cuts3, x, y)

    # Halfway intersected such that y can still be reached (timelike, one intersection with diamond edges)
    cuts4 = [(CausalSets.Coordinates{2}((0.0, 0.1)), CausalSets.Coordinates{2}((1., 0.2)))]
    @test QuantumGrav.in_wedge_of(manifold, cuts4, x, y)

    # Halfway intersected such that y can still be reached (timelike, two intersections with diamond edges)
    cuts5 = [(CausalSets.Coordinates{2}((0.0, 0.5)), CausalSets.Coordinates{2}((2., 0.5)))]
    @test QuantumGrav.in_wedge_of(manifold, cuts5, x, y)

    # Intersected by two cuts which don't cross diamond alone, but conspire to inhibit reaching y
    cuts6 = [(CausalSets.Coordinates{2}((0.0, 0.5)), CausalSets.Coordinates{2}((0.9, 0.5))), (CausalSets.Coordinates{2}((1.0, 0.8)), CausalSets.Coordinates{2}((1.0, -2.0)))]
    @test !QuantumGrav.in_wedge_of(manifold, cuts6, x, y)

    # Intersected by two cuts which don't cross diamond alone and do not conspire to inhibit reaching y
    cuts7 = [(CausalSets.Coordinates{2}((0.0, 0.5)), CausalSets.Coordinates{2}((0.9, 0.5))), (CausalSets.Coordinates{2}((1.0, 0.5)), CausalSets.Coordinates{2}((1.0, -2.0)))]
    @test QuantumGrav.in_wedge_of(manifold, cuts7, x, y)
end

@testitem "in_wedge_of_throws" tags = [:branchedcsetgeneration, :branch_points, :throws] setup = [branchedtests] begin
    x = CausalSets.Coordinates{2}((0.0, 0.0))
    y = CausalSets.Coordinates{2}((2.0, 0.0))
    manifold = CausalSets.MinkowskiManifold{2}()
    no_cuts = Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}[]
    # Throws for bad tolerance
    tolerance = 0.
    @test_throws ArgumentError QuantumGrav.in_wedge_of(manifold, no_cuts, x, y; tolerance=tolerance)
end

@testitem "in_past_of (branched)" tags=[:branchedcsetgeneration] setup=[branchedtests] begin
    x = CausalSets.Coordinates{2}((0.0, 0.0))
    y = CausalSets.Coordinates{2}((2.0, 0.0))
    manifold = CausalSets.MinkowskiManifold{2}()
    # No cuts: straight without obstructions
    no_timelike_cuts = CausalSets.Coordinates{2}[]
    no_finite_cuts = Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}[]
    @test CausalSets.in_past_of(manifold, (no_timelike_cuts, no_finite_cuts), x, y)

    # Completely intersected (spacelike)
    cuts1 = [(CausalSets.Coordinates{2}((1.0, -1.1)), CausalSets.Coordinates{2}((1.0, 1.1)))]
    @test !CausalSets.in_past_of(manifold, (no_timelike_cuts, cuts1), x, y)

    # Completely intersected (timelike)
    cuts2 = [(CausalSets.Coordinates{2}((0., -0.1)), CausalSets.Coordinates{2}((2.0, 0.1)))]
    @test !CausalSets.in_past_of(manifold, (no_timelike_cuts, cuts2), x, y)

    # Halfway intersected such that y can still be reached (spacelike)
    cuts3 = [(CausalSets.Coordinates{2}((1.0, -0.9)), CausalSets.Coordinates{2}((1.0, 1.1)))]
    @test CausalSets.in_past_of(manifold, (no_timelike_cuts, cuts3), x, y)

    # Halfway intersected such that y can still be reached (timelike, one intersection with diamond edges)
    cuts4 = [(CausalSets.Coordinates{2}((0.0, 0.1)), CausalSets.Coordinates{2}((1., 0.2)))]
    @test CausalSets.in_past_of(manifold, (no_timelike_cuts, cuts4), x, y)

    # Halfway intersected such that y can still be reached (timelike, two intersections with diamond edges)
    cuts5 = [(CausalSets.Coordinates{2}((0.0, 0.5)), CausalSets.Coordinates{2}((2., 0.5)))]
    @test CausalSets.in_past_of(manifold, (no_timelike_cuts, cuts5), x, y)

    # Intersected by two cuts which don't cross diamond alone, but conspire to inhibit reaching y
    cuts6 = [(CausalSets.Coordinates{2}((0.0, 0.5)), CausalSets.Coordinates{2}((0.9, 0.5))), (CausalSets.Coordinates{2}((1.0, 0.8)), CausalSets.Coordinates{2}((1.0, -2.0)))]
    @test !CausalSets.in_past_of(manifold, (no_timelike_cuts, cuts6), x, y)

    # Intersected by two cuts which don't cross diamond alone and do not conspire to inhibit reaching y
    cuts7 = [(CausalSets.Coordinates{2}((0.0, 0.5)), CausalSets.Coordinates{2}((0.9, 0.5))), (CausalSets.Coordinates{2}((1.0, 0.5)), CausalSets.Coordinates{2}((1.0, -2.0)))]
    @test CausalSets.in_past_of(manifold, (no_timelike_cuts, cuts7), x, y)

    # fully intersected by timelike boundary-connecting cut
    y2 = CausalSets.Coordinates((2.0, 1.0))
    timelike_cuts1 = [CausalSets.Coordinates{2}((0.0, 0.5))]
    @test !CausalSets.in_past_of(manifold, (timelike_cuts1, no_finite_cuts), x, y2)

    # fully intersected by timelike boundary-connecting cut
    y2 = CausalSets.Coordinates((2.0, 1.0))
    timelike_cuts1 = [CausalSets.Coordinates{2}((0.0, 0.5))]
    @test !CausalSets.in_past_of(manifold, (timelike_cuts1, no_finite_cuts), x, y2)
end

@testitem "in_past_of (branched) throws" tags=[:branchedcsetgeneration, :throws] setup=[branchedtests] begin
    # Throws for N != 2
    x3 = CausalSets.Coordinates{3}((0.0,0.0,0.0))
    y3 = CausalSets.Coordinates{3}((1.0,0.0,0.0))
    manifold3 = CausalSets.MinkowskiManifold{3}()
    branch_info3 = (CausalSets.Coordinates{3}[], Tuple{CausalSets.Coordinates{3},CausalSets.Coordinates{3}}[])
    @test_throws ArgumentError CausalSets.in_past_of(manifold3, branch_info3, x3, y3)
    # Throws for bad tolerance
    x = CausalSets.Coordinates{2}((0.0, 0.0))
    y = CausalSets.Coordinates{2}((2.0, 0.0))
    no_timelike_cuts = CausalSets.Coordinates{2}[]
    no_finite_cuts = Tuple{CausalSets.Coordinates{2},CausalSets.Coordinates{2}}[]
    manifold = CausalSets.MinkowskiManifold{2}()
    tolerance = 0.0
    @test_throws ArgumentError CausalSets.in_past_of(manifold, (no_timelike_cuts, no_finite_cuts), x, y; tolerance=tolerance)
end

@testitem "BranchedManifoldCauset constructor and in_past_of_unchecked" tags=[:branchedcsetgeneration] setup=[branchedtests] begin
    polym = CausalSets.PolynomialManifold{2}(randn(rng, 3, 3))
    coords = [CausalSets.Coordinates{2}((i / 10, 0.0)) for i in 1:10]
    branch_points = ([coords[3], coords[4]], [(coords[3], coords[4])])
    causet = QuantumGrav.BranchedManifoldCauset(polym, branch_points, coords)
    @test causet isa QuantumGrav.BranchedManifoldCauset
    @test causet.atom_count == length(coords)
    @test causet.sprinkling[1] isa CausalSets.Coordinates{2}
    # in_past_of_unchecked
    @test CausalSets.in_past_of_unchecked(causet, 1, 3)
    @test CausalSets.in_past_of_unchecked(causet, 1, 2)
    @test !CausalSets.in_past_of_unchecked(causet, 3, 2)
end

@testitem "convert to BitArrayCauset" tags=[:branchedcsetgeneration] setup=[branchedtests] begin
    polym = CausalSets.PolynomialManifold{2}(randn(rng, 3, 3))
    x = CausalSets.Coordinates{2}((0.0, 0.0))
    y = CausalSets.Coordinates{2}((0.0, 0.1))
    z = CausalSets.Coordinates{2}((0.1, 0.2))
    sprinkling = [x, y, z]
    branch_points = ([CausalSets.Coordinates{2}((0.0, 0.15))], [(CausalSets.Coordinates{2}((0.1, 0.0)),CausalSets.Coordinates{2}((0.1, 0.3)))])
    causet = QuantumGrav.BranchedManifoldCauset(polym, branch_points, sprinkling)
    bitcset = CausalSets.BitArrayCauset(causet)
    @test bitcset isa CausalSets.BitArrayCauset
    @test bitcset.atom_count == 3
    @test bitcset.future_relations[1][3] == false
    @test bitcset.past_relations[3][1] == false
end

@testitem "convert to BitArrayCauset throws" tags=[:branchedcsetgeneration, :throws] setup=[branchedtests] begin
    # Throws for bad tolerance
    polym = CausalSets.PolynomialManifold{2}(randn(rng, 3, 3))
    x = CausalSets.Coordinates{2}((0.0, 0.0))
    y = CausalSets.Coordinates{2}((0.0, 0.1))
    z = CausalSets.Coordinates{2}((0.1, 0.2))
    sprinkling = [x, y, z]
    branch_points = ([CausalSets.Coordinates{2}((0.0, 0.15))], [(CausalSets.Coordinates{2}((0.1, 0.0)),CausalSets.Coordinates{2}((0.1, 0.3)))])
    causet = QuantumGrav.BranchedManifoldCauset(polym, branch_points, sprinkling)
    tolerance = 0.
    @test_throws ArgumentError CausalSets.BitArrayCauset(causet; tolerance = tolerance)
end

@testitem "make_branched_manifold_cset" tags=[:branchedcsetgeneration] setup=[branchedtests] begin
    npoints = 30
    n_vertical_cuts = 3
    n_finite_cuts = 2
    order = 3
    r = 1.2
    cset, sprinkling, branch_points, coefs = QuantumGrav.make_branched_manifold_cset(npoints, n_vertical_cuts, n_finite_cuts, rng, order, r)
    @test cset isa CausalSets.BitArrayCauset
    @test length(sprinkling) <= npoints
    @test length(branch_points[1]) == 3
    @test length(branch_points[2]) == 2
    @test size(coefs) == (order, order)
    @test cset.atom_count == npoints
end

@testitem "make_branched_manifold_cset" tags=[:branchedcsetgeneration, :throws] setup=[branchedtests] begin
    # Throws for bad arguments
    npoints = 30
    n_vertical_cuts = 3
    n_finite_cuts = 2
    order = 3
    r = 1.2
    @test_throws ArgumentError QuantumGrav.make_branched_manifold_cset(0, 0, 1, rng, order, r)
    @test_throws ArgumentError QuantumGrav.make_branched_manifold_cset(10, -3, 1, rng, order, r)
    @test_throws ArgumentError QuantumGrav.make_branched_manifold_cset(10, 0, -1, rng, order, r)
    @test_throws ArgumentError QuantumGrav.make_branched_manifold_cset(10, 5, 1, rng, 0, r)
    @test_throws ArgumentError QuantumGrav.make_branched_manifold_cset(10, 5, 1, rng, order, 0.5)
    @test_throws ArgumentError QuantumGrav.make_branched_manifold_cset(10, 5, 1, rng, order, r; d=3)
end