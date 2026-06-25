
@testsnippet MinkowskiKRInsertionTests begin
    import CausalSets
    import Random

    rng = Random.Xoshiro(42)
    manifold = CausalSets.MinkowskiManifold{2}()

    function relations_are_consistent(cset::CausalSets.AbstractCauset)::Bool
        for i = 1:cset.atom_count
            for j = 1:cset.atom_count
                cset.future_relations[i][j] == cset.past_relations[j][i] || return false
            end
        end
        return true
    end

    function is_transitively_closed(cset::CausalSets.AbstractCauset)::Bool
        for source = 1:cset.atom_count
            for middle = 1:cset.atom_count
                cset.future_relations[source][middle] || continue
                for target = 1:cset.atom_count
                    if cset.future_relations[middle][target] &&
                       !cset.future_relations[source][target]
                        return false
                    end
                end
            end
        end
        return true
    end

    function is_strictly_upper_triangular(cset::CausalSets.AbstractCauset)::Bool
        for i = 1:cset.atom_count
            any(cset.future_relations[i][1:i]) && return false
        end
        return true
    end

    function matrix_is_transitively_closed(adj::BitMatrix)::Bool
        n = size(adj, 1)
        for source = 1:n
            for middle = 1:n
                adj[source, middle] || continue
                for target = 1:n
                    if adj[middle, target] && !adj[source, target]
                        return false
                    end
                end
            end
        end
        return true
    end
end

@testitem "test_offset_causal_diamond_boundary_constructor" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    center = CausalSets.Coordinates{2}((0.0, 0.0))
    boundary = QuantumGrav.OffsetCausalDiamondBoundary{2}(0.8, center)

    @test boundary.duration == 0.8
    @test boundary.center == center
end

@testitem "test_get_tips" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    center = CausalSets.Coordinates{2}((0.0, 0.0))
    boundary = QuantumGrav.OffsetCausalDiamondBoundary{2}(0.8, center)

    past_tip, future_tip = QuantumGrav.get_tips(boundary, manifold)
    @test past_tip == CausalSets.Coordinates{2}((-0.4, 0.0))
    @test future_tip == CausalSets.Coordinates{2}((0.4, 0.0))
end

@testitem "test_offset_causal_diamond_is_in_boundary" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    center = CausalSets.Coordinates{2}((0.0, 0.0))
    boundary = QuantumGrav.OffsetCausalDiamondBoundary{2}(0.8, center)

    @test CausalSets.is_in_boundary(
        manifold,
        boundary,
        CausalSets.Coordinates{2}((-0.2, 0.0)),
    )
    @test CausalSets.is_in_boundary(
        manifold,
        boundary,
        CausalSets.Coordinates{2}((0.2, 0.0)),
    )
    @test !CausalSets.is_in_boundary(
        manifold,
        boundary,
        CausalSets.Coordinates{2}((-0.5, 0.0)),
    )
    @test !CausalSets.is_in_boundary(
        manifold,
        boundary,
        CausalSets.Coordinates{2}((0.5, 0.0)),
    )
end

@testitem "test_offset_causal_diamond_boundary_volume" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    center = CausalSets.Coordinates{2}((0.0, 0.0))
    boundary = QuantumGrav.OffsetCausalDiamondBoundary{2}(0.8, center)

    volume = CausalSets.boundary_volume(manifold, boundary)
    @test volume ≈ 0.32 # comparison to value computed by hand
end

@testitem "test_causal_diamond_duration_from_volume" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    center = CausalSets.Coordinates{2}((0.0, 0.0))

    @test QuantumGrav.causal_diamond_duration_from_volume(manifold, center, 0.32) ≈ 0.8 # value .32 has been computed by hand
end

@testitem "test_count_elements_in_boundary" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    sprinkling = CausalSets.Coordinates{2}[
        (-0.8, 0.0),
        (-0.2, 0.0),
        (0.0, 0.0),
        (0.2, 0.0),
        (0.8, 0.0),
    ]
    manifold_causet = CausalSets.ManifoldCauset(manifold, sprinkling)
    center = CausalSets.Coordinates{2}((0.0, 0.0))
    boundary = QuantumGrav.OffsetCausalDiamondBoundary{2}(0.8, center)

    @test QuantumGrav.count_elements_in_boundary(manifold_causet, boundary) == 3
end

@testitem "test_boundary_from_contained_element_count" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    sprinkling = CausalSets.Coordinates{2}[
        (-0.8, 0.0),
        (-0.2, 0.0),
        (0.0, 0.0),
        (0.2, 0.0),
        (0.8, 0.0),
    ]
    manifold_causet = CausalSets.ManifoldCauset(manifold, sprinkling)
    center = CausalSets.Coordinates{2}((0.0, 0.0))

    sprinkling_boundary = CausalSets.CausalDiamondBoundary{2}(2.0)
    sprinkling_density =
        length(sprinkling) / CausalSets.boundary_volume(manifold, sprinkling_boundary)
    counted_boundary = QuantumGrav.boundary_from_contained_element_count(
        manifold_causet,
        center,
        3,
        sprinkling_density,
    )

    @test counted_boundary isa QuantumGrav.OffsetCausalDiamondBoundary{2}
    @test QuantumGrav.count_elements_in_boundary(manifold_causet, counted_boundary) == 3
end

@testitem "test_transitive_closure" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    adj = falses(4, 4)
    adj[1, 2] = true
    adj[2, 3] = true
    adj[3, 4] = true

    cset = CausalSets.BitArrayCauset(adj)
    QuantumGrav.transitive_closure!(cset)

    @test cset.future_relations[1] == BitVector([false, true, true, true])
    @test cset.future_relations[2] == BitVector([false, false, true, true])
    @test cset.future_relations[3] == BitVector([false, false, false, true])
    @test relations_are_consistent(cset)
    @test is_transitively_closed(cset)
    @test is_strictly_upper_triangular(cset)
end

@testitem "test_bool_mul2" tags = [:minkowski_kr_insertion] begin
    A = BitMatrix([1 0 1; 0 1 0])
    B = BitMatrix([0 1; 1 0; 0 1])

    @test QuantumGrav.bool_mul2(A, B) == BitMatrix([0 1; 1 0])
end

@testitem "test_generate_KR_poset_adjacency_matrix" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    n = 40
    adj = QuantumGrav.generate_KR_poset_adjacency_matrix(n; rng = rng)

    @test adj isa BitMatrix
    @test size(adj) == (n, n)
    @test all(!adj[i, j] for i = 1:n for j = 1:i)
    @test matrix_is_transitively_closed(adj)
end

@testitem "test_generate_KR_poset_adjacency_matrix_layers" tags = [
    :minkowski_kr_insertion,
] setup = [MinkowskiKRInsertionTests] begin

    n = 1000
    adj = QuantumGrav.generate_KR_poset_adjacency_matrix(n; rng = rng)

    bottom_layer = findall(i -> !any(adj[:, i]), 1:n)
    top_layer = findall(i -> !any(adj[i, :]), 1:n)
    middle_layer = setdiff(1:n, union(bottom_layer, top_layer))

    @test !isempty(bottom_layer)
    @test !isempty(middle_layer)
    @test !isempty(top_layer)
    @test length(bottom_layer) + length(middle_layer) + length(top_layer) == n
    @test isapprox(length(bottom_layer) / n, 0.25; atol = 0.02)
    @test isapprox(length(middle_layer) / n, 0.5; atol = 0.02)
    @test isapprox(length(top_layer) / n, 0.25; atol = 0.02)
    @test all(!adj[i, j] for i in bottom_layer for j in bottom_layer)
    @test all(!adj[i, j] for i in middle_layer for j in middle_layer)
    @test all(!adj[i, j] for i in top_layer for j in top_layer)
    @test all(!adj[i, j] for i in middle_layer for j in bottom_layer)
    @test all(!adj[i, j] for i in top_layer for j in bottom_layer)
    @test all(!adj[i, j] for i in top_layer for j in middle_layer)
end

@testitem "test_generate_KR_poset_adjacency_matrix_connectivity" tags = [
    :minkowski_kr_insertion,
] setup = [MinkowskiKRInsertionTests] begin

    n = 1000
    adj = QuantumGrav.generate_KR_poset_adjacency_matrix(n; rng = rng)

    bottom_layer = findall(i -> !any(adj[:, i]), 1:n)
    top_layer = findall(i -> !any(adj[i, :]), 1:n)
    middle_layer = setdiff(1:n, union(bottom_layer, top_layer))

    bottom_to_middle_connectivity =
        sum(adj[i, j] for i in bottom_layer for j in middle_layer) /
        (length(bottom_layer) * length(middle_layer))
    middle_to_top_connectivity =
        sum(adj[i, j] for i in middle_layer for j in top_layer) /
        (length(middle_layer) * length(top_layer))

    @test isapprox(bottom_to_middle_connectivity, 0.5; atol = 0.01)
    @test isapprox(middle_to_top_connectivity, 0.5; atol = 0.01)
end

@testitem "test_BitArrayCauset_from_adjacency_matrix" tags = [:minkowski_kr_insertion] begin
    import CausalSets

    adj = BitMatrix([
        0 1 1
        0 0 0
        0 0 0
    ])
    cset = CausalSets.BitArrayCauset(adj)

    @test cset.atom_count == 3
    @test cset.future_relations == [BitVector(adj[i, :]) for i = 1:3]
    @test cset.past_relations == [BitVector(adj[:, j]) for j = 1:3]
end

@testitem "test_generate_KR_poset" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    n = 40
    cset = QuantumGrav.generate_KR_poset(n; rng = rng)

    @test cset isa CausalSets.BitArrayCauset
    @test cset.atom_count == n
    @test relations_are_consistent(cset)
    @test is_transitively_closed(cset)
    @test is_strictly_upper_triangular(cset)
    @test CausalSets.count_chains(cset, 4) == 0
end

@testitem "test_place_causet_in_manifold_causet" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    sprinkling = CausalSets.Coordinates{2}[
        (-0.8, 0.0),
        (-0.2, 0.0),
        (0.0, 0.0),
        (0.2, 0.0),
        (0.8, 0.0),
    ]
    manifold_causet = CausalSets.ManifoldCauset(manifold, sprinkling)
    boundary = QuantumGrav.OffsetCausalDiamondBoundary{2}(
        0.8,
        CausalSets.Coordinates{2}((0.0, 0.0)),
    )
    inserted_cset = CausalSets.BitArrayCauset(falses(3, 3))

    cset = QuantumGrav.place_causet_in_manifold_causet(
        manifold_causet,
        boundary,
        inserted_cset,
    )

    @test cset isa CausalSets.BitArrayCauset
    @test cset.atom_count == 5
    @test !any(cset.future_relations[i][j] for i = 2:4 for j = 2:4)
    @test relations_are_consistent(cset)
    @test is_transitively_closed(cset)
    @test is_strictly_upper_triangular(cset)
end

@testitem "test_is_subset" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    reference_boundary = CausalSets.CausalDiamondBoundary{2}(1.0)
    inner_boundary = QuantumGrav.OffsetCausalDiamondBoundary{2}(
        0.5,
        CausalSets.Coordinates{2}((0.0, 0.0)),
    )
    overlapping_boundary = QuantumGrav.OffsetCausalDiamondBoundary{2}(
        0.8,
        CausalSets.Coordinates{2}((0.4, 0.0)),
    )

    @test QuantumGrav.is_subset(inner_boundary, reference_boundary, manifold)
    @test !QuantumGrav.is_subset(overlapping_boundary, reference_boundary, manifold)
end

@testitem "test_replace_region_with_KR_poset" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    n = 30
    element_count = 5
    sprinkling_boundary = CausalSets.CausalDiamondBoundary{2}(1.0)
    sprinkling_density =
        n / CausalSets.boundary_volume(manifold, sprinkling_boundary)
    manifold_causet = CausalSets.ManifoldCauset(
        manifold,
        CausalSets.generate_sprinkling(manifold, sprinkling_boundary, n; rng = rng),
    )

    cset = QuantumGrav.replace_region_with_KR_poset(
        manifold_causet,
        sprinkling_boundary,
        element_count,
        sprinkling_density;
        require_region_fully_in_boundary = false,
        rng = rng,
    )

    @test cset isa CausalSets.BitArrayCauset
    @test cset.atom_count == n
    @test relations_are_consistent(cset)
    @test is_transitively_closed(cset)
    @test is_strictly_upper_triangular(cset)
end

@testitem "test_replace_region_with_KR_poset_return_KR_poset" tags = [
    :minkowski_kr_insertion,
] setup = [MinkowskiKRInsertionTests] begin

    n = 30
    element_count = 5
    sprinkling_boundary = CausalSets.CausalDiamondBoundary{2}(1.0)
    sprinkling_density =
        n / CausalSets.boundary_volume(manifold, sprinkling_boundary)
    manifold_causet = CausalSets.ManifoldCauset(
        manifold,
        CausalSets.generate_sprinkling(manifold, sprinkling_boundary, n; rng = rng),
    )

    cset, kr_poset = QuantumGrav.replace_region_with_KR_poset(
        manifold_causet,
        sprinkling_boundary,
        element_count,
        sprinkling_density;
        require_region_fully_in_boundary = false,
        return_KR_poset = true,
        rng = rng,
    )

    @test cset isa CausalSets.BitArrayCauset
    @test kr_poset isa CausalSets.BitArrayCauset
    @test cset.atom_count == n
    @test kr_poset.atom_count == element_count
    @test relations_are_consistent(cset)
    @test is_transitively_closed(cset)
    @test is_strictly_upper_triangular(cset)
end

@testitem "test_replace_region_with_KR_poset_require_region_fully_in_boundary_branch" tags = [
    :minkowski_kr_insertion,
] setup = [MinkowskiKRInsertionTests] begin

    n = 30
    element_count = 5
    sprinkling_boundary = CausalSets.CausalDiamondBoundary{2}(1.0)
    sprinkling_density =
        n / CausalSets.boundary_volume(manifold, sprinkling_boundary)
    manifold_causet = CausalSets.ManifoldCauset(
        manifold,
        CausalSets.generate_sprinkling(manifold, sprinkling_boundary, n; rng = rng),
    )

    cset = QuantumGrav.replace_region_with_KR_poset(
        manifold_causet,
        sprinkling_boundary,
        element_count,
        sprinkling_density;
        require_region_fully_in_boundary = true,
        rng = rng,
    )

    @test cset isa CausalSets.BitArrayCauset
    @test cset.atom_count == n
    @test relations_are_consistent(cset)
    @test is_transitively_closed(cset)
    @test is_strictly_upper_triangular(cset)
end

@testitem "test_generate_causet_with_KR_defect" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    n = 30
    m = 5
    sprinkling_boundary = CausalSets.CausalDiamondBoundary{2}(1.0)
    cset = QuantumGrav.generate_causet_with_KR_defect(
        n,
        m,
        sprinkling_boundary,
        manifold;
        rng = rng,
    )

    @test cset isa CausalSets.BitArrayCauset
    @test cset.atom_count == n
    @test relations_are_consistent(cset)
    @test is_transitively_closed(cset)
    @test is_strictly_upper_triangular(cset)
end
