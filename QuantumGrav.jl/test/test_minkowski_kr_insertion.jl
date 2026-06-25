
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

    function clone_causet(cset::CausalSets.BitArrayCauset)
        return CausalSets.BitArrayCauset(
            cset.atom_count,
            copy.(cset.future_relations),
            copy.(cset.past_relations),
        )
    end
end

@testitem "test_offset_causal_diamond_boundary_constructor" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    center = CausalSets.Coordinates{2}((0.0, 0.0))
    boundary = QuantumGrav.OffsetCausalDiamondBoundary{2}(0.8, center)

    @test boundary.duration == 0.8
    @test boundary.center == center
end

@testitem "test_offset_causal_diamond_boundary_constructor_throws" tags = [
    :minkowski_kr_insertion,
    :throws,
] setup = [MinkowskiKRInsertionTests] begin

    center = CausalSets.Coordinates{2}((0.0, 0.0))

    @test_throws ArgumentError QuantumGrav.OffsetCausalDiamondBoundary{2}(0.0, center)
    @test_throws ArgumentError QuantumGrav.OffsetCausalDiamondBoundary{2}(-0.1, center)
    @test_throws ArgumentError QuantumGrav.OffsetCausalDiamondBoundary{2}(Inf, center)
end

@testitem "test_get_tips" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    center = CausalSets.Coordinates{2}((0.0, 0.0))
    boundary = QuantumGrav.OffsetCausalDiamondBoundary{2}(0.8, center)

    past_tip, future_tip = QuantumGrav.get_tips(boundary, manifold)
    @test past_tip == CausalSets.Coordinates{2}((-0.4, 0.0))
    @test future_tip == CausalSets.Coordinates{2}((0.4, 0.0))
end

@testitem "test_get_tips_d3" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    manifold = CausalSets.MinkowskiManifold{3}()
    center = CausalSets.Coordinates{3}((0.0, 0.2, -0.3))
    boundary = QuantumGrav.OffsetCausalDiamondBoundary{3}(0.8, center)

    past_tip, future_tip = QuantumGrav.get_tips(boundary, manifold)
    @test past_tip == CausalSets.Coordinates{3}((-0.4, 0.2, -0.3))
    @test future_tip == CausalSets.Coordinates{3}((0.4, 0.2, -0.3))
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

@testitem "test_offset_causal_diamond_is_in_boundary_d3" tags = [
    :minkowski_kr_insertion,
] setup = [MinkowskiKRInsertionTests] begin

    manifold = CausalSets.MinkowskiManifold{3}()
    center = CausalSets.Coordinates{3}((0.0, 0.0, 0.0))
    boundary = QuantumGrav.OffsetCausalDiamondBoundary{3}(1.0, center)

    @test CausalSets.is_in_boundary(
        manifold,
        boundary,
        CausalSets.Coordinates{3}((0.0, 0.2, 0.2)),
    )
    @test !CausalSets.is_in_boundary(
        manifold,
        boundary,
        CausalSets.Coordinates{3}((0.0, 0.4, 0.4)),
    )
end

@testitem "test_offset_causal_diamond_boundary_volume" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    center = CausalSets.Coordinates{2}((0.0, 0.0))
    boundary = QuantumGrav.OffsetCausalDiamondBoundary{2}(0.8, center)

    volume = CausalSets.boundary_volume(manifold, boundary)
    @test volume ≈ 0.32 # comparison to value computed by hand
end

@testitem "test_offset_causal_diamond_boundary_volume_d3" tags = [
    :minkowski_kr_insertion,
] setup = [MinkowskiKRInsertionTests] begin

    manifold = CausalSets.MinkowskiManifold{3}()
    boundary = QuantumGrav.OffsetCausalDiamondBoundary{3}(
        1.2,
        CausalSets.Coordinates{3}((0.0, 0.0, 0.0)),
    )

    @test CausalSets.boundary_volume(manifold, boundary) ≈ π * 1.2^3 / 12 # comparison to value computed by hand
end

@testitem "test_causal_diamond_duration_from_volume" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    center = CausalSets.Coordinates{2}((0.0, 0.0))

    @test QuantumGrav.causal_diamond_duration_from_volume(manifold, center, 0.32) ≈ 0.8 # value .32 has been computed by hand
end

@testitem "test_causal_diamond_duration_from_volume_d3" tags = [
    :minkowski_kr_insertion,
] setup = [MinkowskiKRInsertionTests] begin

    manifold = CausalSets.MinkowskiManifold{3}()
    center = CausalSets.Coordinates{3}((0.0, 0.0, 0.0))

    @test QuantumGrav.causal_diamond_duration_from_volume(
        manifold,
        center,
        π * 1.2^3 / 12,
    ) ≈ 1.2 # volume has been computed by hand
end

@testitem "test_causal_diamond_duration_from_volume_throws" tags = [
    :minkowski_kr_insertion,
    :throws,
] setup = [MinkowskiKRInsertionTests] begin

    center = CausalSets.Coordinates{2}((0.0, 0.0))

    @test_throws ArgumentError QuantumGrav.causal_diamond_duration_from_volume(
        manifold,
        center,
        0.0,
    )
    @test_throws ArgumentError QuantumGrav.causal_diamond_duration_from_volume(
        manifold,
        center,
        -0.1,
    )
    @test_throws ArgumentError QuantumGrav.causal_diamond_duration_from_volume(
        manifold,
        center,
        Inf,
    )
end

@testitem "test_required_offset_causal_diamond_duration" tags = [
    :minkowski_kr_insertion,
] setup = [MinkowskiKRInsertionTests] begin

    center = CausalSets.Coordinates{2}((0.0, 0.0))

    @test QuantumGrav.required_offset_causal_diamond_duration(
        center,
        CausalSets.Coordinates{2}((0.0, 0.0)),
    ) == 0.0
    @test QuantumGrav.required_offset_causal_diamond_duration(
        center,
        CausalSets.Coordinates{2}((0.2, 0.0)),
    ) ≈ 0.4
    @test QuantumGrav.required_offset_causal_diamond_duration(
        center,
        CausalSets.Coordinates{2}((0.0, 0.3)),
    ) ≈ 0.6
    @test QuantumGrav.required_offset_causal_diamond_duration(
        center,
        CausalSets.Coordinates{2}((-0.2, 0.3)),
    ) ≈ 1.0
end

@testitem "test_required_offset_causal_diamond_duration_d3" tags = [
    :minkowski_kr_insertion,
] setup = [MinkowskiKRInsertionTests] begin

    center = CausalSets.Coordinates{3}((0.0, 0.0, 0.0))

    @test QuantumGrav.required_offset_causal_diamond_duration(
        center,
        CausalSets.Coordinates{3}((0.2, 0.3, 0.4)),
    ) ≈ 2 * (0.2 + 0.5)
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

@testitem "test_count_elements_in_boundary_d3" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    manifold = CausalSets.MinkowskiManifold{3}()
    sprinkling = CausalSets.Coordinates{3}[
        (-0.8, 0.0, 0.0),
        (-0.2, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.8, 0.0, 0.0),
    ]
    manifold_causet = CausalSets.ManifoldCauset(manifold, sprinkling)
    center = CausalSets.Coordinates{3}((0.0, 0.0, 0.0))
    boundary = QuantumGrav.OffsetCausalDiamondBoundary{3}(0.8, center)

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

    counted_boundary = QuantumGrav.boundary_from_contained_element_count(
        manifold_causet,
        center,
        3,
    )

    @test counted_boundary isa QuantumGrav.OffsetCausalDiamondBoundary{2}
    @test QuantumGrav.count_elements_in_boundary(manifold_causet, counted_boundary) == 3
end

@testitem "test_boundary_from_contained_element_count_d3" tags = [
    :minkowski_kr_insertion,
] setup = [MinkowskiKRInsertionTests] begin

    manifold = CausalSets.MinkowskiManifold{3}()
    sprinkling = CausalSets.Coordinates{3}[
        (-0.8, 0.0, 0.0),
        (-0.2, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.8, 0.0, 0.0),
    ]
    manifold_causet = CausalSets.ManifoldCauset(manifold, sprinkling)
    center = CausalSets.Coordinates{3}((0.0, 0.0, 0.0))
    counted_boundary = QuantumGrav.boundary_from_contained_element_count(
        manifold_causet,
        center,
        3,
    )

    @test counted_boundary isa QuantumGrav.OffsetCausalDiamondBoundary{3}
    @test QuantumGrav.count_elements_in_boundary(manifold_causet, counted_boundary) == 3
end

@testitem "test_boundary_from_contained_element_count_throws" tags = [
    :minkowski_kr_insertion,
    :throws,
] setup = [MinkowskiKRInsertionTests] begin

    sprinkling = CausalSets.Coordinates{2}[
        (-0.2, 0.0),
        (0.2, 0.0),
        (0.6, 0.0),
    ]
    manifold_causet = CausalSets.ManifoldCauset(manifold, sprinkling)
    center = CausalSets.Coordinates{2}((0.0, 0.0))

    @test_throws ArgumentError QuantumGrav.boundary_from_contained_element_count(
        manifold_causet,
        center,
        0,
    )
    @test_throws ArgumentError QuantumGrav.boundary_from_contained_element_count(
        manifold_causet,
        center,
        4,
    )
    @test_throws ArgumentError QuantumGrav.boundary_from_contained_element_count(
        manifold_causet,
        center,
        1,
    )
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

@testitem "test_local_transitive_closure" tags = [:minkowski_kr_insertion] setup =
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
    region_indices = findall(
        CausalSets.is_in_boundary.(
            Ref(manifold_causet.manifold),
            Ref(boundary),
            manifold_causet.sprinkling,
        ),
    )
    inserted_cset = CausalSets.BitArrayCauset(falses(length(region_indices), length(region_indices)))
    preclosed_cset = CausalSets.BitArrayCauset(manifold_causet.manifold, manifold_causet.sprinkling)
    for (inner_idx, global_idx) in enumerate(region_indices)
        preclosed_cset.future_relations[global_idx][region_indices] =
            inserted_cset.future_relations[inner_idx]
        preclosed_cset.past_relations[global_idx][region_indices] =
            inserted_cset.past_relations[inner_idx]
    end

    global_cset = clone_causet(preclosed_cset)
    local_cset = clone_causet(preclosed_cset)
    geometric_local_cset = clone_causet(preclosed_cset)

    QuantumGrav.transitive_closure!(global_cset)
    QuantumGrav.local_transitive_closure!(local_cset, region_indices)
    QuantumGrav.local_transitive_closure!(
        geometric_local_cset,
        region_indices,
        manifold_causet,
        boundary,
    )

    @test local_cset.future_relations == global_cset.future_relations
    @test local_cset.past_relations == global_cset.past_relations
    @test geometric_local_cset.future_relations == global_cset.future_relations
    @test geometric_local_cset.past_relations == global_cset.past_relations
    @test relations_are_consistent(local_cset)
    @test is_transitively_closed(local_cset)
    @test is_strictly_upper_triangular(local_cset)
    @test relations_are_consistent(geometric_local_cset)
    @test is_transitively_closed(geometric_local_cset)
    @test is_strictly_upper_triangular(geometric_local_cset)
end

@testitem "test_affected_rows_for_region" tags = [:minkowski_kr_insertion] setup =
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
    region_indices = [2, 3, 4]
    cset = CausalSets.BitArrayCauset(manifold_causet.manifold, manifold_causet.sprinkling)

    affected = QuantumGrav.affected_rows_for_region(cset, region_indices)
    geometric_affected =
        QuantumGrav.affected_rows_for_region(cset, region_indices, manifold_causet, boundary)

    @test affected == BitVector([true, true, true, true, false])
    @test geometric_affected == BitVector([false, true, true, true, false])
end

@testitem "test_complete_future_rows_for_region" tags = [:minkowski_kr_insertion] setup =
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

    @test QuantumGrav.complete_future_rows_for_region(manifold_causet, boundary) ==
          BitVector([false, false, false, false, true])
end

@testitem "test_side_future_target_masks_for_region" tags = [
    :minkowski_kr_insertion,
] setup = [MinkowskiKRInsertionTests] begin

    sprinkling = CausalSets.Coordinates{2}[
        (-0.8, 0.0),
        (-0.2, -0.1),
        (0.0, 0.0),
        (0.2, 0.1),
        (0.45, -0.2),
        (0.45, 0.2),
        (0.8, 0.0),
    ]
    manifold_causet = CausalSets.ManifoldCauset(manifold, sprinkling)
    boundary = QuantumGrav.OffsetCausalDiamondBoundary{2}(
        0.8,
        CausalSets.Coordinates{2}((0.0, 0.0)),
    )
    region_indices = [2, 3, 4]

    generic_targets, left_targets, right_targets, region_mask =
        QuantumGrav.side_future_target_masks_for_region(
            manifold_causet,
            boundary,
            region_indices,
        )

    @test generic_targets == BitVector([true, true, true, true, true, true, false])
    @test left_targets == BitVector([false, true, true, true, true, false, false])
    @test right_targets == BitVector([false, true, true, true, false, true, false])
    @test region_mask == BitVector([false, true, true, true, false, false, false])
end

@testitem "test_local_transitive_closure_input_throws" tags = [
    :minkowski_kr_insertion,
    :throws,
] setup = [MinkowskiKRInsertionTests] begin

    sprinkling = CausalSets.Coordinates{2}[
        (-0.8, 0.0),
        (-0.2, 0.0),
        (0.0, 0.0),
        (0.2, 0.0),
        (0.8, 0.0),
    ]
    manifold_causet = CausalSets.ManifoldCauset(manifold, sprinkling)
    cset = CausalSets.BitArrayCauset(manifold_causet.manifold, manifold_causet.sprinkling)
    boundary = QuantumGrav.OffsetCausalDiamondBoundary{2}(
        0.8,
        CausalSets.Coordinates{2}((0.0, 0.0)),
    )
    smaller_cset = CausalSets.BitArrayCauset(manifold, sprinkling[1:4])

    @test_throws ArgumentError QuantumGrav.affected_rows_for_region(cset, Int[])
    @test_throws ArgumentError QuantumGrav.affected_rows_for_region(cset, [0])
    @test_throws ArgumentError QuantumGrav.affected_rows_for_region(cset, [6])
    @test_throws ArgumentError QuantumGrav.affected_rows_for_region(
        smaller_cset,
        [2, 3, 4],
        manifold_causet,
        boundary,
    )
    @test_throws ArgumentError QuantumGrav.side_future_target_masks_for_region(
        manifold_causet,
        boundary,
        Int[],
    )
    @test_throws ArgumentError QuantumGrav.side_future_target_masks_for_region(
        manifold_causet,
        boundary,
        [6],
    )
    @test_throws ArgumentError QuantumGrav.local_transitive_closure!(cset, Int[])
    @test_throws ArgumentError QuantumGrav.local_transitive_closure!(
        smaller_cset,
        [2, 3, 4],
        manifold_causet,
        boundary,
    )
    @test_throws ArgumentError QuantumGrav.local_transitive_closure!(cset, falses(4), nothing)
    @test_throws ArgumentError QuantumGrav.local_transitive_closure!(
        cset,
        trues(5),
        falses(4),
    )
end

@testitem "test_bool_mul2" tags = [:minkowski_kr_insertion] begin
    A = BitMatrix([1 0 1; 0 1 0])
    B = BitMatrix([0 1; 1 0; 0 1])

    @test QuantumGrav.bool_mul2(A, B) == BitMatrix([0 1; 1 0])
end

@testitem "test_bool_mul2_throws" tags = [:minkowski_kr_insertion, :throws] begin
    A = falses(2, 3)
    B = falses(2, 2)

    @test_throws ArgumentError QuantumGrav.bool_mul2(A, B)
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

@testitem "test_generate_KR_poset_throws" tags = [
    :minkowski_kr_insertion,
    :throws,
] setup = [MinkowskiKRInsertionTests] begin

    @test_throws ArgumentError QuantumGrav.generate_KR_poset_adjacency_matrix(2; rng = rng)
    @test_throws ArgumentError QuantumGrav.generate_KR_poset(2; rng = rng)
end

@testitem "test_BitArrayCauset_from_adjacency_matrix" tags = [
    :minkowski_kr_insertion,
] setup = [MinkowskiKRInsertionTests] begin

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

@testitem "test_BitArrayCauset_from_adjacency_matrix_throws" tags = [
    :minkowski_kr_insertion,
    :throws,
] setup = [MinkowskiKRInsertionTests] begin

    @test_throws ArgumentError CausalSets.BitArrayCauset(falses(2, 3))
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

@testitem "test_place_causet_in_manifold_causet_throws" tags = [
    :minkowski_kr_insertion,
    :throws,
] setup = [MinkowskiKRInsertionTests] begin

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
    wrong_size_cset = CausalSets.BitArrayCauset(falses(4, 4))

    @test_throws ArgumentError QuantumGrav.place_causet_in_manifold_causet(
        manifold_causet,
        boundary,
        wrong_size_cset,
    )
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
    manifold_causet = CausalSets.ManifoldCauset(
        manifold,
        CausalSets.generate_sprinkling(manifold, sprinkling_boundary, n; rng = rng),
    )

    cset = QuantumGrav.replace_region_with_KR_poset(
        manifold_causet,
        sprinkling_boundary,
        element_count,
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
    manifold_causet = CausalSets.ManifoldCauset(
        manifold,
        CausalSets.generate_sprinkling(manifold, sprinkling_boundary, n; rng = rng),
    )

    cset, kr_poset = QuantumGrav.replace_region_with_KR_poset(
        manifold_causet,
        sprinkling_boundary,
        element_count,
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
    manifold_causet = CausalSets.ManifoldCauset(
        manifold,
        CausalSets.generate_sprinkling(manifold, sprinkling_boundary, n; rng = rng),
    )

    cset = QuantumGrav.replace_region_with_KR_poset(
        manifold_causet,
        sprinkling_boundary,
        element_count,
        require_region_fully_in_boundary = true,
        rng = rng,
    )

    @test cset isa CausalSets.BitArrayCauset
    @test cset.atom_count == n
    @test relations_are_consistent(cset)
    @test is_transitively_closed(cset)
    @test is_strictly_upper_triangular(cset)
end

@testitem "test_replace_region_with_KR_poset_throws" tags = [
    :minkowski_kr_insertion,
    :throws,
] setup = [MinkowskiKRInsertionTests] begin

    n = 30
    sprinkling_boundary = CausalSets.CausalDiamondBoundary{2}(1.0)
    manifold_causet = CausalSets.ManifoldCauset(
        manifold,
        CausalSets.generate_sprinkling(manifold, sprinkling_boundary, n; rng = rng),
    )

    @test_throws ArgumentError QuantumGrav.replace_region_with_KR_poset(
        manifold_causet,
        sprinkling_boundary,
        2;
        rng = rng,
    )
    @test_throws ArgumentError QuantumGrav.replace_region_with_KR_poset(
        manifold_causet,
        sprinkling_boundary,
        n + 1;
        rng = rng,
    )
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

@testitem "test_generate_causet_with_KR_defect_throws" tags = [
    :minkowski_kr_insertion,
    :throws,
] setup = [MinkowskiKRInsertionTests] begin

    sprinkling_boundary = CausalSets.CausalDiamondBoundary{2}(1.0)

    @test_throws ArgumentError QuantumGrav.generate_causet_with_KR_defect(
        2,
        2,
        sprinkling_boundary,
        manifold;
        rng = rng,
    )
    @test_throws ArgumentError QuantumGrav.generate_causet_with_KR_defect(
        30,
        2,
        sprinkling_boundary,
        manifold;
        rng = rng,
    )
    @test_throws ArgumentError QuantumGrav.generate_causet_with_KR_defect(
        30,
        31,
        sprinkling_boundary,
        manifold;
        rng = rng,
    )
end

@testitem "test_generate_causet_with_KR_defect_d3" tags = [:minkowski_kr_insertion] setup =
    [MinkowskiKRInsertionTests] begin

    manifold_d3 = CausalSets.MinkowskiManifold{3}()
    n = 30
    m = 5
    sprinkling_boundary = CausalSets.CausalDiamondBoundary{3}(1.0)
    cset = QuantumGrav.generate_causet_with_KR_defect(
        n,
        m,
        sprinkling_boundary,
        manifold_d3;
        rng = rng,
    )

    @test cset isa CausalSets.BitArrayCauset
    @test cset.atom_count == n
    @test relations_are_consistent(cset)
    @test is_transitively_closed(cset)
    @test is_strictly_upper_triangular(cset)
end
