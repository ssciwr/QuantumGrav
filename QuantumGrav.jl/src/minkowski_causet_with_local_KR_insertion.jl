"""
    OffsetCausalDiamondBoundary{N}(duration::Float64, center::Coordinates{N}) <: AbstractBoundary{N}

Represent an `N`-dimensional causal diamond in Minkowski space with arbitrary
center coordinate.

# Arguments
- `duration`: Timelike distance between the past and future tips. Must be finite
  and positive.
- `center`: Center coordinate of the diamond.

# Throws
- `ArgumentError`: If `duration` is not finite and positive.
"""
struct OffsetCausalDiamondBoundary{N} <: CausalSets.AbstractBoundary{N}
    duration::Float64
    center::CausalSets.Coordinates{N}

    function OffsetCausalDiamondBoundary{N}(
        duration::Real,
        center::CausalSets.Coordinates{N},
    ) where {N}
        isfinite(duration) && duration > 0 ||
            throw(ArgumentError("duration must be finite and positive, got $duration."))
        return new{N}(Float64(duration), center)
    end
end

"""
    OffsetCausalDiamondBoundary(duration::Real, center::Coordinates{N})

Construct an `OffsetCausalDiamondBoundary{N}` from the dimension of `center`.

# Arguments
- `duration`: Timelike distance between the past and future tips. Must be finite
  and positive.
- `center`: Center coordinate of the diamond.

# Throws
- `ArgumentError`: If `duration` is not finite and positive.
"""
OffsetCausalDiamondBoundary(duration::Real, center::CausalSets.Coordinates{N}) where {N} =
    OffsetCausalDiamondBoundary{N}(duration, center)

"""
    get_tips(boundary::OffsetCausalDiamondBoundary{N}, manifold::MinkowskiManifold{N})

Return the past and future tips of `boundary`.

# Arguments
- `boundary`: Offset causal diamond.
- `manifold`: Minkowski manifold with the same dimension as `boundary`.
"""
function get_tips(boundary::OffsetCausalDiamondBoundary{N}, manifold::CausalSets.MinkowskiManifold{N}) where {N}
    past_tip = (boundary.center[1]-boundary.duration/2, boundary.center[2:end]...)
    future_tip = (boundary.center[1]+boundary.duration/2, boundary.center[2:end]...)
    (past_tip, future_tip)
end

"""
    CausalSets.is_in_boundary(manifold::MinkowskiManifold{N}, boundary::OffsetCausalDiamondBoundary{N}, coord::Coordinates{N})::Bool

Return whether `coord` lies inside `boundary` in `manifold`.

# Arguments
- `manifold`: Minkowski manifold containing the coordinate.
- `boundary`: Offset causal diamond to test.
- `coord`: Coordinate to test for containment.
"""
function CausalSets.is_in_boundary(manifold::CausalSets.MinkowskiManifold{N}, boundary::OffsetCausalDiamondBoundary{N}, coord::CausalSets.Coordinates{N})::Bool where {N}
    past_tip, future_tip = get_tips(boundary, manifold)
    
    if coord[1] > boundary.center[1]
        # future half of causal diamond:
        return CausalSets.in_past_of(manifold, coord, future_tip)
    else
        # past half of causal diamond:
        return CausalSets.in_past_of(manifold, past_tip, coord)
    end
end

"""
    CausalSets.boundary_volume(manifold::MinkowskiManifold{N}, boundary::OffsetCausalDiamondBoundary{N})

Compute the spacetime volume of `boundary` in `manifold`.

# Arguments
- `manifold`: Minkowski manifold of the boundary.
- `boundary`: Offset causal diamond whose volume is computed.
"""
function CausalSets.boundary_volume(manifold::CausalSets.MinkowskiManifold{N}, boundary::OffsetCausalDiamondBoundary{N}) where {N}
    return pi^((N-1)/2) / (N*2^(N-1)*SpecialFunctions.gamma((N+1)/2)) * boundary.duration^N
end

"""
    causal_diamond_duration_from_volume(manifold::MinkowskiManifold{N}, center::Coordinates{N}, target_volume::Float64)

Return the duration of an offset causal diamond with spacetime volume
`target_volume`.

# Arguments
- `manifold`: Minkowski manifold in which the volume is measured.
- `center`: Center coordinate of the diamond. The duration is independent of
  this coordinate in flat Minkowski space.
- `target_volume`: Desired spacetime volume. Must be finite and positive.

# Throws
- `ArgumentError`: If `target_volume` is not finite and positive.
"""
function causal_diamond_duration_from_volume(manifold::CausalSets.MinkowskiManifold{N}, center::CausalSets.Coordinates{N}, target_volume::Float64) where {N}
    isfinite(target_volume) && target_volume > 0 ||
        throw(ArgumentError("target_volume must be finite and positive, got $target_volume."))
    return (N*2^(N-1)*SpecialFunctions.gamma((N+1)/2) / (pi^((N-1)/2)) * target_volume)^(1/N)
end

"""
    count_elements_in_boundary(manifold_causet::ManifoldCauset{N}, boundary::OffsetCausalDiamondBoundary{N})

Count the elements of `manifold_causet` that lie in `boundary`.

# Arguments
- `manifold_causet`: Manifold causet whose sprinkling is counted.
- `boundary`: Offset causal diamond used for containment.
"""
function count_elements_in_boundary(manifold_causet::CausalSets.ManifoldCauset{N}, boundary::OffsetCausalDiamondBoundary{N}) where {N}
    sum(CausalSets.is_in_boundary.(Ref(manifold_causet.manifold), Ref(boundary), manifold_causet.sprinkling))
end

"""
    required_offset_causal_diamond_duration(center::Coordinates{N}, coord::Coordinates{N})

Return the smallest offset causal-diamond duration centered at `center` that
contains `coord`.

# Arguments
- `center`: Center coordinate of the offset causal diamond.
- `coord`: Coordinate that must be contained.
"""
function required_offset_causal_diamond_duration(
    center::CausalSets.Coordinates{N},
    coord::CausalSets.Coordinates{N},
) where {N}
    spatial_distance_sq = zero(Float64)
    @inbounds for dim = 2:N
        spatial_distance_sq += (coord[dim] - center[dim])^2
    end
    return 2 * (abs(coord[1] - center[1]) + sqrt(spatial_distance_sq))
end

"""
    boundary_from_contained_element_count(manifold_causet::ManifoldCauset{N}, center::Coordinates{N}, target_count::Integer)

Return an offset causal diamond centered at `center` containing exactly
`target_count` elements of `manifold_causet`.

# Arguments
- `manifold_causet`: Minkowski manifold causet whose sprinkling is searched.
- `center`: Center coordinate of the returned offset causal diamond.
- `target_count`: Required number of contained elements.

# Throws
- `ArgumentError`: If `target_count` is not between 1 and the atom count.
- `ArgumentError`: If no floating-point duration separates exactly
  `target_count` elements.
"""
function boundary_from_contained_element_count(
    manifold_causet::CausalSets.ManifoldCauset{N,<:CausalSets.MinkowskiManifold{N}},
    center::CausalSets.Coordinates{N},
    target_count::Integer,
) where {N}
    1 <= target_count <= manifold_causet.atom_count ||
        throw(ArgumentError("target_count must be between 1 and the atom count."))

    required_durations = Vector{Float64}(undef, manifold_causet.atom_count)
    @inbounds for idx in eachindex(manifold_causet.sprinkling)
        required_durations[idx] =
            required_offset_causal_diamond_duration(center, manifold_causet.sprinkling[idx])
    end

    if target_count == manifold_causet.atom_count
        duration = nextfloat(partialsort!(required_durations, target_count))
        return OffsetCausalDiamondBoundary{N}(duration, center)
    end

    lower, upper = partialsort!(required_durations, target_count:(target_count + 1))
    if lower == upper
        throw(ArgumentError("No boundary duration contains exactly $target_count elements."))
    end

    duration = (lower + upper) / 2
    if !(lower < duration < upper)
        duration = nextfloat(lower)
    end
    duration < upper ||
        throw(ArgumentError("No floating-point duration separates the requested boundary."))

    return OffsetCausalDiamondBoundary{N}(duration, center)
end

# Based on CausalSets.jl/src/causets/sampler.jl
#=
function transitive_closure!(causet::CausalSets.BitArrayCauset)
    for i in 1:causet.atom_count-1
        causet.future_relations[i][i] = 1
        for j in i:causet.atom_count
            if causet.future_relations[i][j]
                for k in j+1:causet.atom_count
                    if causet.future_relations[j][k]
                        causet.future_relations[i][k] = 1
                        causet.past_relations[k][i] = 1
                    end
                end
            end
        end
        causet.future_relations[i][i] = 0
    end
end
=#

# Optimized by walking backwards, so that due to natural labeling, only the nearest future points have to be completed, the rest follows from them already having been transitively completed
"""
    transitive_closure!(causet::BitArrayCauset)

Transitively close `causet` in place.

# Arguments
- `causet`: Causal set to close in place.
"""
function transitive_closure!(causet::CausalSets.BitArrayCauset)
    n = causet.atom_count
    seeds = BitVector(undef, n)
    @inbounds for i in n-1:-1:1
        copyto!(seeds, causet.future_relations[i])
        j = findfirst(seeds)
        while j !== nothing
            causet.future_relations[i] .|= causet.future_relations[j]
            j = findnext(seeds, j+1)
        end
    end

    @inbounds for i in 1:n
        k = findfirst(causet.future_relations[i])
        while k !== nothing
            causet.past_relations[k][i] = true
            k = findnext(causet.future_relations[i], k+1)
        end
    end
end

"""
    validate_region_indices(causet::BitArrayCauset, region_indices)

Validate that `region_indices` is nonempty and contains valid element indices of
`causet`.

# Arguments
- `causet`: Causal set whose element range is used for validation.
- `region_indices`: Indices to validate.

# Throws
- `ArgumentError`: If `region_indices` is empty or contains an out-of-bounds
  index.
"""
function validate_region_indices(
    causet::CausalSets.BitArrayCauset,
    region_indices::AbstractVector{<:Integer},
)
    isempty(region_indices) && throw(ArgumentError("region_indices must not be empty."))
    for idx in region_indices
        1 <= idx <= causet.atom_count ||
            throw(ArgumentError("region index $idx is outside 1:$(causet.atom_count)."))
    end
    return nothing
end

"""
    affected_rows_for_region(causet::BitArrayCauset, region_indices) -> BitVector

Return a mask for rows that can change after replacing the induced subcauset on
`region_indices`.

# Arguments
- `causet`: Causal set after replacing the induced region.
- `region_indices`: Indices of the replaced region.

# Throws
- `ArgumentError`: If `region_indices` is empty or contains an out-of-bounds
  index.
"""
function affected_rows_for_region(
    causet::CausalSets.BitArrayCauset,
    region_indices::AbstractVector{<:Integer},
)
    validate_region_indices(causet, region_indices)
    affected = falses(causet.atom_count)
    affected[region_indices] .= true

    @inbounds for idx in region_indices
        affected .|= causet.past_relations[idx]
    end

    return affected
end

"""
    affected_rows_for_region(causet, region_indices, manifold_causet, boundary) -> BitVector

Return the affected-row mask with geometric pruning by the past tip of
`boundary`.

# Arguments
- `causet`: Causal set after replacing the induced region.
- `region_indices`: Indices of the replaced region.
- `manifold_causet`: Original manifold causet with the same atom count as
  `causet`.
- `boundary`: Offset causal diamond enclosing the replaced region.

# Throws
- `ArgumentError`: If `causet` and `manifold_causet` have different atom counts.
- `ArgumentError`: If `region_indices` is invalid.
"""
function affected_rows_for_region(
    causet::CausalSets.BitArrayCauset,
    region_indices::AbstractVector{<:Integer},
    manifold_causet::CausalSets.ManifoldCauset,
    boundary::OffsetCausalDiamondBoundary,
)
    causet.atom_count == manifold_causet.atom_count ||
        throw(ArgumentError("causet and manifold_causet must have the same atom count."))
    affected = affected_rows_for_region(causet, region_indices)
    region_mask = falses(causet.atom_count)
    region_mask[region_indices] .= true
    past_tip, _ = get_tips(boundary, manifold_causet.manifold)

    @inbounds for source in eachindex(affected)
        if affected[source] &&
           !region_mask[source] &&
           CausalSets.in_past_of(
               manifold_causet.manifold,
               manifold_causet.sprinkling[source],
               past_tip,
           )
            affected[source] = false
        end
    end

    return affected
end

"""
    complete_future_rows_for_region(manifold_causet, boundary) -> BitVector

Return a mask for elements in the complete future of `boundary`, i.e. elements
above the future tip.

# Arguments
- `manifold_causet`: Original manifold causet.
- `boundary`: Offset causal diamond whose future tip defines the complete
  future.
"""
function complete_future_rows_for_region(
    manifold_causet::CausalSets.ManifoldCauset,
    boundary::OffsetCausalDiamondBoundary,
)
    _, future_tip = get_tips(boundary, manifold_causet.manifold)
    complete_future = falses(manifold_causet.atom_count)

    @inbounds for target in eachindex(complete_future)
        complete_future[target] = CausalSets.in_past_of(
            manifold_causet.manifold,
            future_tip,
            manifold_causet.sprinkling[target],
        )
    end

    return complete_future
end

"""
    side_future_target_masks_for_region(manifold_causet, boundary, region_indices)

Return target masks used by the 2D local closure: all non-complete-future
targets, the left side targets, the right side targets, and the region mask.

# Arguments
- `manifold_causet`: Two-dimensional Minkowski manifold causet.
- `boundary`: Two-dimensional offset causal diamond enclosing the replaced
  region.
- `region_indices`: Indices of the replaced region.

# Throws
- `ArgumentError`: If `region_indices` is empty or contains an out-of-bounds
  index.
"""
function side_future_target_masks_for_region(
    manifold_causet::CausalSets.ManifoldCauset{2,<:CausalSets.MinkowskiManifold{2}},
    boundary::OffsetCausalDiamondBoundary{2},
    region_indices::AbstractVector{<:Integer},
)
    isempty(region_indices) && throw(ArgumentError("region_indices must not be empty."))
    for idx in region_indices
        1 <= idx <= manifold_causet.atom_count ||
            throw(ArgumentError("region index $idx is outside 1:$(manifold_causet.atom_count)."))
    end
    _, future_tip = get_tips(boundary, manifold_causet.manifold)
    complete_future = complete_future_rows_for_region(manifold_causet, boundary)
    region_mask = falses(manifold_causet.atom_count)
    region_mask[region_indices] .= true
    generic_targets = .!complete_future
    left_targets = copy(region_mask)
    right_targets = copy(region_mask)
    future_tip_left = future_tip[1] - future_tip[2]
    future_tip_right = future_tip[1] + future_tip[2]

    @inbounds for target in eachindex(generic_targets)
        coord = manifold_causet.sprinkling[target]
        if generic_targets[target] && coord[1] - coord[2] >= future_tip_left
            left_targets[target] = true
        end
        if generic_targets[target] && coord[1] + coord[2] >= future_tip_right
            right_targets[target] = true
        end
    end

    return generic_targets, left_targets, right_targets, region_mask
end

"""
    local_transitive_closure!(causet::BitArrayCauset, region_indices)

Transitively closes a causet after replacing only the induced relations on
`region_indices`. Only rows in the causal past of the replaced region can gain
new future relations, and `past_relations` are updated incrementally.

# Arguments
- `causet`: Causal set after replacing the induced region.
- `region_indices`: Indices of the replaced region.

# Throws
- `ArgumentError`: If `region_indices` is invalid.
"""
function local_transitive_closure!(
    causet::CausalSets.BitArrayCauset,
    region_indices::AbstractVector{<:Integer},
)
    affected = affected_rows_for_region(causet, region_indices)
    return local_transitive_closure!(causet, affected, nothing)
end

"""
    local_transitive_closure!(causet, region_indices, manifold_causet, boundary)

Transitively close `causet` locally using the 2D geometric pruning masks.

# Arguments
- `causet`: Causal set after replacing the induced region.
- `region_indices`: Indices of the replaced region.
- `manifold_causet`: Original two-dimensional Minkowski manifold causet.
- `boundary`: Offset causal diamond enclosing the replaced region.

# Throws
- `ArgumentError`: If `causet` and `manifold_causet` have different atom counts.
- `ArgumentError`: If `region_indices` is invalid.
"""
function local_transitive_closure!(
    causet::CausalSets.BitArrayCauset,
    region_indices::AbstractVector{<:Integer},
    manifold_causet::CausalSets.ManifoldCauset{2,<:CausalSets.MinkowskiManifold{2}},
    boundary::OffsetCausalDiamondBoundary{2},
)
    causet.atom_count == manifold_causet.atom_count ||
        throw(ArgumentError("causet and manifold_causet must have the same atom count."))
    affected = affected_rows_for_region(causet, region_indices, manifold_causet, boundary)
    generic_targets, left_targets, right_targets, region_mask =
        side_future_target_masks_for_region(manifold_causet, boundary, region_indices)
    past_tip, _ = get_tips(boundary, manifold_causet.manifold)
    past_tip_left = past_tip[1] - past_tip[2]
    past_tip_right = past_tip[1] + past_tip[2]

    n = causet.atom_count
    seeds = BitVector(undef, n)
    old_future = BitVector(undef, n)
    added_future = BitVector(undef, n)

    @inbounds for source = n-1:-1:1
        affected[source] || continue
        copyto!(old_future, causet.future_relations[source])
        copyto!(seeds, causet.future_relations[source])
        coord = manifold_causet.sprinkling[source]
        if region_mask[source]
            seeds .&= generic_targets
        elseif coord[1] - coord[2] <= past_tip_left
            seeds .&= left_targets
        elseif coord[1] + coord[2] <= past_tip_right
            seeds .&= right_targets
        else
            seeds .&= generic_targets
        end

        target = findfirst(seeds)
        while target !== nothing
            causet.future_relations[source] .|= causet.future_relations[target]
            target = findnext(seeds, target + 1)
        end

        added_future .= causet.future_relations[source]
        added_future .&= .!old_future
        target = findfirst(added_future)
        while target !== nothing
            causet.past_relations[target][source] = true
            target = findnext(added_future, target + 1)
        end
    end

    return causet
end

"""
    local_transitive_closure!(causet, region_indices, manifold_causet, boundary)

Transitively close `causet` locally using dimension-independent geometric
pruning.

# Arguments
- `causet`: Causal set after replacing the induced region.
- `region_indices`: Indices of the replaced region.
- `manifold_causet`: Original manifold causet with the same atom count as
  `causet`.
- `boundary`: Offset causal diamond enclosing the replaced region.

# Throws
- `ArgumentError`: If `causet` and `manifold_causet` have different atom counts.
- `ArgumentError`: If `region_indices` is invalid.
"""
function local_transitive_closure!(
    causet::CausalSets.BitArrayCauset,
    region_indices::AbstractVector{<:Integer},
    manifold_causet::CausalSets.ManifoldCauset,
    boundary::OffsetCausalDiamondBoundary,
)
    causet.atom_count == manifold_causet.atom_count ||
        throw(ArgumentError("causet and manifold_causet must have the same atom count."))
    affected = affected_rows_for_region(causet, region_indices, manifold_causet, boundary)
    complete_future = complete_future_rows_for_region(manifold_causet, boundary)
    allowed_future_targets = .!complete_future
    return local_transitive_closure!(causet, affected, allowed_future_targets)
end

"""
    local_transitive_closure!(causet, affected, allowed_future_targets)

Transitively close rows selected by `affected`, optionally using a target mask
for closure seeds.

# Arguments
- `causet`: Causal set to close in place.
- `affected`: Row mask selecting rows that may change.
- `allowed_future_targets`: Optional mask selecting future targets whose rows
  should be used as closure seeds.

# Throws
- `ArgumentError`: If `affected` or `allowed_future_targets` has the wrong
  length.
"""
function local_transitive_closure!(
    causet::CausalSets.BitArrayCauset,
    affected::BitVector,
    allowed_future_targets::Union{Nothing,BitVector},
)
    n = causet.atom_count
    length(affected) == n ||
        throw(ArgumentError("affected must have length $(n), got $(length(affected))."))
    if allowed_future_targets !== nothing
        length(allowed_future_targets) == n || throw(ArgumentError(
            "allowed_future_targets must have length $(n), got $(length(allowed_future_targets)).",
        ))
    end
    seeds = BitVector(undef, n)
    old_future = BitVector(undef, n)
    added_future = BitVector(undef, n)

    @inbounds for source = n-1:-1:1
        affected[source] || continue
        copyto!(old_future, causet.future_relations[source])
        copyto!(seeds, causet.future_relations[source])
        if allowed_future_targets !== nothing
            seeds .&= allowed_future_targets
        end

        target = findfirst(seeds)
        while target !== nothing
            causet.future_relations[source] .|= causet.future_relations[target]
            target = findnext(seeds, target + 1)
        end

        added_future .= causet.future_relations[source]
        added_future .&= .!old_future
        target = findfirst(added_future)
        while target !== nothing
            causet.past_relations[target][source] = true
            target = findnext(added_future, target + 1)
        end
    end

    return causet
end

"""
    bool_mul2(A::BitMatrix, B::BitMatrix) -> BitMatrix

Compute the Boolean matrix product of `A` and `B`.

# Arguments
- `A`: Left Boolean matrix.
- `B`: Right Boolean matrix.

# Throws
- `ArgumentError`: If the inner matrix dimensions do not agree.
"""
function bool_mul2(A::BitMatrix, B::BitMatrix)
    mA, nA = size(A)
    mB, nB = size(B)
    nA == mB || throw(ArgumentError(
        "inner dimensions must agree for Boolean matrix multiplication, got $(size(A)) and $(size(B)).",
    ))
    AB = BitArray(undef, mA, nB)
    for i in 1:mA, j in 1:nB
        AB[i,j] = any(A[i,k] && B[k,j] for k in 1:nA)
    end
    AB
end

"""
    generate_KR_poset_adjacency_matrix(n_elements::Integer; rng=Random.default_rng()) -> BitMatrix

Generate the adjacency matrix of a random three-layer KR poset.

# Arguments
- `n_elements`: Number of elements in the KR poset. Must be at least 3.
- `rng`: Random number generator.

# Throws
- `ArgumentError`: If `n_elements < 3`.
"""
function generate_KR_poset_adjacency_matrix(n_elements::Integer; rng::Random.AbstractRNG = Random.default_rng())
    n_elements >= 3 ||
        throw(ArgumentError("n_elements must be at least 3, got $n_elements."))
    # Using this approach rather than simply sampling 
    layer_counts = rand(rng, Distributions.Multinomial(n_elements, [0.25, 0.5, 0.25]));
    # NOTE: Should be equivalent to connecting from the bottom, but am not totaly certain
    bottom_to_middle = Random.bitrand(rng, layer_counts[1], layer_counts[2]);
    middle_to_top = Random.bitrand(rng, layer_counts[2], layer_counts[3]);
    bottom_to_top = bool_mul2(bottom_to_middle, middle_to_top);
    A = falses(n_elements, n_elements);
    A[1:layer_counts[1], (layer_counts[1]+1):(layer_counts[1]+layer_counts[2])] = bottom_to_middle;
    A[(layer_counts[1]+1):(layer_counts[1]+layer_counts[2]),(layer_counts[1]+layer_counts[2]+1):end] = middle_to_top;
    A[1:layer_counts[1], (layer_counts[1]+layer_counts[2]+1):end] = bottom_to_top;
    A
end

"""
    BitArrayCauset(A::BitMatrix)

Create a `BitArrayCauset` from a square adjacency matrix.

# Arguments
- `A`: Square adjacency matrix in natural labelling.

# Throws
- `ArgumentError`: If `A` is not square.
"""
function CausalSets.BitArrayCauset(A::BitMatrix)
    size(A, 1) == size(A, 2) ||
        throw(ArgumentError("adjacency matrix must be square, got size $(size(A))."))
    n = size(A, 1);
    future_relations = [A[i,:] for i in 1:n];
    past_relations   = [A[:,j] for j in 1:n];
    CausalSets.BitArrayCauset(n, future_relations, past_relations);
end

"""
    generate_KR_poset(n_elements::Integer; rng=Random.default_rng()) -> BitArrayCauset

Generate a random three-layer KR poset as a `BitArrayCauset`.

# Arguments
- `n_elements`: Number of elements in the KR poset. Must be at least 3.
- `rng`: Random number generator.

# Throws
- `ArgumentError`: If `n_elements < 3`.
"""
function generate_KR_poset(n_elements::Integer; rng::Random.AbstractRNG = Random.default_rng())
    CausalSets.BitArrayCauset(generate_KR_poset_adjacency_matrix(n_elements; rng = rng))
end

"""
    place_causet_in_manifold_causet(manifold_causet::ManifoldCauset{N}, boundary::OffsetCausalDiamondBoundary{N}, small_causet::BitArrayCauset)

Return a copy of `manifold_causet` with the induced subcauset inside `boundary`
replaced by `small_causet`, followed by local transitive closure.

# Arguments
- `manifold_causet`: Original manifold causet.
- `boundary`: Offset causal diamond selecting the region to replace.
- `small_causet`: Replacement causet. Its atom count must match the number of
  elements in `boundary`.

# Throws
- `ArgumentError`: If the number of elements in `boundary` differs from
  `small_causet.atom_count`.
"""
function place_causet_in_manifold_causet(manifold_causet::CausalSets.ManifoldCauset{N}, boundary::OffsetCausalDiamondBoundary{N}, small_causet::CausalSets.BitArrayCauset) where {N}
    # findall returns indices ordered
    in_boundary_indices = findall(CausalSets.is_in_boundary.(Ref(manifold_causet.manifold), Ref(boundary), manifold_causet.sprinkling))
    length(in_boundary_indices) == small_causet.atom_count || throw(ArgumentError(
        "boundary contains $(length(in_boundary_indices)) elements, but small_causet has $(small_causet.atom_count).",
    ))
    out_causet = CausalSets.BitArrayCauset(manifold_causet.manifold, manifold_causet.sprinkling)
    # Both the elements in the manifold_causet and the small_causet are already ordered by time coordinate, so this will 
    # match the elements of the n-element top layer with the 
    for (i,idx) in enumerate(in_boundary_indices)
        out_causet.future_relations[idx][in_boundary_indices] = small_causet.future_relations[i];
        out_causet.past_relations[idx][in_boundary_indices] = small_causet.past_relations[i];
    end
    local_transitive_closure!(out_causet, in_boundary_indices, manifold_causet, boundary)
    out_causet
end

"""
    is_subset(test_boundary, reference_boundary, manifold) -> Bool

Return whether both tips of `test_boundary` lie inside `reference_boundary`.

# Arguments
- `test_boundary`: Offset causal diamond to test.
- `reference_boundary`: Boundary that should contain `test_boundary`.
- `manifold`: Minkowski manifold used for containment.
"""
function is_subset(test_boundary::OffsetCausalDiamondBoundary, reference_boundary::CausalSets.AbstractBoundary, manifold::CausalSets.MinkowskiManifold)
    return all(CausalSets.is_in_boundary.(Ref(manifold), Ref(reference_boundary), get_tips(test_boundary, manifold)))
end

"""
    replace_region_with_KR_poset(manifold_causet::ManifoldCauset{N}, sprinkling_boundary::AbstractBoundary, element_count::Int64; require_region_fully_in_boundary::Bool=true, return_KR_poset=false)

Replace a randomly selected causal diamond in `manifold_causet` by a random KR
poset with `element_count` elements.

# Arguments
- `manifold_causet`: Original manifold causet.
- `sprinkling_boundary`: Boundary from which candidate region centers are
  sampled.
- `element_count`: Size of the KR poset to insert. Must be between 3 and the
  atom count.
- `require_region_fully_in_boundary`: If true, only accept regions whose tips
  lie in `sprinkling_boundary`.
- `return_KR_poset`: If true, return the inserted KR poset together with the
  resulting causet.
- `rng`: Random number generator.

# Throws
- `ArgumentError`: If `element_count < 3` or exceeds the atom count.
"""
function replace_region_with_KR_poset(manifold_causet::CausalSets.ManifoldCauset{N}, sprinkling_boundary::CausalSets.AbstractBoundary, element_count::Int64; require_region_fully_in_boundary::Bool=true, return_KR_poset=false, rng::Random.AbstractRNG = Random.default_rng()) where {N}
    3 <= element_count <= manifold_causet.atom_count || throw(ArgumentError(
        "element_count must be between 3 and $(manifold_causet.atom_count), got $element_count.",
    ))
    KR_poset = generate_KR_poset(element_count; rng = rng)
    if require_region_fully_in_boundary
        while true
            center = CausalSets.generate_sprinkling(manifold_causet.manifold, sprinkling_boundary, 1; rng = rng)[1]
            inner_boundary = boundary_from_contained_element_count(manifold_causet, center, element_count)
            if is_subset(inner_boundary, sprinkling_boundary, manifold_causet.manifold)
                break
            end
        end
    else
        center = CausalSets.generate_sprinkling(manifold_causet.manifold, sprinkling_boundary, 1; rng = rng)[1]
        inner_boundary = boundary_from_contained_element_count(manifold_causet, center, element_count)
    end
    if return_KR_poset
        return place_causet_in_manifold_causet(manifold_causet, inner_boundary::OffsetCausalDiamondBoundary{N}, KR_poset), KR_poset
    else
        return place_causet_in_manifold_causet(manifold_causet, inner_boundary::OffsetCausalDiamondBoundary{N}, KR_poset)
    end
end

"""
    generate_causet_with_KR_defect(n, m, sprinkling_boundary, manifold; rng=Random.default_rng())

Generate an `n`-element manifold causet and replace an `m`-element local region
by a random KR poset.

# Arguments
- `n`: Number of elements in the generated manifold causet.
- `m`: Number of elements in the inserted KR poset.
- `sprinkling_boundary`: Boundary used for the initial sprinkling.
- `manifold`: Manifold used for the initial sprinkling.
- `rng`: Random number generator.

# Throws
- `ArgumentError`: If `n < 3`, `m < 3`, or `m > n`.
"""
function generate_causet_with_KR_defect(n::Int64, m::Int64, sprinkling_boundary::CausalSets.AbstractBoundary, manifold::CausalSets.AbstractManifold; rng::Random.AbstractRNG = Random.default_rng())
    n >= 3 || throw(ArgumentError("n must be at least 3, got $n."))
    3 <= m <= n || throw(ArgumentError("m must be between 3 and n=$n, got $m."))
    manifold_causet = CausalSets.ManifoldCauset(manifold, CausalSets.generate_sprinkling(manifold, sprinkling_boundary, n; rng = rng))
    combined_causet = replace_region_with_KR_poset(manifold_causet, sprinkling_boundary, m; rng = rng)
    return combined_causet
end

# example
#=
manifold = MinkowskiManifold{4}()
sprinkling_boundary = CausalDiamondBoundary{4}(1.)

generate_causet_with_KR_defect(2048, 256, sprinkling_boundary, manifold)
=#
