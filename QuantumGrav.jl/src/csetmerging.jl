

"""
insert_cset(cset1Raw::AbstractCauset, cset2Raw::AbstractCauset, link_probability::Float64; rng::AbstractRNG=Random.GLOBAL_RNG, position::Union{Nothing, Int64}=nothing) 
    -> BitArrayCauset

Insert `cset2Raw` into `cset1Raw` at a random or specified position. All atoms are reindexed accordingly.
Random links are added *across* the insertion boundary — i.e. from atoms before the inserted block
to atoms inside it, and from the inserted block to atoms after it — with probability `link_probability`.
Transitive closure is applied after insertion.

# Arguments
- `cset1Raw`: First input causal set (converted to `BitArrayCauset` if necessary)
- `cset2Raw`: Second input causal set (converted to `BitArrayCauset` if necessary)
- `link_probability`: Probability with which to add links across the insertion boundary (must be in [0, 1])
- `rng`: Random number generator (default: `Random.GLOBAL_RNG`)
- `position`: Optional insertion index in `0:n1`; if `nothing`, insertion is random

# Returns
- A `BitArrayCauset` representing the merged and transitively closed causal set

# Throws
- `ArgumentError` if `link_probability` is not in the interval [0, 1]
- `ArgumentError` if `position` is not in the valid range `0 ≤ position ≤ atom_count of cset1Raw`
"""
function insert_cset(
    cset1Raw::CausalSets.AbstractCauset,
    cset2Raw::CausalSets.AbstractCauset,
    link_probability::Float64;
    rng::Random.AbstractRNG = Random.GLOBAL_RNG,
    position::Union{Nothing,Int64} = nothing,
)::CausalSets.BitArrayCauset
    if link_probability < 0.0 || link_probability > 1.0
        throw(
            ArgumentError(
                "link_probability must be between 0 and 1. Got $link_probability.",
            ),
        )
    end

    cset1 =
        isa(cset1Raw, CausalSets.BitArrayCauset) ? cset1Raw :
        convert(CausalSets.BitArrayCauset, cset1Raw)
    cset2 =
        isa(cset2Raw, CausalSets.BitArrayCauset) ? cset2Raw :
        convert(CausalSets.BitArrayCauset, cset2Raw)

    n1, n2 = cset1.atom_count, cset2.atom_count
    N = n1 + n2

    insert_pos = isnothing(position) ? rand(rng, 0:n1) : position  # Insert cset2 between indices insert_pos and insert_pos + 1

    (insert_pos < 0 || insert_pos > n1) && throw(
        ArgumentError(
            "position must be between 0 and atom_count of cset1Raw = $(n1). Got $(insert_pos).",
        ),
    )

    # Compute new index mapping
    # Atoms before insert_pos stay the same
    # Then cset2 atoms
    # Then remaining cset1 atoms get shifted by n2
    idx_map_cset1 = [i <= insert_pos ? i : i + n2 for i = 1:n1]
    idx_map_cset2 = insert_pos .+ (1:n2)

    graph_merged = CausalSets.empty_graph(N)
    tcg_merged = CausalSets.empty_graph(N)

    # Copy edges from cset1
    for (i_old, i_new) in enumerate(idx_map_cset1)
        row = falses(N)
        for (j_old, val) in enumerate(cset1.future_relations[i_old])
            if val
                j_new = idx_map_cset1[j_old]
                row[j_new] = true
            end
        end
        graph_merged.edges[i_new] = row
    end

    # Copy edges from cset2
    for (i_old, i_new) in enumerate(idx_map_cset2)
        row = falses(N)
        for (j_old, val) in enumerate(cset2.future_relations[i_old])
            if val
                j_new = idx_map_cset2[j_old]
                row[j_new] = true
            end
        end
        graph_merged.edges[i_new] = row
    end

    # Link insertion
    inserted_range = insert_pos .+ (1:n2)

    # Links from before to inserted
    for i = 1:insert_pos
        for j in inserted_range
            if !graph_merged.edges[i][j] && rand(rng) < link_probability
                graph_merged.edges[i][j] = true
            end
        end
    end

    # Links from inserted to after
    for i in inserted_range
        for j = (insert_pos+n2+1):N
            if !graph_merged.edges[i][j] && rand(rng) < link_probability
                graph_merged.edges[i][j] = true
            end
        end
    end

    CausalSets.transitive_closure!(graph_merged, tcg_merged)
    return CausalSets.to_bitarray_causet(tcg_merged)
end


"""
merge_csets(cset1Raw::AbstractCauset, cset2Raw::AbstractCauset, link_probability::Float64) 
    -> BitArrayCauset

Merge two causal sets `cset1Raw` and `cset2Raw` into a single causal set by placing them
on the diagonal of a larger causet and connecting them with random links in the upper-right
block with probability `link_probability`. Transitive closure is applied after merging.

Note: The actual degree of connectivity is underestimated unless `link_probability` is 0 or 1,
since transitive completion will overwrite the sparsity induced by random linking.

# Arguments
- `cset1Raw`: First input causal set (converted to `BitArrayCauset` if necessary)
- `cset2Raw`: Second input causal set (converted to `BitArrayCauset` if necessary)
- `link_probability`: Probability with which to add links from `cset1Raw` to `cset2Raw` (must be in [0, 1])

# Returns
- A `BitArrayCauset` representing the merged and transitively closed causal set

# Throws
- `ArgumentError` if `link_probability` is not in the interval [0, 1]
"""
function merge_csets(
    cset1Raw::CausalSets.AbstractCauset,
    cset2Raw::CausalSets.AbstractCauset,
    link_probability::Float64;
    rng::Random.AbstractRNG = Random.GLOBAL_RNG,
)::CausalSets.BitArrayCauset
    return insert_cset(
        cset1Raw,
        cset2Raw,
        link_probability;
        rng = rng,
        position = cset1Raw.atom_count,
    )
end

"""
insert_KR_into_manifoldlike(npoints::Int64, order::Int64, r::Float64, link_probability::Float64; 
                            rng::AbstractRNG=Random.GLOBAL_RNG, position::Union{Nothing, Int64}=nothing,
                            d::Int64=2, type::Type=Float32, p::Float64=0.5)
    -> Tuple{BitArrayCauset, Bool, Matrix{T}}

Generate a manifoldlike causal set with `npoints` elements and insert into it a KR-order (random layered)
causal set containing 5% of the elements. The insertion point is chosen randomly (or specified via `position`).
Random links are added across the insertion boundary with probability `link_probability`.
Transitive closure is applied to ensure consistency.

Returns the merged causet, a dummy `true`, and the coordinate matrix used for the manifoldlike causet.

# Arguments
- `npoints`: Number of elements in the manifoldlike causal set
- `order`: Sprinkling order for the manifoldlike causet
- `r`: Interaction scale for manifoldlike causet generation
- `link_probability`: Probability for adding links across the insertion boundary (must be in [0, 1])
- `rng`: Random number generator (default: `Random.GLOBAL_RNG`)
- `n2_rel`: Size of KR order relative to size of manifoldlike causet
- `position`: Optional insertion index in `0:npoints`; if `nothing`, insertion is random
- `d`: Dimension of the manifoldlike causal set (default: 2)
- `type`: Coordinate type (default: Float32)
- `p`: Link-probability within KR-order (default: 0.5)

# Returns
- A tuple `(cset, true, coords)` where:
    - `cset` is the merged and transitively completed `BitArrayCauset`
    - `true` is a placeholder flag
    - `coords` is a `Matrix{T}` of coordinates used for the manifoldlike causet

# Throws
- `ArgumentError` if `link_probability` is not in the interval [0, 1]
- `ArgumentError` if `position` is not in the valid range `0 ≤ position ≤ npoints`
"""
function insert_KR_into_manifoldlike(
    npoints::Int64,
    order::Int64,
    r::Float64,
    link_probability::Float64;
    rng::Random.AbstractRNG = Random.GLOBAL_RNG,
    position::Union{Nothing,Int64} = nothing,
    n2_rel::Float64 = 0.05,
    d::Int64 = 2,
    type::Type{T} = Float32,
    p::Float64 = 0.5,
)::Tuple{CausalSets.BitArrayCauset,Bool,Matrix{T}} where {T}

    n2_rel <= 0 && throw(ArgumentError("n2_rel must be larger than 0, is $n2_rel."))

    cset1Raw, _, _ = make_manifold_cset(npoints, rng, order, r; d = d, type = type)
    n2 = max(1, round(Int, n2_rel * npoints))  # Ensure at least 1

    cset2Raw, _ = create_random_layered_causet(n2, 3; p = p)

    return insert_cset(
        cset1Raw,
        cset2Raw,
        link_probability;
        rng = rng,
        position = position,
    ),
    true,
    stack(make_pseudosprinkling(npoints + n2, d, -1.0, 1.0, type; rng = rng), dims = 1)
end
