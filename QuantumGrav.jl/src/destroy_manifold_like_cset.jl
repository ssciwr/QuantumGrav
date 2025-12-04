"""
    destroy_manifold_cset(size::Int64, num_flips::Int64, rng::Random.AbstractRNG, order::Int64, r::Float64; d::Int64=2, type::Type{T}=Float32) -> CausalSets.BitArrayCauset

Starts from a manifold-like causal set and destroys its manifoldlike structure by flipping a given number of randomly chosen edges, then applies transitive closure.

# Arguments
- `size::Int64`: The number of elements in the causal set.
- `num_flips::Int64`: The number of edges to flip at random in the set (each flip toggles the presence/absence of a relation).
- `rng::Random.AbstractRNG`: Random number generator to use for reproducibility.
- `order::Int64`: The order parameter passed to the manifold causal set generator.
- `r::Float64`: The sprinkling density or radius parameter for the manifold causal set.

# Keyword Arguments
- `d::Int64=2`: The dimension of the spacetime manifold to sprinkle into (default: 2).
- `type::Type{T}=Float32`: The numeric type for coordinates (default: Float32).


# Returns
- `CausalSets.BitArrayCauset`: The resulting causal set as a BitArrayCauset, which may be non-manifoldlike after random edge flips.
- `sprinkling`: The sprinkling of the geometry underlying the destroyed causal set.
- `chebyshev_coefs`: The Chebyshev coefficients of the geometry underlying the destroyed causal set.

# Throws
- `ArgumentError`: if `num_flips < 1`.
- `ArgumentError`: if `num_flips > size*(size-1)/2`.

# Behavior
This function generates a manifold-like causal set using `make_polynomial_manifold_cset`, then randomly selects `num_flips` pairs of elements (edges) and flips their causal relation (adding or removing the edge). After all flips, it applies transitive closure to ensure the causality structure is consistent, and returns the final causal set as a `BitArrayCauset`.
"""
function destroy_manifold_cset(
    size::Int64,
    num_flips::Int64,
    rng::Random.AbstractRNG,
    order::Int64,
    r::Float64;
    d::Int64 = 2,
    type::Type{T} = Float32,
)::Tuple{CausalSets.BitArrayCauset,Vector{Tuple{T,Vararg{T}}},Matrix{T}} where {T<:Number}
    if num_flips < 1
        throw(ArgumentError("num_flips must be at least 1, is $(num_flips)"))
    end
    if num_flips > size * (size - 1) ÷ 2
        throw(
            ArgumentError(
                "num_flips cannot exceed number of possible edges $(size*(size-1)/2), is $(num_flips)",
            ),
        )
    end

    if size < 2 
        throw(ArgumentError("size must be at least 2 to perform edge flips, is $(size)"))
    end
    
    cset, sprinkling, chebyshev_coefs =
        make_polynomial_manifold_cset(size, rng, order, r; d = d, type = type)

    destroyed_cset = CausalSets.empty_graph(size)
    destroyed_tcg = CausalSets.empty_graph(size)
    for i = 1:size
        for j = (i+1):size
            destroyed_cset.edges[i][j] = cset.future_relations[i][j]
        end
    end

    for flip = 1:num_flips
        i = rand(rng, 1:(size-1))
        j = rand(rng, (i+1):size)
        destroyed_cset.edges[i][j] = ! destroyed_cset.edges[i][j]
    end

    CausalSets.transitive_closure!(destroyed_cset, destroyed_tcg)

    return CausalSets.to_bitarray_causet(destroyed_tcg), sprinkling, chebyshev_coefs
end
