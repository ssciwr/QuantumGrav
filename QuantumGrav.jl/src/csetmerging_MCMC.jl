"""
```julia
count_edges_upper_right_corner(graph::CausalSets.ToposortedDAG, atom_count_1::Int) -> Int
```

Count the number of directed edges from the first `atom_count_1` rows
into the upper-right block of the adjacency matrix, i.e., from `cset1` to `cset2`.

# Arguments
- `graph`: The full graph with merged edges.
- `atom_count_1`: The size of the first (top-left) submatrix corresponding to `cset1`.

# Returns
- `Int`: Number of edges in the upper-right corner.
"""
function count_edges_upper_right_corner(graph::CausalSets.ToposortedDAG, atom_count_1::Int64)
    edges = 0
    for i in 1:atom_count_1
        edges += CausalSets.bitvector_count_ones(graph.edges[i][atom_count_1+1:end])
    end
    return edges
end

"""
```julia
merge_csets_MCMC(flip_param::Float64, cset1Raw::AbstractCauset, cset2Raw::AbstractCauset,
                 upper_right_connectivity_goal::Float64, markov_steps::Int64, rng::AbstractRNG;
                 rel_tol=nothing, abs_tol=nothing, acceptance=5e5) -> Tuple
```

Merge two causal sets via MCMC to reach a target connectivity in the upper-right corner.
Applies random edge flips followed by transitive closure and Metropolis acceptance.

# Arguments
- `flip_param`: Scaling parameter controlling number of edge flips per step.
- `cset1Raw`, `cset2Raw`: Input causal sets (converted to BitArrayCauset if needed).
- `upper_right_connectivity_goal`: Target fraction of edges in the upper-right block.
- `markov_steps`: Number of MCMC iterations.
- `rng`: Random number generator.
- `rel_tol`: Optional relative tolerance for early stopping.
- `abs_tol`: Optional absolute tolerance for early stopping.
- `acceptance`: Controls steepness of Metropolis probability.

# Returns
- A tuple `(merged_cset, success::Bool, final_error::Float64)`
"""
function merge_csets_MCMC(  flip_param::Float64,
                            cset1Raw::CausalSets.AbstractCauset, 
                            cset2Raw::CausalSets.AbstractCauset, 
                            upper_right_connectivity_goal::Float64,
                            markov_steps::Int64,
                            rng::Random.AbstractRNG;
                            rel_tol::Union{Float64,Nothing}=nothing,
                            abs_tol::Union{Float64,Nothing}=nothing,
                            acceptance::Float64=5e5,)
    cset1 = typeof(cset1Raw) === BitArrayCauset ? cset1Raw : convert(BitArrayCauset, cset1Raw)
    cset2 = typeof(cset2Raw) === BitArrayCauset ? cset2Raw : convert(BitArrayCauset, cset2Raw)

    atom_count1 = cset1.atom_count
    atom_count2 = cset2.atom_count
    atom_count_merged = atom_count1 + atom_count2

    graph_merged = CausalSets.empty_graph(atom_count_merged)
    tcg_merged = CausalSets.empty_graph(atom_count_merged)
    tcg_merged_new = CausalSets.empty_graph(atom_count_merged)

    for i in 1:cset1.atom_count
        row = falses(atom_count_merged)
        row[1:cset1.atom_count] .= cset1.future_relations[i]
        graph_merged.edges[i] = row
    end
    for i in 1:cset2.atom_count
        row = falses(atom_count_merged)
        row[cset1.atom_count + 1:end] .= cset2.future_relations[i]
        graph_merged.edges[cset1.atom_count + i] = row
    end

    CausalSets.transitive_closure!(graph_merged, tcg_merged)

    # Add random links in upper right block per merge_csets_MCMC
    prev_upper_right_connectivity = 0.

    e_old = 0
    step = 1
    while step < markov_steps
        e_old = abs(prev_upper_right_connectivity - upper_right_connectivity_goal)

        flips_per_step = Int64(ceil(flip_param *
                                    e_old *
                                    atom_count1 * atom_count2))

        # Randomly select edges to flip
        i = [rand(rng, 1:atom_count1) for flip in 1:flips_per_step]
        j = [rand(rng, (atom_count1 + 1):atom_count_merged) for flip in 1:flips_per_step]

        # Store previous edge states for possible rollback
        prev_edges = [graph_merged.edges[i[flip]][j[flip]] for flip in 1:flips_per_step]

        # Flip selected edges
        for flip in 1:flips_per_step
            graph_merged.edges[i[flip]][j[flip]] = !prev_edges[flip]
        end

        # Restore transitivity
        CausalSets.transitive_closure!(graph_merged, tcg_merged_new)

        # Compute new connectivity after flips
        new_upper_right_connectivity = count_edges_upper_right_corner(tcg_merged_new, atom_count1) / (atom_count1 * atom_count2)
        e_new = abs(new_upper_right_connectivity - upper_right_connectivity_goal)

        if abs_tol !== nothing && e_new < abs_tol
            return CausalSets.to_bitarray_causet(tcg_merged_new), true, e_new
        end

        if rel_tol !== nothing &&
           e_new / upper_right_connectivity_goal < rel_tol
            return CausalSets.to_bitarray_causet(tcg_merged_new), true, e_new
        end

        # Decide whether to accept or reject the new state based on connectivity and Metropolis criterion
        delta_e = e_old - e_new

        if e_new <= e_old || rand(rng) < min(1.0, exp(acceptance * delta_e))
            # Accept the modification:
            tcg_merged = tcg_merged_new
            prev_upper_right_connectivity = new_upper_right_connectivity
        else
            # Reject the modification: revert flipped edges
            for flip in 1:flips_per_step
                graph_merged.edges[i[flip]][j[flip]] = prev_edges[flip]
            end
        end
        step += 1
    end

    @warn "Relative precision $(rel_tol) or absolute precision $(abs_tol) not reached after $(markov_steps) steps. Final connectivity error: $(e_old)"
    return CausalSets.to_bitarray_causet(tcg_merged), false, e_old
end