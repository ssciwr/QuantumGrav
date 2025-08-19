"""
merge_csets(cset1Raw::AbstractCauset, cset2Raw::AbstractCauset, link_probability::Float64) -> BitArrayCauset

Merge two causal sets `cset1Raw` and `cset2Raw` into a single causal set by
placing them on the diagonal of a larger causet and connecting them with random
links in the upper-right block with probability `link_probability`. Note that the 
probability is applied before transitive completion. Therefore, it is not a good
measure of connection of the two sub-csets, but underestimates it (unless 
link_probability equals 0 or 1).

# Arguments
- `cset1Raw`: First input causal set (converted to `BitArrayCauset` if necessary)
- `cset2Raw`: Second input causal set (converted to `BitArrayCauset` if necessary)
- `link_probability`: Probability with which to add links from `cset1Raw` to `cset2Raw` (must be in [0, 1])

# Returns
- A `BitArrayCauset` representing the merged and transitively closed causal set
"""
function merge_csets(cset1Raw::CausalSets.AbstractCauset, cset2Raw::CausalSets.AbstractCauset, link_probability::Float64)
    if link_probability < 0.0 || link_probability > 1.0
        throw(ArgumentError("link_probability must be between 0 and 1. Got $link_probability."))
    end

    cset1 = typeof(cset1Raw) === CausalSets.BitArrayCauset ? cset1Raw : convert(CausalSets.BitArrayCauset, cset1Raw)
    cset2 = typeof(cset2Raw) === CausalSets.BitArrayCauset ? cset2Raw : convert(CausalSets.BitArrayCauset, cset2Raw)

    atom_count_merged = cset1.atom_count + cset2.atom_count
    graph_merged = CausalSets.empty_graph(atom_count_merged)
    tcg_merged = CausalSets.empty_graph(atom_count_merged)

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

    # Add random links in upper right block
    for i in 1:cset1.atom_count
        for j in 1:cset2.atom_count
            if rand() < link_probability
                graph_merged.edges[i][cset1.atom_count + j] = true
            end
        end
    end

    CausalSets.transitive_closure!(graph_merged, tcg_merged)
    
    return CausalSets.to_bitarray_causet(tcg_merged)
end