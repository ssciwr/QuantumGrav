using Random
using CausalSets
using CairoMakie

function alt_sample_bitarray_causet(size::Int64, connectivity_goal::Float64, markov_steps::Int64, rng::AbstractRNG)
    graph = CausalSets.empty_graph(size)
    tcg = CausalSets.empty_graph(size)
    trg = CausalSets.empty_graph(size)
    tcg_new = CausalSets.empty_graph(size)
    trg_new = CausalSets.empty_graph(size)

    CausalSets.transitive_closure!(graph, tcg)
    prev_connectivity = CausalSets.count_edges(tcg)/(size*(size-1)/2)

    for step in 1:markov_steps
        i = rand(rng, 1:size-1)
        j = rand(rng, i+1:size)
        prev_edge = graph.edges[i][j]
        graph.edges[i][j] = !prev_edge
        if prev_edge || !tcg.edges[i][j]
            CausalSets.transitive_closure!(graph, tcg_new)
        end
        new_connectivity = CausalSets.count_edges(tcg_new)/(size*(size-1)/2)
        if (new_connectivity - connectivity_goal)^2 <= (prev_connectivity - connectivity_goal)^2 || rand(rng) < 2. ^(1e5*(abs(prev_connectivity - connectivity_goal)-abs(new_connectivity - connectivity_goal)))
            # Accept the modification:
            tcg = tcg_new
            trg = trg_new
            prev_connectivity = new_connectivity
        else
            # Reject the modification:
            graph.edges[i][j] = !graph.edges[i][j]
        end
    end
    return CausalSets.to_bitarray_causet(tcg)
end

size = 2^9;
offset = 0;
stepsize = 1000;
max_step = 5000;
maxi = Int64((max_step - offset) / stepsize)
resrel = zeros(maxi);
markov_chain_step = zeros(maxi);
for i in 1:maxi
    rng = MersenneTwister(1234);
    cset = alt_sample_bitarray_causet(size, .45, stepsize * i + offset, rng);
    resrel[i] = count_relations(cset)
    markov_chain_step[i] = stepsize * i + offset
end
connectivity = resrel ./ (size*(size+1)/2)
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Markov Steps", ylabel="Connectivity",title = "MCMC sampling of $(size)-atom non-manifoldlike causets")
lines!(ax, markov_chain_step, connectivity)
fig