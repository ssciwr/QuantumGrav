size = 2^6;
offset = 0;
stepsize = 20;
max_step = 1000;
maxi = Int64((max_step - offset) / stepsize)
resrel = zeros(maxi);
resrel2 = zeros(maxi);
markov_chain_step = zeros(maxi);
for i in 1:maxi
    rng = MersenneTwister(1234);
    # Sample causet with 2 flips per step
    cset = alt_sample_bitarray_causet(size, .55, stepsize * i + offset, rng,20);
    # Sample causet with default flips per step (missing argument treated as error, but kept as is)
    cset2 = alt_sample_bitarray_causet(size, .55, stepsize * i + offset, rng);
    # Count relations in sampled causets
    resrel[i] = count_relations(cset)
    resrel2[i] = count_relations(cset2)
    markov_chain_step[i] = stepsize * i + offset
end
# Compute connectivity ratios
connectivity = resrel ./ (size*(size+1)/2)
connectivity2 = resrel2 ./ (size*(size+1)/2)
# Plot results
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Markov Steps", ylabel="Connectivity",title = "MCMC sampling of $(size)-atom non-manifoldlike causets")
lines!(ax, markov_chain_step, connectivity)
lines!(ax, markov_chain_step, connectivity2)
fig