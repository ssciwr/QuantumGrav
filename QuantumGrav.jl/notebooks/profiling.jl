using Pkg: Pkg
Pkg.activate("./QuantumGrav.jl/notebooks")
# Pkg.activate(".")
Pkg.update()
Pkg.resolve()
Pkg.instantiate()

using Random: Random
using Distributions: Distributions
using SparseArrays: SparseArrays
using StatsBase: StatsBase
using CausalSets: CausalSets

include("../src/daggeneration.jl") # 
include("../src/datageneration.jl") # for QuantumGrav.jl

function future_deg(rng, i, future, atom_count)
    distr = Distributions.Binomial(length(future), 0.5)
    return Distributions.rand(rng, distr)
end

function link_prob(rng, i, j)
    if i == j
        return 1e-15
    end

    return 1e-7 * exp(-abs(i - j) / 10.0) + 1e-15
end

rng = Random.Xoshiro(234)

# to compile the functions in question in order to remove compile time from measurements
# do this to compile the function first, so the compilation time doesn't show up in the measurements
create_random_cset(100,
                   future_deg,
                   link_prob,
                   rng);

@profview create_random_cset(3000,
                             future_deg,
                             link_prob,
                             rng);

# sizes = 10:250:25010
# times = []
# told = 0

# for size in sizes
#     time = @elapsed create_random_cset(size,
#                                        future_deg,
#                                        link_prob,
#                                        rng)
#     delta = time - told
#     println("Size: $size, Time: $time seconds, Delta: $delta seconds")
#     push!(times, time)
#     global told = time
# end

# using Plots
# using CSV
# dt = vcat([0], diff(times))
# CSV.write("times_normal_sampling.csv", (s = sizes, t = times, dt = dt))

# scatter(sizes, times; xlabel="Size", ylabel="Time (seconds)",
#         title="Causal Set Creation Time", legend=false, ylim=[0, 150], markersize=3)
# savefig("causal_set_creation_time_normal_sampling.png")

# scatter(log.(sizes), log.(times); xlabel="log(Size)", ylabel="log(Time) (log(seconds))",
#         title="Causal Set Creation Time", legend=false, markersize=3)
# savefig("causal_set_creation_time_log_normal_sampling.png")

# scatter(log.(sizes), diff(log.(times)); xlabel="log(Size) Difference",
#         ylabel="log(Time) Difference (log(seconds))",
#         title="Causal Set Creation Time Differences", legend=false, markersize=3,
#         ylim=[-1, 1])
# savefig("causal_set_creation_time_log_diffs_normal_sampling.png")

# scatter(sizes, diff(times); xlabel="size Difference", ylabel="time Difference (seconds)",
#         title="Causal Set Creation Time Differences", legend=false, markersize=3,
#         ylim=[-10, 10])
# savefig("causal_set_creation_time_diffs_normal_sampling.png")
