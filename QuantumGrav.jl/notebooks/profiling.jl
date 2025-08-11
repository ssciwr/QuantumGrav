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
create_random_cset(100, future_deg, link_prob, rng);

@profview create_random_cset(3000, future_deg, link_prob, rng);

sizes = [1000, 2000, 3000, 4000]
times = []
for size in sizes
    t = @elapsed create_random_cset(size, future_deg, link_prob, rng)
    push!(times, t)
end

println(times)
