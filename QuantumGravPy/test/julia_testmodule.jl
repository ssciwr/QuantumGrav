import QuantumGrav as QG
import Random
import Distributions

struct Generator
    seed::Int64

end

function Generator(config::Dict{String,Any})
    return Generator(config["seed"])
end

# this is a test function to create a single datapoint of a dataset
function (gen::Generator)()
    manifolds =
        ["Minkowski", "DeSitter", "AntiDeSitter", "HyperCylinder", "Torus", "Random"]
    boundaries = ["CausalDiamond", "TimeBoundary", "BoxBoundary"]
    min_atomcount = 50
    max_atomcount = 600
    rng = Random.MersenneTwister(gen.seed)
    dim_distr = Distributions.DiscreteUniform(2, 4)
    manifold_distr = Distributions.DiscreteUniform(1, 6)
    boundary_distr = Distributions.DiscreteUniform(0.1, 1.0)
    atomcount_distr = Distributions.DiscreteUniform(min_atomcount, max_atomcount)

    # make a bunch of datapoints
    batch = []
    for i in 1:5
        data = Dict{String,Any}()
        ok = false
        max_iter = 20
        type = Float32
        while ok == false && max_iter > 0
            max_iter -= 1
            dimension = rand(rng, dim_distr)
            manifold = manifolds[Distributions.rand(rng, manifold_distr)]
            boundary = boundaries[Distributions.rand(rng, boundary_distr)]
            atomcount = Distributions.rand(rng, atomcount_distr)
            # make dataset 
            try
                # make data needed 
                cset = make_cset(manifold, boundary, atomcount, dimension, rng; type=type)

                link_matrix = QG.datageneration.make_link_matrix(cset, type=type)
                adjacency_matrix = QG.datageneration.make_adj(cset, type=type)


                data[i]["manifold"] = manifold
                data[i]["boundary"] = boundary
                data[i]["dimension"] = dimension
                data[i]["atomcount"] = atomcount
                data[i]["adjacency_matrix"] = adjacency_matrix
                data[i]["link_matrix"] = link_matrix


                ok = true
            catch e
                ok = false
            end
            if max_iter <= 0
                throw(
                    ErrorException(
                        "Failed to create a valid causal set after multiple attempts.",
                    ),
                )
            end
        end
        push!(batch, data)
    end
    return batch
end



