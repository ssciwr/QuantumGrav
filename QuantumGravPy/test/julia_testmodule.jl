import QuantumGrav as QG
import Random
import Distributions

struct Generator
    seed::Int64
end

function Generator(config::AbstractDict)
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
    boundary_distr = Distributions.DiscreteUniform(1, 3)
    atomcount_distr = Distributions.DiscreteUniform(min_atomcount, max_atomcount)

    # make a bunch of datapoints
    batch = []
    for _ = 1:5
        data = Dict{String,Any}()
        ok = false
        max_iter = 20
        type = Float32
        e = nothing
        while ok == false && max_iter > 0
            max_iter -= 1
            dimension = rand(rng, dim_distr)
            manifold_id = Distributions.rand(rng, manifold_distr)
            manifold = manifolds[manifold_id]
            boundary_id = Distributions.rand(rng, boundary_distr)
            boundary = boundaries[boundary_id]
            atomcount = Distributions.rand(rng, atomcount_distr)
            # make dataset 
            try
                # make data needed 
                cset, sprinkling =
                    QG.make_cset(manifold, boundary, atomcount, dimension, rng; type = type)

                link_matrix = QG.make_link_matrix(cset, type = type)
                adjacency_matrix = QG.make_adj(cset, type = type)

                data["manifold"] = manifold_id
                data["boundary"] = boundary_id
                data["dimension"] = dimension
                data["atomcount"] = atomcount
                data["adjacency_matrix"] = adjacency_matrix
                data["link_matrix"] = link_matrix

                ok = true
            catch error
                println("Error generating data: ", error)
                ok = false
                e = error
            end
            if max_iter <= 0
                println("Max iterations reached, breaking out of loop.")
                break
            end
        end
        push!(batch, data)
    end
    return batch
end
