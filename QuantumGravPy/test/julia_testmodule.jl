import QuantumGrav as QG
import Random

seed = 1234
rng = Random.MersenneTwister(1234)

# this is a test function to create a single datapoint of a dataset
function create_datapoint(
    num_datapoints::Int,
    min_atomcount::Int,
    max_atomcount::Int,
    type = Float32,
)
    manifolds =
        ["Minkowski", "DeSitter", "AntiDeSitter", "HyperCylinder", "Torus", "Random"]
    boundaries = ["CausalDiamond", "TimeBoundary", "BoxBoundary"]

    dim_distr = Distributions.DiscreteUniform(2, 4)
    manifold_distr = Distributions.DiscreteUniform(1, 6)
    boundary_distr = Distributions.DiscreteUniform(0.1, 1.0)
    atomcount_distr = Distributions.DiscreteUniform(min_atomcount, max_atomcount)

    # make a bunch of datapoints
    data = [Dict{String,Any}() for _ = 1:num_datapoints]
    for i = 1:num_datapoints
        ok = false
        while ok == false
            dimension = rand(rng, dim_distr)
            manifold = manifolds[Distributions.rand(rng, manifold_distr)]
            boundary = boundaries[Distributions.rand(rng, boundary_distr)]
            atomcount = Distributions.rand(rng, atomcount_distr)
            # make dataset 
            try
                # make data needed 
                cset = make_cset(manifold, boundary, atomcount, dimension, rng; type = type)

                link_matrix = QG.datageneration.make_link_matrix(cset, type = type)
                adjacency_matrix = QG.datageneration.make_adj(cset, type = type)

                max_pathlen_future_adj = Vector{Int}(undef, size(adjacency_matrix, 1))
                max_pathlen_past_adj = Vector{Int}(undef, size(adjacency_matrix, 1))
                max_pathlen_future_link = Vector{Int}(undef, size(adjacency_matrix, 1))
                max_pathlen_past_link = Vector{Int}(undef, size(adjacency_matrix, 1))

                # the atoms are toposorted, so it should be fine to use the 
                # indexes as the topo order
                for i = 1:size(adj, 1)
                    @inbounds max_pathlen_future_adj[i] =
                        maxpathlen_fast(adj, 1:size(adj, 1), i)
                    @inbounds max_pathlen_past_adj[i] =
                        maxpathlen_fast(adj', 1:size(adj, 2), i)
                    @inbounds max_pathlen_future_link[i] =
                        maxpathlen_fast(link_matrix, 1:size(adj, 1), i)
                    @inbounds max_pathlen_past_link[i] =
                        maxpathlen_fast(link_matrix', 1:size(adj, 2), i)
                end

                data[i]["manifold"] = manifold
                data[i]["boundary"] = boundary
                data[i]["dimension"] = dimension
                data[i]["atomcount"] = atomcount
                data[i]["adjacency_matrix"] = adjacency_matrix
                data[i]["link_matrix"] = link_matrix
                data[i]["max_pathlen_future_adj"] = max_pathlen_future_adj
                data[i]["max_pathlen_past_adj"] = max_pathlen_past_adj
                data[i]["max_pathlen_future_link"] = max_pathlen_future_link
                data[i]["max_pathlen_past_link"] = max_pathlen_past_link

                ok = true
            catch e
                ok = false
            end
        end
    end
    return data
end
