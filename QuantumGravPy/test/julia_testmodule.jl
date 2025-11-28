import QuantumGrav as QG
import Random
import Distributions

"""
This is a dummy module only used for testing purposes. It shows how to create a Julia module that can be called from Python using the `jlcall` package.
In detail, this module defines a `Generator` struct and a call operator that generates a batch of data points for testing purposes.
"""

"""
    Generator

A struct that represents a random data generator for causal sets.

# Fields:
- `seed::Int64`: Random seed for reproducibility.
"""
struct Generator
    seed::Int64
end

"""
    Generator(config::AbstractDict)

Initialize the generator with a configuration dictionary containing a seed.

Args:
- config::AbstractDict: A dictionary containing the configuration parameters, including a "seed" key.
Returns:
- `Generator`: An instance of the `Generator` struct initialized with the provided seed.
"""
function Generator(config::AbstractDict)
    return Generator(config["seed"])
end

"""
    gen::Generator(batchsize::Int)

Call the generator to create a batch of data points.

Args:
- `batchsize::Int`: The number of data points to generate.

Returns:
- `Array{Dict{String,Any}}`: An array of dictionaries, each containing the generated data for a causal set. Each dictionary includes:
  - `manifold`: An integer representing the manifold type (1-6).
  - `boundary`: An integer representing the boundary type (1-3).
  - `dimension`: An integer representing the dimension of the causal set (2-4).
  - `atomcount`: An integer representing the number of atoms in the causal set (5-15).
  - `adjacency_matrix`: A matrix representing the adjacency relations in the causal set.
  - `link_matrix`: A matrix representing the link relations in the causal set.
  - `max_pathlen_future`: A list of maximum path lengths into the future for each atom in the causal set.
  - `max_pathlen_past`: A list of maximum path lengths into the past for each atom in the causal set.
"""
function (gen::Generator)(batchsize::Int)

    # set of manifolds and boundaries to choose from
    # min, max sizes of the causal sets.
    min_atomcount = 5
    max_atomcount = 15

    # rng and distributions for the random generation
    rng = Random.Xoshiro(gen.seed)
    atomcount_distr = Distributions.DiscreteUniform(min_atomcount, max_atomcount)

    # make a bunch of datapoints
    batch = Vector{Dict{String,Any}}(undef, batchsize)
    for i = 1:batchsize
        data = Dict{String,Any}()
        type = Float32
        atomcount = rand(rng, atomcount_distr)
        dim = 2
        # choose all the parameters randomly for the causal set
        cset, _, __ =
            QG.make_polynomial_manifold_cset(atomcount, rng, 5, 2.0; d = dim, type = type)

        # make the data: adjacency matrix and the other stuff
        adjacency_matrix = QG.make_adj(cset, type = Matrix, eltype = type)

        link_matrix = deepcopy(adjacency_matrix)
        QG.transitive_reduction!(link_matrix)

        max_pathlen_future = [
            QG.max_pathlen(adjacency_matrix, collect(1:size(adjacency_matrix, 1)), i)
            for i = 1:cset.atom_count
        ]

        max_pathlen_past = [
            QG.max_pathlen(adjacency_matrix', collect(1:size(adjacency_matrix, 2)), i)
            for i = 1:cset.atom_count
        ]

        # fill the data dictionary with the generated data
        data["dimension"] = dim
        data["atomcount"] = atomcount
        data["adjacency_matrix"] = Matrix(adjacency_matrix)
        data["link_matrix"] = Matrix(link_matrix)
        data["max_pathlen_future"] = max_pathlen_future
        data["max_pathlen_past"] = max_pathlen_past

        batch[i] = data
    end
    return batch
end
