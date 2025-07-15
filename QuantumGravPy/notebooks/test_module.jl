import Random
import Distributions
import QuantumGrav as QG

struct Generator
    rng::Random.MersenneTwister
    atom_distr::Distributions.AbstractDistribution
    manifolds::Vector{String}
    boundaries::Vector{String}
    manifold_distr::Distributions.AbstractDistribution
    boundary_distr::Distributions.AbstractDistribution
    dimension_distr::Distributions.AbstractDistribution
    n_samples::Int64
    parallel::Bool
end


function Generator(
    seed::Int64,
    atom_min::Int64,
    atom_max::Int64,
    manifolds::Vector{String},
    boundaries::Vector{String},
    dimensions::Vector{Int64},
    n_samples::Int64 = 1,
    parallel::Bool = false,
)
    rng = Random.MersenneTwister(seed)
    atom_distr = Distributions.DiscreteUniform(atom_min, atom_max)
    manifold_distr = Distributions.DiscreteUniform(1:length(manifolds))
    boundary_distr = Distributions.DiscreteUniform(1:length(boundaries))
    dimension_distr =
        Distributions.DiscreteUniform(minimum(dimensions), maximum(dimensions))

    return Generator(
        rng,
        atom_distr,
        manifolds,
        boundaries,
        manifold_distr,
        boundary_distr,
        dimension_distr,
        n_samples,
        parallel,
    )
end

function (gen::Generator)()

    atom_count = rand(gen.rng, gen.atom_distr)
    manifold = gen.manifolds[rand(gen.rng, gen.manifold_distr)]
    boundary = gen.boundaries[rand(gen.rng, gen.boundary_distr)]
    dimension = rand(gen.rng, gen.dimension_distr)
    cset = nothing
    sprinkling = nothing
    working = false
    while working == false
        try
            cset, sprinkling =
                QG.make_cset(manifold, boundary, atom_count, dimension, rng = gen.rng)
            working = true
        catch _
            working = false
            cset = nothing
            sprinkling = nothing
        end
    end

    # use closure to parallelize the sample generation
    function generate_single_sample()
        adj = QG.make_adj(cset)
        link = QG.make_link_matrix(cset)
        max_pathlen_future_adj =
            [QG.max_pathlen(adj, 1:size(adj, 1), i) for i = 1:size(adj, 1)]
        max_pathlen_past_adj =
            [QG.max_pathlen(adj', i, 1:size(adj, 2)) for i = 1:size(adj, 2)]
        max_pathlen_future_link =
            [QG.max_pathlen(link, 1:size(link, 1), i) for i = 1:size(link, 1)]
        max_pathlen_past_link =
            [QG.max_pathlen(link', 1:size(link, 2), i) for i = 1:size(link, 2)]
        in_degree_adj = sum(adj, dims = 1)
        out_degree_adj = sum(adj, dims = 2)
        in_degree_link = sum(link, dims = 1)
        out_degree_link = sum(link, dims = 2)

        data = Dict(
            "atom_count" => atom_count,
            "manifold" => manifold,
            "boundary" => boundary,
            "dimension" => dimension,
            "sprinkling" => sprinkling,
            "adjacency_matrix" => adj,
            "link_matrix" => link,
            "max_pathlen_future_adj" => max_pathlen_future_adj,
            "max_pathlen_past_adj" => max_pathlen_past_adj,
            "max_pathlen_future_link" => max_pathlen_future_link,
            "max_pathlen_past_link" => max_pathlen_past_link,
            "in_degree_adj" => in_degree_adj,
            "out_degree_adj" => out_degree_adj,
            "in_degree_link" => in_degree_link,
            "out_degree_link" => out_degree_link,
        )
        return data
    end

    if gen.parallel
        data = [[] for _ = 1:Threads.nthreads()]
        for d in data
            sizehint!(d, Int(round(gen.n_samples / Threads.nthreads())))
        end
        Threads.@threads for i = 1:gen.n_samples
            push!(data[i], generate_single_sample())
        end
        data = vcat(data...)
    else
        data = [generate_single_sample() for _ = 1:gen.n_samples]
    end

    return data
end
