import Random
import Distributions
import QuantumGrav as QG

struct Generator
    rng::Random.MersenneTwister
    atom_distr::Distributions.DiscreteUniform
    manifolds::Vector{Int64}
    boundaries::Vector{Int64}
    manifold_distr::Distributions.DiscreteUniform
    boundary_distr::Distributions.DiscreteUniform
    dimension_distr::Distributions.DiscreteUniform
    n_samples::Int64
    parallel::Bool
end

function Generator(
    seed::Int64,
    atom_min::Int64,
    atom_max::Int64,
    manifolds::Vector{Int64},
    boundaries::Vector{Int64},
    dimensions::Vector{Int64},
    n_samples::Int64 = 1,
    num_threads::Int64 = Threads.nthreads(),
)
    rng = Random.MersenneTwister(seed)
    atom_distr = Distributions.DiscreteUniform(atom_min, atom_max)
    manifold_distr = Distributions.DiscreteUniform(1, length(manifolds))
    boundary_distr = Distributions.DiscreteUniform(1, length(boundaries))
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
        num_threads > 1,
    )
end

function (gen::Generator)()
    println("call generator")
    # use closure to parallelize the sample generation
    function generate_single_sample()
        cset = nothing
        sprinkling = nothing
        working = false

        get_manifold_encoding = Dict(
            1 => "Minkowski",
            2 => "DeSitter",
            3 => "AntiDeSitter",
            4 => "HyperCylinder",
            5 => "Torus",
            6 => "Random",
        )

        get_boundary_encoding =
            Dict(1 => "CausalDiamond", 2 => "TimeBoundary", 3 => "BoxBoundary")
        atom_count = 0
        manifold_id = 0
        boundary_id = 0
        manifold = ""
        boundary = ""
        dimension = 0
        maxiter = 20
        while working == false && maxiter > 0
            atom_count = rand(gen.rng, gen.atom_distr)
            manifold_id = rand(gen.rng, gen.manifold_distr)
            boundary_id = rand(gen.rng, gen.boundary_distr)
            manifold = gen.manifolds[manifold_id]
            boundary = gen.boundaries[boundary_id]
            dimension = rand(gen.rng, gen.dimension_distr)
            try
                cset, sprinkling = QG.make_cset(
                    get_manifold_encoding[manifold],
                    get_boundary_encoding[boundary],
                    atom_count,
                    dimension,
                    gen.rng,
                )
                working = true
            catch e
                println("Error creating causal set: $e")
                working = false
            end
            maxiter -= 1
        end

        if maxiter <= 0
            throw(
                ErrorException(
                    "Failed to create a valid causal set after multiple attempts.",
                ),
            )
        end

        adj = QG.make_adj(cset)
        link = QG.make_link_matrix(cset)
        max_pathlen_future_adj =
            [QG.max_pathlen(adj, collect(1:size(adj, 1)), i) for i = 1:size(adj, 1)]
        max_pathlen_past_adj =
            [QG.max_pathlen(adj', collect(1:size(adj, 2)), i) for i = 1:size(adj, 2)]
        max_pathlen_future_link =
            [QG.max_pathlen(link, collect(1:size(link, 1)), i) for i = 1:size(link, 1)]
        max_pathlen_past_link =
            [QG.max_pathlen(link', collect(1:size(link, 2)), i) for i = 1:size(link, 2)]
        in_degree_adj = sum(adj, dims = 1)
        out_degree_adj = sum(adj, dims = 2)
        in_degree_link = sum(link, dims = 1)
        out_degree_link = sum(link, dims = 2)

        data = Dict(
            "atom_count" => atom_count,
            "manifold" => manifold_id,
            "boundary" => boundary_id,
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
