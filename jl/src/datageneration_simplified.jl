struct PseudoManifold{N} <: CausalSets.AbstractManifold{N} end

function make_cset_features(cset::CausalSets.AbstractCauset, adj::AbstractMatrix,
        in_degrees::AbstractVector, out_degrees::AbstractVector,
        manifold::CausalSets.AbstractManifold, d::Int, type::Type{T}) where {T <: Number}
    # modify this as is needed. We will need more and different positional ang structural embeddings
    # make features for nodes
    manifold_name = get_manifold_name(typeof(manifold), d)

    max_path_lengths_future = Vector{Int32}(undef, size(adj, 1))

    max_path_lengths_past = Vector{Int32}(undef, size(adj, 1))

    # I don´t think we need this b/c the atoms are already topsorted
    topo_order_future = topsort(adj, in_degrees)
    topo_order_past = topsort(adj', out_degrees)

    for i in 1:size(adj, 1)
        @inbounds max_path_lengths_future[i] = maxpathlen(adj, topo_order_future, i)
        @inbounds max_path_lengths_past[i] = maxpathlen(adj', topo_order_past, i)
    end

    @inbounds dim = CausalSets.estimate_relation_dimension(cset)

    return manifold_name,
    max_path_lengths_future, max_path_lengths_past, topo_order_future, topo_order_past, dim
end

function make_data(seed, n_csets, d, num_event_dist; with_resize = false, node_max::Int = 0)
    thread_range = 1:Threads.nthreads()
    rngs = [Random.MersenneTwister(seed + i) for i in thread_range]

    # 1: Minkowski, 2: HyperCylinder, 3: DeSitter, 4: AntiDeSitter, 5: Torus, 6: PseudoManifold
    choose_manifolds = [Distributions.DiscreteUniform(1, 6) for i in thread_range]
    num_event_dists = [num_event_dist for i in thread_range]
    lm_data = [Vector{Matrix{Float32}}() for i in thread_range]
    adj_data = [Vector{SparseArrays.SparseMatrixCSC{Float32}}() for i in thread_range]
    in_degrees_data = [Vector{Vector{Float32}}() for i in thread_range]
    out_degrees_data = [Vector{Vector{Float32}}() for i in thread_range]
    manifold_ids = [Vector{Int64}() for i in thread_range]
    max_path_lengths_future_data = [Vector{Vector{Int32}}() for i in thread_range]
    max_path_lengths_past_data = [Vector{Vector{Int32}}() for i in thread_range]
    topo_order_future_data = [Vector{Vector{Int64}}() for i in thread_range]
    topo_order_past_data = [Vector{Vector{Int64}}() for i in thread_range]
    relation_dim_data = [Vector{Float32}() for i in thread_range]

    size_per_thread = Int(ceil(n_csets / Threads.nthreads()))
    sizehint!.(lm_data, size_per_thread)
    sizehint!.(adj_data, size_per_thread)
    sizehint!.(in_degrees_data, size_per_thread)
    sizehint!.(out_degrees_data, size_per_thread)
    sizehint!.(manifold_ids, size_per_thread)
    sizehint!.(max_path_lengths_future_data, size_per_thread)
    sizehint!.(max_path_lengths_past_data, size_per_thread)
    sizehint!.(topo_order_future_data, size_per_thread)
    sizehint!.(topo_order_past_data, size_per_thread)
    sizehint!.(relation_dim_data, size_per_thread)

    p = ProgressMeter.Progress(n_csets)

    Threads.@threads for i in 1:n_csets
        k = Threads.threadid()
        rng = rngs[k]
        num_event_dist = num_event_dists[k]
        choose_manifold = choose_manifolds[k]
        local n = Int(ceil(rand(rng, num_event_dist(d))))
        local manifold = make_manifold(rand(rng, choose_manifold), d)
        local boundary = nothing
        if manifold isa CausalSets.TorusManifold{d}
            boundary = CausalSets.BoxBoundary{d}((
                ([-0.49 for i in 1:d]...,), ([0.49 for i in 1:d]...,)))
        else
            boundary = CausalSets.CausalDiamondBoundary{d}(1.0)
        end

        local cset = make_cset(manifold, boundary, n, rng)
        local (lm, adj, in_degrees,
            out_degrees) = make_cset_matrices(
            cset, with_resize, node_max, Float32)
        local (manifold_name,
            max_path_lengths_future,
            max_path_lengths_past,
            topo_order_future,
            topo_order_past,
            relation_dim) = make_cset_features(
            cset, adj, in_degrees, out_degrees, manifold, d, Float32)

        push!(lm_data[k], lm)
        push!(adj_data[k], adj)
        push!(in_degrees_data[k], in_degrees)
        push!(out_degrees_data[k], out_degrees)
        push!(manifold_ids[k], get_manifold_encoding[manifold_name])
        push!(max_path_lengths_future_data[k], max_path_lengths_future)
        push!(max_path_lengths_past_data[k], max_path_lengths_past)
        push!(topo_order_future_data[k], topo_order_future)
        push!(topo_order_past_data[k], topo_order_past)
        push!(relation_dim_data[k], relation_dim)
        ProgressMeter.next!(p)
    end
    ProgressMeter.finish!(p)
    return vcat(lm_data...),
    vcat(adj_data...), vcat(in_degrees_data...), vcat(out_degrees_data...),
    vcat(manifold_ids...), vcat(max_path_lengths_future_data...),
    vcat(max_path_lengths_past_data...), vcat(topo_order_future_data...),
    vcat(topo_order_past_data...), vcat(relation_dim_data...)
end
