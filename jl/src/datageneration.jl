
"""
    generate_data_for_manifold(
            ; dimension = 2,
            seed = 329478,
            num_datapoints = 1500,
            choose_num_events = d -> Distributions.Uniform(0.7 * 10^(d + 1), 1.1 * 10^(d + 1)),
            make_diamond = d -> CS.CausalDiamondBoundary{d}(1.0),
            make_box = d -> CS.BoxBoundary{d}((
                ([-0.49 for i in 1:d]...,), ([0.49 for i in 1:d]...,)))
    )

Generates a cset and a variet of data for a given manifold and dimension.

# Keyword Arguments
- `dimension::Int` (default: `2`): The dimension of the manifold.
- `seed::Int` (default: `329478`): The random seed for reproducibility.
- `num_datapoints::Int` (default: `1500`): The number of data points to generate.
- `choose_num_events::Function` (default: `d -> 10^(d + 1)`): A function that defines the size of the caussets, i.e., the number of events.
- `make_diamond::Function` (default: `d -> CausalDiamondBoundary{d}(1.0)`): A function to create a causal diamond boundary for the manifold.
- `make_box::Function` (default: `d -> BoxBoundary{d}((([-0.49 for i in 1:d]...,), ([0.49 for i in 1:d]...,))))`): A function to create a box boundary for the manifold.

# Returns
- `data::Dict`: A dictionary containing the generated data with the following keys:
  - `"idx"`: Indices of the data points.
  - `"n"`: Sizes of the sprinklings.
  - `"dimension"`: Dimensions of the manifold.
  - `"manifold"`: Names of the manifolds.
  - `"coords"`: Coordinates of the sprinklings.
  - `"future_relations"`: Future relations of the causet.
  - `"past_relations"`: Past relations of the causet.
  - `"link_matrix"`: Link matrices of the causet.
  - `"relation_count"`: Counts of relations in the causet.
  - `"chains_3"`: Counts of 3-element chains in the causet.
  - `"chains_4"`: Counts of 4-element chains in the causet.
  - `"chains_10"`: Counts of 10-element chains in the causet.
  - `"cardinality_abundances"`: Cardinality abundances of the causet.
  - `"relation_dimension"`: Estimated relation dimensions of the causet.
  - `"chain_dimension_3"`: Estimated chain dimensions for 3-element chains.
  - `"chain_dimension_4"`: Estimated chain dimensions for 4-element chains.

# Notes
- The function uses multithreading (`Threads.@threads`) to parallelize the generation of data points.
- The results from all threads are concatenated into the final `data` dictionary.

"""
function generate_data_for_manifold(
        ; dimension = 2,
        seed = 329478,
        num_datapoints = 1500,
        choose_num_events = d -> Distributions.Uniform(0.7 * 10^(d + 1), 1.1 * 10^(d + 1)),
        make_diamond = d -> CS.CausalDiamondBoundary{d}(1.0),
        make_box = d -> CS.BoxBoundary{d}((
            ([-0.49 for i in 1:d]...,), ([0.49 for i in 1:d]...,)))
)
    field_types = Dict(
        "idx" => Int64,
        "n" => Float32,
        "nmax" => Float32,
        "dimension" => Float32,
        "manifold" => String,
        "coords" => Vector{Vector{Float32}},
        "future_relations" => Vector{Vector{Int64}},
        "past_relations" => Vector{Vector{Int64}},
        "link_matrix" => SparseArrays.SparseMatrixCSC{Float32, Int32},
        "relation_count" => Float32,
        "chains_3" => Float32,
        "chains_4" => Float32,
        "chains_10" => Float32,
        "cardinality_abundances" => Vector{Float32},
        "relation_dimension" => Float32,
        "chain_dimension_3" => Float32,
        "chain_dimension_4" => Float32
    )

    thread_data = Dict(k => [begin
                                 x = T[]
                                 sizehint!(
                                     x, Int(ceil(num_datapoints / Threads.nthreads())))
                                 x
                             end
                             for _ in 1:Threads.nthreads()] for (k, T) in field_types)

    # build helper stuff
    thread_rngs = [Random.MersenneTwister(seed + 2*i) for i in 1:Threads.nthreads()]

    nmax = Int(ceil(maximum(choose_num_events(dimension))))

    # Use Threads.@threads to parallelize the loop, put everything into 
    # arrays indexed with threadid 
    Threads.@threads for p in 1:num_datapoints
        tid = Threads.threadid()

        n = Int(ceil(rand(thread_rngs[tid], choose_num_events(dimension))))

        manifoldname = valid_manifolds[rand(thread_rngs[tid], 1:length(valid_manifolds))]

        boundary = (manifoldname == "torus") ? make_box(dimension) : make_diamond(dimension)

        manifold = get_manifolds_of_dim(dimension)[manifoldname]

        sprinkling = CS.generate_sprinkling(
            manifold, boundary, n; rng = thread_rngs[tid])

        c = CS.BitArrayCauset(manifold, sprinkling)

        @inbounds push!(thread_data["nmax"][Threads.threadid()], nmax)

        @inbounds push!(thread_data["idx"][Threads.threadid()], p)

        @inbounds push!(thread_data["n"][Threads.threadid()], n)

        @inbounds push!(thread_data["dimension"][Threads.threadid()], dimension)

        @inbounds push!(thread_data["manifold"][Threads.threadid()], manifoldname)

        @inbounds push!(
            thread_data["coords"][Threads.threadid()], map(x -> [x...], sprinkling))

        @inbounds push!(thread_data["past_relations"][Threads.threadid()],
            convert.(Vector{Int64}, c.past_relations))

        @inbounds push!(thread_data["future_relations"][Threads.threadid()],
            convert.(Vector{Int64}, c.future_relations))

        @inbounds push!(thread_data["link_matrix"][Threads.threadid()], make_link_matrix(c))

        # TODO: add full matrix here 

        # TODO: add neighborhood structure here per node 
        @inbounds push!(
            thread_data["relation_count"][Threads.threadid()], CS.count_relations(c))

        @inbounds push!(thread_data["chains_3"][Threads.threadid()], CS.count_chains(c, 3))

        @inbounds push!(thread_data["chains_4"][Threads.threadid()], CS.count_chains(c, 4))

        @inbounds push!(
            thread_data["chains_10"][Threads.threadid()], CS.count_chains(c, 10))

        cp = CS.cardinality_abundances(c)
        if isnothing(cp)
            cp = Float32(0)
        end

        @inbounds push!(thread_data["cardinality_abundances"][Threads.threadid()],
            convert.(eltype(field_types["cardinality_abundances"]), cp))

        @inbounds push!(thread_data["relation_dimension"][Threads.threadid()],
            CS.estimate_relation_dimension(c))

        @inbounds push!(thread_data["chain_dimension_3"][Threads.threadid()],
            CS.estimate_chain_dimension(c, 3))

        @inbounds push!(thread_data["chain_dimension_4"][Threads.threadid()],
            CS.estimate_chain_dimension(c, 4))
    end

    # Concatenate the results from all threads into the main data dictionary
    d = Dict(Symbol(k) => vcat(thread_data[k]...) for (k, v) in field_types)

    return d
end
