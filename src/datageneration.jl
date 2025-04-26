module DataGeneration
using CausalSets
using SparseArrays
using Distributions
using Random

export generateDataForManifold

function get_manifolds_of_dim(d::Int64)
    Dict(
        "minkowski" => MinkowskiManifold{d}(),
        "hypercylinder" => HypercylinderManifold{d}(1.0),
        "deSitter" => DeSitterManifold{d}(1.0),
        "antiDeSitter" => AntiDeSitterManifold{d}(1.0),
        "torus" => TorusManifold{d}(1.0)
    )
end

"""
    makeLinkMatrix(cset::AbstractCauset) -> SparseMatrixCSC{Float32}

Generates a sparse link matrix for the given causal set (`cset`). The link matrix
is a square matrix where each entry `(i, j)` is `1` if there is a causal link
from element `i` to element `j` in the causal set, and `0` otherwise.

# Arguments
- `cset::AbstractCauset`: The causal set for which the link matrix is to be generated.
It must have the properties `atom_count` (number of elements in the set) and
a function `is_link(cset, i, j)` that determines if there is a causal link
between elements `i` and `j`.

# Returns
- A sparse matrix (`SparseMatrixCSC{Float32}`) representing the link structure
of the causal set.

# Notes
- The function uses a nested loop to iterate over all pairs of elements in the
causal set, so its complexity is quadratic in the number of elements.
- The sparse matrix representation is used to save memory, as most entries
are expected to be zero in typical causal sets.
"""
function makeLinkMatrix(cset::AbstractCauset)
    link_matrix = spzeros(Float32, cset.atom_count, cset.atom_count)
    for i in 1:(cset.atom_count)
        for j in 1:(cset.atom_count)
            if is_link(cset, i, j)
                link_matrix[i, j] = 1
            end
        end
    end
    return link_matrix
end

"""
    makeCardinalityMatrix(cset::AbstractCauset) -> SparseMatrixCSC{Float32, Int}

Constructs a sparse matrix representing the cardinality relationships between 
atoms in the given causal set (`cset`). The matrix is of type 
`SparseMatrixCSC{Float32, Int}` and has dimensions equal to the number of atoms 
in the causal set.

# Arguments
- `cset::AbstractCauset`: The causal set for which the cardinality matrix is to 
  be generated. It must have an `atom_count` property and support the 
  `cardinality_of` function.

# Returns
- A sparse matrix of type `SparseMatrixCSC{Float32, Int}` where each entry 
  `(i, j)` contains the cardinality value between atom `i` and atom `j` in the 
  causal set. If no cardinality value exists for a pair `(i, j)`, the entry 
  remains zero.

# Notes
- The function uses `spzeros` to initialize the sparse matrix.
- The `cardinality_of` function is expected to return `nothing` if no 
  cardinality value exists for a given pair `(i, j)`.
"""
function makeCardinalityMatrix(cset::AbstractCauset)::SparseMatrixCSC{Float32, Int}
    if cset.atom_count == 0
        throw(ArgumentError("The causal set must not be empty."))
    end

    cardinality_matrix = spzeros(Float32, cset.atom_count, cset.atom_count)

    for i in 1:(cset.atom_count)
        for j in 1:(cset.atom_count)
            ca = cardinality_of(cset, i, j)
            if isnothing(ca) == false
                cardinality_matrix[i, j] = ca
            end
        end
    end
    return cardinality_matrix
end

"""
    makeBdMatrix(cset, ds::Array{Int64}, maxCardinality::Int64=10) -> Array{Float32, 2}

Generates a matrix of size `(maxCardinality, ds[end])` filled with coefficients computed using the `bd_coef` function.

# Arguments
- `ds::Array{Int64}`: An array of integers representing the dimensions or parameters for which the coefficients are computed.
- `maxCardinality::Int64`: The maximum cardinality (number of rows in the matrix). Defaults to `10`.

# Returns
- A 2D array of type `Float32` where each element at position `(c, d)` is the coefficient computed by `bd_coef(c, d, CausalSets.Discrete())`. If the coefficient is `0`, the corresponding matrix element remains `0`.

# Notes
- The function iterates over all combinations of `c` (from `1` to `maxCardinality`) and `d` (elements of `ds`).
- The `bd_coef` function is expected to return a coefficient for the given `c` and `d`. If the coefficient is `0`, the matrix element is not updated.
- The `CausalSets.Discrete()` object is passed to `bd_coef` as a parameter, which may influence the computation of the coefficients.

"""
function makeBdMatrix(ds::Array{Int64}, maxCardinality::Int64 = 10)
    if length(ds) == 0
        throw(ArgumentError("The dimensions must not be empty."))
    end

    if maxCardinality <= 0
        throw(ArgumentError("maxCardinality must be a positive integer."))
    end

    mat = spzeros(Float32, maxCardinality, length(ds))
    for c in 1:maxCardinality
        for d in 1:length(ds)
            bd = bd_coef(c, ds[d], CausalSets.Discrete())
            if bd != 0
                mat[c, d] = bd
            end
        end
    end
    return mat
end

"""
    generateDataForManifold(; dimension=2, manifoldname="minkowski", seed=329478, 
                            num_datapoints=1500, equal_size=false, 
                            size_distr=d -> Uniform(0.7 * 10^(d + 1), 1.1 * 10^(d + 1)), 
                            make_diamond=d -> CausalDiamondBoundary{d}(1.0), 
                            make_box=d -> BoxBoundary{d}((([-0.49 for i in 1:d]...,), ([0.49 for i in 1:d]...,))))

Generates a cset and a variet of data for a given manifold and dimension.

# Keyword Arguments
- `dimension::Int` (default: `2`): The dimension of the manifold.
- `manifoldname::String` (default: `"minkowski"`): The name of the manifold to generate data for.
- `seed::Int` (default: `329478`): The random seed for reproducibility.
- `num_datapoints::Int` (default: `1500`): The number of data points to generate.
- `equal_size::Bool` (default: `false`): If `true`, all sprinklings will have the same size; otherwise, sizes are sampled from `size_distr`.
- `size_distr::Function` (default: `d -> Uniform(0.7 * 10^(d + 1), 1.1 * 10^(d + 1))`): A function that defines the size distribution of the sprinklings.
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
  - `"linkMatrix"`: Link matrices of the causet.
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
function generateDataForManifold(
        ; dimension = 2,
        manifoldname = "minkowski",
        seed = 329478,
        num_datapoints = 1500,
        equal_size = false,
        size_distr = d -> Uniform(0.7 * 10^(d + 1), 1.1 * 10^(d + 1)),
        make_diamond = d->CausalDiamondBoundary{d}(1.0),
        make_box = d->BoxBoundary{d}((
            ([-0.49 for i in 1:d]...,), ([0.49 for i in 1:d]...,)))
)
    data = Dict(
        "idx" => Int64[],
        "n" => Float32[],
        "dimension" => Float32[],
        "manifold" => String[],
        "coords" => Vector{Vector{Float32}}[],
        "future_relations" => Vector{Vector{Int8}}[],
        "past_relations" => Vector{Vector{Int8}}[],
        "linkMatrix" => SparseMatrixCSC{Float32, Int32}[],
        "relation_count" => Float32[],
        "chains_3" => Float32[],
        "chains_4" => Float32[],
        "chains_10" => [],
        "cardinality_abundances" => Vector{Float32}[],
        "relation_dimension" => Float32[],
        "chain_dimension_3" => Float32[],
        "chain_dimension_4" => Float32[]
    )

    # make nested vectors for thread safe writing which later will be concatenated
    idxs = [Int64[] for i in 1:Threads.nthreads()]
    ns = [Float32[] for i in 1:Threads.nthreads()]
    dimensions = [Float32[] for i in 1:Threads.nthreads()]
    manifolds = [String[] for i in 1:Threads.nthreads()]
    coordss = [Vector{Vector{Float32}}[] for i in 1:Threads.nthreads()]
    past_relations = [Vector{Vector{Int8}}[] for i in 1:Threads.nthreads()]
    future_relations = [Vector{Vector{Int8}}[] for i in 1:Threads.nthreads()]
    linkMatrixs = [SparseMatrixCSC{Float32, Int32}[] for i in 1:Threads.nthreads()]
    relation_counts = [Float32[] for i in 1:Threads.nthreads()]
    chains_3s = [Float32[] for i in 1:Threads.nthreads()]
    chains_4s = [Float32[] for i in 1:Threads.nthreads()]
    chains_10s = [Float32[] for i in 1:Threads.nthreads()]
    cardinality_abundancess = [Vector{Float32}[] for i in 1:Threads.nthreads()]
    relation_dimensions = [Float32[] for i in 1:Threads.nthreads()]
    chain_dimension_3s = [Float32[] for i in 1:Threads.nthreads()]
    chain_dimension_4s = [Float32[] for i in 1:Threads.nthreads()]

    # build helper stuff 
    manifold = get_manifolds_of_dim(dimension)[manifoldname]
    boundary = manifoldname == "torus" ? make_box(dimension) : make_diamond(dimension)
    rng = Random.MersenneTwister(seed)
    distr = size_distr(dimension)
    idx = 1

    # Use Threads.@threads to parallelize the loop, put everything into 
    # arrays indexed with threadid 
    Threads.@threads for p in 1:num_datapoints
        n = 10^(dimension + 1)
        if equal_size == false
            n = Int64(floor(rand(rng, distr)))
        end

        sprinkling = generate_sprinkling(
            manifold, boundary, Int(n), rng = rng)

        c = BitArrayCauset(manifold, sprinkling)

        push!(idxs[Threads.threadid()], idx)

        push!(ns[Threads.threadid()], n)

        push!(dimensions[Threads.threadid()], dimension)

        push!(manifolds[Threads.threadid()], manifoldname)

        push!(coordss[Threads.threadid()], map(x -> [x...], sprinkling))

        push!(past_relations[Threads.threadid()], convert.(Vector{Int8}, c.past_relations))

        push!(future_relations[Threads.threadid()],
            convert.(Vector{Int8}, c.future_relations))

        push!(linkMatrixs[Threads.threadid()], makeLinkMatrix(c))

        push!(relation_counts[Threads.threadid()], count_relations(c))

        push!(chains_3s[Threads.threadid()], count_chains(c, 3))

        push!(chains_4s[Threads.threadid()], count_chains(c, 4))

        push!(chains_10s[Threads.threadid()], count_chains(c, 10))

        push!(cardinality_abundancess[Threads.threadid()],
            convert.(Float32, cardinality_abundances(c)))

        push!(relation_dimensions[Threads.threadid()], estimate_relation_dimension(c))

        push!(chain_dimension_3s[Threads.threadid()], estimate_chain_dimension(c, 3))

        push!(chain_dimension_4s[Threads.threadid()], estimate_chain_dimension(c, 4))

        idx += 1
    end

    # Concatenate the results from all threads into the main data dictionary
    data["idx"] = vcat(idxs...)
    data["n"] = vcat(ns...)
    data["dimension"] = vcat(dimensions...)
    data["manifold"] = vcat(manifolds...)
    data["coords"] = vcat(coordss...)
    data["past_relations"] = vcat(past_relations...)
    data["future_relations"] = vcat(future_relations...)
    data["linkMatrix"] = vcat(linkMatrixs...)
    data["relation_count"] = vcat(relation_counts...)
    data["chains_3"] = vcat(chains_3s...)
    data["chains_4"] = vcat(chains_4s...)
    data["chains_10"] = vcat(chains_10s...)
    data["cardinality_abundances"] = vcat(cardinality_abundancess...)
    data["relation_dimension"] = vcat(relation_dimensions...)
    data["chain_dimension_3"] = vcat(chain_dimension_3s...)
    data["chain_dimension_4"] = vcat(chain_dimension_4s...)

    return data
end

end
