
"""
    make_cset(manifold, boundary, n, d, rng, type) -> (cset, coordinates)

Creates a causet and its coordinate representation from a manifold and boundary.

# Arguments
- `manifold::CausalSets.AbstractManifold`: The spacetime manifold
- `boundary::CausalSets.AbstractBoundary`: The boundary type  
- `n::Int64`: Number of points in the causet
- `d::Int`: Dimension of the spacetime
- `rng::Random.AbstractRNG`: Random number generator
- `type::Type{T}`: Numeric type for coordinates

# Returns
- `Tuple`: (causet, coordinates) where coordinates is a matrix of point positions

# Notes
Special handling for PseudoManifold which generates random causets,
while other manifolds use CausalSets sprinkling generation.
"""
function make_cset(
        manifold::CausalSets.AbstractManifold, boundary::CausalSets.AbstractBoundary, n::Int64, d::Int,
        rng::Random.AbstractRNG, type::Type{T}) where {T <: Number}
    if manifold isa PseudoManifold
        return CausalSets.sample_random_causet(CausalSets.BitArrayCauset, n, 300, rng),
        stack(make_pseudosprinkling(n, d, -0.49, 0.49, type; rng = rng), dims = 1)
    else
        sprinkling = CausalSets.generate_sprinkling(manifold, boundary, n; rng = rng)
        cset = CausalSets.BitArrayCauset(manifold, sprinkling)
        return cset, stack(collect.(sprinkling), dims = 1)
    end
end

"""
    make_link_matrix(cset::AbstractCauset) -> SparseMatrixCSC{Float32}

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
function make_link_matrix(cset::CausalSets.AbstractCauset)
    link_matrix = SparseArrays.spzeros(
        Float32, CausalSets.atom_count, CausalSets.atom_count)
    for i in 1:(CausalSets.atom_count)
        for j in 1:(CausalSets.atom_count)
            if CausalSets.is_link(cset, i, j)
                @inbounds link_matrix[i, j] = 1
            end
        end
    end
    return link_matrix
end

"""
    calculate_angles(sprinkling, node_idx, neighbors, num_nodes, type) -> SparseMatrix

Calculates angles between vectors from a central node to its neighbors in a sprinkling.

# Arguments
- `sprinkling::AbstractMatrix`: Matrix where each row represents a point's coordinates
- `node_idx::Int`: Index of the central node
- `neighbors::AbstractVector`: Vector of neighbor node indices
- `num_nodes::Int`: Total number of nodes
- `type::Type{T}`: Numeric type for the angle values

# Returns
- `SparseMatrix{T}`: Sparse matrix of angles between neighbor pairs, where entry (i,j) 
  contains the angle between vectors from the central node to neighbors i and j

# Notes
- Returns angles in radians using acos function
- Uses dot product and vector norms to calculate angles
- Diagonal entries (same neighbor) are zero
- Empty neighbor list returns zero matrix
"""
function calculate_angles(
        sprinkling::AbstractMatrix, node_idx::Int, neighbors::AbstractVector,
        num_nodes::Int, type::Type{T}; multithreading::Bool = false) where {T <: Number}
    # what about the metric?
    angles = SparseArrays.spzeros(type, num_nodes, num_nodes)
    if isempty(neighbors)
        return angles
    end
    # TODO: dispatch on multithreading variable
    for (i, neighbor_i) in enumerate(neighbors), (j, neighbor_j) in enumerate(neighbors)
        if neighbor_i != neighbor_j
            v_i = sprinkling[neighbor_i, :] - sprinkling[node_idx, :]
            v_j = sprinkling[neighbor_j, :] - sprinkling[node_idx, :]
            angles[i,
            j] = acos(clamp(
                LinearAlgebra.dot(
                    v_i / LinearAlgebra.norm(v_i), v_j / LinearAlgebra.norm(v_j)),
                -1.0,
                1.0))
        end
    end
    return angles
end

"""
    calculate_distances(sprinkling, node_idx, neighbors, num_nodes, type) -> SparseVector

Calculates Euclidean distances from a central node to its neighbors in a sprinkling.

# Arguments
- `sprinkling::AbstractMatrix`: Matrix where each row represents a point's coordinates
- `node_idx::Int`: Index of the central node
- `neighbors::AbstractVector`: Vector of neighbor node indices  
- `num_nodes::Int`: Total number of nodes
- `type::Type{T}`: Numeric type for the distance values

# Returns
- `SparseVector{T}`: Sparse vector where entry i contains the Euclidean distance 
  from the central node to neighbor i

# Notes
- Uses Euclidean norm to calculate distances
- Distance to self (same node) is zero
- Empty neighbor list returns zero vector
- Distances are always non-negative
"""
function calculate_distances(
        sprinkling::AbstractMatrix, node_idx::Int, neighbors::AbstractVector,
        num_nodes::Int, type::Type{T}; multithreading::Bool = false) where {T <: Number}
    distances = SparseArrays.spzeros(type, num_nodes)
    # TODO: what about the metric?
    if isempty(neighbors)
        return distances
    end

    # TODO: check if multithreading is needed here
    for (i, neighbor_i) in enumerate(neighbors)
        if neighbor_i != node_idx
            distances[i] = LinearAlgebra.norm(sprinkling[neighbor_i, :] -
                                              sprinkling[node_idx, :])
        end
    end
    return distances
end

"""
    make_cardinality_matrix(cset::AbstractCauset) -> SparseMatrixCSC{Float32, Int}

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
function make_cardinality_matrix(cset::CausalSets.AbstractCauset;
        multithreading::Bool = false)::SparseArrays.SparseMatrixCSC{
        Float32, Int}
    if CausalSets.atom_count == 0
        throw(ArgumentError("The causal set must not be empty."))
    end

    cardinality_matrix = SparseArrays.spzeros(Float32, cset.atom_count, cset.atom_count)
    # TODO: dispatch on multithreading variable
    for i in 1:(cset.atom_count)
        for j in 1:(cset.atom_count)
            ca = CausalSets.cardinality_of(cset, i, j)
            if isnothing(ca) == false
                @inbounds cardinality_matrix[i, j] = ca
            end
        end
    end
    return cardinality_matrix
end

"""
    make_Bd_matrix(cset, ds::Array{Int64}, maxCardinality::Int64=10) -> Array{Float32, 2}

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
# TODO: check again if this is correct, lookup in paper!
function make_Bd_matrix(ds::Array{Int64}, maxCardinality::Int64, type::Type{T};
        multithreading::Bool = false) where {T <: Real}
    if length(ds) == 0
        throw(ArgumentError("The dimensions must not be empty."))
    end

    if maxCardinality <= 0
        throw(ArgumentError("maxCardinality must be a positive integer."))
    end

    mat = SparseArrays.spzeros(T, maxCardinality, length(ds))

    for c in 1:maxCardinality
        for d in 1:length(ds)
            bd = CSet.bd_coef(c, ds[d], CSet.Discrete()) #does this work?
            if bd != 0
                @inbounds mat[c, d] = bd
            end
        end
    end

    return mat
end

"""
    make_adj(c::CausalSets..AbstractCauset, type::Type{T}) -> SparseMatrixCSC{T}

Creates an adjacency matrix from a causet's future relations.

# Arguments
- `c::CausalSets..AbstractCauset`: The causet object containing future relations
- `type::Type{T}`: Numeric type for the adjacency matrix entries

# Returns
- `SparseMatrixCSC{T}`: Sparse adjacency matrix representing the causal structure

# Notes
Converts the causet's future_relations to a sparse matrix format by 
horizontally concatenating, transposing, and converting to the specified type.
"""
function make_adj(c::CausalSets .. AbstractCauset, type::Type{T}) where {T <: Number}
    c.future_relations |>
    x -> hcat(x...) |>
         transpose |>
         SparseArrays.SparseMatrixCSC{type}
end

"""
    maxpathlen(adj_matrix, topo_order::Vector{Int}, source::Int) -> Int32

Calculates the maximum path length from a source node in a directed acyclic graph.

# Arguments
- `adj_matrix`: Adjacency matrix representing the directed graph
- `topo_order::Vector{Int}`: Topologically sorted order of vertices
- `source::Int`: Source vertex index to calculate distances from

# Returns
- `Int32`: Maximum finite distance from the source node, or 0 if no paths exist

# Notes
Uses dynamic programming with topological ordering for efficient longest path computation.
Processes vertices in topological order to ensure optimal substructure property.
Returns 0 if no finite distances exist from the source.
"""
function max_pathlen(adj_matrix, topo_order::Vector{Int}, source::Int)
    n = size(adj_matrix, 1)

    # Dynamic programming for longest paths
    dist = fill(-Inf, n)
    dist[source] = 0

    # Process vertices in topological order
    @inbounds for u in topo_order
        if @inbounds dist[u] != -Inf
            @inbounds for v in SparseArrays.findnz(adj_matrix[u, :])[1]
                @inbounds dist[v] = max(dist[v], dist[u] + 1)
            end
        end
    end

    # Return max finite distance
    @inbounds finite_dists = filter(d -> d != -Inf, dist)
    return @inbounds isempty(finite_dists) ? 0 : Int 32(maximum(finite_dists))
end

"""
    make_data(transform::Function, prepare_output::Function, write_data::Function, config::Dict{String, Any})

Create data using the transform function and write it to an HDF5 file. The HDF5 file is prepared using the `prepare_output` function, and the data is written using the `write_data` function.
In order to work, transform must return a `Dict{String, Any}` containing the name and individual data elements. Only primitive types and arrays thereof are supported as values.
The write_data function must accept the HDF5 file and the data dictionary to be written as arguments.

# Arguments:
- `transform`: Function to generate a single datapoint. It should accept a configuration dictionary and a random number generator, and return a `Dict{String, Any}` with the data.
- `prepare_output`: Function to prepare the HDF5 file for writing data. It should accept the HDF5 file and configuration dictionary as arguments.
- `write_data`: Function to write the generated data to the HDF5 file. It should accept the HDF5 file and a the data dictionary as arguments.
- `config`: Configuration dictionary containing various settings for data generation. The settings contained are not completely specified a priori and can be specific to the passed-in functions. This dictionary will be augmented with information about the current commit hash and branch name of the QuantumGrav package, and written to a YAML file in the output directory. It is expected to contain the number of datapoints as a node `num_datapoints`, the output directory as a node `output`, the file mode as a node `file_mode`, and the number of threads to use as a node `num_threads`. The seed for the random number generator is expected to be passed in as a node `seed`. If the data generation should be chunked, the number of chunks can be specified with the node `chunks`.
"""
function make_data(transform::Function, prepare_output::Function,
        write_data::Function, config::Dict{String, Any})::Nothing

    # check return type 
    if Dict in Base.return_types(transform, (Dict{String, Any}, Random.MersenneTwister)) ==
       false
        throw(ArgumentError("The transform function must return a Dict{String, Any} containing the name and individual data element. Only primitive types and arrays thereof are supported as values."))
    end

    file = HDF5.HDF5File(
        joinpath(abspath(expanduser(config["output"])), "data.h5"), config["file_mode"])

    # get the source code of the transform function and write them to the data folder 
    for func in [transform, prepare_output, write_data]
        funcdata = first(methods(func))
        filepath = String(funcdata.file)

        if isfile(abspath(expanduser(filepath)))
            cp(filepath,
                joinpath(abspath(expanduser(config["output"])), filepath))
        end
    end

    # get the current commit hash of the QuantumGrav package and put it into the config
    config["commit_hash"] = run(`git rev-parse HEAD`) |> strip

    config["branch_name"] = run(`git rev-parse --abbrev-ref HEAD`) |> strip

    # write out config to the specified output and add it to the hdf5 file
    HDF5.write(
        file, "config", config)

    YAML.write(joinpath(abspath(expanduser(config["output"])), "config.yaml"), config)

    # prepare the file 
    prepare_output(file, config)

    num_datapoints = config["num_datapoints"]

    # check if the data generation should be chunked
    if "chunks" in config
        num_chunks = config["chunks"]
        num_datapoints = div(num_datapoints, num_chunks)
    else
        num_chunks = 1
    end

    # create data either single-threaded or multi-threaded 
    if Threads.nthreads != config["num_threads"]
        throw(ArgumentError("Number of available threads does not match the configuration."))
    end

    # Multithreading enabled, use Threads.@threads for parallel data generation
    rngs = [Random.MersenneTwister(config["seed"] + i) for i in 1:Threads.nthreads()]

    for _ in 1:num_chunks
        data = [[] for _ in 1:Threads.nthreads()]

        Threads.@threads for _ in 1:num_datapoints
            t = Threads.threadid()
            rng = rngs[t]
            data_point = transform(config, rng)
            # Store or process the generated data point as needed
            push!(data[t], data_point)
        end

        data = reduce(vcat, data)  # Combine results from all threads

        write_data(file, data)
    end

    HDF5.close(file)
end

"""
    make_data(transform::Function, prepare_output::Function, write_data::Function, configpath::String)
Creates data using the specified transform function, prepares the output using the prepare_output function, and writes the data using the write_data function. The configuration is read from a YAML file at the specified path.

# Arguments:
- `transform`: Function to generate a single datapoint. It should accept a configuration dictionary and
- `prepare_output`: Function to prepare the HDF5 file for writing data. It should accept the HDF5 file and configuration dictionary as arguments.
- `write_data`: Function to write the generated data to the HDF5 file. It should accept the HDF5 file and a the data dictionary as arguments.
- `configpath`: String path to the YAML configuration file containing various settings for data generation. The settings contained are not completely specified a priori and can be specific to the passed-in functions. This dictionary will be augmented with information about the current commit hash and branch name of the QuantumGrav package, and written to a YAML file in the output directory. It is expected to contain the number of datapoints as a node `num_datapoints`, the output directory as a node `output`, the file mode as a node `file_mode`, and the number of threads to use as a node `num_threads`. The seed for the random number generator is expected to be passed in as a node `seed`. If the data generation should be chunked, the number of chunks can be specified with the node `chunks`.
"""
function make_data(
        transform::Function,
        prepare_output::Function,
        write_data::Function,
        configpath::String
)
    config = YAML.read(abspath(expanduser(configpath)))

    return make_data(
        transform,
        prepare_output,
        write_data,
        config
    )
end
