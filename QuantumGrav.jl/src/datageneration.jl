
"""
    make_cset(manifold, boundary, n, d, rng, type) -> (cset, coordinates)

Creates a causet and its coordinate representation from a manifold and boundary.

# Arguments
- `manifold::String`: The spacetime manifold
- `boundary::String`: The boundary type  
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
    manifold::String,
    boundary::String,
    n::Int64,
    d::Int,
    rng::Random.AbstractRNG;
    type::Type{T} = Float32,
) where {T<:Number}
    manifold = make_manifold(manifold, d)
    boundary = make_boundary(boundary, d)

    if manifold isa PseudoManifold
        return CausalSets.sample_random_causet(CausalSets.BitArrayCauset, n, 300, rng),
        stack(make_pseudosprinkling(n, d, -0.49, 0.49, type; rng = rng), dims = 1)
    else
        sprinkling = CausalSets.generate_sprinkling(manifold, boundary, n; rng = rng)
        cset = CausalSets.BitArrayCauset(manifold, sprinkling)
        return cset, type.(stack(collect.(sprinkling), dims = 1))
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
function make_link_matrix(
    cset::CausalSets.AbstractCauset;
    type::Type{T} = Float32,
) where {T<:Number}
    link_matrix = SparseArrays.spzeros(T, cset.atom_count, cset.atom_count)
    for i = 1:(cset.atom_count)
        for j = 1:(cset.atom_count)
            if CausalSets.is_link(cset, i, j)
                link_matrix[i, j] = 1
            end
        end
    end
    return link_matrix
end

"""
    calculate_angles(sprinkling, node_idx, neighbors, num_nodes, type) -> SparseMatrix

Calculates angles between vectors from a central node given by `node_idx` to its neighbors in a sprinkling. This uses a euclidean metric to compute angles between pairs of neighbors relative to the node given by `node_idx`. It thus assumes that the manifold is represented as a `point cloud` in a euclidean space. 

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
    sprinkling::AbstractMatrix,
    node_idx::Int,
    neighbors::AbstractVector;
    type::Type{T} = Float32,
    multithreading::Bool = false,
) where {T<:Number}

    function compute_angle(sprinkling::AbstractMatrix, node_idx::Int, i::Int, j::Int)
        v_i = sprinkling[i, :] - sprinkling[node_idx, :]
        v_j = sprinkling[j, :] - sprinkling[node_idx, :]

        angle = acos(
            clamp(
                LinearAlgebra.dot(
                    v_i / LinearAlgebra.norm(v_i),
                    v_j / LinearAlgebra.norm(v_j),
                ),
                -1.0,
                1.0,
            ),
        )

        return angle
    end

    if multithreading
        Is = [Vector{Int}() for _ = 1:Threads.nthreads()]
        Js = [Vector{Int}() for _ = 1:Threads.nthreads()]
        angles = [Vector{type}() for _ = 1:Threads.nthreads()]
        sizehint!.(Is, length(neighbors))
        sizehint!.(Js, length(neighbors))
        sizehint!.(angles, length(neighbors))

        Threads.@threads for ((i, neighbor_i), (j, neighbor_j)) in collect(
            Iterators.product(enumerate(neighbors), enumerate(neighbors)),
        )
            if neighbor_i != neighbor_j
                angle = compute_angle(sprinkling, node_idx, i, j)
                push!(Is[Threads.threadid()], i)
                push!(Js[Threads.threadid()], j)
                push!(angles[Threads.threadid()], type(angle))

            end
        end

        Is = vcat(Is...)
        Js = vcat(Js...)
        angles = vcat(angles...)
    else
        Is = Vector{Int}()
        Js = Vector{Int}()
        angles = Vector{type}()
        sizehint!(Is, length(neighbors))
        sizehint!(Js, length(neighbors))
        sizehint!(angles, length(neighbors))

        for ((i, neighbor_i), (j, neighbor_j)) in
            Iterators.product(enumerate(neighbors), enumerate(neighbors))
            if neighbor_i != neighbor_j
                angle = compute_angle(sprinkling, node_idx, i, j)
                Is = push!(Is, i)
                Js = push!(Js, j)
                angles = push!(angles, type(angle))
            end
        end
    end
    return SparseArrays.sparse(Is, Js, angles, length(neighbors), length(neighbors), type)
end

"""
    calculate_distances(sprinkling, node_idx, neighbors, num_nodes, type) -> SparseVector

Calculates Euclidean distances from a central node to its neighbors in a sprinkling. Since this uses a euclidean metric, it assumes that the manifold is represented as a `point cloud` in a euclidean space.

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
    sprinkling::AbstractMatrix,
    node_idx::Int,
    neighbors::AbstractVector,
    num_nodes::Int;
    type::Type{T} = Float32,
    multithreading::Bool = false,
) where {T<:Number}

    if isempty(neighbors)
        return SparseArrays.spzeros(type, num_nodes)
    end

    if multithreading

        Is = [Vector{Int}() for _ = 1:Threads.nthreads()]
        distances = [Vector{type}() for _ = 1:Threads.nthreads()]

        sizehint!.(Is, num_nodes)
        sizehint!.(distances, num_nodes)

        Threads.@threads for (i, _) in collect(enumerate(neighbors))
            if i != node_idx
                push!(Is[Threads.threadid()], i)
                push!(
                    distances[Threads.threadid()],
                    LinearAlgebra.norm(sprinkling[i, :] - sprinkling[node_idx, :]),
                )
            end
        end

        Is = vcat(Is...)
        distances = vcat(distances...)
    else

        Is = Vector{Int}()
        distances = Vector{type}()

        sizehint!(Is, num_nodes)
        sizehint!(distances, num_nodes)

        for (i, _) in enumerate(neighbors)
            if i != node_idx
                push!(Is, i)
                push!(
                    distances,
                    LinearAlgebra.norm(sprinkling[i, :] - sprinkling[node_idx, :]),
                )
            end
        end
    end
    return SparseArrays.sparsevec(Is, type.(distances), num_nodes)
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
function make_cardinality_matrix(
    cset::CausalSets.AbstractCauset;
    type::Type{T} = Float32,
    multithreading::Bool = false,
)::SparseArrays.SparseMatrixCSC{T,Int} where {T<:Number}
    if cset.atom_count == 0
        throw(ArgumentError("The causal set must not be empty."))
    end

    if multithreading

        Is = [Vector{Int}() for _ = 1:Threads.nthreads()]
        Js = [Vector{Int}() for _ = 1:Threads.nthreads()]
        Vs = [Vector{type}() for _ = 1:Threads.nthreads()]

        sizehint!.(Is, cset.atom_count)
        sizehint!.(Js, cset.atom_count)
        sizehint!.(Vs, cset.atom_count)

        Threads.@threads for i in collect(1:cset.atom_count)
            for j = 1:cset.atom_count
                ca = CausalSets.cardinality_of(cset, i, j)
                if isnothing(ca) == false
                    push!(Is[Threads.threadid()], i)
                    push!(Js[Threads.threadid()], j)
                    push!(Vs[Threads.threadid()], type(ca))
                end
            end
        end

        Is = vcat(Is...)
        Js = vcat(Js...)
        Vs = vcat(Vs...)
    else

        Is = Vector{Int}()
        Js = Vector{Int}()
        Vs = Vector{type}()

        sizehint!(Is, cset.atom_count)
        sizehint!(Js, cset.atom_count)
        sizehint!(Vs, cset.atom_count)

        for i = 1:cset.atom_count
            for j = 1:cset.atom_count
                ca = CausalSets.cardinality_of(cset, i, j)
                if isnothing(ca) == false
                    push!(Is, i)
                    push!(Js, j)
                    push!(Vs, type(ca))
                end
            end
        end
    end
    return SparseArrays.sparse(Is, Js, Vs, cset.atom_count, cset.atom_count, type)
end


"""
    make_adj(c::CausalSets.AbstractCauset, type::Type{T}) -> SparseMatrixCSC{T}

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
function make_adj(c::CausalSets.AbstractCauset; type::Type{T} = Float32) where {T<:Number}
    if c.atom_count == 0
        throw(ArgumentError("The causal set must not be empty."))
    end

    c.future_relations |> x -> hcat(x...) |> transpose |> SparseArrays.SparseMatrixCSC{type}
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
    for u in topo_order
        if dist[u] != -Inf
            if adj_matrix isa SparseArrays.AbstractSparseMatrix
                indices = SparseArrays.findnz(adj_matrix[u, :])[1]

            else
                indices = [v for v = 1:n if adj_matrix[u, v] != 0]
            end

            for v in indices
                dist[v] = max(dist[v], dist[u] + 1)
            end
        end
    end

    # Return max finite distance
    finite_dists = filter(d -> d != -Inf, dist)
    return isempty(finite_dists) ? 0 : Int32(maximum(finite_dists))
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
function make_data(
    transform::Function,
    prepare_output::Function,
    write_data::Function;
    config::Dict{String,Any} = Dict{String,Any}(
        "num_datapoints" => 1000,
        "output" => "~/QuantumGrav/data",
        "file_mode" => "w",
        "num_threads" => Threads.nthreads(),
        "seed" => 42,
    ),
)
    for key in ["num_datapoints", "output", "file_mode", "num_threads", "seed"]
        if !haskey(config, key)
            throw(ArgumentError("Configuration must contain the key: $key"))
        end
    end

    # check return type 
    if Dict in
       Base.return_types(transform, (Dict{String,Any}, Random.MersenneTwister)) ==
       false
        throw(
            ArgumentError(
                "The transform function must return a Dict{String, Any} containing the name and individual data element. Only primitive types and arrays thereof are supported as values.",
            ),
        )
    end

    if !isdir(abspath(expanduser(config["output"])))
        mkpath(abspath(expanduser(config["output"])))
    end

    HDF5.h5open(
        joinpath(abspath(expanduser(config["output"])), "data.h5"),
        config["file_mode"],
    ) do file

        # get the source code of the transform function and write them to the data folder 
        for func in [transform, prepare_output, write_data]
            funcdata = first(methods(func))
            filepath = String(funcdata.file)
            if isfile(abspath(expanduser(filepath))) == false
                cp(
                    filepath,
                    joinpath(abspath(expanduser(config["output"])), basename(filepath)),
                )
            end
        end

        # get the current commit hash of the QuantumGrav package and put it into the config
        config["commit_hash"] = read(`git rev-parse HEAD`, String)

        config["branch_name"] = read(`git rev-parse --abbrev-ref HEAD`, String)

        YAML.write_file(
            joinpath(abspath(expanduser(config["output"])), "config.yaml"),
            config,
        )

        # prepare the file 
        prepare_output(file, config)

        num_datapoints = config["num_datapoints"]

        # check if the data generation should be chunked
        if "chunks" in keys(config)
            num_chunks = config["chunks"]
            num_datapoints = div(num_datapoints, num_chunks)
        else
            num_chunks = 1
        end

        # create data either single-threaded or multi-threaded 
        if Threads.nthreads() != config["num_threads"]
            throw(
                ArgumentError(
                    "Number of available threads does not match the configuration.",
                ),
            )
        end

        # Multithreading enabled, use Threads.@threads for parallel data generation
        rngs = [Random.MersenneTwister(config["seed"] + i) for i = 1:Threads.nthreads()]
        for _ = 1:num_chunks
            data = [[] for _ = 1:Threads.nthreads()]

            Threads.@threads for _ = 1:num_datapoints
                t = Threads.threadid()
                rng = rngs[t]
                data_point = transform(config, rng)
                # Store or process the generated data point as needed
                push!(data[t], data_point)
            end

            final_data = Dict{String,Any}()
            for thread_local_data in data
                for datapoint in thread_local_data
                    for (key, value) in datapoint
                        if haskey(final_data, key)
                            final_data[key] =
                                cat(final_data[key], value; dims = ndims(value)+1)
                        else
                            final_data[key] = value
                        end
                    end
                end
            end

            write_data(file, final_data)
        end
    end
end
