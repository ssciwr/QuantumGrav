
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
    link_matrix = SparseArrays.spzeros(type, cset.atom_count, cset.atom_count)
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
    calculate_angles(sprinkling, node_idx, neighbors, type, multithreading) -> SparseMatrix

Calculates angles between vectors from a central node given by `node_idx` to its neighbors in a sprinkling. This uses a euclidean metric to compute angles between pairs of neighbors relative to the node given by `node_idx`. It thus assumes that the manifold is represented as a `point cloud` in a euclidean space. 

# Arguments
- `sprinkling::AbstractMatrix`: Matrix where each row represents a point's coordinates
- `node_idx::Int`: Index of the central node
- `neighbors::AbstractVector`: Vector of neighbor node indices
- `type::Type{T}`: Numeric type for the angle values
- `multithreading::Bool`: Whether to use multithreading for angle calculations
# Returns
- `Vector{T}`: Flat vector of angles between neighbor pairs, where each entry corresponds to the angle between vectors from the central node to a pair of neighbors.

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
)::Vector{T} where {T<:Number}

    # actual angle calculation function for two neighbors i and j
    # in euclidean space with respect to the node given by node_idx
    function inner(sprinkling, i, j, n)
        @inbounds v_i = sprinkling[i, :] - sprinkling[n, :]
        @inbounds v_j = sprinkling[j, :] - sprinkling[n, :]
        angle = type(
            acos(
                clamp(
                    LinearAlgebra.dot(
                        v_i / LinearAlgebra.norm(v_i),
                        v_j / LinearAlgebra.norm(v_j),
                    ),
                    -1.0,
                    1.0,
                ),
            ),
        )
        if isnan(angle) || isinf(angle)
            throw(ArgumentError("Angle is NaN or Inf. This should not happen."))
        end
        return angle
    end

    idxs = findall(x -> x > 0, neighbors)
    n = length(idxs)
    num_pairs = div(n * (n - 1), 2)
    if multithreading
        # to avoid race conditions and locking with multithreading, have a separate target 
        # vector for each thread and later concatenate them. Not needed for the 
        # non-multithreading branch
        angles = [Vector{type}() for _ = 1:Threads.nthreads()]

        # preallocate vectors to avoid data copying due to reallocation
        sizehint!.(angles, length(neighbors))

        Threads.@threads for idx1 = 1:length(idxs)
            for idx2 = (idx1+1):length(idxs)
                i = idxs[idx1]
                j = idxs[idx2]
                if i != j && i != node_idx && j != node_idx
                    angle = inner(sprinkling, i, j, node_idx)
                    push!(angles[Threads.threadid()], angle)
                end
            end
        end
        angles = vcat(angles...)
    else
        angles = Vector{type}(undef, num_pairs)
        k = 1
        for idx1 = 1:length(idxs)
            for idx2 = (idx1+1):length(idxs)
                i = idxs[idx1]
                j = idxs[idx2]
                if i != j && i != node_idx && j != node_idx
                    angle = inner(sprinkling, i, j, node_idx)
                    angles[k] = angle
                    k += 1
                end
            end
        end
    end
    return angles
end

"""
    calculate_distances(sprinkling, node_idx, neighbors; type, multithreading) -> Vector{T}

Calculates Euclidean distances from a central node to its neighbors in a sprinkling. Since this uses a euclidean metric, it assumes that the manifold is represented as a `point cloud` in a euclidean space.

# Arguments
- `sprinkling::AbstractMatrix`: Matrix where each row represents a point's coordinates
- `node_idx::Int`: Index of the central node
- `neighbors::AbstractVector`: Vector of neighbor node indices  
- `type::Type{T}`: Numeric type for the distance values
- `multithreading::Bool`: Whether to use multithreading for distance calculations
# Returns
- `Vector{T}`: Vector where entry i contains the Euclidean distance 
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
    neighbors::AbstractVector;
    type::Type{T} = Float32,
    multithreading::Bool = false,
)::Vector{T} where {T<:Number}
    if isempty(neighbors)
        return T[]
    end

    if multithreading
        # to avoid race conditions and locking with multithreading, have a separate target 
        # vector for each thread and later concatenate them. Not needed for the 
        # non-multithreading branch
        distances = [Vector{type}() for _ = 1:Threads.nthreads()]

        # preallocate vectors to avoid data copying due to reallocation
        sizehint!.(distances, length(neighbors))

        Threads.@threads for (i, _) in collect(enumerate(neighbors))
            if i != node_idx
                push!(
                    distances[Threads.threadid()],
                    LinearAlgebra.norm(sprinkling[i, :] - sprinkling[node_idx, :]),
                )
            end
        end

        distances = vcat(distances...)
    else
        distances = Vector{type}()

        # preallocate vectors to avoid data copying due to reallocation
        sizehint!(distances, length(neighbors))

        for (i, _) in enumerate(neighbors)
            if i != node_idx
                push!(
                    distances,
                    LinearAlgebra.norm(sprinkling[i, :] - sprinkling[node_idx, :]),
                )
            end
        end
    end
    return type.(distances)
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
        # to avoid race conditions and locking with multithreading, have a separate target 
        # vector for each thread and later concatenate them. Not needed for the 
        # non-multithreading branch
        Is = [Vector{Int}() for _ = 1:Threads.nthreads()]
        Js = [Vector{Int}() for _ = 1:Threads.nthreads()]
        Vs = [Vector{type}() for _ = 1:Threads.nthreads()]

        # preallocate vectors to avoid data copying due to reallocation
        sizehint!.(Is, cset.atom_count)
        sizehint!.(Js, cset.atom_count)
        sizehint!.(Vs, cset.atom_count)

        Threads.@threads for i in collect(1:(cset.atom_count))
            for j = 1:(cset.atom_count)
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

        # preallocate vectors to avoid data copying due to reallocation
        sizehint!(Is, cset.atom_count)
        sizehint!(Js, cset.atom_count)
        sizehint!(Vs, cset.atom_count)

        for i = 1:(cset.atom_count)
            for j = 1:(cset.atom_count)
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
function make_adj(
    c::CausalSets.AbstractCauset;
    type::Type{T} = Float32,
)::AbstractMatrix{T} where {T<:Number}
    if c.atom_count == 0
        throw(ArgumentError("The causal set must not be empty."))
    end

    return (x -> SparseArrays.SparseMatrixCSC{type}(transpose(hcat(x...))))(
        c.future_relations,
    )
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


function prepare_dataproduction(config::Dict{String,Any}, funcs_to_copy::Vector{Any})
    # consistency checks
    for key in ["num_datapoints", "output", "seed", "output_format"]
        if !haskey(config, key)
            throw(ArgumentError("Configuration must contain the key: $key"))
        end
    end


    # make directory to put data into
    if !isdir(abspath(expanduser(config["output"])))
        mkpath(abspath(expanduser(config["output"])))
    end

    # get the source code of the prepare/write functions and write them to the data folder 
    # to document how the data has been created
    for func_to_copy in funcs_to_copy
        funcdata = first(methods(func_to_copy)) # this assumes that all overloads of the passed functions are part of the same file
        filepath = String(funcdata.file)
        targetpath =
            joinpath(abspath(expanduser(config["output"])), splitext(basename(filepath))[1])
        if isfile(targetpath) == false
            cp(filepath, targetpath)
        end
    end

    datetime = Dates.now()
    datetime = Dates.format(datetime, "yyyy-mm-dd_HH-MM-SS")

    # build the config file 
    # get git info of QuantumGrav package 
    pkg_id = Base.identify_package("QuantumGrav")
    info = Pkg.dependencies()[pkg_id.uuid]
    git_source = info.git_source
    git_branch = info.git_revision
    git_tree_hash = info.tree_hash
    config["QuantumGrav"] = Dict(
        "git_source" => git_source,
        "git_branch" => git_branch,
        "git_tree_hash" => git_tree_hash,
    )

    YAML.write_file(
        joinpath(
            abspath(expanduser(config["output"])),
            "config_$(getpid())_$(datetime).yaml",
        ),
        config,
    )

    if config["output_format"] == "hdf5"
        filepath = joinpath(
            abspath(expanduser(config["output"])),
            "data_$(getpid())_$(datetime).h5",
        )

        file = HDF5.h5open(filepath, get(config, "file_mode", "w"))
    elseif config["output_format"] == "zarr"
        filepath = joinpath(
            abspath(expanduser(config["output"])),
            "data_$(getpid())_$(datetime).zarr",
        )

        file = Zarr.DirectoryStore(filepath)
    else
        throw(ArgumentError("output_format must be either 'hdf5' or 'zarr'"))
    end

    return filepath, file

end
