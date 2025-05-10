module DataLoader

export Dataset, loadData

import Arrow
import Tables
import Flux
import JLD2


"""
    Dataset

A mutable struct for efficiently loading and caching simulated causal set data from Arrow files, based on MLUtils.DataLoader. This loader assumes that each file contains multiple rows of data, and each row is a matrix of Float32 values. The data is loaded in batches, and the loader caches the loaded data to minimize file I/O operations. The amount of caching is configurable.

# Fields
- `base_path::String`: The base directory path where the data files are located
- `file_paths::Vector{String}`: Relative paths to the Arrow files containing the data
- `indices::Dict{Int, Tuple{Int, Int}}`: Mapping from global indices to (file_index, row_index) tuples
- `file_length::Int`: Number of rows in each file (assumed to be consistent across files)
- `buffer::Dict{Int, Array{Matrix{Float32}}}`: Cache for loaded data to minimize file I/O
- `max_buffer_size::Int64`: Maximum number of files to keep in memory
- `mode::String`: The mode of data loading, either "jld2" or "arrow"
"""
mutable struct Dataset
    base_path::String
    file_paths::Union{Vector{String}, String}
    indices::Dict{Int, Tuple{Int, Int}}
    file_length::Int
    buffer::Dict{Int, Any}
    max_buffer_size::Int64
    mode::String  # jld2 or arrow
end

"""
    Dataset(base_path::String, file_paths::Vector{String}; cache_size::Int=5)

Construct a `Dataset` object from Arrow files for quantum gravity simulations.

# Arguments
- `base_path::String`: Base directory containing Arrow data files.
- `file_paths::Vector{String}`: Paths to Arrow files relative to `base_path`.

# Keywords
- `cache_size::Int=5`: Maximum number of files to keep in memory cache.

# Returns
- `Dataset`: A Dataset object for accessing quantum gravity simulation data.
"""
function Dataset(
        base_path::String, file_paths::Union{Vector{String}, String};
        mode::String = "arrow",
        cache_size::Int = 5)

    chunk_size = 0
    
    indices = Dict{Int, Tuple{Int, Int}}()
    
    idx = 1

    if mode == "arrow"
        chunk_size = length(Tables.getcolumn(
            Arrow.Table(base_path*"/"*file_paths[1]), :linkMatrix))

        for f in 1:length(file_paths)
            for i in 1:chunk_size
                indices[idx] = (f, i)
                idx += 1
            end
        end

    elseif mode == "jld2"

        # in jld2 everything is stored in a single file with 
        chunks, chunk_size = JLD2.jldopen(
            base_path*"/"*file_paths, "r") do file

            length(file), length(file["chunk1"]["linkMatrix"])
        end

        for f in 1:chunks
            for i in 1:chunk_size
                indices[idx] = (f, i)
                idx += 1
            end
        end
    else 
        throw(ArgumentError("Unsupported mode: $mode"))
    end

    return Dataset(base_path, file_paths, indices, chunk_size,
        Dict{Int, Array{Matrix{Float32}}}(), cache_size, mode)
end


"""
    loadData(d::Dataset, i::Int)

Load data from simulated causal set data from an Arrow file the path of which is stored in the passed `Dataset` object.

# Arguments
- `d::Dataset`: Dataset to load from
- `i::Int`: Index of the file to load

# Returns
Array of matrices of Float32 values from the specified column in the Arrow file.
"""
function loadData(d::Dataset, i::Int)
    if d.mode == "arrow"
        Arrow.Table(d.base_path*"/"*d.file_paths[i])
    else 
        JLD2.jldopen(
            d.base_path*"/"*d.file_paths, "r") do file
                group = file["chunk$i"]
                Dict(Symbol(k): group[k] for k in keys(group))
        end
    end
end

"""
    length(d::Dataset)
Get the number of files in the Dataset.
"""
Base.length(d::Dataset) = length(d.indices)

"""
    getindex(data::Dataset, i::Int) -> Any

Retrieve a specific element from a `Dataset`, which is a particular entry in a file.

This method implements a buffer mechanism to efficiently load and retrieve data:
1. Converts the linear index `i` to file and member indices (`f_idx`, `m_idx`)
2. Checks if the file is already in buffer
3. If so, returns the specific member directly
4. If not, checks if buffer is at capacity and removes oldest entry if needed
5. Loads the required file into the buffer
6. Returns the requested member from the newly loaded file

# Arguments
- `data::Dataset`: The Dataset to access
- `i::Int`: The linear index of the element to retrieve

# Returns
- The data element at the specified index
"""
function Base.getindex(data::Dataset, i::Int)
    f_idx, m_idx = data.indices[i]

    if f_idx in keys(data.buffer)
        return data.buffer[f_idx][m_idx]
    else
        if length(data.buffer) == data.max_buffer_size
            delete!(data.buffer, first(keys(data.buffer)))
        end
        data.buffer[f_idx] = loadData(data, f_idx)
    end

    return data.buffer[f_idx][m_idx]
end

"""
    getindex(d::Dataset, is::Vector{Int}) -> Vector

Return a vector of data points from the Dataset `d` at the indices specified in `is`.

# Arguments
- `d::Dataset`: The Dataset to index into.
- `is::Vector{Int}`: A vector of indices.

# Returns
- `Vector`: A vector containing the data points at the specified indices.

"""
function Base.getindex(d::Dataset, is::Union{Vector{Integer}, AbstractRange{Integer}})
    return [d[i] for i in is]
end


end
