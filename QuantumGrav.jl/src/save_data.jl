"""
	default_chunks(data::AbstractArray)

Default chunking strategy for Zarr arrays. Chunks of size 128 along each dimension, or smaller if the dimension size is less than 128.
"""
function default_chunks(data::AbstractArray)
    if ndims(data) == 1
        return (min(size(data, 1), 1024),)
    elseif ndims(data) == 2
        return Tuple(min(size(data, i), 64) for i = 1:ndims(data))
    elseif ndims(data) == 3
        return Tuple(min(size(data, i), 16) for i = 1:ndims(data))
    else
        return Tuple(min(size(data, i), 8) for i = 1:ndims(data))
    end
end

"""
	write_arraylike_to_zarr(group::Zarr.ZGroup, key::String, data::AbstractArray; type = eltype(data), chunks = nothing, compressor_kwargs = Dict(:clevel => 9, :cname => "lz4", :shuffle => 2))

Write a Julia AbstractArray to a Zarr group.

# Arguments:
- `group`: Group to write the array to
- `key`: Key under which to store the array
- `data`: Array to write
- `type`: Data type of the array elements
- `compressor_kwargs`: Compression options for the array
- `chunking_strategy`: Function mapping input data to chunks in which it should be written to disk. When writing this function, be aware of the curse of dimensionality
"""
function write_arraylike_to_zarr(
    group::Zarr.ZGroup,
    key::String,
    data::AbstractArray;
    type = eltype(data),
    compressor_kwargs = Dict(:clevel => 9, :cname => "lz4", :shuffle => 2),
    chunking_strategy = default_chunks,
)
    arr = Zarr.zcreate(
        type,
        group,
        key,
        size(data)...;
        chunks = chunking_strategy(data),
        compressor = Zarr.BloscCompressor(; compressor_kwargs...),
    )

    arr[:] = data
end

"""
	dict_to_zarr(file_or_group::Union{Zarr.DirectoryStore, Zarr.ZGroup}, data::Dict{String, Any}, compressor_kwargs = Dict(clevel => 9, cname => "lz4", shuffle => 2), chunking_strategy::Union{Dict{String, Function}, Function, Nothing} = default_chunks)

Recursively write a nested dictionary to a Zarr group.

# Arguments:
- `file_or_group`: Zarr DirectoryStore or Group to write the data to
- `data`: Nested dictionary to write
- `compressor_kwargs`: Compression options for the arrays
- `chunking_strategy`: Chunking strategy for the arrays
"""
function dict_to_zarr(
    file_or_group::Union{Zarr.DirectoryStore,Zarr.ZGroup},
    data::Dict;
    compressor_kwargs = Dict(:clevel => 9, :cname => "lz4", :shuffle => 2),
    chunking_strategy::Union{Dict{String,Function},Function,Nothing} = default_chunks,
)
    for (key, value) in data
        if value isa Dict
            print("going deeper: ", key)
            group = Zarr.zgroup(file_or_group, string(key))
            dict_to_zarr(
                group,
                value;
                compressor_kwargs = compressor_kwargs,
                chunking_strategy = chunking_strategy,
            )
        else
            print("key: ", key, ", valuetype: ", typeof(value))
            chunking_strategy_key =
                chunking_strategy isa Dict ? get(chunking_strategy, key, default_chunks) :
                chunking_strategy

            if value isa AbstractArray
                write_arraylike_to_zarr(
                    file_or_group,
                    key,
                    value;
                    type = eltype(value),
                    compressor_kwargs = compressor_kwargs,
                    chunking_strategy = chunking_strategy_key,
                )
            elseif value isa String
                write_arraylike_to_zarr(
                    file_or_group,
                    key,
                    [value];
                    type = String,
                    compressor_kwargs = compressor_kwargs,
                    chunking_strategy = chunking_strategy_key,
                )
            elseif value isa Number || value isa Bool || value isa Char
                write_arraylike_to_zarr(
                    file_or_group,
                    key,
                    [value];
                    type = typeof(value),
                    compressor_kwargs = compressor_kwargs,
                    chunking_strategy = chunking_strategy_key,
                )
            else
                throw(
                    ArgumentError(
                        "Unsupported data type for key '$(key)': $(typeof(value))",
                    ),
                )
            end
        end
    end
end
