module GraphDataset

export GraphDataset

using AutomaticDocstrings

"""
    GDataset

A dataset wrapper for graph neural network (GNN) data, allowing transformation of tabular data into GNNGraph objects.

# Fields:
- `dset::TDataset`: The underlying tabular dataset.
- `transform_fn::Function`: A function that converts a table row into a `GNNGraph` object.
"""
struct GDataset
    dset::TDataset #table dataset as base
    transform_fn::Function #function that makes table into vector of GNNGraph
end

"""
    GDataset(path::String; transform_fn::Function = identity, mode::String = "arrow", cache_size::Int = 5)

Construct a `GDataset` from a dataset file at the given path, with optional transformation and loading options.

# Arguments:
- `path`: Path to the dataset file.
- `transform_fn`: Function to transform each row of the dataset into a `GNNGraph`. Defaults to `identity`.
- `mode`: Loading mode for the dataset (e.g., "arrow").
- `cache_size`: Number of dataset chunks to cache in memory.
"""
function GDataset(path::String; transform_fn::Function = identity,
        mode::String = "arrow", cache_size::Int = 5)
    return GDataset(
        Dataset(path; mode = mode, cache_size = cache_size),
        transform_fn
    )
end

"""
    Base.getindex(d::GDataset, i::Int)

Return the `i`-th graph in the dataset as a `GNNGraph` object, applying the transformation function.
"""
Base.getindex(d::GDataset, i::Int)::GNN.GNNGraph = d.transform_fn(d.dset[i])

"""
    Base.getindex(d::GDataset, is::AbstractVector{Int})

Return a vector of `GNNGraph` objects corresponding to the indices in `is`, applying the transformation function to each.
"""
Base.getindex(d::GDataset,
    is::Union{Vector{Int}, AbstractRange{Int}})::Vector{GNN.GNNGraph} = [d.transform_fn(d.dset[i])
                                                                         for i in is]

"""
    Base.length(d::GDataset)

Return the number of graphs (rows) in the dataset.
"""
Base.length(d::GDataset)::Int = length(d.dset)

"""
    Base.iterate(d::GDataset, state = 1)

Iterate over the dataset, yielding pairs of (`GNNGraph`, index) for each graph in the dataset.
Returns `nothing` when iteration is complete.
"""
Base.iterate(d::GDataset,
    state = 1)::Union{
    Nothing, Tuple{GNN.GNNGraph, Int}} = state > length(d) ? nothing : (d[state], state + 1)

end
