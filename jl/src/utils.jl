"""
    make_manifold(i::Int, d::Int)

DOCSTRING
"""
function make_manifold(i::Int, d::Int)::CausalSets.AbstractManifold
    if i == 1
        return CausalSets.MinkowskiManifold{d}()
    elseif i == 2
        return CausalSets.HypercylinderManifold{d}(1.0)
    elseif i == 3
        return CausalSets.DeSitterManifold{d}(1.0)
    elseif i == 4
        return CausalSets.AntiDeSitterManifold{d}(1.0)
    elseif i == 5
        return CausalSets.TorusManifold{d}(1.0)
    elseif i == 6
        return PseudoManifold{d}()
    else
        error("Unsupported manifold: $i")
    end
end

"""
    get_manifold_name(type::Type, d)

DOCSTRING
"""
function get_manifold_name(type::Type, d)
    Dict(
        CausalSets.MinkowskiManifold{d} => "Minkowski",
        CausalSets.DeSitterManifold{d} => "DeSitter",
        CausalSets.AntiDeSitterManifold{d} => "AntiDeSitter",
        CausalSets.HypercylinderManifold{d} => "HyperCylinder",
        CausalSets.TorusManifold{d} => "Torus",
        PseudoManifold{2} => "Random")[type]
end

get_manifold_encoding = Dict(
    "Minkowski" => 1,
    "DeSitter" => 3,
    "AntiDeSitter" => 4,
    "HyperCylinder" => 2,
    "Torus" => 5,
    "Random" => 6
)

# TODO: are the atoms are already topsorted, so we can use that order directly and don´t have to recompute it?

"""
    topsort(adj_matrix::AbstractMatrix, in_degree::Vector{Float32})

DOCSTRING
"""
function topsort(adj_matrix::AbstractMatrix, in_degree::Vector{Float32})::Vector{Int}
    n = size(adj_matrix, 1)

    # Topological sort using Kahn's algorithm --> will be needed later for the topo order of the csets
    queue = Vector{Int64}()
    sizehint!(queue, n)
    for i in 1:n
        if in_degree[i] == 0
            @inbounds push!(queue, i)
        end
    end

    topo_order = Vector{Int64}()
    sizehint!(topo_order, n)
    while !isempty(queue)
        @inbounds u = popfirst!(queue)
        @inbounds push!(topo_order, u)

        if adj_matrix isa SparseArrays.AbstractSparseMatrix
            nodes = SparseArrays.findnz(adj_matrix[u, :])[1]
        else
            nodes = findall(.!isapprox.(adj_matrix[u, :], 0, atol = 1e-10))
        end
        # For each neighbor v of u
        @inbounds for v in nodes
            @inbounds in_degree[v] -= 1
            if in_degree[v] == 0
                @inbounds push!(queue, v)
            end
        end
    end

    return topo_order
end

"""
    maxpathlen(adj_matrix::AbstractMatrix{T}, topo_order::Vector{Int}, source::Int)

DOCSTRING

# Arguments:
- `adj_matrix`: DESCRIPTION
- `topo_order`: DESCRIPTION
- `source`: DESCRIPTION
"""
function maxpathlen(adj_matrix::AbstractMatrix{T}, topo_order::Vector{Int},
        source::Int) where {T <: Number}
    n = size(adj_matrix, 1)

    # Dynamic programming for longest paths
    dist = fill(-Inf, n)
    dist[source] = 0

    # Process vertices in topological order
    @inbounds for u in topo_order
        if @inbounds dist[u] != -Inf
            if adj_matrix isa SparseArrays.AbstractSparseMatrix
                nodes = SparseArrays.findnz(adj_matrix[u, :])[1]
            else
                nodes = findall(.!isapprox.(adj_matrix[u, :], 0, atol = 1e-10))
            end
            @inbounds for v in nodes
                @inbounds dist[v] = max(dist[v], dist[u] + 1)
            end
        end
    end

    # Return max finite distance
    @inbounds finite_dists = filter(d -> d != -Inf, dist)
    return @inbounds isempty(finite_dists) ? 0 : Int32(maximum(finite_dists))
end

"""
    make_adj(c::CausalSets.AbstractCauset, type::Type{T})

DOCSTRING
"""
function make_adj(c::CausalSets.AbstractCauset, type::Type{T}) where {T <: Number}
    c.future_relations |> x -> hcat(x...) |> transpose |> SparseArrays.SparseMatrixCSC{type}
end

"""
    topsort(adj_matrix, in_degree::Vector{Float32})

DOCSTRING
"""
function topsort(adj_matrix, in_degree::Vector{Float32})::Vector{Int}

    # TODO: the atoms are already topsorted, so we can use that order directly and don´t have to recompute it?

    n = size(adj_matrix, 1)

    # Topological sort using Kahn's algorithm --> will be needed later for the topo order of the CausalSets
    queue = Vector{Int64}()
    sizehint!(queue, n)
    for i in 1:n
        if in_degree[i] == 0
            @inbounds push!(queue, i)
        end
    end

    topo_order = Vector{Int64}()
    sizehint!(topo_order, n)
    while !isempty(queue)
        @inbounds u = popfirst!(queue)
        @inbounds push!(topo_order, u)

        # For each neighbor v of u
        @inbounds for v in SparseArrays.findnz(adj_matrix[u, :])[1]
            @inbounds in_degree[v] -= 1
            if in_degree[v] == 0
                @inbounds push!(queue, v)
            end
        end
    end

    return topo_order
end

"""
    make_link_matrix(cset::CausalSets.AbstractCauset, type::Type{T})

DOCSTRING
"""
function make_link_matrix(
        cset::CausalSets.AbstractCauset, type::Type{T}) where {T <: Number}
    link_matrix = SparseArrays.spzeros(type, cset.atom_count, cset.atom_count)
    for i in 1:(cset.atom_count)
        for j in 1:(cset.atom_count)
            if CausalSets.is_link(cset, i, j)
                @inbounds link_matrix[i, j] = type(1)
            end
        end
    end
    return link_matrix
end

"""
    make_cset(manifold::CausalSets.AbstractManifold, boundary::CausalSets.AbstractBoundary, n::Int64, rng::Random.AbstractRNG; markov_convergence::Int = 300)

DOCSTRING

# Arguments:
- `manifold`: DESCRIPTION
- `boundary`: DESCRIPTION
- `n`: DESCRIPTION
- `rng`: DESCRIPTION
- `markov_convergence`: DESCRIPTION
"""
function make_cset(
        manifold::CausalSets.AbstractManifold, boundary::CausalSets.AbstractBoundary,
        n::Int64, rng::Random.AbstractRNG; markov_convergence::Int = 300)
    if manifold isa PseudoManifold
        return CausalSets.sample_random_causet(
            CausalSets.BitArrayCauset, n, markov_convergence, rng)

    else
        sprinkling = CausalSets.generate_sprinkling(manifold, boundary, n; rng = rng)
        cset = CausalSets.BitArrayCauset(manifold, sprinkling)
        return cset
    end
end

"""
    resize(m::AbstractArray{T}, new_size::Tuple(Int))

DOCSTRING
"""
function resize(m::AbstractArray{T}, new_size::Tuple(Int,)) where {T <: Number}
    if all(size(m) .< new_size)
        resized_m = m isa SparseArrays.AbstractSparseArray ?
                    SparseArrays.spzeros(T, new_size...) : zeros(T, new_size...)
        @inbounds resized_m[tuple([1:n for n in size(m)]...)...] .= m
        return resized_m
    else
        return @inbounds m[tuple([1:n for n in new_size]...)...]
    end
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
function make_cardinality_matrix(cset::CS.AbstractCauset)::SparseArrays.SparseMatrixCSC{
        Float32, Int}
    if cset.atom_count == 0
        throw(ArgumentError("The causal set must not be empty."))
    end

    cardinality_matrix = SparseArrays.spzeros(Float32, cset.atom_count, cset.atom_count)

    for i in 1:(cset.atom_count)
        for j in 1:(cset.atom_count)
            ca = CS.cardinality_of(cset, i, j)
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
# TODO: check again if this is correct, lookup in paper
function make_Bd_matrix(ds::Array{Int64}, maxCardinality::Int64 = 10)
    if length(ds) == 0
        throw(ArgumentError("The dimensions must not be empty."))
    end

    if maxCardinality <= 0
        throw(ArgumentError("maxCardinality must be a positive integer."))
    end

    mat = SparseArrays.spzeros(Float32, maxCardinality, length(ds))

    for c in 1:maxCardinality
        for d in 1:length(ds)
            bd = CS.bd_coef(c, ds[d], CS.Discrete()) #does this work?
            if bd != 0
                @inbounds mat[c, d] = bd
            end
        end
    end

    return mat
end

"""
    make_cset_matrices(cset::CausalSets.AbstractCauset, with_resize::Bool, max_nodes::Int, type::Type{T})

DOCSTRING

# Arguments:
- `cset`: The causal set for which the matrices are to be generated.
- `with_resize`: A boolean indicating whether to resize the matrices.
- `max_nodes`: The maximum number of nodes (atoms) in the causal set.
- `type`: The numeric type to be used for the matrices.
"""
function make_cset_matrices(cset::CausalSets.AbstractCauset, with_resize::Bool,
        max_nodes::Int, type::Type{T}) where {T <: Number}
    link_matrix = make_link_matrix(cset, type)
    adj = make_adj(cset, type)

    if with_resize
        link_matrix = resize(link_matrix, (max_nodes, max_nodes))
        adj = resize(adj, (max_nodes, max_nodes))
    end

    in_degrees = sum(adj, dims = 1)
    out_degrees = sum(adj, dims = 2)

    is_sparse = SparseArrays.issparse(adj)
    if is_sparse
        in_degrees = SparseArrays.sparse(in_degrees)
        out_degrees = SparseArrays.sparse(out_degrees)
        in_degree_laplacian = SparseArrays.spdiagm(in_degrees) - adj
        out_degree_laplacian = SparseArrays.spdiagm(out_degrees) - adj
    else
        in_degrees = vec(in_degrees)
        out_degrees = vec(out_degrees)
        in_degree_laplacian = LinearAlgebra.Diagonal(in_degrees) - adj
        out_degree_laplacian = LinearAlgebra.Diagonal(out_degrees) - adj
    end

    return link_matrix, adj, in_degrees, out_degrees, in_degree_laplacian,
    out_degree_laplacian
end
