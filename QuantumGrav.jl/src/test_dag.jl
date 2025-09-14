using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Random: Random
using LinearAlgebra
using Distributions

function transitive_reduction!(mat::AbstractArray{T,N}) where {T,N}
    n = size(mat, 1)
    for i in 1:n
        for j in (i + 1):n
            if mat[i, j] == 1
                # If any intermediate node k exists with i → k and k → j, remove i → j
                for k in (i + 1):(j - 1)
                    if mat[i, k] == 1 && mat[k, j] == 1
                        mat[i, j] = 0 # remove intermediate nodes
                        break
                    end
                end
            end
        end
    end
end

function transitive_closure!(adj::AbstractArray{T,N}) where {T,N}
end

function check_num_paths(i::Int64, j::Int64, adj::Matrix{T})::Int64 where {T}
    # FIXME: check that this thing works as it should! Currently, transitive closure still changes 
    # the final matrix when this is used to check for existing paths

    # this assumes topological ordering of nodes
    # check that there is not path from i to j <=> a possible link i -> j is not redundant
    function check_row(row::Vector{T}, j::Int64, num_paths::Int64)::Int64
        @inbounds for k in 1:length(row)
            if row[k] > 0
                if k == j
                    num_paths += 1
                end
                num_paths = check_row(adj[k, :], j, num_paths)
            else
                continue
            end
        end
        return num_paths
    end

    return check_row(adj[i, :], j, 0) #FIXME: this is probably the issue
end

testmat = zeros(Int64, 10, 10)
testmat[1, 4] = 1
testmat[2, 4] = 1
testmat[3, 9] = 1
testmat[4, 5] = 1
testmat[4, 7] = 1
testmat[4, 6] = 1
testmat[4, 9] = 1
testmat[5, 6] = 1
testmat[6, 7] = 1
testmat[7, 8] = 1
testmat[3, 9] = 1
testmat[9, 10] = 1

@assert check_num_paths(4, 7, testmat) == 3

function generate_dag(p_ij::Function, adj::T;
                      rng=Random.Xoshiro(1234)) where {T<:AbstractArray}
    @inbounds for i in Random.shuffle(rng, 1:size(adj, 1))
        for j in Random.shuffle(rng, (i + 1):size(adj, 2)) # ensure that 1:num_nodes is topologically sorted with the i+1:num_nodes
            if Random.rand(rng) < p_ij(i, j, adj) && check_num_paths(i, j, adj) == 0 # i can never be the same as j here, hence not checked
                adj[i, j] = 1
            end
        end
    end

    return adj
end

# try out various distributions and understand the impact on the generated DAG wrt connectivity 
# and all that shit. 
uniform = Distributions.Normal(1.0, 1.0)
rng = Random.Xoshiro(1234)
p_ij(i, j, adj) = rand(rng, uniform)

for i in 1:10
    display(generate_dag(p_ij, zeros(Int64, 25, 25); rng=rng))
end

adj = generate_dag(p_ij, zeros(Int64, 10, 10); rng=rng)
paths = zeros(Int64, 10, 10)
for i in 1:10
    for j in (i + 1):10
        p = check_num_paths(i, j, adj)
        paths[i, j] = p
    end
end

display(adj)
display(paths)

reduced = deepcopy(adj)
transitive_reduction!(reduced)
display(reduced)

paths = zeros(Int64, 10, 10)
for i in 1:10
    for j in (i + 1):10
        p = check_num_paths(i, j, reduced)
        paths[i, j] = p
    end
end
display(paths)

paths = zeros(Int64, 10, 10)
for i in 1:10
    for j in (i + 1):10
        p = check_num_paths(i, j, adj)
        paths[i, j] = p
    end
end
display(paths)
