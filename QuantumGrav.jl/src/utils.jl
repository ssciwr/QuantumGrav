

"""
    make_pseudosprinkling(n, d, box_min, box_max, type; rng) -> Vector{Vector{T}}

Generates random points uniformly distributed in a d-dimensional box.

# Arguments
- `n::Int64`: Number of points to generate
- `d::Int64`: Dimension of each point
- `box_min::Float64`: Minimum coordinate value
- `box_max::Float64`: Maximum coordinate value
- `type::Type{T}`: Numeric type for coordinates
- `rng`: Random number generator (default: Xoshiro(1234))

# Returns
- `Vector{Vector{T}}`: Vector of n points, each point is a d-dimensional vector

# Notes
Used for creating pseudo-sprinklings in PseudoManifold when geometric
manifold sprinkling is not applicable.
"""
function make_pseudosprinkling(
    n::Int64,
    d::Int64,
    box_min::Float64,
    box_max::Float64,
    type::Type{T};
    rng = Random.Xoshiro(1234),
)::Vector{Vector{T}} where {T<:Number}
    if box_min >= box_max
        throw(ArgumentError("box_min must be less than box_max"))
    end
    distr = Distributions.Uniform(box_min, box_max)

    return [[type(rand(rng, distr)) for _ = 1:d] for _ = 1:n]
end

function transitive_reduction!(mat::AbstractMatrix)
	n = size(mat, 1)
	@inbounds for i ∈ 1:n
		for j ∈ (i+1):n
			if mat[i, j] == 1
				# If any intermediate node k exists with i → k and k → j, remove i → j
				for k ∈ (i+1):(j-1)
					if mat[i, k] == 1 && mat[k, j] == 1
						mat[i, j] = 0 # remove intermediate nodes
						break
					end
				end
			end
		end
	end
end