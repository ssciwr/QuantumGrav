module QuantumGrav

using LinearAlgebra
using Random
using SparseArrays
using CausalSets
using Distributions
using YAML
using HDF5
using ProgressMeter
using Dates
using StatsBase

include("utils.jl")
include("csetgeneration.jl")
include("layeredgeneration.jl")
include("datageneration.jl")
include("daggeneration.jl")

export make_simple_cset,
       make_manifold_cset,
       make_general_cset,
       make_link_matrix,
       make_adj,
       max_pathlen,
       calculate_angles,
       calculate_distances,
       create_random_cset_from_dag,
       transitive_closure!,
       transitive_reduction!
gaussian_dist_cuts,
create_random_layered_causet
end # module QuantumGrav
