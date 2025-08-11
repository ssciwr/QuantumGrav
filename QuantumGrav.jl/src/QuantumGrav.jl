module QuantumGrav

using LinearAlgebra: LinearAlgebra
using Random: Random
using SparseArrays: SparseArrays
using CausalSets: CausalSets
using Distributions: Distributions
using YAML: YAML
using HDF5: HDF5
using ProgressMeter: ProgressMeter
using Dates: Dates
using StatsBase: StatsBase

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
