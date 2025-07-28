module QuantumGrav

import LinearAlgebra
import Random
import Graphs
import SparseArrays
import CausalSets
import Distributions
import YAML
import HDF5
import ProgressMeter
import StaticArrays

include("utils.jl")
include("csetgeneration.jl")
include("datageneration.jl")

export make_simple_cset,
    make_manifold_cset,
    make_general_cset,
    make_link_matrix,
    make_adj,
    max_pathlen,
    calculate_angles,
    calculate_distances

end # module QuantumGrav
