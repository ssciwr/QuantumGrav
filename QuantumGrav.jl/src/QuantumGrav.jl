module QuantumGrav

import LinearAlgebra
import Random
import Graphs
import SparseArrays
import CausalSets
import Distributions
import YAML
import HDF5
import Zarr
import ProgressMeter
import Dates
import Pkg ## needed for make_data

include("utils.jl")
include("csetgeneration.jl")
include("layeredgeneration.jl")
include("datageneration.jl")
include("csetgenerationbyconnectivity.jl")
include("csetmerging.jl")
include("destroy_manifold_like_cset.jl")
include("grid_like_causets.jl")

export make_simple_cset,
    make_manifold_cset,
    make_general_cset,
    make_link_matrix,
    make_adj,
    max_pathlen,
    calculate_angles,
    calculate_distances,
    sample_bitarray_causet_by_connectivity,
    gaussian_dist_cuts,
    create_random_layered_causet,
    merge_csets,
    insert_cset,
    insert_KR_into_manifoldlike,
    prepare_dataproduction,
    random_causet_by_connectivity_distribution,
    destroy_manifold_cset,
    generate_grid_from_brillouin_cell,
    generate_grid_2d,
    create_grid_causet_2D,
    create_grid_causet_2D_polynomial_manifold
end # module QuantumGrav
