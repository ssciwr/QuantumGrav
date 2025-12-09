module QuantumGrav

import LinearAlgebra
import Random
import Graphs
import SparseArrays
import CausalSets
import Distributions
import YAML
import Zarr
import Dates
import Pkg
import JSONSchema

include("utils.jl")
include("csetgeneration.jl")
include("layeredgeneration.jl")
include("preparation.jl")
include("csetgenerationbyconnectivity.jl")
include("branchedcsetgeneration.jl")
include("csetmerging.jl")
include("destroy_manifold_like_cset.jl")
include("grid_like_causets.jl")
include("graph_utils.jl")
include("curvature_on_manifold.jl")
include("cset_factories.jl")
include("save_data.jl")

export make_adj,
    max_pathlen,
    transitive_reduction!,
    make_polynomial_manifold_cset,
    Ricci_scalar_2D,
    Ricci_scalar_2D_of_sprinkling,
    sample_bitarray_causet_by_connectivity,
    gaussian_dist_cuts,
    create_random_layered_causet,
    BranchedManifoldCauset,
    make_branched_manifold_cset,
    merge_csets,
    insert_cset,
    insert_KR_into_manifoldlike,
    prepare_dataproduction,
    random_causet_by_connectivity_distribution,
    destroy_manifold_cset,
    generate_grid_from_brillouin_cell,
    generate_grid_2d,
    create_grid_causet_2D,
    create_grid_causet_2D_polynomial_manifold,
    prepare_dataproduction,
    setup_mp,
    setup_config
produce_data,
copy_sourcecode,
get_git_info!,
PolynomialCsetMaker,
RandomCsetMaker,
LayeredCsetMaker,
DestroyedCsetMaker,
GridCsetMakerPolynomial,
MergedCsetMaker,
ComplexTopCsetMaker,
CsetFactory,
encode_csettype,# data saving
dict_to_zarr,
validate_config,
make_pseudosprinkling
end # module QuantumGrav
