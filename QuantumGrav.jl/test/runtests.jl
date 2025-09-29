using TestItemRunner
using TestItems
using Test
include("./test_utils.jl")
include("./test_csetgeneration.jl")
include("./test_csetgenerationbyconnectivity.jl")
include("./test_layeredgeneration.jl")
include("./test_branchedcsetgeneration.jl")
include("./test_csetmerging.jl")
include("./test_destroy_cset.jl")
include("./test_grid_like_causets.jl")
include("./test_graph_utils.jl")
include("./test_prepare_dataproduction.jl")

@run_package_tests
