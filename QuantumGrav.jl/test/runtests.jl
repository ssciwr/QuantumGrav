using TestItemRunner
using TestItems
using Test
include("./test_utils.jl")
include("./test_datageneration.jl")
include("./test_csetgeneration.jl")
include("./test_layeredgeneration.jl")
include("./test_daggeneration.jl")
@run_package_tests
