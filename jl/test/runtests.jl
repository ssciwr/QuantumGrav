using TestItemRunner

include("./test_datageneration.jl")
include("./test_dataloader.jl")
@run_package_tests
