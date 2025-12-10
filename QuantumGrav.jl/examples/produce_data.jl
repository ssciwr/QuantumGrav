using Distributed

# add processes
addprocs(4; exeflags = ["--threads=2", "--optimize=3"], enable_threaded_blas = true)

# use @everywhere to include necessary modules on all workers
@everywhere using QuantumGrav
@everywhere using Random
@everywhere using YAML
@everywhere using Zarr
@everywhere using LinearAlgebra
@everywhere using Dates
@everywhere using ProgressMeter
@everywhere using CausalSets

# make some dummy data. this must return a dictionary
# the make data fucntion must be defined at the top level 
function make_data(factory::CsetFactory)
    cset, _ = factory("random", 32, factory.rng)
    return Dict("n" => cset.atom_count)
end

# make a temporary output path
targetpath = mktempdir()
try
    println("starting produce_data_mp_testitem.jl")

    # read the default config and modify it to use the temporary output path
    defaultconfigpath = joinpath(dirname(@__DIR__), "configs", "createdata_default.yaml")
    cfg = YAML.load_file(defaultconfigpath)
    cfg["output"] = targetpath
    configpath = joinpath(targetpath, "config.yaml")

    # .. then write back again
    open(configpath, "w") do io
        YAML.write(io, cfg)
    end

    # produce 20 data points using multiprocessing
    # hole max. 10 datapoints in the writing queue at once 
    # and use the make data function that we had defined above
    QuantumGrav.produce_data(10, configpath, make_data)

finally
    println("done with produce_data_mp_testitem.jl")
    rmprocs(workers()...)
end
