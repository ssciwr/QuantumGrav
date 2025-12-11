################################################################################
# command line args
config = nothing
configpath = nothing
num_workers = 1
num_threads = 1
num_blas_threads = 1
chunksize = 1

args = ARGS

if length(args) == 0
    @warn "No command line arguments provided. Using default configuration."
end

# go through command line arguments and assign them
for (i, arg) in enumerate(args)
    if arg == "--config"
        if i + 1 <= length(args)
            global configpath = args[i+1]
        else
            println("Error: --config requires a file path argument.")
            exit(1)
        end
    end

    if arg == "--num_workers"
        if i + 1 <= length(args)
            global num_workers = parse(Int64, args[i+1])
        else
            print("Error: --num_workers requires an integer argument")
            exit(1)
        end
    end

    if arg == "--num_threads"
        if i + 1 <= length(args)
            global num_threads = parse(Int64, args[i+1])
        else
            print("Error: --num_threads requires an integer argument")
            exit(1)
        end
    end

    if arg == "--num_blas_threads"
        if i + 1 <= length(args)
            global num_blas_threads = parse(Int64, args[i+1])
        else
            print("Error: --num_blas_threads requires an integer argument")
            exit(1)
        end
    end

    if arg == "--chunksize"
        if i + 1 <= length(args)
            global chunksize = parse(Int64, args[i+1])
        else
            print("Error: --chunksize requires an integer argument")
            exit(1)
        end
    end

    if arg == "--help" || arg == "-h"
        println("Usage: julia create_data.jl [--config <path_to_config_file>]")
        println("Options:")
        println("  --config <path_to_config_file>  Path to the configuration file.")
        println("  --num_processes <int> number of processes to run on.")
        println("  --num_threads <int> number of threads **per process**")
        println(
            "  --num_blas_threads <int> number of threads the blas library should use. Only needed if you use LinearAlgebra.jl algorithms. This is independent of --num_threads!",
        )
        println(
            "  --chunksize <int> number of csets to generate and write in one go. Setting this to something smaller than the total number of csets to be generated will result in the dataset being generated in chunks so that it doesn't have to be kept in memory all at once.",
        )
        println("  --help, -h                      Show this help message.")
        exit(0)
    end
end

################################################################################
# we need the distributed package to add multiprocessing
import Distributed

# add processes
Distributed.addprocs(
    num_workers,
    execflags = ["--threads=$(num_threads)", "-O3", "--project=$(dirname(@__DIR__))"],
    with_blas_threads = true,
)

################################################################################
# import things we need everywhere
Distributed.@everywhere import QuantumGrav as QG
Distributed.@everywhere import LinearAlgebra

# include further packages that might be needed here. Make sure to add the @everywhere part to have them available on all workers
# e.g. Distributed.@everywhere import StatsBase

# set BLAS threads in linear algebra so things like eigendecomposition run multithreaded
@everywhere LinearAlgebra.BLAS.set_num_threads($num_blas_threads)

################################################################################
# functions for data production

# encode type of csets numerically in data
cset_type_encoder = Dict(
    "polynomial" => 1,
    "complex_topology" => 2,
    "destroyed" => 3,
    "merged" => 4,
    "grid" => 5,
    "layered" => 6,
    "random" => 7,
    "merged_ambiguous" => 8,
    "destroyed_ambiguous" => 9,
)

# actual make cset. Must eat a worker_factory as sole input and return a dict of named data features
function make_cset_data(worker_factory)::Dict{String,Any}

    # reference the globally build thing
    config = worker_factory.conf
    rng = worker_factory.rng
    n = 0 # placeholder, filled later
    cset_type = config["cset_type"]
    cset = nothing
    counter = 0
    while isnothing(cset) && counter < 20
        n = rand(rng, worker_factory.npoint_distribution)
        @debug "    Generating cset data with $n atoms"
        try
            cset, _ = worker_factory(cset_type, n, rng)
        catch e
            @warn "cset generator threw an exception: $(e)"
            cset = nothing
        end
        counter += 1
    end

    if isnothing(cset)
        throw(ErrorException("Couldn't create a working cset"))
    end

    @debug "    computing adjacency matrix"
    adj = QG.make_adj(cset, type = Matrix, eltype = UInt8)

    @debug "    make link matrix as transitive reduction of adjacency"
    linkmat = deepcopy(adj)
    QG.transitive_reduction!(linkmat)

    return Dict(
        "adj" => adj,
        "manifold_like" => cset_type in ["polynomial", "complex_topology"],
    )
end

################################################################################
# run the main dataproduction part
try
    QG.produce_data(chunksize, configpath, make_data)
catch e
    @error "An error occured: $(e). Data production cancelled"
finally
    # don´t forget to remove workers processes after being done
    Distributed.rmprocs(Distributed.workers()...)
end
