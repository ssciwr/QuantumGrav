
"""
	copy_sourcecode(funcs_to_copy::Vector{Any}, targetpath::String)

Copy the sourcecode files of the functions in the arguments to a targetpath.

# Arguments:
funcs_to_copy::Vector list of functions to copy the sourcecode files of
targetpath::String local path to copy the files to
"""
function copy_sourcecode(funcs_to_copy::Vector, targetpath::String)

    # get the source code of the prepare/write functions and write them to the data folder
    # to document how the data has been created
    for func_to_copy in funcs_to_copy
        funcdata = first(methods(func_to_copy)) # this assumes that all overloads of the passed functions are part of the same file
        filepath = String(funcdata.file)

        tgt = joinpath(targetpath, splitext(basename(filepath))[1] * ".jl")
        if isfile(tgt) == false
            cp(filepath, tgt)
        end
    end

end

"""
	get_git_info!(config::AbstractDict)
Get git info (branch, source, tree_hash) of the QuantumGrav package

# Arguments
config::AbstractDict Config dictionary to put the git data into
"""
function get_git_info!(config::AbstractDict)
    pkg_id = Base.identify_package("QuantumGrav")

    if pkg_id === nothing
        throw(ErrorException("QuantumGrav not found in current environment"))
    end

    info = get(Pkg.dependencies(), pkg_id.uuid, nothing)

    if info === nothing
        throw(ErrorException("QuantumGrav dependency info not found"))
    end

    git_source = info.git_source
    git_branch = info.git_revision
    git_tree_hash = info.tree_hash
    config["QuantumGrav"] = Dict(
        "git_source" => git_source,
        "git_branch" => git_branch,
        "git_tree_hash" => git_tree_hash,
    )
end

"""
	prepare_dataproduction(config::AbstractDict, funcs_to_copy::Vector{Any})::Tuple{str, Zarr.DirectoryStore}

Prepare the data production process from the config dict supplied.
- create a target directory
- copy over the source files of the passed functions. The user has to select these based on their importance for data production and whether they should be retained together with the data in the target directory
- store the git info of the QuantumGrav package (branch, tree hash, source) used
- save the config file, augmented with git info,  to the target directory
- create a zarr directorystore based on the config file.

# Arguments
config::AbstractDict Config file defining the data generation system
funcs_to_copy::Vector{Any} Functions which are to be used and whose source files are to be copied to be retained

# Keyword arguments
name::String Prefix for the created zarr file. Will be followed by the caller process pid and the date in yyyy-mm-dd_HH-MM-SS

# Returns
Tuple{String, Zarr.DirectoryStore} The path to the output file and the output file itself.
"""
function prepare_dataproduction(
    config::AbstractDict,
    funcs_to_copy::Vector;
    name::String = "data",
)::Tuple{String,Zarr.DirectoryStore}
    # consistency checks
    for key in ["num_datapoints", "output", "seed"]
        if !haskey(config, key)
            throw(ArgumentError("Configuration must contain the key: $key"))
        end
    end

    if length(funcs_to_copy) == 0
        throw(ArgumentError("No functions to copy"))
    end

    targetpath = abspath(expanduser(config["output"]))

    # make directory to put data into
    if !isdir(targetpath)
        mkpath(targetpath)
    end

    datetime = Dates.now()
    datetime = Dates.format(datetime, "yyyy-mm-dd_HH-MM-SS")

    # get git info of QuantumGrav package and write into config file
    get_git_info!(config)

    # copy the source code of the used functions into the target directory
    copy_sourcecode(funcs_to_copy, targetpath)

    # write the config file to target
    YAML.write_file(
        joinpath(
            abspath(expanduser(config["output"])),
            "$(name)_$(getpid())_$(datetime).yaml",
        ),
        config,
    )

    # create the output file
    filepath = joinpath(
        abspath(expanduser(config["output"])),
        "$(name)_$(getpid())_$(datetime).zarr",
    )

    file = Zarr.DirectoryStore(filepath)

    # create root group
    Zarr.zgroup(file, "")

    return filepath, file

end


# """
# 	setup_mp(config::Dict, num_blas_threads::Int, make_data::Function)

# Set up the multiprocessing environment for data production. 
# This initializes a CsetFactory on each worker process with a different seed and 
# copies the config dict to each worker. It also imports needed modules.
# """
# function setup_mp(config::Dict, num_blas_threads::Int, make_data::Function)

#     @sync for p in Distributed.workers()
#         println("process: ", p)
#         # runs on the main process
#         seed_per_process = rand(1:1_000_000_000_000)
#         @info "setting up worker $p with seed $seed_per_process"

#         # deepcopy config and set seed in the copy to not contaminate the config
#         worker_conf = deepcopy(config)
#         worker_conf["seed"] = seed_per_process

#         Distributed.@spawnat p begin
#             println("setting up worker: ", myid())
#             #execute this on each worker
#             @eval Main begin
#                 import Pkg
#                 Pkg.activate($(Base.active_project()))
#                 println(
#                     "importing modules on worker: ",
#                     myid(),
#                     ", ",
#                     Base.active_project(),
#                 )
#                 using CausalSets
#                 using YAML
#                 using Random
#                 using Zarr
#                 using ProgressMeter
#                 using Statistics
#                 using LinearAlgebra
#                 using Distributions
#                 using SparseArrays
#                 using StatsBase
#                 using JSON

#                 LinearAlgebra.BLAS.set_num_threads(num_blas_threads)

#                 println("setting up cset factory on worker: ", myid())
#                 global worker_factory
#                 global make_cset_data

#                 global make_cset_data = $make_data
#                 global worker_factory = CsetFactory(worker_conf)
#                 println("done setting up cset factory on worker: ", myid())
#             end
#         end
#     end
# end

"""
    setup_config(configpath::Union{String, Nothing})::Dict{Any,Any}

Set up the configuration for data production.
- Loads the default configuration from `configs/createdata_default.yaml`.
- If a user-provided config path is given, loads it and merges with the default (overlapping keys in the user config override defaults).

# Arguments
- `configpath::Union{String, Nothing}`: Path to a YAML config file to merge with the defaults. If `nothing`, only the default config is used. Relative paths and paths with `~` are supported.

# Returns
- `Dict{Any,Any}`: The resulting configuration dictionary after applying defaults and optional overrides.
"""
function setup_config(configpath::Union{String,Nothing})::Dict{Any,Any}
    defaultconfigpath = joinpath(dirname(@__DIR__), "configs", "createdata_default.yaml")
    default_config = YAML.load_file(defaultconfigpath)

    if configpath === nothing
        config = default_config
        return config
    else
        # Normalize user-provided path to improve robustness
        normalized = abspath(expanduser(configpath))
        if isfile(normalized)
            loaded_config = YAML.load_file(normalized)
            # this replaces the overlapping entries (shallow merge is fine for our config structure)
            config = merge(default_config, loaded_config)
            return config
        else
            throw(ArgumentError("Error: Config file not found at $(normalized)"))
        end
    end
end


"""
	produce_data(num_workers::Int64, num_threads::Int64, num_blas_threads::Int64, chunksize::Int64, configpath::String, make_data::Function)

Produce data on num_workers in parallel using the supplied make_data function. This works by spawning the data production workers and have them
fill a Channel with data that is drained by a writer task concurrently. Therefore, data writing and data production happens at the same time. At most `chunksize` datapoints can be held in the queue. When it's full the workers will
wait until a slot in it becomes free. In the same way, the writer task will wait until a new datapoint is available if the queue is empty.
How many datapoints are written is defined in the config. See the documentation of the config file for this.
Worker processes will be removed after the data production task is done.

# Arguments:
- `num_workers`: Number of data production workers
- `num_threads`: Number of threads per data production worker. Make sure you coordinate this with the number of workers and number of blas threads to avoid oversubscription
- `num_blas_threads`: Number of threads the BLAS library uses for LinearAlgebra tasks. Make sure you coordinate this with the number of workers and number of threads to avoid oversubscription
- `chunksize`: Maximum datapoints to be held in memory at any one time.
- `configpath`: path on disk to the config file to load
- `make_data`: Function with the signature: functionname(worker_factory::CsetFactory). No other signature is permissible and supplying one will throw an error.

# Returns
Nothing
"""
function produce_data(
    num_workers::Int64,
    num_threads::Int64,
    num_blas_threads::Int64,
    chunksize::Int64,
    configpath::String,
    make_data::Function,
)::Nothing

    try
        if length(workers()) > 1
            @warn "Warning, there are multiple workers already initialized: $(length(workers()))"
        end

        Distributed.addprocs(
            num_workers;
            exeflags = [
                "--project=$(Base.active_project())",
                "--threads=$(num_threads)",
                "--optimize=3",
            ],
            enable_threaded_blas = true,
        )


        @info "setup config"
        config = setup_config(configpath)

        # set the global rng seed in the main process
        @info "set seed"
        Random.seed!(config["seed"])

        # setup multiprocessing
        setup_mp(config, num_blas_threads, make_data)

        # get cset type
        cset_type = config["cset_type"]
        @info "generating data for cset type $(cset_type)"

        # make file, prepare config, output dir
        @info "preparing output zarr store"
        filepath, file = QG.prepare_dataproduction(config, [make_data], name = cset_type)

        # get number of csets to create
        n = config["num_datapoints"]

        # split indices across workers and set up data queue
        # only hold chunksize many csets in memory at any given time
        queue = RemoteChannel(() -> Channel{Tuple{Int,Dict{String,Any}}}(max(1, chunksize)))

        # start async writer task
        writer = @async begin
            @info "Writer started on pid=$(myid())"
            root = Zarr.zopen(file, "w"; path = "")
            for _ ∈ 1:n # write exactly n times => num_datasets write operations
                i, cset_data = take!(queue) # blocks when queue is empty
                csetdata = Dict("cset_$(i)" => cset_data)
                QG.dict_to_zarr(root, csetdata)
            end
            @info "Writer finished"
        end

        # start producers
        @info "Producing data"
        p = Progress(n, barglyphs = BarGlyphs("[=> ]"))
        progress_pmap(1:n, progress = p, batch_size = 1) do i
            cset_data = make_cset_data(worker_factory)
            put!(queue, (i, cset_data)) # blocks when queue is full
        end

        @info "All producers finished. Waiting for writer to finish"
        wait(writer)
        @info "Finished cset type $(cset_type)"
    catch e
        @error "Error during data generation: $e"
    finally
        @info "Finished data generation"
        rmprocs(workers())
    end
end
