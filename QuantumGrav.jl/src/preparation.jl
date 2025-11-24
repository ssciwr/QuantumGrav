
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
            "config_$(getpid())_$(datetime).yaml",
        ),
        config,
    )

    # create the output file
    filepath = joinpath(
        abspath(expanduser(config["output"])),
        "$(name)_$(getpid())_$(datetime).zarr",
    )

    file = Zarr.DirectoryStore(filepath)

    return filepath, file

end
