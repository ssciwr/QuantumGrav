import Distributed
import QuantumGrav: CsetFactory

function setup_mp(
    config::Dict,
    num_workers::Int,
    num_threads::Int,
    num_blas_threads::Int,
    make_data::Function,
)

    Distributed.addprocs(
        num_workers;
        exeflags = [
            "--project=$(Base.active_project())",
            "--threads=$(num_threads)",
            "--optimize=3",
        ],
        enable_threaded_blas = true,
    )

    @sync for p in Distributed.workers()
        println("process: ", p)

        # runs on the main process
        seed_per_process = rand(1:1_000_000_000_000)
        @info "setting up worker $p with seed $seed_per_process"

        # deepcopy config and set seed in the copy to not contaminate the config
        worker_conf = deepcopy(config)
        worker_conf["seed"] = seed_per_process

        Distributed.@spawnat p begin
            println("setting up worker: ", myid())
            #execute this on each worker
            @eval Main begin
                println("importing modules on worker: ", myid(), Base.active_project())
                using CausalSets
                using YAML
                using Random
                using Zarr
                using ProgressMeter
                using Statistics
                using LinearAlgebra
                using Distributions
                using SparseArrays
                using StatsBase

                LinearAlgebra.BLAS.set_num_threads($num_blas_threads)

                println("settin gup cset factory on worker: ", myid())
                global worker_factory
                global make_cset_data

                global make_cset_data = $make_data
                global worker_factory = CsetFactory($worker_conf)
            end
        end
    end
end


try
    setup_mp(Dict(), 2, 2, 2, x->Dict())
finally
    Distributed.rmprocs(Distributed.workers()...)
end
