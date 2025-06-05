module QuantumGrav

include("datageneration.jl")
include("tabledataset.jl")
include("graphdataset.jl")

export TDataset, load_data
export GDataset
export generate_data_for_manifold

end # module QuantumGrav
