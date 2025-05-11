module QuantumGrav

include("datageneration.jl")
include("dataloader.jl")

export Dataset, collateMatrices, encodeMatrix
export generate_data_for_manifold

end # module QuantumGrav
