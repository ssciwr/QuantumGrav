module QuantumGrav

include("datageneration.jl")
include("dataloader.jl")

export Dataset, collateMatrices, encodeMatrix
export generateDataForManifold

end # module QuantumGrav
