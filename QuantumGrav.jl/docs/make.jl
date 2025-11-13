push!(LOAD_PATH,"../src")

using Documenter, QuantumGrav

makedocs(sitename="QuantumGrav.jl",
    repo="https://github.com/ssciwr/QuantumGrav/blob/{commit}{path}#{line}",
    # modules=[QuantumGrav],
    # pages=[
    #     "Home" => "index.md",
    #     "Getting Started" => "getting_started.md",
    #     "Causal Set Generation" => [
    #     ],
    #     "Utilities" => [
    #     ],
    #     "Data" => [],
    #     "Visualization Tools" => "visualization_tools.md",
    #     "Contributing" => "contributing.md",
    # ],
    format=Documenter.HTML(
        prettyurls=true,
        repolink="https://github.com/ssciwr/QuantumGrav",
        edit_link="write_julia_docs")
)