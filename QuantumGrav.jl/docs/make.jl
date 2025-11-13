push!(LOAD_PATH,"../src")

using Documenter, QuantumGrav

makedocs(sitename="QuantumGrav.jl",
    repo="https://github.com/ssciwr/QuantumGrav/blob/{commit}{path}#{line}",
    format=Documenter.HTML(
        prettyurls=true,
        repolink="https://github.com/ssciwr/QuantumGrav",
        edit_link="main")
)