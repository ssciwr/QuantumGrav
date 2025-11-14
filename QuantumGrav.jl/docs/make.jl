using Documenter

# Ensure the package source in this monorepo is on LOAD_PATH. This file lives in
# QuantumGrav.jl/docs, so the package source is at ../src relative to this file.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using QuantumGrav

# Build docs. In a monorepo where the Julia package is in `QuantumGrav.jl/`, set
# `repo` to the top-level repo (owner/repo). We will deploy the built site into a
# subdirectory on the target branch (see deploydocs below) so it won't clobber any
# other documentation (e.g. Python docs).
makedocs(
    sitename = "QuantumGrav.jl",
    repo = "ssciwr/QuantumGrav",
    format = Documenter.HTML(
        prettyurls = true,
        # repolink should point to the repository root for the project
        repolink = "https://github.com/ssciwr/QuantumGrav",
        # edit_link can be a branch name or full edit URL base; keep simple here
        edit_link = "main",
    ),
)

# Deploy the built docs into the repository's gh-pages branch under the
# `julia/` subdirectory. That way the Python docs can live in a different
# subdirectory (e.g. `python/`) and each docs pipeline can safely overwrite
# only its own subtree.
deploydocs(
    repo = "ssciwr/QuantumGrav",
    branch = "gh-pages",
    forcepush = false,   # do not clobber other content on gh-pages
    dirname = "julia",
)
