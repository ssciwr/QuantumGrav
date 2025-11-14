[![test](https://github.com/ssciwr/QuantumGrav/actions/workflows/ci.yml/badge.svg)](https://github.com/ssciwr/QuantumGrav/actions/workflows/ci.yml)
[![Docs — Python package for modelling](https://img.shields.io/badge/docs-python-blue?logo=python)](https://ssciwr.github.io/QuantumGrav/python/)
[![Docs — Julia package for data generation](https://img.shields.io/badge/docs-julia-purple?logo=julia)](https://ssciwr.github.io/QuantumGrav/julia/)

# QuantumGrav
Quantum gravity project experimental repo.

## Installation

### Julia
1. Clone the repository from [here](https://github.com/ssciwr/QuantumGrav).

We will assume you want to use the QuantumGrav.jl package in another environment that you create yourself.

2. Open a terminal and start the Julia REPL

3. Activate a project environment in which you want to work

```julia
# press ] to enter the package manager prompt, then:
activate path/to/your/project
# press backspace or Ctrl+C to leave the pkg prompt
```

4. Add the QuantumGrav package as a dependency. This must be done from a local path

```julia
# press ] to enter the package manager prompt, then:
add /path/to/QuantumGrav/QuantumGrav.jl
```
This is only necessary as long as QuantumGrav.jl is not in the official package repository of julia.


### Python
For the python installation instructions, see the [documentation](https://ssciwr.github.io/QuantumGrav/getting_started/)

## Installation as a developer
1. Clone the repository from [here](https://ssciwr.github.io/QuantumGrav/getting_started/).

### Julia (developer workflow)
We will assume you want to use the QuantumGrav.jl package in another environment that you create yourself.

2. Open a terminal, start the Julia REPL and activate your target environment
```bash
activate path/to/your/project
```

3. Check out `QuantumGrav.jl` for development
```bash
# press ] to enter the package manager prompt, then:
develop path/to/QuantumGrav/QuantumGrav.jl
```
This will track the changes you made automatically.

4. Running tests from the Julia REPL (recommended for detailed output):
```julia
using TestItemRunner
@run_package_tests
```

5. For building the documentation locally
- Go o the `docs` directory in the QuantumGrav.jl subdirectory
- run `julia --color=yes --project make.jl`
- if you want to have debug output, set the `JULIA_DEBUG` environment variable to `Documenter`:
```bash
export JULIA_DEBUG=Documenter
```
and then run the above command again to get all the debug output.

This runs the tests defined under `QuantumGrav.jl/test` using the activated environment.

### Python (developer workflow)
For the python installation instructions, see the [documentation](https://ssciwr.github.io/QuantumGrav/getting_started/)

### Building the documentation locally
The documentation is generated with MkDocs and `mkdocstrings`. MkDocs needs to be able to import the `QuantumGrav` package so either install the package in the same environment (editable install above) or add the `src` path to `PYTHONPATH` before running mkdocs.

Quick serve (from repository root):

```bash
cd QuantumGravPy
# if you didn't install the package, export PYTHONPATH to include the src dir
mkdocs serve
```
Follow the instructions on screen to open the documentation. More on `mkdocs` can be found [here](https://www.mkdocs.org/), and on the `Documenter.jl` package used in the Julia package documentation [here](https://documenter.juliadocs.org/stable/).


## Notes and troubleshooting

- PyTorch and PyTorch-Geometric installation is platform and CUDA-version specific; consult the official installation docs if you encounter wheel or binary compatibility errors.

### Contribution guide

Tbd.



