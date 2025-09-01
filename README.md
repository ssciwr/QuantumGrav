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

4. Add the QuantumGrav package as a dependendy. This must be done from a local path

```julia 
# press ] to enter the package manager prompt, then:
add /path/to/QuantumGrav/QuantumGrav.jl
```
This is only necessary as long as QuantumGrav.jl is not in the official package repository of julia. 

### Python
For the python installation instructions, see the [documentation](https://ssciwr.github.io/QuantumGrav/)

## Installation as a developer
1. Clone the repository from [here](https://github.com/ssciwr/QuantumGrav). 

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

This runs the tests defined under `QuantumGrav.jl/test` using the activated environment.

### Python (developer workflow)
For the python installation instructions, see the [documentation](https://ssciwr.github.io/QuantumGrav/)

### Building the documentation locally
The documentation is generated with MkDocs and `mkdocstrings`. MkDocs needs to be able to import the `QuantumGrav` package so either install the package in the same environment (editable install above) or add the `src` path to `PYTHONPATH` before running mkdocs.

Quick serve (from repository root):

```bash
cd QuantumGravPy
# if you didn't install the package, export PYTHONPATH to include the src dir
mkdocs serve
```
Follow the instructions on screen to open the documentation. More on `mkdocs` can be found [here](https://www.mkdocs.org/), and on the `Documenter.jl` package used in the Julia package [here](https://documenter.juliadocs.org/stable/).

Training and evaluation scripts live under `QuantumGravPy/src/QuantumGrav/` (`train.py`, `train_ddp.py`) and can be run once dependencies are installed.
See the [training section](./training_a_model.md) for more. 

## Notes and troubleshooting

- PyTorch and PyTorch-Geometric installation is platform and CUDA-version specific; consult the official installation docs if you encounter wheel or binary compatibility errors.

### Contribution guide

Tbd.



