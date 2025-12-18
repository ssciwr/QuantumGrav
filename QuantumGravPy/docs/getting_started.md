# Getting started
This document explains how to get the project running for development and for usage. It covers the two supported ecosystems used in this repository: the Julia package (data generation and low-level causal-set utilities) and the Python package (model code, training and evaluation).

## Installation
Note: Install PyTorch and PyTorch Geometric according to your platform and available CUDA version. Follow their official installers to ensure compatible wheels. There are currently two predefined requirements files, one for CPU-only installations and one for CUDA-enabled installations.

The Python package lives in `QuantumGravPy/` and follows a standard packaging layout (sources under `QuantumGravPy/src/QuantumGrav`). The project uses PyTorch and PyTorch Geometric; installation of those dependencies depends on your OS and available hardware (CUDA version).

Basic steps (UNIX-like shells):

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Upgrade pip:

```bash
python -m pip install --upgrade pip
```


5. Install torch and torch_geometric dependencies (here we use cpu requirements; for CUDA use `requirements-cuda.txt`):
```bash
cd QuantumGravPy
pip install -r requirements-cpu.txt  # or requirements-cuda.txt
python -m pip install -e .
```

If you are on another platform, please refer to the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) and the [PyTorch Geometric installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for guidance on installing compatible versions of those packages.

After installation you can import the package as `import QuantumGrav` from Python code or a REPL.

## Installation as a developer
1. Clone the repository from [here](https://github.com/ssciwr/QuantumGrav).


Do an editable install so local changes in `QuantumGravPy/src/QuantumGrav` are available immediately:

```bash
cd QuantumGravPy
python -m pip install -e .[dev,docs]
```

Run the Python unit tests (with the virtualenv activated):

```bash
cd QuantumGravPy
pytest
```

### Building the documentation locally

The documentation is generated with MkDocs and `mkdocstrings`. MkDocs needs to be able to import the `QuantumGrav` package so either install the package in the same environment (editable install above) or add the `src` path to `PYTHONPATH` before running mkdocs.

Quick serve (from repository root):

```bash
cd QuantumGravPy
# if you didn't install the package, export PYTHONPATH to include the src dir
mkdocs serve
```
Follow the instructions on screen to open the documentation. More on `mkdocs` can be found [here](https://www.mkdocs.org/), and on the `Documenter.jl` package used in the Julia package [here](https://documenter.juliadocs.org/stable/).

Training and evaluation are driven via configs and the `Trainer`. See [Model training](./training_a_model.md) for a minimal example.

## Notes and troubleshooting

- PyTorch and PyTorch-Geometric installation is platform and CUDA-version specific; consult the official installation docs if you encounter wheel or binary compatibility errors.

### Contribution guide
Tbd.
