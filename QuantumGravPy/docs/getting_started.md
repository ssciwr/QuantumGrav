# Getting started
This document explains how to get the project running for development and for usage. It covers the two supported ecosystems used in this repository: the Julia package (data generation and low-level causal-set utilities) and the Python package (model code, training and evaluation).

## Installation

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

3. Install PyTorch (follow the instructions at https://pytorch.org/get-started/locally/ for the correct wheel for your platform and CUDA). Example (CPU-only wheel). 

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

4. For PyTorch Geometric, follow the project documentation for the correct platform-specific wheels: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

5. Install the Python package (recommended editable install for development):

```bash
cd QuantumGravPy
python -m pip install -e .
```

After installation you can import the package as `import QuantumGrav` from Python code or a REPL.

## Installation as a developer
1. Clone the repository from [here](https://github.com/ssciwr/QuantumGrav). 


Do an editable install so local changes in `QuantumGravPy/src/QuantumGrav` are available immediately:

```bash
cd QuantumGravPy
python -m pip install -e .[dev]
```

Run the Python unit tests from the repository root (with the virtualenv activated):

```bash
cd QuantumGravPy
pytest test
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

Training and evaluation scripts live under `QuantumGravPy/src/QuantumGrav/` (`train.py`, `train_ddp.py`) and can be run once dependencies are installed.
See the [training section](./training_a_model.md) for more. 

## Notes and troubleshooting

- PyTorch and PyTorch-Geometric installation is platform and CUDA-version specific; consult the official installation docs if you encounter wheel or binary compatibility errors.

### Contribution guide

Tbd.



