# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This is a monorepo with two independent but complementary subprojects:

- **[QuantumGrav.jl/](QuantumGrav.jl/)** — Julia package for generating causal-set (cset) datasets and saving them as Zarr archives.
- **[QuantumGravPy/](QuantumGravPy/)** — Python package (`src/QuantumGrav/`) for GNN model training, evaluation, dataset loading, and hyperparameter tuning.

A shared virtual environment lives at `.venv/` in the repo root.

## Setup

### Python
```bash
# Platform-specific PyTorch + PyG first, e.g.:
cd QuantumGravPy
pip install -r requirements-cpu.txt     # or -cuda.txt / -cuda12.8.txt / -rocm.txt / -macos.txt
pip install -e .[dev,docs]              # editable install with test and docs extras
```

### Julia
```julia
# In the Julia REPL, activate a project environment and add QuantumGrav.jl as a dev dependency:
] develop path/to/QuantumGrav/QuantumGrav.jl
```

## Commands

### Python tests
```bash
cd QuantumGravPy
pytest                          # all tests
pytest test/test_gnn_model.py   # single file
pytest -k test_trainer          # match by name
```

### Julia tests
```julia
# Recommended (verbose output):
using TestItemRunner
@run_package_tests

# Or from terminal (activating the QuantumGrav.jl environment):
julia --project=QuantumGrav.jl -e "using Pkg; Pkg.test()"
```

### Linting and formatting
```bash
# Run all pre-commit hooks (ruff format/lint for Python, julia-format for Julia, nbstripout)
pre-commit run --all-files

# Python only
cd QuantumGravPy && ruff format src/ && ruff check src/

# Julia docs (from QuantumGrav.jl/docs/):
julia --color=yes --project make.jl
# With debug output: JULIA_DEBUG=Documenter julia --color=yes --project make.jl

# Python docs (from QuantumGravPy/):
mkdocs serve
```

### Dependencies
We are managing dependencies with a combination of uv and requirements files for specific hardware and pytorch versions. The `requirements-*.txt` files specify the exact versions of PyTorch and PyTorch Geometric for different platforms (CPU, CUDA, ROCm, macOS). The `pyproject.toml` specifies the other dependencies with version ranges.

## Architecture

### QuantumGravPy — Python package (`src/QuantumGrav/`)

**Core abstraction: `Configurable` (`base.py`)** — abstract base class requiring `from_config(cls, config)`. All major classes (`GNNModel`, `Trainer`, `Evaluator`, `DefaultEarlyStopping`) implement this interface, enabling fully YAML-driven instantiation.

**Config system (`config_utils.py`)** — extends PyYAML with custom tags:
- `!pyobject torch_geometric.nn.SAGEConv` — imports a Python type by dotted path at load time; must use full module path, not aliases
- `!sweep` / `!coupled-sweep` — declares a hyperparameter sweep dimension (Cartesian product) or zips values with a sweep target
- `!range` / `!random_uniform` — shorthand for numeric ranges and random samples for Optuna tuning
- `!reference` — late-bound pointer to another config key (resolved after load)

`ConfigHandler` expands a single config with `!sweep` nodes into the full list of per-run configs. `get_loader()` returns the customized `yaml.SafeLoader`.

**Model (`gnn_model.py`)** — `GNNModel` is a `torch.nn.Module` composing:
- `encoder`: any torch module (e.g. a `GNNBlock` stack)
- `pooling_layers` + `aggregate_pooling` **or** `latent_model` (mutually exclusive) — maps node embeddings to a graph embedding
- `graph_features_net` + `aggregate_graph_features` — optional branch for scalar graph-level features, concatenated with the pooled embedding before downstream tasks
- `downstream_tasks` — `ModuleList` of task heads; tasks can be selectively activated via `set_task_active/inactive`

**Built-in model components (`models/`)** — `GNNBlock` (stacked conv layers with skip connections), `GPSTransformer`, `SkipConnection`, `Sequential`, `LinearSequential`.

**Training (`train.py`)** — `Trainer` takes fully-constructed components (model, optimizer, criterion, validator, tester, early stopper) and exposes `run_training(train_loader, val_loader)` and `run_test(test_loader)`. `Snapshot` serializes/deserializes all trainer state for checkpointing and resume. `Trainer.from_config` and `Trainer.load_checkpoint` are the entry points from YAML.

**Data (`dataset_ondisk.py`, `dataloaders.py`)** — `QGDataset` wraps Zarr archives as a PyTorch Geometric `Dataset`. `DataLoaderFactory` / `DistributedDataLoaderFactory` build train/val/test loaders from config. Raw data is stored as Zarr; `load_zarr.py` provides helpers to convert Zarr groups to Python dicts.

**Tuning (`QGTune/tune.py`)** — Optuna-based hyperparameter search driven by `!range` / `!random_uniform` / `!sweep` tags in the config.

### QuantumGrav.jl — Julia package (`src/`)

**Entry point for data generation: `save_data.jl` → `produce_data`** — reads a validated YAML config, builds a `CsetFactory`, generates csets in parallel (via `Distributed`), and writes results to a Zarr archive. `preparation.jl` handles directory setup, source-code copying for reproducibility, and git provenance recording.

**`CsetFactory` (`cset_factories.jl`)** — umbrella callable struct that bundles all cset-type-specific maker structs. Each maker is a callable struct (Julia functor pattern) that draws from parameterised distributions and returns a `(cset, metadata)` tuple. Makers:
- `PolynomialCsetMaker` — Chebyshev polynomial 2D manifold sprinkling
- `LayeredCsetMaker` — random layered causal set
- `RandomCsetMaker` — connectivity-targeted random cset
- `DestroyedCsetMaker` — polynomial cset with edge flips
- `GridCsetMakerPolynomial` — 2D grid-type csets (quadratic / rectangular / rhombic / hexagonal / triangular / oblique)
- `MergedCsetMaker` — polynomial manifold with inserted random KR-order
- `ComplexTopCsetMaker` — polynomial manifold with vertical/finite causality cuts

Every factory struct validates its config section against an embedded `JSONSchema.Schema` on construction.

**Graph representation** — csets are stored as `CausalSets.BitArrayCauset` (from the external `CausalSets.jl` package). `graph_utils.jl` provides adjacency matrix construction and transitive reduction. Data is saved to Zarr via `dict_to_zarr` (`save_data.jl`).

**Config validation** — all Julia constructors call `validate_config(schema, config)` before use. Each struct type owns its own JSON schema constant (e.g. `PolynomialCsetMaker_schema`). The top-level `csetfactory_schema` validates the full factory config.

### Cross-language data flow

```
QuantumGrav.jl: YAML config → CsetFactory → Zarr archive
       ↓
QuantumGravPy: QGDataset(zarr) → DataLoaderFactory → Trainer → checkpoints
```

The Zarr format is the contract between the two subprojects. The Python reader function passed to `QGDataset` decodes the Zarr groups written by Julia.

## Design Principles and how to work with this code base

- **All branching at config load time** — runtime hot paths (training loop, data loading) must remain branch-free. Architecture decisions (which encoder, which pooling, which tasks are active) are resolved during construction from config, not during forward passes.
- **`Configurable` contract** — every user-facing class that can be specified in a YAML file implements `from_config`. The `!pyobject` tag is the mechanism for embedding Python types directly in YAML.
- **Single responsibility** — `Trainer` orchestrates; it does not build models or loaders itself. Construction is delegated to `from_config` class methods. Validation logic lives in JSON schemas, not in constructors.

The code here aims to build on a set of principles which constitute central design goals for the project:
- **Simplicity**: The code should be as simple as possible, avoiding unnecessary abstractions or complex patterns. This makes it easier to understand and modify.
- **Modularity**: The code should be organized into clear, self-contained modules that have well-defined responsibilities. This allows for easier maintenance and the ability to swap out components without affecting the rest of the system.
- **Configurability**: The code should be highly configurable through external YAML files, allowing for easy experimentation with different model architectures, training parameters, and data sources without changing the code itself.
- **Reproducibility**: The use of DVC ensures that experiments are reproducible, with clear tracking of data, code, and metrics. This allows for easy comparison of different runs and configurations.

In particular, we follow the SOLID principles of software design, with a focus on open-closed principles, single responsibility and separation of concerns. We are using clean code principles where appropriate, but not at the cost of over-engineering or unnecessary abstraction. We are also following the principle of YAGNI (You Aren't Gonna Need It) to avoid adding features or abstractions that are not currently needed.

Furthermore, we aim to take decisions and choose paths on the highest level of code organization possible, which means in particular before and outside the hot path, such that the hot paths stay as free as possible of branching points. When reviewing any code, violation of these principles is to be flagged as a code smell and be brought to the users attention.
