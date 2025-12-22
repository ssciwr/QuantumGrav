# API Reference

## Models

### GNNModel
Top-level model that ties together an encoder, optional graph-feature networks, pooling/aggregation, and downstream tasks. It is fully configurable via a JSON/YAML schema.

::: QuantumGrav.gnn_model
    handler: python
    options:
      show_source: true

### Sequential encoder
Composable GNN encoder built from PyG layers with a declarative layer spec and optional skip connections.

::: QuantumGrav.models.sequential
    handler: python
    options:
      show_source: true

### LinearSequential
Lightweight MLP builder used for graph-feature nets and task heads.

::: QuantumGrav.models.linear_sequential
    handler: python
    options:
      show_source: true

### GNN block (optional)
Block-style GNN backbone. This can also be build with `Sequential`, but gives a useful example and baseline.

::: QuantumGrav.models.gnn_block
    handler: python
    options:
      show_source: true

## Training and Evaluation

### Trainer
Configuration-driven training loop: prepares datasets/dataloaders, initializes model/optimizer/scheduler, runs train/validate/test, and manages checkpoints.

::: QuantumGrav.train
    handler: python
    options:
      show_source: true

### Distributed training (DDP)
Data-parallel trainer built on PyTorch DDP.

::: QuantumGrav.train_ddp
    handler: python
    options:
      show_source: true

### Evaluators
Evaluation framework with `Evaluator`, `Validator`, and `Tester` plus configurable monitor tasks.

::: QuantumGrav.evaluate
    handler: python
    options:
      show_source: true

### Early stopping
Multi-task early-stopping utilities.

::: QuantumGrav.early_stopping
    handler: python
    options:
      show_source: true

## Data

### Dataset base class
Common interface and preprocessing pipeline.

::: QuantumGrav.dataset_base
    handler: python
    options:
      show_source: true

### On-disk dataset
Persistent processed dataset with lazy loading.

::: QuantumGrav.dataset_ondisk
    handler: python
    options:
      show_source: true

### Zarr loaders
Helpers for loading Zarr stores.

::: QuantumGrav.load_zarr
    handler: python
    options:
      show_source: true

## Configuration utilities
YAML helpers for sweeps, references, ranges, and Python objects.

::: QuantumGrav.config_utils
    handler: python
    options:
      show_source: true

## Julia integration
Bridge for user-supplied Julia code.

::: QuantumGrav.julia_worker
    handler: python
    options:
      show_source: true

## Utilities
General utilities used throughout the package.

::: QuantumGrav.utils
    handler: python
    options:
      show_source: true