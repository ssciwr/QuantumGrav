# Distributed Training with DDP

This guide demonstrates how to use the `TrainerDDP` class for distributed training across multiple NVIDIA GPUs using PyTorch's Distributed Data Parallel (DDP) framework.

## Overview

DDP (Distributed Data Parallel) allows you to train models across multiple GPUs efficiently by:

- **Parallelizing computation** across multiple GPUs
- **Automatically distributing data** so each GPU processes a unique subset
- **Synchronizing gradients** during backpropagation
- **Scaling batch size** as total_batch_size = batch_size × world_size

### Key Concepts

- **World Size**: Total number of processes (GPUs)
- **Rank**: Unique identifier for each process (0 to world_size-1)
- **Master Node**: Coordinates communication between processes
- **DistributedSampler**: Automatically splits data across ranks

## Setup Requirements

This guide assumes:

- Multiple NVIDIA GPUs are available (e.g., GPUs 3, 4, and 5)
- `CUDA_VISIBLE_DEVICES` is set to limit visible GPUs
- PyTorch with CUDA support is installed
- The training script will be launched using `torchrun` or `torch.distributed.launch`

## Running DDP Training

### Option 1: Using `torchrun` (Recommended - PyTorch ≥ 1.10)

```bash
export CUDA_VISIBLE_DEVICES=3,4,5
torchrun --nproc_per_node=3 your_training_script.py
```

### Option 2: Using `torch.distributed.launch` (PyTorch < 1.10)

```bash
export CUDA_VISIBLE_DEVICES=3,4,5
python -m torch.distributed.launch --nproc_per_node=3 your_training_script.py
```

### Option 3: Standalone Script with `torch.multiprocessing`

For local development and testing, you can use `torch.multiprocessing.spawn()` without external launchers.

## Example Training Script

Here's a complete example script using `torch.multiprocessing` for process spawning:

```python
#!/usr/bin/env python3
"""
Standalone DDP training script for QuantumGrav models.
This script demonstrates how to set up and run distributed training on multiple GPUs.

Usage:
    python train_ddp_standalone.py

Or with CUDA_VISIBLE_DEVICES:
    CUDA_VISIBLE_DEVICES=3,4,5 python train_ddp_standalone.py
"""

import os
import sys
from pathlib import Path
import yaml
import logging
from typing import Any
import torch
import torch.multiprocessing as mp

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import QuantumGrav as QG
from QuantumGrav.train_ddp import TrainerDDP, initialize_ddp, cleanup_ddp


def setup_logging(rank: int) -> None:
    """Setup logging for the current rank."""
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[RANK {rank}] %(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def train_ddp(
    rank: int,
    world_size: int,
    config: dict[str, Any],
    master_addr: str = "localhost",
    master_port: str = "12355",
) -> None:
    """
    Main training function to be run in each process.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        config: Training configuration dictionary
        master_addr: Address of the master node
        master_port: Port for master node communication
    """
    setup_logging(rank)

    logger = logging.getLogger(__name__)
    logger.info(f"Process {rank}/{world_size} started")

    try:
        # Initialize distributed training
        initialize_ddp(
            rank=rank,
            worldsize=world_size,
            master_addr=master_addr,
            master_port=master_port,
            backend="nccl" if torch.cuda.is_available() else "gloo",
        )

        # Create DDP trainer
        trainer = TrainerDDP(rank=rank, config=config)

        # Prepare data loaders for distributed training
        if rank == 0:
            logger.info("Preparing data loaders...")

        train_loader, val_loader, test_loader = trainer.prepare_dataloaders()

        if rank == 0:
            logger.info(
                f"Training set size: {len(train_loader.dataset)}, "
                f"Validation set size: {len(val_loader.dataset)}, "
                f"Test set size: {len(test_loader.dataset)}"
            )

        # Run training
        if rank == 0:
            logger.info("Starting distributed training...")

        train_results, val_results = trainer.run_training(train_loader, val_loader)

        # Only rank 0 performs testing
        if rank == 0:
            logger.info("Starting testing phase...")
            test_results = trainer.run_test(test_loader)
            logger.info("Training and testing completed successfully!")

    finally:
        # Always cleanup
        cleanup_ddp()
        logger.info(f"Process {rank} cleanup completed")


def main():
    """
    Main entry point for DDP training.
    Handles process spawning and configuration loading.
    """
    # Configuration
    world_size = 3  # Number of GPUs to use
    config_path = Path(__file__).parent / "configs/train_ddp.yaml"
    master_addr = "localhost"
    master_port = "12355"

    # Load configuration
    if not config_path.exists():
        print(f"Config file not found at {config_path}")
        print("Please create a DDP config file first.")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.load(f, QG.get_loader())

    # Update world_size in config if needed
    if "parallel" not in config:
        print("Config must have 'parallel' section for DDP training")
        sys.exit(1)

    config["parallel"]["world_size"] = world_size

    # Spawn processes
    print(f"Starting DDP training with {world_size} processes")
    mp.spawn(
        train_ddp,
        args=(world_size, config, master_addr, master_port),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
```

## Example Script for torchrun

If you prefer to use `torchrun` or `torch.distributed.launch`, here's a simpler script that relies on environment variables set by the launcher:

```python
#!/usr/bin/env python3
"""
DDP training script for use with torchrun or torch.distributed.launch.

Usage with torchrun:
    export CUDA_VISIBLE_DEVICES=3,4,5
    torchrun --nproc_per_node=3 train_ddp_torchrun.py

Usage with torch.distributed.launch:
    export CUDA_VISIBLE_DEVICES=3,4,5
    python -m torch.distributed.launch --nproc_per_node=3 train_ddp_torchrun.py
"""

import os
import sys
from pathlib import Path
import yaml
import logging
import torch

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import QuantumGrav as QG
from QuantumGrav.train_ddp import TrainerDDP, initialize_ddp, cleanup_ddp


def setup_logging(rank: int) -> None:
    """Setup logging for the current rank."""
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[RANK {rank}] %(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main():
    """
    Main training function.

    When launched with torchrun or torch.distributed.launch, the following
    environment variables are automatically set:
    - RANK: Global rank of this process
    - LOCAL_RANK: Local rank on this node
    - WORLD_SIZE: Total number of processes
    - MASTER_ADDR: Address of master node
    - MASTER_PORT: Port for communication
    """
    # Get DDP settings from environment (set by torchrun/torch.distributed.launch)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    setup_logging(rank)

    logger = logging.getLogger(__name__)
    logger.info(f"Process {rank}/{world_size} started")

    try:
        # Initialize DDP (environment variables are already set by launcher)
        initialize_ddp(
            rank=rank,
            worldsize=world_size,
            backend="nccl" if torch.cuda.is_available() else "gloo",
        )

        # Load configuration
        config_path = Path(__file__).parent / "configs/train_ddp.yaml"

        if not config_path.exists():
            logger.error(f"Config file not found at {config_path}")
            sys.exit(1)

        with open(config_path, "r") as f:
            config = yaml.load(f, QG.get_loader())

        # Validate config
        if "parallel" not in config:
            logger.error("Config must have 'parallel' section for DDP training")
            sys.exit(1)

        # Update world_size in config
        config["parallel"]["world_size"] = world_size

        # Create DDP trainer
        trainer = TrainerDDP(rank=rank, config=config)

        # Prepare data loaders
        if rank == 0:
            logger.info("Preparing data loaders...")

        train_loader, val_loader, test_loader = trainer.prepare_dataloaders()

        if rank == 0:
            logger.info(
                f"Training set size: {len(train_loader.dataset)}, "
                f"Validation set size: {len(val_loader.dataset)}, "
                f"Test set size: {len(test_loader.dataset)}"
            )

        # Run training
        if rank == 0:
            logger.info("Starting distributed training...")

        train_results, val_results = trainer.run_training(train_loader, val_loader)

        # Only rank 0 performs testing
        if rank == 0:
            logger.info("Starting testing phase...")
            test_results = trainer.run_test(test_loader)
            logger.info("Training and testing completed successfully!")

    finally:
        # Cleanup
        cleanup_ddp()
        logger.info(f"Process {rank} cleanup completed")


if __name__ == "__main__":
    main()
```

### Running the torchrun Script

```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=3,4,5

# Run with torchrun (recommended)
torchrun --nproc_per_node=3 train_ddp_torchrun.py

# Or with torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=3 train_ddp_torchrun.py
```

### Key Differences Between torchrun and multiprocessing Scripts

| Aspect | torchrun/launch | multiprocessing |
|--------|-----------------|-----------------|
| Process spawning | Handled by launcher | Manual with `mp.spawn()` |
| Environment variables | Set by launcher | Set manually in script |
| config["parallel"]["world_size"] | From launcher | Must be hardcoded |
| Backend selection | In initialize_ddp() | In initialize_ddp() |
| Easier for production | ✅ Yes | ❌ No |
| Easier for local testing | ❌ No | ✅ Yes |

The torchrun approach is recommended for production because:
- The launcher handles process management
- Environment variables are automatically set
- Better error handling and logging
- Easier to use on clusters

## Configuration for DDP Training

The configuration file needs a `parallel` section with DDP-specific settings:

```yaml
criterion: !pyobject __main__.compute_loss

# REQUIRED: Parallel training configuration for DDP
parallel:
  world_size: 3              # Number of GPUs to use
  rank: null                 # Will be set automatically by launcher
  master_addr: "localhost"   # localhost for single machine, node IP for clusters
  master_port: "12355"       # Unique port for inter-process communication
  output_device: null        # null = automatic, or specify GPU device ID
  find_unused_parameters: false  # Set to true if model has unused parameters

training:
  seed: 42
  device: "cuda"  # Use CUDA for distributed training
  path: /path/to/output/directory
  num_epochs: 50
  batch_size: 64  # Batch size PER GPU (total batch size = batch_size × world_size)
  optimizer_type: !pyobject torch.optim.AdamW
  optimizer_args: []
  optimizer_kwargs:
    lr: 0.001
    weight_decay: 0.00005
  num_workers: 4  # DataLoader worker processes per GPU
  pin_memory: true
  drop_last: true
  prefetch_factor: 2
  checkpoint_at: 10

model:
  # Your model configuration here
  type: "GNNModel"

validation:
  batch_size: 64
  num_workers: 2
  pin_memory: true
  drop_last: false
  validator:
    device: "cuda"

testing:
  batch_size: 64
  num_workers: 2
  pin_memory: true
  drop_last: false
  tester:
    device: "cuda"

early_stopping:
  type: "DefaultEarlyStopping"
  args: []
  kwargs:
    patience: 5
```

### Important Configuration Notes

1. **Batch Size**: The `batch_size` in the config is **per GPU**
   - Total batch size = batch_size × world_size
   - Example: batch_size: 64 with 3 GPUs = 192 total samples per iteration

2. **Device**: Must be set to `"cuda"` for CUDA training, not `"cpu"`

3. **World Size**: Should match the number of GPUs (`--nproc_per_node`)

## Key Differences from Single-GPU Training

| Aspect | Single-GPU | DDP |
|--------|-----------|-----|
| Config section | Not needed | `parallel` required |
| Data distribution | Manual | Automatic via `DistributedSampler` |
| Validation | All ranks | Rank 0 only |
| Checkpointing | All ranks | Rank 0 only |
| Early stopping | Single decision | Broadcast from rank 0 |
| Model wrapping | Direct | Wrapped in `DistributedDataParallel` |

## Rank-0 Only Operations

Some operations should only run on rank 0 (the master process) to avoid redundant computation and file conflicts:

```python
if rank == 0:
    validator.report()        # Report validation metrics
    save_checkpoint()         # Write model to disk
    run_test()               # Test phase
```

The `TrainerDDP` class handles this automatically.

## Troubleshooting

### Port Already in Use

If you get "Address already in use" error:

```python
master_port = "12356"  # Use a different port (must be unique)
```

### CUDA Out of Memory

Remember that the configured batch size is **per GPU**:

```yaml
batch_size: 64  # Each GPU processes 64 samples
# Total: 64 × 3 GPUs = 192 samples per iteration
```

Reduce `batch_size` if you encounter OOM errors.

### Only One GPU is Used

Ensure `CUDA_VISIBLE_DEVICES` is set correctly before running:

```bash
export CUDA_VISIBLE_DEVICES=3,4,5
torchrun --nproc_per_node=3 your_script.py
```

Verify GPU visibility:

```bash
python -c "import torch; print(f'Visible GPUs: {torch.cuda.device_count()}')"
```

### Processes Not Synchronizing

The `TrainerDDP` class includes proper synchronization barriers. If processes hang:

1. Check network connectivity between processes
2. Verify `master_addr` and `master_port` are correct
3. Check logs for errors on individual ranks

### Data Not Distributed Properly

`DistributedSampler` is automatically used in `TrainerDDP.prepare_dataloaders()`. It ensures:

- Each rank gets a unique subset of data
- No data duplication or gaps
- Independent shuffling per rank (with seed control)

## Environment Variables

When using `torchrun` or `torch.distributed.launch`, these are automatically set:

- `RANK`: Global rank of this process (0 to world_size-1)
- `LOCAL_RANK`: Local rank on this node
- `WORLD_SIZE`: Total number of processes
- `MASTER_ADDR`: Address of the master node
- `MASTER_PORT`: Communication port

The `TrainerDDP` class uses these to configure itself.

## Monitoring Training

### GPU Usage

Monitor GPU utilization in real-time:

```bash
watch -n 1 nvidia-smi
```

You should see activity on all 3 GPUs simultaneously during training.

### Log Output

With proper logging configuration, you'll see outputs like:

```
[RANK 0] 2025-01-15 14:23:47 INFO: Training set size: 80,000
[RANK 0] 2025-01-15 14:23:49 INFO: Starting distributed training...
[RANK 0] 2025-01-15 14:24:15 INFO: Validation: loss=0.524 ± 0.012
[RANK 0] 2025-01-15 14:24:15 INFO: Saving checkpoint for model at epoch 0
```

Only rank 0 logs most operations. Other ranks only log warnings and errors.

## Performance Considerations

### Scaling Efficiency

- **Communication overhead**: DDP adds gradient synchronization at each step
- **Batch size scaling**: Larger batch sizes (enabled by DDP) can improve throughput
- **Data loading**: Use appropriate `num_workers` to avoid I/O bottlenecks

### Optimization Tips

1. **Tune batch size**: Larger per-GPU batches often improve throughput
2. **Adjust workers**: Set `num_workers` based on your CPU cores and I/O
3. **Pin memory**: Set `pin_memory: true` for GPU data transfer speed
4. **Prefetch**: Adjust `prefetch_factor` for your data pipeline

## Cluster Training

For multi-node training on clusters:

1. Set `master_addr` to the IP address of the master node (not "localhost")
2. Use the same `master_port` on all nodes
3. Launch with `torchrun` or `torch.distributed.launch` with appropriate `--nnodes` and `--nproc_per_node`

Example for 2 nodes with 4 GPUs each:

```bash
# On master node (IP: 192.168.1.10)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
  --master_addr=192.168.1.10 --master_port=12355 your_script.py

# On worker node (IP: 192.168.1.11)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
  --master_addr=192.168.1.10 --master_port=12355 your_script.py
```

## See Also

- [PyTorch DDP Documentation](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [Training Guide](training_a_model.md)
- [Configuration Reference](../api.md)
