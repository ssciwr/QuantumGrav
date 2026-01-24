#!/usr/bin/env python3
"""
Standalone DDP training script for QuantumGrav models.

This script demonstrates how to set up and run distributed training on multiple GPUs
using torch.multiprocessing for process spawning.

Usage:
    python train_ddp_standalone_example.py

Or with specific GPUs:
    CUDA_VISIBLE_DEVICES=3,4,5 python train_ddp_standalone_example.py

This example uses torch.multiprocessing to spawn processes, which is simpler than using
the torch.distributed.launch or torchrun launchers, but the latter are more suitable
for production/cluster setups.
"""

import os
import sys
from pathlib import Path
import yaml
import logging
from typing import Any
import torch
import torch.multiprocessing as mp

# Add the src directory to Python path if needed
# Adjust this path based on your project structure
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent  # Assumes script is in QuantumGravPy/
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

import QuantumGrav as QG
from QuantumGrav.train_ddp import TrainerDDP, initialize_ddp, cleanup_ddp


def setup_logging(rank: int) -> None:
    """
    Setup logging for the current rank.

    Only rank 0 logs at INFO level; others log at WARNING to reduce verbosity.

    Args:
        rank: Process rank (0 to world_size-1)
    """
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

    This function runs in each spawned process and handles:
    1. DDP initialization
    2. Trainer creation
    3. Data preparation
    4. Training loop
    5. Testing (rank 0 only)
    6. Cleanup

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
        # ===== Step 1: Initialize Distributed Training =====
        logger.info(f"Initializing DDP for rank {rank}...")
        initialize_ddp(
            rank=rank,
            worldsize=world_size,
            master_addr=master_addr,
            master_port=master_port,
            # Use NCCL backend for CUDA, or GLOO for CPU-only training
            backend="nccl" if torch.cuda.is_available() else "gloo",
        )
        logger.info(f"DDP initialization complete for rank {rank}")

        # ===== Step 2: Create DDP Trainer =====
        logger.info(f"Creating TrainerDDP for rank {rank}...")
        trainer = TrainerDDP(rank=rank, config=config)

        # ===== Step 3: Prepare Data Loaders for Distributed Training =====
        # The DistributedSampler in prepare_dataloaders() automatically
        # distributes the dataset across all processes such that:
        # - Each rank gets a unique subset of the data
        # - No data is duplicated or missed
        # - Training data is shuffled independently per rank
        if rank == 0:
            logger.info("Preparing data loaders...")

        train_loader, val_loader, test_loader = trainer.prepare_dataloaders()

        if rank == 0:
            logger.info(
                f"Data split complete:\n"
                f"  Training samples:   {len(train_loader.dataset):,}\n"
                f"  Validation samples: {len(val_loader.dataset):,}\n"
                f"  Test samples:       {len(test_loader.dataset):,}\n"
                f"  Total batch size:   {trainer.config['training']['batch_size'] * world_size} "
                f"({trainer.config['training']['batch_size']} per GPU × {world_size} GPUs)"
            )

        # ===== Step 4: Run Training Loop =====
        if rank == 0:
            logger.info("Starting distributed training...")

        train_results, val_results = trainer.run_training(train_loader, val_loader)

        # ===== Step 5: Run Testing (Rank 0 Only) =====
        # Testing is only performed on rank 0 to avoid redundant computation
        # and to ensure a single, consistent test report
        if rank == 0:
            logger.info("Starting testing phase...")
            test_results = trainer.run_test(test_loader)
            logger.info("Training and testing completed successfully!")

        # Ensure all processes wait before cleanup
        torch.distributed.barrier()

    except Exception as e:
        logger.error(f"Error in training process {rank}: {e}", exc_info=True)
        raise
    finally:
        # ===== Step 6: Cleanup Distributed Training =====
        logger.info(f"Cleaning up rank {rank}...")
        cleanup_ddp()
        logger.info(f"Process {rank} cleanup completed")


def main():
    """
    Main entry point for the DDP training script.

    This function:
    1. Loads the configuration
    2. Sets up the distributed training environment
    3. Spawns worker processes using torch.multiprocessing
    4. Waits for all processes to complete
    """
    # ===== Configuration =====
    world_size = 3  # Number of GPUs to use
    config_path = Path(__file__).parent / "configs/train_ddp.yaml"
    master_addr = "localhost"
    master_port = "12355"

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info(f"DDP Training with {world_size} processes")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    # ===== Load Configuration =====
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        logger.error("Please create a DDP config file or adjust the path in this script.")
        sys.exit(1)

    logger.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.load(f, QG.get_loader())

    # ===== Validate Configuration =====
    if "parallel" not in config:
        logger.error("Config must have 'parallel' section for DDP training")
        logger.error("Add this to your config file:")
        logger.error("""
parallel:
  world_size: 3
  rank: null
  master_addr: "localhost"
  master_port: "12355"
  output_device: null
  find_unused_parameters: false
        """)
        sys.exit(1)

    # Update world_size in config to match script setting
    config["parallel"]["world_size"] = world_size

    # ===== Spawn Processes =====
    logger.info(f"Spawning {world_size} processes for DDP training...")
    logger.info(f"Master address: {master_addr}, Master port: {master_port}")

    # torch.multiprocessing.spawn() spawns 'nprocs' processes, each calling
    # train_ddp() with rank 0, 1, 2, ..., nprocs-1
    mp.spawn(
        train_ddp,
        args=(world_size, config, master_addr, master_port),
        nprocs=world_size,
        join=True,  # Wait for all processes to complete
    )

    logger.info("All training processes completed!")


if __name__ == "__main__":
    # This is important for Windows compatibility with multiprocessing
    mp.set_start_method('spawn', force=True)
    main()
