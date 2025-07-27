import torch.distributed as dist
import torch
from torch_geometric.data import DataLoader, DistributedSampler, Data
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Callable
import os
from . import evaluate as ev


# TODO:
# - add support for monitoring training progress
# - add data parallel testing
# - make sure the evaluate function works with DDP as they should


def initialize(
    rank: int,
    worldsize: int,
    master_addr: str = "localhost",
    master_port: str = "12345",
    backend: str = "nccl",
) -> None:
    """Initialize the distributed process group.

    Args:
        rank (int): The rank of the current process.
        worldsize (int): The total number of processes.
        master_addr (str, optional): The address of the master process. Defaults to "localhost".
        master_port (str, optional): The port of the master process. Defaults to "12345".
        backend (str, optional): The backend to use for distributed training. Defaults to "nccl".

    Raises:
        RuntimeError: If the environment variables MASTER_ADDR and MASTER_PORT are already set.
    """
    if "MASTER_ADDR" in os.environ or "MASTER_PORT" in os.environ:
        raise RuntimeError(
            "Environment variables MASTER_ADDR and MASTER_PORT are already set. Please unset them before initializing."
        )
    torch.cuda.set_device(rank)  # Set the device for this process
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=worldsize)


# FIXME: this is probably not the right abstraction level. probably should be removed
def train_ddp(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    loss_fn: Callable[[torch.Tensor, torch.Tensor | Data], torch.Tensor],
    rank: int,
    worldsize: int,
    output_device: int | None = None,
    hyperparams: dict | None = None,
    ddp_kwargs: dict | None = None,
    train_epoch_kwargs: dict | None = None,
) -> None:
    """Train a PyTorch model using Distributed Data Parallel (DDP). This assumes that the multiprocessing environment has been initialized already.
    Make sure to use `torch.multiprocessing.spawn` to start the training processes, because the `nncl` and `gloo` backends are not fork-safe.
    Args:
        model (torch.nn.Module): The model to train.
        dataset (torch.utils.data.Dataset): The dataset to use for training.
        loss_fn (Callable[[torch.Tensor, torch.Tensor | Data], torch.Tensor]): The loss function.
        rank (int): The rank of the current process.
        worldsize (int): The total number of processes.
        output_device (int | None, optional): The device to output results to. Defaults to None.
        hyperparams (dict, optional): Hyperparameters for training. Defaults to None.
        ddp_kwargs (dict, optional): Additional arguments for DDP. Defaults to None.
        train_epoch_kwargs (dict, optional): Additional arguments for the training epoch. Defaults to None.

    Raises:
        ValueError: If hyperparams are not provided.

    # Example:
    # TODO
    """
    print(f"Process {rank} initialized with world size {worldsize}.")

    if hyperparams is None:
        raise ValueError("hyperparams must be provided.")

    if ddp_kwargs is None:
        ddp_kwargs = {}

    if train_epoch_kwargs is None:
        train_epoch_kwargs = {}

    sampler = DistributedSampler(
        dataset, num_replicas=worldsize, rank=rank, shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=hyperparams["batch_size"],
        num_workers=hyperparams["num_workers"],
        pin_memory=True,
        sampler=sampler,
    )

    # put model onto the appropriate device
    # and set up DistributedDataParallel
    model = model.to(rank)  # move model to this GPU
    ddp_model = DDP(
        model, device_ids=[rank], output_device=output_device, **ddp_kwargs
    )  # wrap model for distributed training

    optimizer = torch.optim.Adam(
        ddp_model.parameters(), lr=hyperparams["learning_rate"]
    )
    for epoch in range(hyperparams["num_epochs"]):
        sampler.set_epoch(epoch)
        ddp_model.train()
        ev.train_epoch(
            model=ddp_model,
            data_loader=dataloader,
            optimizer=optimizer,
            criterion=loss_fn,
            device=rank,
            **train_epoch_kwargs,
        )


def cleanup() -> None:
    """Clean up the distributed process group."""
    dist.destroy_process_group()
