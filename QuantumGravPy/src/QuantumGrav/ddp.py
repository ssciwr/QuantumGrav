import torch.distributed as dist
import torch
import torch_geometric
from torch.nn.parallel import DistributedDataParallel as DDP
import os


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
    """Initialize the distributed process group. This assumes one process per GPU.

    Args:
        rank (int): The rank of the current process.
        worldsize (int): The total number of processes.
        master_addr (str, optional): The address of the master process. Defaults to "localhost". This needs to be the ip of the master node if you are running on a cluster.
        master_port (str, optional): The port of the master process. Defaults to "12345". Choose a high port if you are running multiple jobs on the same machine to avoid conflicts. If running on a cluster, this should be the port that the master node is listening on.
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


# this function behaves like a factory function, which is why it is capitalized
def DistributedDataLoader(
    dataset: torch.utils.data.Dataset
    | torch_geometric.data.Dataset
    | list[torch_geometric.data.Data],
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
    drop_last: bool = False,
    seed: int = 42,
) -> torch_geometric.data.DataLoader:
    """Create a distributed data loader for training.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to load.
        batch_size (int): The batch size to use.
        num_workers (int, optional): The number of worker processes to use for data loading. Defaults to 0.
        pin_memory (bool, optional): Whether to pin memory for the data loader. Defaults to True.
        rank (int, optional): The rank of the current process. Defaults to 0.
        world_size (int, optional): The total number of processes. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to False.
        seed (int, optional): The random seed for shuffling. Defaults to 42.

    Returns:
        torch_geometric.data.DataLoader: The data loader for the distributed training.
    """

    sampler = torch_geometric.data.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )

    dataloader = torch_geometric.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
    )

    return dataloader, sampler


def DistributedModel(
    model: torch.nn.Module, rank: int, output_device: int | None = None, **ddp_kwargs
) -> torch.nn.Module:
    """Create a distributed data parallel model for training.

    Args:
        model (torch.nn.Module): The model to train.
        rank (int): The rank of the current process.
        output_device (int | None, optional): The device to output results to. Defaults to None.

    Returns:
        torch.nn.Module: The distributed data parallel model.
    """
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=output_device, **ddp_kwargs)
    return ddp_model


def cleanup() -> None:
    """Clean up the distributed process group."""
    dist.destroy_process_group()
