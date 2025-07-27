import torch.distributed as dist
import os


# we are using the same logic as in jax.sharding --> single function mulitple data
def initialize(
    rank: int,
    worldsize: int,
    master_addr: str = "localhost",
    master_port: str = "12345",
    backend: str = "gloo",
) -> None:
    if "MASTER_ADDR" in os.environ or "MASTER_PORT" in os.environ:
        raise RuntimeError(
            "Environment variables MASTER_ADDR and MASTER_PORT are already set. Please unset them before initializing."
        )

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=worldsize)


def cleanup() -> None:
    dist.destroy_process_group()
