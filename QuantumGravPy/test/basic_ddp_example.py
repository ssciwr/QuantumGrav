# example by chatgpt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from glob import glob


# ---------------------------
# Custom Dataset: loads each sample from a separate .pt file lazily
# ---------------------------
class GraphDataset(Dataset):
    def __init__(self, data_dir):
        self.paths = sorted(glob(os.path.join(data_dir, "*.pt")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.load(self.paths[idx])  # returns (graph_tensor, label)


# ---------------------------
# Model definition (replace with your real one)
# ---------------------------
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, x):
        return self.net(x)


# ---------------------------
# One training process per GPU
# ---------------------------
def train(rank, world_size, data_dir):
    print(f"[Rank {rank}] Starting training process...")

    # Set environment variables for torch.distributed
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    # Use NCCL for GPU (fastest, but not fork-safe → all init must happen inside this function)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Pin this process to its GPU
    torch.cuda.set_device(rank)

    # ---------------------------
    # Build dataset and sharded DataLoader
    # ---------------------------
    dataset = GraphDataset(data_dir)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        sampler=sampler,
        num_workers=2,  # Don't go too high here
        pin_memory=True,
    )

    # ---------------------------
    # Initialize model and wrap in DDP
    # ---------------------------
    model = DummyModel().to(rank)  # move to this GPU only
    ddp_model = DDP(model, device_ids=[rank])  # wrap with sync hooks

    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(5):
        sampler.set_epoch(epoch)  # ensure different shuffling per epoch
        ddp_model.train()
        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(rank, non_blocking=True)
            y = y.to(rank, non_blocking=True)

            optimizer.zero_grad()
            logits = ddp_model(x)
            loss = loss_fn(logits, y)
            loss.backward()  # DDP syncs gradients here across GPUs
            optimizer.step()

            if rank == 0 and batch_idx % 10 == 0:
                print(f"[Epoch {epoch} | Batch {batch_idx}] Loss: {loss.item():.4f}")

    # ---------------------------
    # Cleanup process group
    # ---------------------------
    dist.destroy_process_group()


# ---------------------------
# Entry point for launching processes
# ---------------------------
def main():
    data_dir = "/path/to/your/pt/files"

    # Safety: only call torch.multiprocessing.spawn from __main__
    world_size = torch.cuda.device_count()

    mp.spawn(
        fn=train,  # the function to run
        args=(world_size, data_dir),  # arguments passed to `train`
        nprocs=world_size,  # how many processes to spawn
        join=True,  # wait for all to finish
    )


if __name__ == "__main__":
    main()
