# Quick Reference: DDP Training with TrainerDDP

## TL;DR - How to Run DDP Training

### Setup (Before Running)
```bash
# Set visible GPUs (change to your desired GPU IDs)
export CUDA_VISIBLE_DEVICES=3,4,5

# Check GPU visibility
python -c "import torch; print(torch.cuda.device_count())"  # Should print 3
```

### Option 1: Using torchrun (Recommended)
```bash
torchrun --nproc_per_node=3 train_ddp_script.py
```

### Option 2: Using Standalone Script
```bash
python QuantumGravPy/examples/train_ddp_standalone_example.py
```

### Option 3: Using torch.distributed.launch (Older PyTorch)
```bash
python -m torch.distributed.launch --nproc_per_node=3 train_ddp_script.py
```

---

## What Was Fixed

| Bug | Symptom | Fixed In |
|-----|---------|----------|
| Missing `self.epoch = 0` | `AttributeError: 'TrainerDDP' has no attribute 'epoch'` | `__init__` |
| Wrong device_ids type | `RuntimeError: device_ids can only be None or contain a single device` | `initialize_model()` |
| Non-rank-0 accessing validation data | `KeyError` or `AttributeError` on ranks 1,2,... | `run_training()` |

---

## Configuration Must Have

```yaml
parallel:
  world_size: 3              # GPUs count
  rank: null                 # Auto-set
  master_addr: "localhost"   # localhost on single machine
  master_port: "12355"       # Unique port
```

---

## Key Points to Remember

1. **Batch size is per GPU** → Total = batch_size × world_size
2. **Only rank 0 saves** → Checkpoints, validation, testing
3. **Data split automatically** → `DistributedSampler` does it
4. **Processes must sync** → `dist.barrier()` for synchronization
5. **CUDA_VISIBLE_DEVICES** → Controls which GPUs are visible

---

## Files Created/Modified

✅ Fixed: `src/QuantumGrav/train_ddp.py` (3 bugs fixed)
✅ Created: `docs/examples/train_ddp_example.ipynb` (tutorial)
✅ Created: `configs/train_ddp.yaml` (config template)
✅ Created: `examples/train_ddp_standalone_example.py` (runnable script)
✅ Created: `TRAINERDP_FIXES_SUMMARY.md` (summary)
✅ Created: `TRAINERDP_DETAILED_FIXES.md` (detailed explanation)

---

## When Things Go Wrong

**Port already in use?**
```python
master_port = "12356"  # Use different port
```

**Only 1 GPU detected?**
```bash
# Verify before running
python -c "import torch; print(torch.cuda.device_count())"
# If <3, check: export CUDA_VISIBLE_DEVICES=3,4,5
```

**Rank 1/2 crashing?**
→ This is now fixed! Update `train_ddp.py` with the fixes above.

**Processes hanging?**
→ Missing `dist.barrier()` or deadlock in broadcast. Check network/IPC.

---

## Architecture Overview

```
Master (localhost:12355)
├─ Rank 0 (GPU 3) - Train, Validate, Checkpoint, Test
├─ Rank 1 (GPU 4) - Train
└─ Rank 2 (GPU 5) - Train

Data split:
├─ Rank 0: 0-36,400 samples
├─ Rank 1: 36,400-72,800 samples
└─ Rank 2: 72,800-109,200 samples

Each epoch:
1. Train in parallel on all ranks
2. Validate on rank 0 only
3. Broadcast stop decision from rank 0 to all
4. All continue/stop together
```

---

## Next Steps (Tomorrow with Real GPUs)

1. Run the standalone example:
   ```bash
   CUDA_VISIBLE_DEVICES=3,4,5 python examples/train_ddp_standalone_example.py
   ```

2. Monitor with:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. Check logs for proper rank synchronization

4. Verify checkpoints appear in output directory

---

## Documentation

- **Tutorial**: `docs/examples/train_ddp_example.ipynb`
- **Config**: `configs/train_ddp.yaml`
- **Fixes Summary**: `TRAINERDP_FIXES_SUMMARY.md`
- **Detailed Explanation**: `TRAINERDP_DETAILED_FIXES.md`
- **Runnable Script**: `examples/train_ddp_standalone_example.py`

All set for testing! 🚀
