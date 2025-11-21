# Getting Started with HPC Starter Kit

This guide will help you get up and running quickly with the HPC PyTorch Starter Kit.

## What's Included

This starter kit contains everything you need to run PyTorch deep learning workloads on HPC clusters:

### Core Files
- **install.sh** - Automated setup script for Python environment and dependencies
- **pyproject.toml** - Python dependencies managed by UV package manager
- **LICENSE** - Apache 2.0 license
- **.python-version** - Specifies Python 3.11.7
- **.gitignore** - Standard ignores for Python/PyTorch/HPC projects

### PBS Job Scripts (`scripts/`)
- **gpu_job.pbs** - Template for 4-GPU distributed training jobs
- **cpu_job.pbs** - Template for CPU-only jobs

### Example Code (`examples/`)
- **basic_benchmark.py** - GPU/CPU performance benchmarking
- **ddp_cnn.py** - Basic distributed training with PyTorch DDP
- **ddp_cnn_wandb.py** - Full example with Weights & Biases logging

## Quick Start (5 minutes)

### 1. Clone and Setup

```bash
# Clone the repository (replace with your URL)
git clone <your-repo-url>
cd hpc_starter_kit

# Run automated installation
bash install.sh
```

The install script will:
- ✓ Verify Python 3.11.7+
- ✓ Install UV package manager
- ✓ Create virtual environment
- ✓ Install PyTorch and dependencies
- ✓ Test GPU availability
- ✓ Set up HPC aliases

### 2. Test Your Setup

```bash
# Activate environment
source .venv/bin/activate

# Run CPU benchmark
python examples/basic_benchmark.py

# Test distributed training locally (if you have GPUs)
torchrun --standalone --nnodes=1 --nproc_per_node=2 examples/ddp_cnn.py
```

### 3. Submit Your First Job

```bash
# Edit the PBS script to point to your paths
nano scripts/gpu_job.pbs

# Submit to cluster
qsub scripts/gpu_job.pbs

# Monitor job
qstat -u $USER
```

## Understanding the Examples

### basic_benchmark.py
- Tests raw compute performance
- Measures TFLOPS on GPUs and CPUs
- Useful for verifying GPU acceleration works

**When to use:** Hardware validation, performance testing

### ddp_cnn.py
- Minimal distributed training example
- Automatically detects network interfaces
- Trains on random data (no dataset needed)

**When to use:** Learning DDP basics, testing multi-GPU setup

### ddp_cnn_wandb.py
- Production-ready distributed training
- Offline Weights & Biases logging
- Model checkpointing and metrics tracking

**When to use:** Actual training runs, experiment tracking

## Adapting for Your Project

### Step 1: Add Your Dependencies

Edit `pyproject.toml`:
```toml
dependencies = [
    "torch>=2.9.1",
    "your-package>=1.0.0",
    # ... add more
]
```

Then run:
```bash
uv sync
```

### Step 2: Create Your Training Script

Use `examples/ddp_cnn_wandb.py` as a template:
1. Replace `RandomDataset` with your real dataset
2. Replace `SimpleCNN` with your model
3. Update hyperparameters
4. Adjust batch sizes for your GPU memory

### Step 3: Customize PBS Script

Edit `scripts/gpu_job.pbs`:
```bash
# Adjust resources
#PBS -l select=1:ncpus=96:mem=700g:ngpus=4

# Adjust walltime
#PBS -l walltime=24:00:00

# Point to your script
torchrun ... your_training_script.py
```

### Step 4: Submit and Monitor

```bash
qsub scripts/gpu_job.pbs
qstat -u $USER
tail -f pytorch_gpu_job.o<jobid>
```

## Common Modifications

### Change Number of GPUs

In PBS script:
```bash
#PBS -l ngpus=2  # Change to 2, 4, 8, etc.
```

In torchrun command:
```bash
torchrun --nproc_per_node=2  # Match ngpus value
```

### Multi-Node Training

PBS script:
```bash
#PBS -l select=2:ncpus=96:mem=700g:ngpus=4  # 2 nodes = 8 GPUs
```

Torchrun command:
```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  your_script.py
```

### Adjust Resource Requests

Common resource patterns:
```bash
# Small job (1 GPU, 12 hours)
#PBS -l select=1:ncpus=24:mem=180g:ngpus=1
#PBS -l walltime=12:00:00

# Medium job (4 GPUs, 24 hours)
#PBS -l select=1:ncpus=96:mem=700g:ngpus=4
#PBS -l walltime=24:00:00

# Large job (8 GPUs across 2 nodes, 48 hours)
#PBS -l select=2:ncpus=96:mem=700g:ngpus=4
#PBS -l walltime=48:00:00
```

## Troubleshooting

### "CUDA not available"
```bash
# Check GPU detection
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### "Job stuck in queue"
```bash
# Check queue status
qstat -Q

# Check node availability
pbsnodes -a | grep -A 5 "state = free"

# View detailed job info
qstat -f <jobid>
```

### "NCCL communication errors"
```bash
# Enable NCCL debugging in PBS script
export NCCL_DEBUG=INFO

# Check network interfaces
ip addr show | grep -E "ib[0-9]|eth[0-9]"
```

### "Out of memory"
- Reduce batch size in training script
- Enable gradient checkpointing
- Use mixed precision training
- Request more GPU memory in PBS script

## Next Steps

1. **Read the full README.md** for comprehensive documentation
2. **Review example scripts** to understand the patterns
3. **Test locally** before submitting to cluster
4. **Start small** with short walltime jobs
5. **Monitor resources** with `nvidia-smi` and `top`

## Learning Resources

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PBS Professional User Guide](https://www.altair.com/pbs-professional/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

## Getting Help

If you encounter issues:
1. Check the **Troubleshooting** section in README.md
2. Enable verbose logging (`NCCL_DEBUG=INFO`)
3. Test with smaller batch sizes/datasets
4. Review PBS job output files (`.o*` and `.e*` files)

## Best Practices Checklist

- [ ] Test scripts locally before submitting jobs
- [ ] Use appropriate walltime (not too long/short)
- [ ] Save checkpoints regularly
- [ ] Log metrics to track training progress
- [ ] Use W&B offline mode on HPC clusters
- [ ] Monitor GPU utilization during training
- [ ] Clean up large temporary files after jobs
- [ ] Use version control for your training scripts

---

**Ready to start?** Run `bash install.sh` and you'll be training in minutes!
