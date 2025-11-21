# HPC PyTorch Starter Kit

A production-ready starter kit for running PyTorch deep learning workloads on HPC clusters with PBS job scheduling. This kit includes automated setup, multi-GPU training examples, and best practices for distributed training.

## Features

- Automated Python 3.11.7+ environment setup with UV package manager
- GPU/CUDA verification and diagnostics
- Multi-GPU distributed training with PyTorch DDP (DistributedDataParallel)
- PBS job submission templates for CPU and GPU clusters
- Weights & Biases integration (offline mode for HPC environments)
- NCCL configuration for InfiniBand networks
- Benchmark scripts for performance testing
- HPC workflow management aliases

## Prerequisites

- Python 3.11.7 or higher
- Access to HPC cluster with PBS job scheduler
- CUDA-capable GPU(s) (for GPU training)
- InfiniBand network (optional, for optimal multi-GPU performance)
- Dependencies: `curl`, `git`

## Quick Start

### 1. Installation

Run the automated installation script:

```bash
# Clone this repository
git clone <your-repo-url>
cd hpc_starter_kit

# Make installation script executable
chmod +x install.sh

# Run installation
bash install.sh
```

The installation script will:
- Check Python version (requires 3.11.7+)
- Install UV package manager (if not already installed)
- Install Python 3.11.7 via UV (if not available)
- Sync all project dependencies using `uv sync --python 3.11.7`
- Create and configure virtual environment at `.venv`
- Verify PyTorch installation with comprehensive GPU/CUDA diagnostics
- Run a quick GPU computation test
- Set up HPC job management aliases

### 2. GPU/CUDA Verification

The installation script provides detailed diagnostics:
- PyTorch version and configuration
- CUDA availability and version
- cuDNN version and status
- GPU information (name, memory, compute capability)
- Functional GPU computation test

### 3. Submit Your First Job

**GPU Job (4 GPUs):**
```bash
qsub scripts/gpu_job.pbs
```

**CPU Job:**
```bash
qsub scripts/cpu_job.pbs
```

**Monitor job status:**
```bash
qstat -u $USER
```

## Project Structure

```
hpc_starter_kit/
├── install.sh                      # Automated installation script
├── scripts/
│   ├── gpu_job.pbs                 # PBS script for GPU training (4 GPUs)
│   └── cpu_job.pbs                 # PBS script for CPU benchmarking
├── examples/
│   ├── basic_benchmark.py          # GPU/CPU performance benchmark
│   ├── ddp_cnn.py                  # Distributed training example (basic)
│   └── ddp_cnn_wandb.py            # Distributed training with W&B logging
├── pyproject.toml                  # Python dependencies
└── README.md                       # This file
```

## Example Scripts

### 1. Basic Benchmark ([examples/basic_benchmark.py](examples/basic_benchmark.py))

Tests GPU and CPU performance with matrix multiplication:
- Individual GPU benchmarks
- Parallel multi-GPU execution
- CPU benchmark with configurable threads
- TFLOPS measurements

**Run locally:**
```bash
source .venv/bin/activate
python examples/basic_benchmark.py
```

### 2. Distributed Training - Basic ([examples/ddp_cnn.py](examples/ddp_cnn.py))

Simple CNN training with PyTorch DistributedDataParallel:
- Random dataset for testing
- SimpleCNN architecture
- Automatic network interface detection
- NCCL configuration for InfiniBand

**Run with torchrun:**
```bash
source .venv/bin/activate
torchrun --standalone --nnodes=1 --nproc_per_node=4 examples/ddp_cnn.py
```

### 3. Distributed Training - with W&B ([examples/ddp_cnn_wandb.py](examples/ddp_cnn_wandb.py))

Next, log in and paste your API key when prompted.

```bash
wandb login
```
And copy past your wandb API key for logging in to the wandb library.





Full-featured distributed training with experiment tracking:
- Weights & Biases integration (offline mode)
- Training metrics logging (loss, accuracy)
- Model checkpointing
- Hyperparameter tracking

**Features:**
- Offline W&B mode (no internet required on HPC)
- Logs saved locally for later syncing
- Model artifact logging
- GPU metrics tracking



## PBS Job Templates

### GPU Job Configuration ([scripts/gpu_job.pbs](scripts/gpu_job.pbs))

```bash
#PBS -l select=1:ncpus=96:mem=700g:ngpus=4
#PBS -q gpu4_std
```

**Configured for:**
- 4 GPUs
- 96 CPU cores
- 700GB RAM
- 1.5 hour walltime
- InfiniBand network optimization

### CPU Job Configuration ([scripts/cpu_job.pbs](scripts/cpu_job.pbs))

```bash
#PBS -l select=1:ncpus=192:mem=700g
#PBS -q cpu_std
```

**Configured for:**
- 192 CPU cores
- 700GB RAM
- 1.5 hour walltime

## Customization Guide

### Modify PBS Scripts

Edit the PBS job scripts to match your cluster's configuration:

1. **Walltime:** Adjust `#PBS -l walltime=HH:MM:SS`
2. **Resources:** Change `ncpus`, `mem`, `ngpus` values
3. **Queue:** Update `#PBS -q <queue_name>`
4. **Script path:** Point to your training script

### Adapt for Your Project

1. **Replace example scripts** with your own training code
2. **Update pyproject.toml** with your dependencies
3. **Modify NCCL settings** for your network topology
4. **Adjust batch sizes** in training scripts

### Configure Weights & Biases

In [examples/ddp_cnn_wandb.py](examples/ddp_cnn_wandb.py#L158-L161):

```python
use_wandb = True  # Set to False to disable
wandb_project = "your-project-name"
wandb_run_name = "experiment-name"
wandb_dir = "/path/to/wandb/logs"
```

**Sync logs later:**
```bash
wandb sync /path/to/wandb/logs/wandb/latest-run
```

## HPC Workflow

### Common Commands

```bash
# Submit a job
qsub scripts/gpu_job.pbs

# Check job status
qstat -u $USER

# Watch job status (auto-refresh)
watch -n 5 "qstat -u $USER"

# View job output in real-time
tail -f <job_name>.o<jobid>

# Cancel a job
qdel <jobid>
```

### Job Status Codes

- `Q` - Job is queued and waiting to run
- `R` - Job is running
- `H` - Job is held
- `E` - Job is exiting after completion
- `C` - Job is completed

### Useful Aliases

Add to your `~/.bashrc` (or installed automatically by `install.sh`):

```bash
# Job monitoring
alias qstat='qstat -u $USER -aw'
alias qwatch='watch -n 1 qstat -u $USER'

# Log viewing
alias qtail='tail -f'
alias qless='less'
```

Activate with: `source ~/.bashrc`

## Dependencies

Core dependencies (auto-installed via `uv sync`):

- **PyTorch** >= 2.9.1 (deep learning framework)
- **NumPy** >= 2.3.4 (numerical computing)
- **Loguru** >= 0.7.3 (logging)
- **wandb** >= 0.23.0 (experiment tracking)

Optional but recommended:
- **fastparquet** >= 2024.11.0 (parquet file support)
- **seaborn** >= 0.13.2 (visualization)

All dependencies are managed in [pyproject.toml](pyproject.toml).

## Troubleshooting

### UV sync fails

```bash
# Update UV package manager
uv self update  # If installed via standalone installer

# Or if installed via pip/conda
pip install --upgrade uv
# or
conda update uv

# Then sync again
uv sync
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

Apache License 2.0