# Wandb Tracking & Multi-GPU Setup

Advanced features for running experiments across machines with full tracking and checkpointing.

---

## Overview

The enhanced `train.py` now supports:

✅ **Weights & Biases (wandb)** — Cloud-based experiment tracking
✅ **Checkpointing** — Save/load experiments to resume across machines  
✅ **Multi-GPU** — Use 2 T4 GPUs on Kaggle simultaneously
✅ **Cross-Machine Resume** — Start on Kaggle Account A, continue on Account B

---

## Part 1: Wandb Setup

### 1.1 Install wandb

```bash
pip install wandb
```

### 1.2 Create Wandb Account

1. Go to https://wandb.ai
2. Sign up (free for research)
3. Create project: `edge-vlm-thesis`

### 1.3 Login

```bash
wandb login
# Paste your API key when prompted
```

Your credentials are saved to `~/.netrc`

### 1.4 Run Experiments with Tracking

```bash
# Basic usage (logs to "edge-vlm-thesis" project)
python train.py --phase threshold_sweep --wandb-project edge-vlm-thesis

# Specify team/entity (if using organization)
python train.py --phase threshold_sweep \
  --wandb-project edge-vlm-thesis \
  --wandb-entity my-team

# Offline mode (sync later when online)
python train.py --phase threshold_sweep --wandb-offline
```

### 1.5 Monitor on Dashboard

1. Go to https://wandb.ai/[your-username]/edge-vlm-thesis
2. Watch experiments in real-time
3. Compare SCE across configurations
4. Download results as CSV

---

## Part 2: Checkpointing

### 2.1 How Checkpointing Works

Each experiment saves a checkpoint:
```
checkpoints/
├── exp_001_phase_threshold_sweep_checkpoint.pt
├── exp_002_phase_threshold_sweep_checkpoint.pt
├── exp_050_phase_ablation_checkpoint.pt
└── ...
```

Checkpoints contain:
- Configuration (τ_low, τ_high, α, β, γ)
- Metrics (SCE, semantic_retention, etc.)
- Status (keep/discard/crash)
- Machine info (node name, GPU count)
- Timestamp

### 2.2 List Checkpoints

```bash
python train.py --list-checkpoints

# Output:
# Available checkpoints:
#   exp_001_phase_threshold_sweep_checkpoint.pt:
#     Exp: 1, Phase: threshold_sweep, SCE: 0.8200
#     Machine: kaggle-001, GPUs: 2
#   exp_002_phase_threshold_sweep_checkpoint.pt:
#     Exp: 2, Phase: threshold_sweep, SCE: 0.7900
#     Machine: kaggle-001, GPUs: 2
```

### 2.3 Resume from Checkpoint

```bash
# On a different machine/Kaggle account:
python train.py --resume-checkpoint checkpoints/exp_050_phase_ablation_checkpoint.pt

# Output:
# ======================================================================
# RESUMING FROM CHECKPOINT: checkpoints/exp_050_phase_ablation_checkpoint.pt
# ======================================================================
# ✓ Loaded checkpoint:
#   Exp ID: 50
#   Phase: ablation
#   SCE: 0.8150
#   Saved on: kaggle-001
#   Original GPUs: 2
```

### 2.4 Custom Checkpoint Directory

```bash
# Save checkpoints to custom location
python train.py --phase threshold_sweep --checkpoint-dir /kaggle/working/my_checkpoints/

# Resume from custom location
python train.py --resume-checkpoint /kaggle/working/my_checkpoints/exp_050_checkpoint.pt
```

---

## Part 3: Multi-GPU Setup

### 3.1 Check GPU Availability

```bash
# In Python or notebook:
import torch
print(f"GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Expected on Kaggle:
# GPUs available: 2
#   GPU 0: Tesla T4
#   GPU 1: Tesla T4
```

### 3.2 Run on Multiple GPUs

#### Option A: Automatic (Recommended)

```bash
# Uses all available GPUs automatically
python train.py --phase threshold_sweep --multi-gpu
```

#### Option B: Specify GPU IDs

```bash
# Use specific GPUs
python train.py --phase threshold_sweep --gpu-ids 0,1
```

#### Option C: Environment Variable

```bash
# Set before running
export CUDA_VISIBLE_DEVICES=0,1
python train.py --phase threshold_sweep
```

### 3.3 Monitor GPU Usage

**During experiment, in another terminal:**

```bash
# Watch GPU utilization in real-time
watch nvidia-smi

# Expected output on Kaggle with 2 T4s:
# GPU 0: 95% utilization, 12.5GB / 16GB memory
# GPU 1: 95% utilization, 12.5GB / 16GB memory
```

### 3.4 Single GPU (Fallback)

If you only have 1 GPU, it works fine:

```bash
python train.py --phase threshold_sweep
# Uses GPU 0 only
```

---

## Part 4: Cross-Machine Workflow

**Scenario**: You have access to 2 Kaggle accounts, each with 2 T4 GPUs. Run 50 experiments continuously across both.

### 4.1 Initial Setup (Both Machines)

**On both Kaggle accounts:**

1. Create same dataset: `edge-vlm-thesis`
2. Create Kaggle notebooks
3. Install dependencies:
   ```bash
   !pip install wandb torch
   ```
4. Login to wandb:
   ```bash
   !wandb login
   # Use same account/API key
   ```

### 4.2 Workflow

**Account A (Kaggle 1):**
```bash
# Run experiments 1-25
python train.py --phase threshold_sweep --wandb-project edge-vlm-thesis --multi-gpu

# After 25 experiments, download checkpoints:
# Go to /kaggle/working/checkpoints/
# Download all *.pt files
```

**Account B (Kaggle 2):**
```bash
# Upload checkpoints from Account A
# Place in /kaggle/working/checkpoints/

# Continue from experiment 25
python train.py --resume-checkpoint checkpoints/exp_025_phase_threshold_sweep_checkpoint.pt --multi-gpu

# Runs experiments 26-50
# Wandb automatically syncs to same project
```

**Back to Account A (or any machine):**
```bash
# View all 50 experiments on wandb dashboard
# All results unified in one project despite 2 machines!
```

### 4.3 Syncing Checkpoints Between Accounts

**Download from Account A:**
```bash
# In Kaggle notebook on Account A:
import shutil
shutil.make_archive('/kaggle/working/checkpoints_backup', 'zip', '/kaggle/working/checkpoints/')
# Download: checkpoints_backup.zip
```

**Upload to Account B:**
```bash
# In Kaggle notebook on Account B:
import shutil
shutil.unpack_archive('/kaggle/input/checkpoints_backup.zip', '/kaggle/working/')
# Now have all checkpoints from Account A
```

**Or via Cloud Storage:**
```bash
# Account A: Upload to Google Drive/OneDrive
# Account B: Download from cloud storage
# Both mount cloud storage automatically on Kaggle
```

---

## Part 5: Kaggle Multi-GPU Notebook Template

Use this notebook structure on Kaggle with 2 T4 GPUs:

```python
# Cell 1: Install & Setup
!pip install -q wandb torch ultralytics opencv-python clip boxmot pyyaml

import sys
sys.path.insert(0, '/kaggle/input/edge-vlm-thesis')
sys.path.insert(0, '/kaggle/working')

print("✓ Dependencies installed")
```

```python
# Cell 2: Check GPUs
import torch
print(f"GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
```

```python
# Cell 3: Wandb Login
!wandb login --reuse
# If this is first time, paste API key
```

```python
# Cell 4: Check Checkpoints
import subprocess
result = subprocess.run(['python', '/kaggle/input/edge-vlm-thesis/autoresearch-master/train.py', 
                        '--list-checkpoints'], 
                       capture_output=True, text=True, cwd='/kaggle/working')
print(result.stdout)
```

```python
# Cell 5: Run Phase 1 (Multi-GPU)
!cd /kaggle/working && \
python /kaggle/input/edge-vlm-thesis/autoresearch-master/train.py \
  --phase threshold_sweep \
  --wandb-project edge-vlm-thesis \
  --multi-gpu \
  --checkpoint-dir /kaggle/working/checkpoints \
  --output-dir /kaggle/working/results

# Monitor in another terminal:
# !nvidia-smi
```

```python
# Cell 6: Download Results & Checkpoints
import shutil
shutil.make_archive('/kaggle/working/results_backup', 'zip', '/kaggle/working/results')
shutil.make_archive('/kaggle/working/checkpoints_backup', 'zip', '/kaggle/working/checkpoints')
print("✓ Download: results_backup.zip and checkpoints_backup.zip")
```

---

## Part 6: Examples

### Example 1: Single Machine, All Experiments

```bash
# Kaggle Account A: Run all 38 experiments
cd autoresearch-master

# Phase 1: 28 threshold experiments
python train.py --phase threshold_sweep --multi-gpu --wandb-project edge-vlm-thesis

# Phase 2: 5 ablations
python train.py --phase ablation --multi-gpu --wandb-project edge-vlm-thesis

# Phase 3: 5 failure scenarios
python train.py --phase failure_analysis --scenario stable --multi-gpu
python train.py --phase failure_analysis --scenario motion --multi-gpu
python train.py --phase failure_analysis --scenario lighting --multi-gpu
python train.py --phase failure_analysis --scenario tracking --multi-gpu
python train.py --phase failure_analysis --scenario transitions --multi-gpu

# All results in wandb dashboard
# All checkpoints in checkpoints/ directory
```

### Example 2: Two Accounts, Split Work

```bash
# Account A: Experiments 1-25
python train.py --phase threshold_sweep --multi-gpu \
  --wandb-project edge-vlm-thesis \
  --checkpoint-dir checkpoints_a

# Account B: Experiments 26-38
# First, copy checkpoints from Account A
python train.py --resume-checkpoint checkpoints_a/exp_025_checkpoint.pt \
  --checkpoint-dir checkpoints_b \
  --wandb-project edge-vlm-thesis

# Wandb unifies both in same project
```

### Example 3: Resume After Interruption

```bash
# Experiment got interrupted at exp_047
# Check what we have
python train.py --list-checkpoints

# Resume from last checkpoint
python train.py --resume-checkpoint checkpoints/exp_047_phase_ablation_checkpoint.pt \
  --multi-gpu \
  --wandb-project edge-vlm-thesis

# Continues as exp_048
```

---

## Part 7: Troubleshooting

### "wandb: no such file or directory"

```bash
pip install wandb
```

### "Could not initialize wandb"

```bash
# Login first
wandb login

# Or run offline
python train.py --phase threshold_sweep --wandb-offline
```

### "CUDA out of memory"

```bash
# Reduce batch size or frame count in pipeline
# Or use single GPU
python train.py --phase threshold_sweep --gpu-ids 0
```

### "Checkpoint loading fails on different GPU setup"

Checkpoints save GPU count but work on any setup:
- Checkpoint saved on 2 GPUs? No problem loading on 1 GPU.
- Checkpoint saved on 1 GPU? Works fine on 2 GPUs.

Just resume normally:
```bash
python train.py --resume-checkpoint checkpoints/exp_025_checkpoint.pt --multi-gpu
```

### "Wandb shows different experiments from two machines"

This is normal! Both log to same project. Wandb dashboard shows all experiments:
```
Project: edge-vlm-thesis
├── Exp 1-25 (Kaggle Account A, Machine: kaggle-001)
├── Exp 26-38 (Kaggle Account B, Machine: kaggle-002)
└── All synced to same project
```

To filter by machine:
```
wandb dashboard → Filters → machine: kaggle-001
```

---

## Part 8: Best Practices

✅ **DO:**
- Login to wandb once per account
- Use `--multi-gpu` on multi-GPU machines
- Save checkpoints regularly
- Use same wandb project across all accounts
- Monitor GPU usage with `nvidia-smi`
- Download checkpoints frequently (backup)

❌ **DON'T:**
- Run two experiments simultaneously on same GPU
- Modify checkpoints manually (they're serialized PyTorch)
- Change experiment configuration mid-phase (use Phase 1 → Phase 2 → Phase 3 order)
- Forget to commit checkpoints to git (keep in checkpoints/ .gitignore'd)

---

## Part 9: Summary

| Feature | Command | Notes |
|---------|---------|-------|
| **Wandb** | `--wandb-project NAME` | Cloud tracking of all experiments |
| **Multi-GPU** | `--multi-gpu` | Uses all available GPUs |
| **Checkpoints** | `--checkpoint-dir DIR` | Save/load experiments |
| **Resume** | `--resume-checkpoint FILE` | Continue on different machine |
| **List** | `--list-checkpoints` | Show all saved experiments |
| **GPU IDs** | `--gpu-ids 0,1` | Specify which GPUs |
| **Offline** | `--wandb-offline` | Sync later when online |

---

## Next Steps

1. **Install wandb**: `pip install wandb`
2. **Create wandb account**: https://wandb.ai
3. **Login locally/Kaggle**: `wandb login`
4. **Run Phase 1**: `python train.py --phase threshold_sweep --multi-gpu --wandb-project edge-vlm-thesis`
5. **Monitor**: https://wandb.ai/[your-username]/edge-vlm-thesis
6. **Scale across machines**: Use checkpoints to resume on different Kaggle accounts

---

**Your research is now cloud-tracked, machine-agnostic, and fully resumable.**
