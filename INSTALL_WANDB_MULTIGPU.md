# Installation & Verification Guide

Get wandb + multi-GPU working in 10 minutes.

---

## Step 1: Install Wandb (2 minutes)

### Local Machine or Kaggle Notebook

```bash
pip install wandb
```

Verify:
```bash
wandb --version
# Should show: wandb version X.X.X
```

---

## Step 2: Create Wandb Account (3 minutes)

1. Go to https://wandb.ai
2. Click "Sign up"
3. Fill in email, password (free for research)
4. Verify email
5. Go to Settings → API Keys
6. Copy your API key

---

## Step 3: Login (2 minutes)

### Option A: Local Machine

```bash
wandb login
# Paste your API key
# Should show: Successfully logged in
```

### Option B: Kaggle Notebook

```python
!wandb login --reuse
# If first time, will prompt for API key
# If already logged in locally, uses cached credentials
```

Verify:
```bash
wandb auth default
# Shows: Successfully logged in
```

---

## Step 4: Verify Multi-GPU (2 minutes)

### Check GPU Availability

```python
import torch
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
```

Expected on Kaggle:
```
GPU count: 2
  GPU 0: Tesla T4
  GPU 1: Tesla T4
```

---

## Step 5: Test First Experiment (2 minutes)

### Run a Single Experiment

```bash
cd autoresearch-master

python train.py \
  --phase threshold_sweep \
  --tau_low 0.15 \
  --tau_high 0.40 \
  --wandb-project edge-vlm-thesis \
  --multi-gpu
```

Expected output:
```
======================================================================
AUTORESEARCH EXPERIMENT #1
======================================================================
Phase:     threshold_sweep
Config:    {'tau_low': 0.15, 'tau_high': 0.40, ...}
Output:    /path/to/results_threshold_sweep_default
GPUs:      [0, 1]
Wandb:     ✓ Enabled
Checkpoint: default
======================================================================

Running on (GPUs: 2 available)
---
sce:                  0.8200
semantic_retention:   0.9400
vlm_call_rate:        0.0420
latency_ms:           45.3
gpu_utilization:      0.42
elapsed_seconds:      127.5
false_positives:      2
false_negatives:      0

✓ Checkpoint saved: exp_001_phase_threshold_sweep_checkpoint.pt
✓ Results logged to results.tsv
✓ Wandb sync complete
```

### Check Wandb Dashboard

1. Go to https://wandb.ai/[your-username]/edge-vlm-thesis
2. Click on latest run
3. See metrics, config, GPU info

---

## Step 6: Verify Checkpoints

```bash
python train.py --list-checkpoints

# Output:
# Available checkpoints:
#   exp_001_phase_threshold_sweep_checkpoint.pt:
#     Exp: 1, Phase: threshold_sweep, SCE: 0.8200
#     Machine: your-machine-name, GPUs: 2
```

---

## Troubleshooting

### Issue: "wandb: command not found"

```bash
pip install wandb --upgrade
```

### Issue: "Could not initialize wandb"

```bash
# Make sure you're logged in
wandb login

# Or run in offline mode
python train.py --phase threshold_sweep --wandb-offline
```

### Issue: "CUDA out of memory"

```bash
# Use single GPU instead
python train.py --phase threshold_sweep --gpu-ids 0
```

### Issue: "No GPUs detected"

```bash
# Check NVIDIA drivers
nvidia-smi

# If no output, drivers not installed
# On Kaggle: should work by default
# On local machine: install NVIDIA drivers
```

### Issue: "Checkpoint loading fails"

```bash
# Make sure checkpoint file exists
ls checkpoints/

# If not found, checkpoint not saved (maybe crashed)
# Check results.tsv for status
```

---

## Verification Checklist

Print this and check off each:

- [ ] `pip install wandb` succeeds
- [ ] `wandb --version` shows version number
- [ ] `wandb login` authenticates
- [ ] `nvidia-smi` shows GPU(s)
- [ ] `torch.cuda.device_count()` ≥ 1
- [ ] `python train.py --phase threshold_sweep` runs
- [ ] Experiment completes without error
- [ ] `results.tsv` has new row
- [ ] `checkpoints/exp_001_*.pt` exists
- [ ] Wandb dashboard shows the experiment
- [ ] `python train.py --list-checkpoints` shows checkpoint

**If all 11 items are checked: ✅ You're ready!**

---

## Quick Test Script

Run this Python script to verify everything:

```python
#!/usr/bin/env python3
"""Verify wandb + multi-GPU setup."""

import sys

print("\n" + "="*60)
print("VERIFICATION SCRIPT")
print("="*60)

# Check wandb
print("\n1. Checking wandb...")
try:
    import wandb
    print("   ✓ wandb installed")
except ImportError:
    print("   ✗ wandb NOT installed")
    print("   → Run: pip install wandb")
    sys.exit(1)

# Check torch
print("\n2. Checking PyTorch...")
try:
    import torch
    print(f"   ✓ PyTorch installed ({torch.__version__})")
except ImportError:
    print("   ✗ PyTorch NOT installed")
    sys.exit(1)

# Check GPUs
print("\n3. Checking GPUs...")
num_gpus = torch.cuda.device_count()
print(f"   ✓ Found {num_gpus} GPU(s)")
for i in range(num_gpus):
    print(f"      GPU {i}: {torch.cuda.get_device_name(i)}")

if num_gpus == 0:
    print("   ⚠ WARNING: No GPUs detected")
    print("   → Check NVIDIA drivers: nvidia-smi")

# Check wandb login
print("\n4. Checking wandb login...")
try:
    api = wandb.Api()
    user = api.viewer()
    print(f"   ✓ Logged in as: {user.username}")
except Exception as e:
    print(f"   ✗ Not logged in: {e}")
    print("   → Run: wandb login")
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL CHECKS PASSED - Ready to run AutoResearch!")
print("="*60)
print("\nNext steps:")
print("  1. cd autoresearch-master")
print("  2. python train.py --phase threshold_sweep --multi-gpu --wandb-project edge-vlm-thesis")
print("  3. Monitor at: https://wandb.ai/[your-username]/edge-vlm-thesis")
```

Save as `verify_setup.py` and run:
```bash
python verify_setup.py
```

---

## Next Steps

✅ Installation complete

→ **Read:** `WANDB_QUICKREF.md` for common commands

→ **Run:** Phase 1 threshold sweep

→ **Monitor:** Wandb dashboard in real-time

→ **Resume:** On different machines using checkpoints

---

## Support

If something doesn't work:

1. Check the **Troubleshooting** section above
2. Read `WANDB_MULTIGPU_SETUP.md` for detailed explanations
3. Check `WANDB_QUICKREF.md` for command examples
4. Verify GPU setup: `nvidia-smi`
5. Verify wandb: `wandb auth default`

---

**Your setup is complete. Happy researching! 🚀**
