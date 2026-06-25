# Wandb + Multi-GPU Quick Reference

Copy-paste commands for common workflows.

---

## 1️⃣ First-Time Setup (5 minutes)

```bash
# Install wandb
pip install wandb

# Login (do this once per machine/account)
wandb login
# Paste your API key from https://wandb.ai

# Verify
wandb --version
```

---

## 2️⃣ Run Single Experiment with Tracking

### Basic (Single GPU)
```bash
cd autoresearch-master
python train.py --phase threshold_sweep --wandb-project edge-vlm-thesis
```

### With Multi-GPU
```bash
python train.py --phase threshold_sweep --wandb-project edge-vlm-thesis --multi-gpu
```

### Check GPU Usage
```bash
nvidia-smi
# Expect: Both GPU 0 and GPU 1 showing ~90%+ utilization
```

---

## 3️⃣ Check Checkpoints

```bash
cd autoresearch-master
python train.py --list-checkpoints

# Output shows all saved experiments
# Copy the checkpoint filename you want
```

---

## 4️⃣ Resume on Different Machine

**On Kaggle Account A, after 25 experiments:**
```bash
# Get checkpoint filename
python train.py --list-checkpoints

# Download results (in Kaggle: click Download button)
# Download checkpoints folder (in Kaggle: click Download button)
```

**On Kaggle Account B:**
```bash
# Upload checkpoints to new account's /kaggle/working/checkpoints/

# Resume from last experiment
python train.py --resume-checkpoint checkpoints/exp_025_phase_threshold_sweep_checkpoint.pt \
  --wandb-project edge-vlm-thesis \
  --multi-gpu

# Continues as exp_026 automatically
# Wandb syncs to SAME project as Account A
```

---

## 5️⃣ Full Phase 1 on Kaggle (Multi-GPU)

Paste this into Kaggle notebook cells:

```python
# Cell 1: Install & Setup
!pip install -q wandb torch ultralytics opencv-python clip boxmot pyyaml
!wandb login --reuse
import torch; print(f"GPUs: {torch.cuda.device_count()}")
```

```python
# Cell 2: Run Phase 1
!cd /kaggle/working && \
  python /kaggle/input/edge-vlm-thesis/autoresearch-master/train.py \
    --phase threshold_sweep \
    --wandb-project edge-vlm-thesis \
    --multi-gpu \
    --checkpoint-dir /kaggle/working/checkpoints

# Monitor in browser: https://wandb.ai/[your-username]/edge-vlm-thesis
```

```python
# Cell 3: Download Results
import shutil
shutil.make_archive('/kaggle/working/results_backup', 'zip', '/kaggle/working/results')
shutil.make_archive('/kaggle/working/checkpoints_backup', 'zip', '/kaggle/working/checkpoints')
# Download both .zip files from Output panel
```

---

## 6️⃣ Monitor Experiments in Real-Time

1. Go to: https://wandb.ai/[your-username]/edge-vlm-thesis
2. Watch experiments appear live
3. See GPU usage, metrics, configs
4. Compare SCE across configs

---

## 7️⃣ Two-Account Parallel Workflow

**Account A:**
```bash
# Experiments 1-25
python train.py --phase threshold_sweep --multi-gpu --wandb-project edge-vlm-thesis
```

**Account B:**
```bash
# Get checkpoints from Account A (download .zip)
# Extract to /kaggle/working/checkpoints/

# Continue from exp 25
python train.py --resume-checkpoint checkpoints/exp_025_phase_threshold_sweep_checkpoint.pt \
  --multi-gpu \
  --wandb-project edge-vlm-thesis

# Runs experiments 26-50
# Wandb shows all 50 in one project!
```

---

## 8️⃣ Kaggle Notebook Workflow (Full)

**Complete notebook for end-to-end thesis:**

```python
# ============================================
# Cell 1: Setup
# ============================================
!pip install -q wandb torch ultralytics opencv-python clip boxmot pyyaml
!wandb login --reuse

import sys
sys.path.insert(0, '/kaggle/input/edge-vlm-thesis')

import torch
print(f"✓ GPUs available: {torch.cuda.device_count()}")
```

```python
# ============================================
# Cell 2: Phase 1 - Threshold Sweep (3 hours)
# ============================================
!cd /kaggle/working && \
  timeout 12000 python /kaggle/input/edge-vlm-thesis/autoresearch-master/train.py \
    --phase threshold_sweep \
    --wandb-project edge-vlm-thesis \
    --multi-gpu \
    --checkpoint-dir /kaggle/working/checkpoints
```

```python
# ============================================
# Cell 3: Analyze Phase 1
# ============================================
import pandas as pd
results = pd.read_csv('/kaggle/working/autoresearch-master/results.tsv', sep='\t')

# Best SCE from Phase 1
best = results.loc[results['sce'].idxmax()]
print(f"Best config from Phase 1:")
print(f"  SCE: {best['sce']:.4f}")
print(f"  Retention: {best['semantic_retention']:.4f}")
print(f"  Call rate: {best['vlm_call_rate']:.4f}")

# Extract best tau_low, tau_high from description
print(f"  Description: {best['description']}")
```

```python
# ============================================
# Cell 4: Phase 2 - Ablations (30 min)
# ============================================
# Extract tau_low, tau_high from Phase 1 best result

!cd /kaggle/working && \
  timeout 3600 python /kaggle/input/edge-vlm-thesis/autoresearch-master/train.py \
    --phase ablation \
    --tau_low 0.15 \
    --tau_high 0.40 \
    --wandb-project edge-vlm-thesis \
    --multi-gpu \
    --checkpoint-dir /kaggle/working/checkpoints
```

```python
# ============================================
# Cell 5: Phase 3 - Failure Analysis (30 min)
# ============================================
scenarios = ['stable', 'motion', 'lighting', 'tracking', 'transitions']

for scenario in scenarios:
    print(f"\nRunning {scenario}...")
    import subprocess
    result = subprocess.run([
        'python', '/kaggle/input/edge-vlm-thesis/autoresearch-master/train.py',
        '--phase', 'failure_analysis',
        '--scenario', scenario,
        '--tau_low', '0.15',
        '--tau_high', '0.40',
        '--wandb-project', 'edge-vlm-thesis',
        '--multi-gpu',
        '--checkpoint-dir', '/kaggle/working/checkpoints'
    ], cwd='/kaggle/working', capture_output=True, text=True, timeout=1200)
    
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
```

```python
# ============================================
# Cell 6: Final Results
# ============================================
results = pd.read_csv('/kaggle/working/autoresearch-master/results.tsv', sep='\t')

print(f"\nTotal experiments: {len(results)}")
print(f"Best SCE: {results['sce'].max():.4f}")
print(f"Phases: {results['description'].str.split(':').str[0].unique()}")

# Export for thesis
results.to_csv('/kaggle/working/thesis_results.csv', index=False)
print("\n✓ Exported: thesis_results.csv")
```

```python
# ============================================
# Cell 7: Download All Results
# ============================================
import shutil

# Backup everything
shutil.make_archive('/kaggle/working/thesis_complete', 'zip', '/kaggle/working', 
                    include=['checkpoints', 'results', 'autoresearch-master/results.tsv'])
print("✓ Download: thesis_complete.zip")

# Also download individual result CSVs
shutil.copy('/kaggle/working/thesis_results.csv', '/kaggle/working/thesis_results_for_download.csv')
print("✓ Download: thesis_results_for_download.csv")
```

---

## 9️⃣ Resume After Interruption

```bash
# Get latest checkpoint
python train.py --list-checkpoints
# Find the highest exp_N checkpoint

# Resume
python train.py --resume-checkpoint checkpoints/exp_048_phase_ablation_checkpoint.pt \
  --wandb-project edge-vlm-thesis \
  --multi-gpu

# Continues as exp_049
```

---

## 🔟 Wandb Dashboard Tips

**On wandb.ai:**

1. **Filter by phase:**
   - Add filter: `phase = "threshold_sweep"`
   - Then: `phase = "ablation"`

2. **Sort by metric:**
   - Click "sce" column header
   - See best configs at top

3. **Export data:**
   - Click hamburger menu → Export
   - Download as CSV for thesis plots

4. **Compare configurations:**
   - Select 2+ runs
   - Click "Compare"
   - See side-by-side metrics

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `wandb: command not found` | `pip install wandb` |
| `CUDA out of memory` | Use `--gpu-ids 0` (single GPU) |
| `Permission denied` checkpoint | `chmod +x checkpoints/` |
| Wandb offline? | `--wandb-offline` then sync later |
| Resume shows wrong exp ID | Expected - each machine counts separately, wandb unifies |
| Different GPU count? | Still works - checkpoint adapts to available GPUs |

---

## Checklists

### ✅ Before Running Phase 1
- [ ] `wandb login` completed
- [ ] Video in `experiments/test/`
- [ ] GPU check: `nvidia-smi`
- [ ] Project created: `edge-vlm-thesis`

### ✅ During Experiments
- [ ] Monitor wandb dashboard live
- [ ] Watch `nvidia-smi` (both GPUs should be >80%)
- [ ] Note any crashes or anomalies

### ✅ After Each Phase
- [ ] Download checkpoints (backup)
- [ ] Download results CSV
- [ ] Analyze best config
- [ ] Plan next phase

### ✅ End of All Experiments
- [ ] All 38 experiments logged to wandb
- [ ] All checkpoints backed up
- [ ] results.tsv downloaded
- [ ] Ready to write Section 4 of thesis

---

## One-Liner Cheat Sheet

```bash
# Check setup
python train.py --list-checkpoints

# Run with everything enabled
python train.py --phase threshold_sweep --multi-gpu --wandb-project edge-vlm-thesis

# Resume on new machine
python train.py --resume-checkpoint checkpoints/exp_025_checkpoint.pt --multi-gpu --wandb-project edge-vlm-thesis

# Single GPU fallback
python train.py --phase ablation --gpu-ids 0 --wandb-project edge-vlm-thesis

# Monitor
watch nvidia-smi
```

---

**You're ready for production-scale thesis research.**

Run experiments on 2 Kaggle accounts simultaneously.
Track everything on wandb.
Resume anywhere, anytime.
