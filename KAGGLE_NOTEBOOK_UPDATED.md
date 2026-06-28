# Kaggle Notebook: Updated for Infrastructure Fixes

Updated notebook cells with proper environment setup, boxmot pinning, and path configuration.

---

## Key Changes from Original

1. **Cell 1:** Pin `boxmot==10.11.62` explicitly
2. **New Cell 2A:** Set environment variables BEFORE imports
3. **Cell 4:** Add environment verification
4. **Cell 5+:** Updated paths use env config

---

## Updated Notebook Cells

### Cell 1: Setup & Dependencies (UPDATED)

```python
# ============================================
# CELL 1: Setup & Dependencies
# ============================================
# Time: ~90 seconds
# CRITICAL: Pin boxmot version for Python 3.12

print("="*70)
print("KAGGLE AUTORESEARCH - SETUP")
print("="*70)

# Install dependencies with correct boxmot version
!pip install -q wandb torch ultralytics opencv-python clip pyyaml matplotlib seaborn pandas
!pip install -q boxmot==10.11.62  # ← CRITICAL: Pinned version for Python 3.12

# Verify installs
import torch
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

print("\n✓ All dependencies installed")
```

---

### NEW Cell 2A: Environment Configuration (NEW!)

```python
# ============================================
# CELL 2A: Configure Environment Variables
# ============================================
# Time: ~10 seconds
# CRITICAL: Set environment BEFORE any src imports

import os
from pathlib import Path

print("\n" + "="*70)
print("CONFIGURING KAGGLE PATHS")
print("="*70)

# !! CRITICAL: Set these FIRST, before importing src.config !!

# 1. Test video directory
# ← CHANGE THIS to your Kaggle dataset name
TEST_VIDEO_DATASET = "srishanmukhasai-vidoes"  # Update to your dataset name

# Detect the actual path
test_video_paths = list(Path(f"/kaggle/input/{TEST_VIDEO_DATASET}").glob("*.mp4")) + \
                   list(Path(f"/kaggle/input/{TEST_VIDEO_DATASET}").glob("*.avi"))

if test_video_paths:
    actual_path = f"/kaggle/input/{TEST_VIDEO_DATASET}"
else:
    # Try to find any video dataset in /kaggle/input
    for dataset in Path("/kaggle/input").iterdir():
        if dataset.is_dir():
            if list(dataset.glob("*.mp4")) or list(dataset.glob("*.avi")):
                actual_path = str(dataset)
                break
    else:
        actual_path = f"/kaggle/input/{TEST_VIDEO_DATASET}"

os.environ["TEST_VIDEO_DIR"] = actual_path
os.environ["OUTPUT_DIR"] = "/kaggle/working/results"
os.environ["CHECKPOINT_DIR"] = "/kaggle/working/checkpoints"

print(f"\n✓ TEST_VIDEO_DIR = {actual_path}")
print(f"✓ OUTPUT_DIR = /kaggle/working/results")
print(f"✓ CHECKPOINT_DIR = /kaggle/working/checkpoints")

# Create directories
os.makedirs("/kaggle/working/results", exist_ok=True)
os.makedirs("/kaggle/working/checkpoints", exist_ok=True)

print("\n✓ Environment variables configured")
```

---

### Cell 2 (RENAMED from 2): Git Setup & Repository

```python
# ============================================
# CELL 2: Git Setup & Repository
# ============================================
# Time: ~30 seconds

import os
import subprocess
from pathlib import Path

# Configure git
os.system('git config --global user.email "kaggle@research.local"')
os.system('git config --global user.name "Kaggle AutoResearch"')

# Navigate to working directory
os.chdir('/kaggle/working')

# Check if repo exists, if not clone
if not Path('/kaggle/working/edge-vlm-thesis').exists():
    print("Cloning repository...")
    os.system('git clone https://github.com/shanmukhasaicentific/edge-vlm-thesis.git')
    os.chdir('/kaggle/working/edge-vlm-thesis')
else:
    print("✓ Repository already exists")
    os.chdir('/kaggle/working/edge-vlm-thesis')
    # Update to latest
    os.system('git pull origin main')

print(f"\n✓ Working directory: {os.getcwd()}")
print(f"✓ Git configured")
```

---

### Cell 3 (RENAMED from 3): Wandb & Checkpoint Setup

```python
# ============================================
# CELL 3: Wandb & Checkpoint Setup
# ============================================
# Time: ~30 seconds

import wandb
import sys
sys.path.insert(0, '/kaggle/working/edge-vlm-thesis')

# Login to wandb
print("\n✓ Wandb version:", wandb.__version__)
print("✓ Attempting wandb login (will use cached credentials or prompt)...")
os.system('wandb login --reuse')

print("\n✓ Wandb configured")
```

---

### NEW Cell 4: Verify Environment Configuration

```python
# ============================================
# CELL 4: Verify Environment Configuration (NEW!)
# ============================================
# Time: ~30 seconds

from src.config import (
    print_environment_info,
    IS_KAGGLE,
    get_test_video_dir,
    get_output_dir,
    get_checkpoint_dir
)

print_environment_info()

# Verify video directory
video_dir = get_test_video_dir()
print(f"\n✓ Test Video Directory: {video_dir}")

# List videos
import glob
videos = glob.glob(f"{video_dir}/*.mp4") + glob.glob(f"{video_dir}/*.avi")
if videos:
    print(f"\n✓ Videos found ({len(videos)}):")
    for v in videos[:3]:
        size_mb = Path(v).stat().st_size / (1024**2)
        print(f"  - {Path(v).name} ({size_mb:.1f} MB)")
else:
    print(f"\n⚠ WARNING: No videos found in {video_dir}")
    print(f"  Make sure your Kaggle dataset is added to the notebook!")

print(f"\n✓ Environment verification complete")
```

---

### Cell 5 (RENAMED from 4): Test Video Verification (SIMPLIFIED)

```python
# ============================================
# CELL 5: Test Video Verification
# ============================================
# Time: ~10 seconds

from src.config import get_test_video_dir

video_dir = get_test_video_dir()
print(f"\nSearching for test video in: {video_dir}")

videos = []
for ext in ['*.mp4', '*.avi', '*.mov']:
    videos.extend(Path(video_dir).glob(ext))

if videos:
    print(f"\n✓ Test videos found:")
    for v in videos[:5]:
        size_mb = v.stat().st_size / (1024**2)
        print(f"  - {v.name} ({size_mb:.1f} MB)")
    print("\n✓ Video verification passed")
else:
    print(f"\n✗ WARNING: No test video found in {video_dir}")
    print(f"  Please add your Kaggle dataset to this notebook")
```

---

### Cell 6 (RENAMED from 5): Check AutoResearch Files

```python
# ============================================
# CELL 6: Check AutoResearch Files
# ============================================
# Time: ~10 seconds

autosearch_dir = Path('/kaggle/working/edge-vlm-thesis/autoresearch-master')

print(f"\nAutoResearch directory: {autosearch_dir}")
print("\nKey files:")

files_to_check = [
    'train.py',
    'program.md',
    'prepare.py',
    'results.tsv'
]

for f in files_to_check:
    path = autosearch_dir / f
    if path.exists():
        size = path.stat().st_size
        print(f"  ✓ {f} ({size} bytes)")
    else:
        print(f"  ✗ {f} NOT FOUND")

print("\n✓ File check complete")
```

---

### Cell 7 (RENAMED from 6): Phase 0 - Baseline Test (UPDATED)

```python
# ============================================
# CELL 7: Phase 0 - Baseline Test (Optional)
# ============================================
# Time: ~5 minutes
# OPTIONAL: Skip if you want to go straight to Phase 1

import subprocess
import os

os.chdir('/kaggle/working/edge-vlm-thesis/autoresearch-master')

print("\n" + "="*70)
print("PHASE 0: BASELINE TEST")
print("="*70)
print("\nRunning single baseline experiment...")
print("This tests that everything works before running full sweep.\n")

cmd = [
    'python', 'train.py',
    '--phase', 'threshold_sweep',
    '--tau_low', '0.15',
    '--tau_high', '0.40',
    '--multi-gpu',
    '--wandb-project', 'edge-vlm-thesis',
]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
print(result.stdout)

if result.returncode != 0:
    print("\nERROR OUTPUT:")
    print(result.stderr)
    print("\n❌ Baseline test FAILED")
else:
    print("\n✓ Baseline test PASSED")
```

---

### Cell 8+ (RENAMED): Phases 1-3 (Use Original Logic)

The Phase 1, 2, 3 cells remain the same as the original notebook.

---

## Critical Changes Summary

| Aspect | Original | Updated |
|--------|----------|---------|
| **BoxMOT** | Generic pip | `pip install -q boxmot==10.11.62` |
| **Env vars** | Not set | Set BEFORE imports in Cell 2A |
| **Path setup** | Hardcoded | Uses env config |
| **Verification** | None | Added Cell 4 |
| **Video detection** | Manual | Auto-detects Kaggle datasets |
| **Error messages** | Generic | Clear guidance |

---

## How to Update Your Kaggle Notebook

### Option 1: Manual Update (5 minutes)

Edit your existing notebook:

1. **Cell 1:** Change pip install to:
   ```python
   !pip install -q boxmot==10.11.62
   ```

2. **Insert New Cell 2A:** Before git setup
   ```python
   import os
   os.environ["TEST_VIDEO_DIR"] = "/kaggle/input/your-dataset"
   os.environ["OUTPUT_DIR"] = "/kaggle/working/results"
   os.environ["CHECKPOINT_DIR"] = "/kaggle/working/checkpoints"
   ```

3. **Insert New Cell 4:** After git setup
   ```python
   from src.config import print_environment_info
   print_environment_info()
   ```

4. **Rename subsequent cells** to account for new Cell 2A and 4

### Option 2: Use Updated Notebook (Copy-Paste)

Replace your entire notebook with cells shown above.

---

## Environment Variables Explained

```python
import os

# 1. Test video directory (CRITICAL)
# Must point to where your test videos are
os.environ["TEST_VIDEO_DIR"] = "/kaggle/input/srishanmukhasai-vidoes"
# Change "srishanmukhasai-vidoes" to your actual Kaggle dataset name

# 2. Output directory (optional, defaults to /kaggle/working/results)
os.environ["OUTPUT_DIR"] = "/kaggle/working/results"

# 3. Checkpoint directory (optional, defaults to /kaggle/working/checkpoints)
os.environ["CHECKPOINT_DIR"] = "/kaggle/working/checkpoints"
```

---

## Finding Your Video Dataset Name

If you don't know your dataset name:

```python
import os
from pathlib import Path

print("Available Kaggle input datasets:")
for dataset in Path("/kaggle/input").iterdir():
    if dataset.is_dir():
        videos = list(dataset.glob("*.mp4")) + list(dataset.glob("*.avi"))
        if videos:
            print(f"  ✓ {dataset.name} ({len(videos)} videos)")
            print(f"    Path: {dataset}")
```

Run this in a cell to find your dataset, then use that path.

---

## Verification Checklist for Updated Notebook

After updating, verify:

- [ ] Cell 1 installs `boxmot==10.11.62`
- [ ] Cell 2A sets `os.environ["TEST_VIDEO_DIR"]` to your dataset
- [ ] Cell 4 prints environment info without errors
- [ ] Cell 5 finds your test video
- [ ] Cell 6 finds AutoResearch files
- [ ] Cell 7 baseline test runs (or can be skipped)
- [ ] Cell 8+ Phase 1-3 run as before

---

## Summary: What Changed and Why

| Change | Why | Impact |
|--------|-----|--------|
| Pin boxmot==10.11.62 | Python 3.12 compatibility | No import errors |
| Set env vars early | Config module needs them | Paths auto-detect |
| Add verification cell | Catch setup issues early | Fast debugging |
| Auto-detect videos | Different dataset names | Works on any Kaggle setup |
| Use src.config | Portable paths | Works locally too |

---

## Running Your Updated Notebook

**Entire notebook flow:**

1. Cell 1: Install deps (with pinned boxmot)
2. Cell 2A: Set environment variables (NEW)
3. Cell 2: Git setup
4. Cell 3: Wandb login
5. Cell 4: Verify environment (NEW)
6. Cell 5: Verify video
7. Cell 6: Verify files
8. Cell 7: Baseline test (optional)
9. Cell 8+: Phase 1-3 experiments

**Time:** ~5-6 hours for all experiments

---

## If You Don't Update the Notebook

The original notebook will:
- ❌ Fail when importing src.config (no env vars set)
- ❌ Use hardcoded paths (may not work on Kaggle)
- ❌ Lose benefits of path auto-detection

**Recommendation:** Update the notebook — it's only 2-3 new cells + small changes to existing cells.

---

**Updated notebook is ready to use on Kaggle with all infrastructure improvements.** ✅
