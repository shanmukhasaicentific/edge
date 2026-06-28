# Kaggle Notebook Update Guide: Quick Reference

Exact changes needed for the Kaggle AutoResearch notebook.

---

## 📋 Quick Summary

**3 Changes Required:**
1. **Cell 1:** Pin boxmot version
2. **Insert Cell 2A:** Set environment variables (NEW)
3. **Insert Cell 4:** Verify setup (NEW)

**Time to update:** 5-10 minutes

---

## 🔧 Change 1: Cell 1 (Dependencies)

### Original Cell 1
```python
!pip install -q wandb torch ultralytics opencv-python clip boxmot pyyaml matplotlib seaborn pandas
```

### Updated Cell 1
```python
!pip install -q wandb torch ultralytics opencv-python clip pyyaml matplotlib seaborn pandas
!pip install -q boxmot==10.11.62  # ← CRITICAL: Pin version for Python 3.12
```

**Why:** BoxMOT v21 breaks the ByteTrack API. v10.11.62 is the last stable version.

---

## ➕ Change 2: INSERT New Cell 2A (After Cell 2)

### Insert AFTER original Cell 2 (Git Setup)

**New Cell 2A: Environment Configuration**

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

# !! CRITICAL: Set these FIRST !!

# Test video directory
# ← CHANGE THIS to your Kaggle dataset name
TEST_VIDEO_DATASET = "srishanmukhasai-vidoes"  # Update to your dataset name

# Auto-detect the actual path
test_video_paths = list(Path(f"/kaggle/input/{TEST_VIDEO_DATASET}").glob("*.mp4")) + \
                   list(Path(f"/kaggle/input/{TEST_VIDEO_DATASET}").glob("*.avi"))

if test_video_paths:
    actual_path = f"/kaggle/input/{TEST_VIDEO_DATASET}"
else:
    # Try to find any video dataset
    for dataset in Path("/kaggle/input").iterdir():
        if dataset.is_dir():
            if list(dataset.glob("*.mp4")) or list(dataset.glob("*.avi")):
                actual_path = str(dataset)
                break
    else:
        actual_path = f"/kaggle/input/{TEST_VIDEO_DATASET}"

# Set environment variables
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

**Why:** Configures paths before importing src.config module.

---

## ➕ Change 3: INSERT New Cell 4 (After Wandb Setup)

### Insert AFTER original Cell 3 (Wandb Setup)

**New Cell 4: Verify Environment**

```python
# ============================================
# CELL 4: Verify Environment Configuration
# ============================================
# Time: ~30 seconds

from src.config import (
    print_environment_info,
    get_test_video_dir
)

print_environment_info()

# Verify video directory
video_dir = get_test_video_dir()

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

print(f"\n✓ Environment verification complete")
```

**Why:** Verifies all paths are configured correctly before running experiments.

---

## 📌 Cell Numbering After Updates

After inserting the 2 new cells, renumber like this:

| Original | After Update | Change |
|----------|--------------|--------|
| Cell 1 | Cell 1 | UPDATED (boxmot pinning) |
| Cell 2 | Cell 2 | UNCHANGED (Git setup) |
| — | Cell 2A | **NEW** (Environment vars) |
| Cell 3 | Cell 3 | UNCHANGED (Wandb login) |
| — | Cell 4 | **NEW** (Verify setup) |
| Cell 4 | Cell 5 | UNCHANGED (Video verify) |
| Cell 5 | Cell 6 | UNCHANGED (File check) |
| Cell 6 | Cell 7 | UNCHANGED (Baseline test) |
| Cell 7 | Cell 8 | UNCHANGED (Phase 1) |
| Cell 8 | Cell 9 | UNCHANGED (Analyze P1) |
| ... | ... | All subsequent cells shift +2 |

---

## ✅ Verification After Updates

Run this in any cell to verify:

```python
from src.config import print_environment_info, IS_KAGGLE
print_environment_info()
assert IS_KAGGLE, "Not running on Kaggle!"
print("\n✅ All checks passed")
```

Expected output:
```
======================================================================
ENVIRONMENT CONFIGURATION
======================================================================

Environment:     KAGGLE
Python:          3.12
Project Root:    /kaggle/working/edge-vlm-thesis
Test Videos:     /kaggle/input/your-dataset
Output Dir:      /kaggle/working/results
Checkpoints:     /kaggle/working/checkpoints

Kaggle Input:    /kaggle/input

======================================================================

✅ All checks passed
```

---

## 🎯 The Key Issue

**Original notebook:**
```python
from src.config import ...  # ← Fails! No TEST_VIDEO_DIR set yet
```

**Updated notebook:**
```python
import os
os.environ["TEST_VIDEO_DIR"] = "/kaggle/input/videos"  # ← Set first

from src.config import ...  # ← Now works!
```

Environment variables must be set **before** importing src.config.

---

## 📋 Update Checklist

Before running experiments:

- [ ] Cell 1 has `!pip install -q boxmot==10.11.62`
- [ ] Cell 2A is inserted with environment variable setup
- [ ] Cell 2A's `TEST_VIDEO_DATASET` matches your Kaggle dataset
- [ ] Cell 4 is inserted and prints environment info
- [ ] Cell 5+ (original test video check) finds your videos
- [ ] All cells run without import errors

---

## 🚀 After Updates

Your notebook will:

✅ Use correct boxmot version (no Python 3.12 issues)
✅ Auto-detect video paths (no hardcoding)
✅ Set environment variables properly (before imports)
✅ Verify setup before running experiments (early error detection)
✅ Work on any Kaggle account (just change dataset name)

---

## If You Get Import Errors

**Error:** `ModuleNotFoundError: No module named src.config`

**Cause:** Environment variables not set before import

**Fix:** Make sure Cell 2A runs before the import fails

---

## If You Get File Not Found Errors

**Error:** `FileNotFoundError: No test video found in /kaggle/input/...`

**Cause:** Wrong dataset name in Cell 2A

**Fix:** 
```python
from pathlib import Path
for d in Path("/kaggle/input").iterdir():
    if d.is_dir():
        print(d.name)
```
Find correct name, update `TEST_VIDEO_DATASET` in Cell 2A

---

## Summary

| Task | Time | Required |
|------|------|----------|
| Update Cell 1 (boxmot) | 1 min | ✅ Yes |
| Insert Cell 2A (env vars) | 2 min | ✅ Yes |
| Insert Cell 4 (verify) | 2 min | ✅ Recommended |
| Test notebook | 5 min | ✅ Yes |

**Total: 10 minutes**

---

**Your notebook will be production-ready and infrastructure-compliant.** ✅
