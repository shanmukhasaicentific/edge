# Environment Setup Guide

Fix hardcoded paths and make your project work reliably on Kaggle, Colab, and local machines.

---

## 1. Environment Variables (Kaggle Notebook)

**In Cell 3 of your Kaggle notebook, add this:**

```python
import os

# Configure paths for Kaggle (before any imports)
os.environ["TEST_VIDEO_DIR"] = "/kaggle/input/your-video-dataset"  # ← CHANGE THIS
os.environ["OUTPUT_DIR"] = "/kaggle/working/results"
os.environ["CHECKPOINT_DIR"] = "/kaggle/working/checkpoints"

# Then the rest of setup
import wandb
import sys
sys.path.insert(0, '/kaggle/working/edge-vlm-thesis')
```

**What to change:**
- Replace `your-video-dataset` with your actual Kaggle dataset name
- If your videos are in `/kaggle/input/srishanmukhasai-vidoes`, use:
  ```python
  os.environ["TEST_VIDEO_DIR"] = "/kaggle/input/srishanmukhasai-vidoes"
  ```

---

## 2. Local Machine (No Changes Needed)

The code **automatically** detects:
- ✅ Local: Uses `experiments/test/` by default
- ✅ Kaggle: Detects `/kaggle/input` and uses env variables
- ✅ Colab: Detects `/content` and adapts paths

Run as normal:
```bash
python autoresearch-master/train.py --phase threshold_sweep --multi-gpu
```

---

## 3. Verify Configuration Works

**In Python:**
```python
from src.config import print_environment_info, IS_KAGGLE, get_test_video_dir

print_environment_info()

if IS_KAGGLE:
    print("Running on Kaggle ✓")
else:
    print("Running locally ✓")

video_dir = get_test_video_dir()
print(f"Videos found in: {video_dir}")
```

**Expected output on Kaggle:**
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
Configs:         /kaggle/working/edge-vlm-thesis/experiments

Kaggle Input:    /kaggle/input

======================================================================

Running on Kaggle ✓
Videos found in: /kaggle/input/your-dataset
```

---

## 4. Environment Variable Reference

| Variable | Default | Kaggle | Purpose |
|----------|---------|--------|---------|
| `TEST_VIDEO_DIR` | `experiments/test` | Your dataset | Where to find test videos |
| `OUTPUT_DIR` | `outputs/` | `/kaggle/working/results` | Where to save results |
| `CHECKPOINT_DIR` | `checkpoints/` | `/kaggle/working/checkpoints` | Where to save experiments |

---

## 5. Kaggle Dataset Setup

**Step 1:** Upload your test video to Kaggle

1. Go to Kaggle.com → Datasets → Create
2. Upload your `test_video.mp4`
3. Name it something memorable (e.g., `edge-vlm-videos`)
4. Note the exact path shown

**Step 2:** In your notebook, set the environment variable

```python
import os

# Get the exact path from your dataset
# View it in: https://www.kaggle.com/datasets/your-username/dataset-name
os.environ["TEST_VIDEO_DIR"] = "/kaggle/input/edge-vlm-videos"
```

---

## 6. BoxMOT Dependency Fix

**The problem:**
- Old: `boxmot>=10.0.0` allows v21 which broke the API
- New: Pin to stable v10

**Solution:**
```bash
pip install boxmot==10.11.62
```

**In Kaggle cell 1:**
```python
!pip install -q boxmot==10.11.62
```

---

## 7. Troubleshooting

### "No test video found"

**Fix:** Verify the environment variable
```python
import os
from src.config import get_test_video_dir

print(f"TEST_VIDEO_DIR = {os.environ.get('TEST_VIDEO_DIR')}")
print(f"Actual directory: {get_test_video_dir()}")

import subprocess
subprocess.run(["ls", "-la", str(get_test_video_dir())])
```

### "from boxmot import ByteTrack" fails

**Fix:** Use pinned version
```bash
pip install boxmot==10.11.62
```

### Paths not found on Kaggle

**Fix:** Set environment variables BEFORE importing
```python
# DO THIS FIRST
import os
os.environ["TEST_VIDEO_DIR"] = "/kaggle/input/..."

# THEN this
from src.config import get_test_video_dir
```

---

## 8. For Your Thesis Evaluation

When your evaluators run your code:

**Local:**
```bash
pip install -r requirements.txt
pip install boxmot==10.11.62
python autoresearch-master/train.py --phase threshold_sweep
```

**Kaggle:**
- They add their video dataset
- Set `os.environ["TEST_VIDEO_DIR"]` to their dataset path
- Rest works automatically

**Colab:**
```python
!git clone https://github.com/shanmukhasaicentific/edge-vlm-thesis.git
!pip install -r edge-vlm-thesis/requirements.txt
!pip install boxmot==10.11.62

os.environ["TEST_VIDEO_DIR"] = "/content/edge-vlm-thesis/experiments/test"
# Run code
```

**Result:** No hardcoded paths, works everywhere.

---

## 9. What Changed

### New Files
- `src/config/env_config.py` — Environment detection and path management
- `src/config/__init__.py` — Config module exports

### Updated Files
- `autoresearch-master/train.py` — Uses `get_test_video_dir()` and env config
- `src/tracking/bytetrack_wrapper.py` — Better error handling for boxmot imports
- `README.md` — Added environment setup section
- `requirements.txt` — Added boxmot pinning note

### No Breaking Changes
- All old code still works
- Local machine works without any environment variables
- Backward compatible

---

## 10. Next Steps

1. **In Kaggle notebook Cell 3, add:**
   ```python
   import os
   os.environ["TEST_VIDEO_DIR"] = "/kaggle/input/your-dataset"
   os.environ["OUTPUT_DIR"] = "/kaggle/working/results"
   os.environ["CHECKPOINT_DIR"] = "/kaggle/working/checkpoints"
   ```

2. **Install correct boxmot in Cell 1:**
   ```python
   !pip install -q boxmot==10.11.62
   ```

3. **Verify with:**
   ```python
   from src.config import print_environment_info
   print_environment_info()
   ```

4. **Run experiments:**
   ```python
   !cd /kaggle/working/edge-vlm-thesis/autoresearch-master && \
    python train.py --phase threshold_sweep --multi-gpu
   ```

---

**Your thesis code is now production-ready, portable, and evaluator-friendly.** 🎓
