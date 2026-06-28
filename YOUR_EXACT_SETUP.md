# Your Exact Kaggle Setup

**Videos:** `/kaggle/input/datasets/srishanmukhasai/videos`
**Repo:** `edge`
**Working dir:** `/kaggle/working/edge`

---

## Copy-Paste This Into Cell 2A

```python
import os

# YOUR EXACT SETUP
os.environ["TEST_VIDEO_DIR"] = "/kaggle/input/datasets/srishanmukhasai/videos"
os.environ["OUTPUT_DIR"] = "/kaggle/working/results"
os.environ["CHECKPOINT_DIR"] = "/kaggle/working/checkpoints"

os.makedirs("/kaggle/working/results", exist_ok=True)
os.makedirs("/kaggle/working/checkpoints", exist_ok=True)

print("✓ Environment configured for your setup")
```

---

## Copy-Paste This Into Cell 2 (Git Clone)

```python
import os
from pathlib import Path

os.chdir('/kaggle/working')

# YOUR REPO: edge
if not Path('/kaggle/working/edge').exists():
    print("Cloning repository...")
    os.system('git clone https://github.com/shanmukhasaicentific/edge.git')
    os.chdir('/kaggle/working/edge')
else:
    print("✓ Repository exists")
    os.chdir('/kaggle/working/edge')
    os.system('git pull origin main')

print(f"✓ Working dir: {os.getcwd()}")
```

---

## The 3 Critical Lines

Replace these in your notebook:

### Line 1: Cell 1 (Dependencies)
```python
!pip install -q boxmot==10.11.62  # ← ADD THIS LINE
```

### Line 2: Cell 2A (Environment) - NEW CELL
```python
os.environ["TEST_VIDEO_DIR"] = "/kaggle/input/datasets/srishanmukhasai/videos"
```

### Line 3: Cell 2 (Git)
```python
os.system('git clone https://github.com/shanmukhasaicentific/edge.git')
```

---

## Verification (Cell 4)

This should print:

```
Test Videos: /kaggle/input/datasets/srishanmukhasai/videos
✓ Videos found (N)
```

If not, you have the wrong path.

---

## That's It!

Everything else from `KAGGLE_NOTEBOOK_FINAL.md` stays the same.

**Your 10-hour Kaggle session is ready to run 38 experiments!** 🚀
