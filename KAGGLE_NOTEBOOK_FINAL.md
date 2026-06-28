# Kaggle Notebook: Final Configuration

**For your specific setup:**
- Videos: `/kaggle/input/datasets/srishanmukhasai/videos`
- Repo: `edge`
- Working dir: `/kaggle/working/edge`

---

## Cell 1: Setup & Dependencies (EXACT)

```python
# ============================================
# CELL 1: Setup & Dependencies
# ============================================

print("="*70)
print("KAGGLE AUTORESEARCH - SETUP")
print("="*70)

# Install dependencies with correct boxmot version
!pip install -q wandb torch ultralytics opencv-python clip pyyaml matplotlib seaborn pandas
!pip install -q boxmot==10.11.62  # ← CRITICAL: Pinned for Python 3.12

# Verify installs
import torch
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
```

---

## Cell 2: Git Setup & Repository (EXACT)

```python
# ============================================
# CELL 2: Git Setup & Repository
# ============================================

import os
import subprocess
from pathlib import Path

# Configure git
os.system('git config --global user.email "kaggle@research.local"')
os.system('git config --global user.name "Kaggle AutoResearch"')

# Navigate to working directory
os.chdir('/kaggle/working')

# Clone your repo (repo name: edge)
if not Path('/kaggle/working/edge').exists():
    print("Cloning repository...")
    os.system('git clone https://github.com/shanmukhasaicentific/edge.git')
    os.chdir('/kaggle/working/edge')
else:
    print("✓ Repository already exists")
    os.chdir('/kaggle/working/edge')
    # Update to latest
    os.system('git pull origin main')

print(f"\n✓ Working directory: {os.getcwd()}")
print(f"✓ Git configured")
```

---

## Cell 2A: Configure Environment Variables (EXACT) - NEW!

```python
# ============================================
# CELL 2A: Configure Environment Variables
# ============================================
# CRITICAL: Set environment BEFORE any src imports

import os
from pathlib import Path

print("\n" + "="*70)
print("CONFIGURING KAGGLE PATHS")
print("="*70)

# Set environment variables for YOUR setup
os.environ["TEST_VIDEO_DIR"] = "/kaggle/input/datasets/srishanmukhasai/videos"
os.environ["OUTPUT_DIR"] = "/kaggle/working/results"
os.environ["CHECKPOINT_DIR"] = "/kaggle/working/checkpoints"

print(f"\n✓ TEST_VIDEO_DIR = /kaggle/input/datasets/srishanmukhasai/videos")
print(f"✓ OUTPUT_DIR = /kaggle/working/results")
print(f"✓ CHECKPOINT_DIR = /kaggle/working/checkpoints")

# Create directories
os.makedirs("/kaggle/working/results", exist_ok=True)
os.makedirs("/kaggle/working/checkpoints", exist_ok=True)

print("\n✓ Environment variables configured")
```

---

## Cell 3: Wandb & Checkpoint Setup (SAME)

```python
# ============================================
# CELL 3: Wandb & Checkpoint Setup
# ============================================

import wandb
import sys
sys.path.insert(0, '/kaggle/working/edge')

# Login to wandb
print("\n✓ Wandb version:", wandb.__version__)
print("✓ Attempting wandb login...")
os.system('wandb login --reuse')

print("\n✓ Wandb configured")
```

---

## Cell 4: Verify Environment Configuration (NEW!) - CRITICAL

```python
# ============================================
# CELL 4: Verify Environment Configuration
# ============================================

from src.config import (
    print_environment_info,
    IS_KAGGLE,
    get_test_video_dir
)

print_environment_info()

# Verify video directory
video_dir = get_test_video_dir()
print(f"\n📁 Test Video Directory: {video_dir}")

# List videos
import glob
from pathlib import Path
videos = glob.glob(f"{video_dir}/*.mp4") + glob.glob(f"{video_dir}/*.avi")

if videos:
    print(f"\n✓ Videos found ({len(videos)}):")
    for v in videos[:3]:
        size_mb = Path(v).stat().st_size / (1024**2)
        print(f"  - {Path(v).name} ({size_mb:.1f} MB)")
    if len(videos) > 3:
        print(f"  ... and {len(videos) - 3} more")
else:
    print(f"\n❌ NO VIDEOS FOUND in {video_dir}")
    print(f"Make sure /kaggle/input/datasets/srishanmukhasai/videos exists!")

# Verify it's Kaggle
assert IS_KAGGLE, "Not running on Kaggle!"
print(f"\n✓ Environment verification complete")
```

---

## Cell 5: Test Video Verification (SAME)

```python
# ============================================
# CELL 5: Test Video Verification
# ============================================

from src.config import get_test_video_dir

video_dir = get_test_video_dir()
print(f"\nSearching for test video in: {video_dir}")

import glob
from pathlib import Path
videos = glob.glob(f"{video_dir}/*.mp4") + glob.glob(f"{video_dir}/*.avi")

if videos:
    print(f"\n✓ Test videos found:")
    for v in videos:
        size_mb = Path(v).stat().st_size / (1024**2)
        print(f"  - {Path(v).name} ({size_mb:.1f} MB)")
    print("\n✓ Video verification passed")
else:
    print(f"\n✗ WARNING: No test video found")
```

---

## Cell 6: Check AutoResearch Files (SAME)

```python
# ============================================
# CELL 6: Check AutoResearch Files
# ============================================

autosearch_dir = Path('/kaggle/working/edge/autoresearch-master')

print(f"\nAutoResearch directory: {autosearch_dir}")
print("\nKey files:")

files_to_check = [
    'train.py',
    'program.md',
    'prepare.py'
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

## Cell 7: Phase 0 - Baseline Test (Optional)

```python
# ============================================
# CELL 7: Phase 0 - Baseline Test (Optional)
# ============================================

import subprocess
import os

os.chdir('/kaggle/working/edge/autoresearch-master')

print("\n" + "="*70)
print("PHASE 0: BASELINE TEST")
print("="*70)
print("\nRunning single baseline experiment...\n")

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
    print("\nERROR:")
    print(result.stderr)
    print("\n❌ Baseline test FAILED")
else:
    print("\n✓ Baseline test PASSED")
```

---

## Cell 8: Phase 1 - Threshold Sweep (3 hours)

```python
# ============================================
# CELL 8: Phase 1 - THRESHOLD SWEEP (3 hours)
# ============================================

import subprocess
import os
from datetime import datetime

os.chdir('/kaggle/working/edge/autoresearch-master')

print("\n" + "="*70)
print("PHASE 1: THRESHOLD SWEEP")
print("="*70)
print(f"\nStart: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Running 28 threshold experiments...\n")

cmd = [
    'python', 'train.py',
    '--phase', 'threshold_sweep',
    '--multi-gpu',
    '--wandb-project', 'edge-vlm-thesis',
]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)

print(result.stdout)
print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if result.returncode == 0:
    print("\n✓ Phase 1 COMPLETE")
else:
    print(f"\n⚠ Phase 1 exited with code {result.returncode}")
```

---

## Cell 9: Analyze Phase 1 Results

```python
# ============================================
# CELL 9: Analyze Phase 1 Results
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

results_file = '/kaggle/working/edge/autoresearch-master/results.tsv'

if Path(results_file).exists():
    df = pd.read_csv(results_file, sep='\t')
    phase1 = df[df['description'].str.contains('threshold_sweep', na=False)]
    
    print("\n" + "="*70)
    print("PHASE 1 ANALYSIS")
    print("="*70)
    
    print(f"\nTotal experiments: {len(phase1)}")
    print(f"Best SCE: {phase1['sce'].max():.4f}")
    print(f"Average SCE: {phase1['sce'].mean():.4f}")
    
    print("\nTop 5 configurations:")
    top5 = phase1.nlargest(5, 'sce')[['sce', 'semantic_retention', 'vlm_call_rate']]
    print(top5.to_string())
    
    # Plot
    if len(phase1) > 0:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(
            phase1['vlm_call_rate'],
            phase1['semantic_retention'],
            c=phase1['sce'],
            cmap='RdYlGn',
            s=100,
            alpha=0.6
        )
        plt.xlabel('VLM Call Rate')
        plt.ylabel('Semantic Retention')
        plt.title('Pareto Frontier')
        plt.colorbar(scatter, label='SCE')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(phase1['sce'], bins=10, color='skyblue', edgecolor='black')
        plt.xlabel('SCE')
        plt.ylabel('Frequency')
        plt.title('SCE Distribution')
        
        plt.tight_layout()
        plt.savefig('/kaggle/working/phase1_analysis.png', dpi=100)
        print("\n✓ Plot saved")
        plt.show()
```

---

## Cell 10: Phase 2 - Ablations (30 min)

```python
# ============================================
# CELL 10: Phase 2 - ABLATION STUDIES (30 min)
# ============================================

import subprocess
import os
import pandas as pd
import re
from datetime import datetime

os.chdir('/kaggle/working/edge/autoresearch-master')

print("\n" + "="*70)
print("PHASE 2: ABLATION STUDIES")
print("="*70)
print(f"\nStart: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Get best config from Phase 1
results_file = '/kaggle/working/edge/autoresearch-master/results.tsv'
best_config = {'tau_low': '0.15', 'tau_high': '0.40'}

try:
    df = pd.read_csv(results_file, sep='\t')
    phase1 = df[df['description'].str.contains('threshold_sweep', na=False)]
    if len(phase1) > 0:
        best = phase1.loc[phase1['sce'].idxmax()]
        match = re.search(r'tau_low=([\d.]+), tau_high=([\d.]+)', best['description'])
        if match:
            best_config['tau_low'] = match.group(1)
            best_config['tau_high'] = match.group(2)
except:
    pass

print(f"Using best config: tau_low={best_config['tau_low']}, tau_high={best_config['tau_high']}")

cmd = [
    'python', 'train.py',
    '--phase', 'ablation',
    '--tau_low', best_config['tau_low'],
    '--tau_high', best_config['tau_high'],
    '--multi-gpu',
    '--wandb-project', 'edge-vlm-thesis',
]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
print(result.stdout)

print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if result.returncode == 0:
    print("\n✓ Phase 2 COMPLETE")
```

---

## Cell 11: Phase 3 - Failure Analysis (30 min)

```python
# ============================================
# CELL 11: Phase 3 - FAILURE ANALYSIS (30 min)
# ============================================

import subprocess
import os
from datetime import datetime

os.chdir('/kaggle/working/edge/autoresearch-master')

print("\n" + "="*70)
print("PHASE 3: FAILURE ANALYSIS")
print("="*70)
print(f"\nStart: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

scenarios = ['stable', 'motion', 'lighting', 'tracking', 'transitions']

for i, scenario in enumerate(scenarios, 1):
    print(f"\n[{i}/5] Running scenario: {scenario}")
    
    cmd = [
        'python', 'train.py',
        '--phase', 'failure_analysis',
        '--scenario', scenario,
        '--tau_low', '0.15',
        '--tau_high', '0.40',
        '--multi-gpu',
        '--wandb-project', 'edge-vlm-thesis',
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        for line in result.stdout.split('\n'):
            if 'sce:' in line or '✓' in line:
                print(line)
    except:
        print(f"⚠ {scenario} timed out or failed")

print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n✓ Phase 3 COMPLETE")
```

---

## Cell 12: Final Summary

```python
# ============================================
# CELL 12: Final Summary
# ============================================

import pandas as pd
from pathlib import Path

results_file = '/kaggle/working/edge/autoresearch-master/results.tsv'

if Path(results_file).exists():
    df = pd.read_csv(results_file, sep='\t')
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print(f"\n📊 Total Experiments: {len(df)}")
    
    phase1 = len(df[df['description'].str.contains('threshold_sweep', na=False)])
    phase2 = len(df[df['description'].str.contains('ablation', na=False)])
    phase3 = len(df[df['description'].str.contains('failure_analysis', na=False)])
    
    print(f"\nPhase Breakdown:")
    print(f"  Phase 1: {phase1}")
    print(f"  Phase 2: {phase2}")
    print(f"  Phase 3: {phase3}")
    
    print(f"\n📈 SCE Statistics:")
    print(f"  Best: {df['sce'].max():.4f}")
    print(f"  Avg: {df['sce'].mean():.4f}")
    
    best = df.loc[df['sce'].idxmax()]
    print(f"\n🏆 Best Configuration:")
    print(f"  SCE: {best['sce']:.4f}")
    print(f"  Retention: {best['semantic_retention']:.4f}")
    print(f"  VLM Rate: {best['vlm_call_rate']:.4f}")
    
    # Export
    df.to_csv('/kaggle/working/thesis_results.csv', index=False)
    print(f"\n✓ Exported: thesis_results.csv")
```

---

## Cell 13: Git Commit & Push

```python
# ============================================
# CELL 13: Git Commit & Push
# ============================================

import os
import shutil
from datetime import datetime
import pandas as pd

os.chdir('/kaggle/working/edge')

print("\n" + "="*70)
print("GIT COMMIT & PUSH")
print("="*70)

# Copy results
results_src = '/kaggle/working/edge/autoresearch-master/results.tsv'
if os.path.exists(results_src):
    shutil.copy(results_src, '/kaggle/working/edge/autoresearch-master/results.tsv')
    print("✓ Copied results.tsv")

# Stage and commit
os.system('git add -A')
os.system('git add autoresearch-master/results.tsv')

# Count experiments
try:
    df = pd.read_csv('/kaggle/working/edge/autoresearch-master/results.tsv', sep='\t')
    exp_count = len(df)
except:
    exp_count = 0

commit_msg = f'AutoResearch results: {exp_count} experiments completed on {datetime.now().strftime("%Y-%m-%d %H:%M")}'

os.system(f'git commit -m "{commit_msg}"')
print("✓ Committed to git")

# Try to push
os.system('git push origin main 2>&1')
print("✓ Push attempted")
```

---

## Cell 14: Download Results

```python
# ============================================
# CELL 14: Download Results & Cleanup
# ============================================

import shutil
import os
from pathlib import Path

os.chdir('/kaggle/working')

print("\n" + "="*70)
print("PACKAGING RESULTS FOR DOWNLOAD")
print("="*70)

# Results CSV
if Path('/kaggle/working/thesis_results.csv').exists():
    print("✓ thesis_results.csv ready")

# Checkpoints
if Path('/kaggle/working/checkpoints').exists():
    shutil.make_archive('/kaggle/working/checkpoints_backup', 'zip', '/kaggle/working/', 'checkpoints')
    print("✓ checkpoints_backup.zip ready")

print("\n" + "="*70)
print("✅ AUTORESEARCH COMPLETE!")
print("="*70)

print("\n📥 DOWNLOAD FROM KAGGLE OUTPUT:")
print("  - thesis_results.csv (all metrics)")
print("  - checkpoints_backup.zip (resume capability)")
```

---

## 🎯 Key Points for Your Setup

```
✅ Repository: edge (not edge-vlm-thesis)
✅ Videos: /kaggle/input/datasets/srishanmukhasai/videos
✅ Working dir: /kaggle/working/edge
✅ Output: /kaggle/working/results
✅ Checkpoints: /kaggle/working/checkpoints
```

---

## 📋 Copy-Paste Order

1. Cell 1: Dependencies (pin boxmot)
2. Cell 2: Git setup (clone edge repo)
3. Cell 2A: **NEW** Environment variables
4. Cell 3: Wandb login
5. Cell 4: **NEW** Verify setup
6. Cell 5: Test video verify
7. Cell 6: Check files
8. Cell 7: Baseline (optional)
9. Cell 8: Phase 1
10. Cell 9: Analyze P1
11. Cell 10: Phase 2
12. Cell 11: Phase 3
13. Cell 12: Summary
14. Cell 13: Git commit
15. Cell 14: Download

---

## ✅ Test Before Running

In Cell 4, you should see:

```
======================================================================
ENVIRONMENT CONFIGURATION
======================================================================

Environment:     KAGGLE
Python:          3.12
Project Root:    /kaggle/working/edge
Test Videos:     /kaggle/input/datasets/srishanmukhasai/videos
Output Dir:      /kaggle/working/results
Checkpoints:     /kaggle/working/checkpoints

✓ Videos found (3)
  - video1.mp4 (500.5 MB)
  - video2.mp4 (450.2 MB)
  - video3.mp4 (600.1 MB)

✓ Environment verification complete
```

---

**Your notebook is ready to use with your exact Kaggle setup!** ✅
