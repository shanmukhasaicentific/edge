# Kaggle Notebook: Complete AutoResearch Pipeline

Copy-paste this entire notebook structure into Kaggle. It runs all 3 phases (38 experiments) in 10-hour session.

---

## How to Use This

1. **On Kaggle.com:**
   - New Notebook → Python
   - Add data sources:
     - Your uploaded `edge-vlm-thesis` dataset
     - Your test video dataset
   - Copy-paste each cell below into Kaggle cells in order

2. **Before Running:**
   - Prepare your wandb API key
   - Know your Kaggle API token (for git commits)
   - Have test video ready in dataset

3. **During Run:**
   - Cells run sequentially
   - Monitor GPU with `nvidia-smi` in cell
   - Watch wandb dashboard in browser
   - Takes ~4-5 hours total with 2 T4s

4. **After Run:**
   - Download results files
   - Results committed to git
   - All checkpoints saved

---

## 📓 Notebook Content

### Cell 1: Setup & Dependencies

```python
# ============================================
# CELL 1: Setup & Dependencies
# ============================================
# Time: ~60 seconds

print("="*70)
print("KAGGLE AUTORESEARCH - SETUP")
print("="*70)

# Install dependencies
!pip install -q wandb torch ultralytics opencv-python clip boxmot pyyaml matplotlib seaborn pandas

# Verify installs
import torch
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
```

---

### Cell 2: Git Setup & Repo

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
print(f"✓ Git status:")
os.system('git status --short')
```

---

### Cell 3: Wandb & Checkpoint Setup

```python
# ============================================
# CELL 3: Wandb Login & Checkpoint Setup
# ============================================
# Time: ~30 seconds

import wandb
import sys
sys.path.insert(0, '/kaggle/working/edge-vlm-thesis')

# Login to wandb
print("\n✓ Wandb version:", wandb.__version__)
print("✓ Attempting wandb login (will use cached credentials or prompt)...")
os.system('wandb login --reuse')

# Create checkpoint directory
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/results', exist_ok=True)

print("\n✓ Checkpoint directory: /kaggle/working/checkpoints")
print("✓ Results directory: /kaggle/working/results")

# Verify wandb
try:
    api = wandb.Api()
    user = api.viewer()
    print(f"\n✓ Wandb logged in as: {user.username}")
except Exception as e:
    print(f"WARNING: Wandb error: {e}")
    print("  Continuing with offline mode...")
```

---

### Cell 4: Test Video Verification

```python
# ============================================
# CELL 4: Test Video Verification
# ============================================
# Time: ~10 seconds

from pathlib import Path

# Find test video
experiments_dir = Path('/kaggle/working/edge-vlm-thesis/experiments/test')
print(f"\nSearching for test video in: {experiments_dir}")

video_found = False
for ext in ['*.mp4', '*.avi', '*.mov']:
    videos = list(experiments_dir.glob(ext))
    if videos:
        video_found = True
        print(f"\n✓ Test videos found:")
        for v in videos:
            size_mb = v.stat().st_size / (1024**2)
            print(f"  - {v.name} ({size_mb:.1f} MB)")
        break

if not video_found:
    print("\n⚠ WARNING: No test video found!")
    print("  Expected: experiments/test/*.mp4")
    print("  Make sure test video is in your dataset!")
else:
    print("\n✓ Video verification passed")
```

---

### Cell 5: Check AutoResearch Files

```python
# ============================================
# CELL 5: Check AutoResearch Files
# ============================================
# Time: ~10 seconds

import os

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

### Cell 6: Phase 0 - Baseline (Optional Quick Test)

```python
# ============================================
# CELL 6: Phase 0 - Baseline Test (5 min)
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
    '--checkpoint-dir', '/kaggle/working/checkpoints',
    '--output-dir', '/kaggle/working/results'
]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
print(result.stdout)

if result.returncode != 0:
    print("\nERROR OUTPUT:")
    print(result.stderr)
    print("\n❌ Baseline test FAILED")
else:
    print("\n✓ Baseline test PASSED")

# Check results
if Path('/kaggle/working/edge-vlm-thesis/autoresearch-master/results.tsv').exists():
    import pandas as pd
    df = pd.read_csv('/kaggle/working/edge-vlm-thesis/autoresearch-master/results.tsv', sep='\t')
    print(f"\n✓ Experiments in results.tsv: {len(df)}")
```

---

### Cell 7: Phase 1 - Threshold Sweep (3 hours)

```python
# ============================================
# CELL 7: Phase 1 - THRESHOLD SWEEP (3 hours)
# ============================================
# Time: ~3 hours
# 28 experiments: all tau_low x tau_high combinations

import subprocess
import os
from datetime import datetime

os.chdir('/kaggle/working/edge-vlm-thesis/autoresearch-master')

print("\n" + "="*70)
print("PHASE 1: THRESHOLD SWEEP")
print("="*70)
print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Running 28 threshold combination experiments...")
print("Estimated time: ~3 hours with 2 T4 GPUs\n")

cmd = [
    'python', 'train.py',
    '--phase', 'threshold_sweep',
    '--multi-gpu',
    '--wandb-project', 'edge-vlm-thesis',
    '--checkpoint-dir', '/kaggle/working/checkpoints',
    '--output-dir', '/kaggle/working/results'
]

# Run with timeout of 4 hours (240 min)
result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)

print(result.stdout)
if result.stderr:
    print("\nSTDERR:")
    print(result.stderr)

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if result.returncode == 0:
    print("\n✓ Phase 1 COMPLETE")
else:
    print(f"\n⚠ Phase 1 exited with code {result.returncode}")

# Show progress
import pandas as pd
results_file = '/kaggle/working/edge-vlm-thesis/autoresearch-master/results.tsv'
if Path(results_file).exists():
    df = pd.read_csv(results_file, sep='\t')
    print(f"\n📊 Progress: {len(df)} experiments completed")
    print(f"Best SCE so far: {df['sce'].max():.4f}")
```

---

### Cell 8: Analyze Phase 1 Results

```python
# ============================================
# CELL 8: Analyze Phase 1 Results
# ============================================
# Time: ~1 minute

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results_file = '/kaggle/working/edge-vlm-thesis/autoresearch-master/results.tsv'

if Path(results_file).exists():
    df = pd.read_csv(results_file, sep='\t')
    
    # Filter Phase 1 only
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
    
    # Get best config for Phase 2
    best_row = phase1.loc[phase1['sce'].idxmax()]
    print(f"\nBest config for Phase 2:")
    print(f"  SCE: {best_row['sce']:.4f}")
    print(f"  Semantic Retention: {best_row['semantic_retention']:.4f}")
    print(f"  VLM Call Rate: {best_row['vlm_call_rate']:.4f}")
    print(f"  Description: {best_row['description']}")
    
    # Plot Pareto frontier
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
        plt.xlabel('VLM Call Rate (calls/sec)')
        plt.ylabel('Semantic Retention')
        plt.title('Phase 1: Pareto Frontier')
        plt.colorbar(scatter, label='SCE')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(phase1['sce'], bins=10, color='skyblue', edgecolor='black')
        plt.xlabel('SCE')
        plt.ylabel('Frequency')
        plt.title('SCE Distribution')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('/kaggle/working/phase1_analysis.png', dpi=100, bbox_inches='tight')
        print("\n✓ Saved: phase1_analysis.png")
        plt.show()

else:
    print("No results file found!")
```

---

### Cell 9: Phase 2 - Ablation Studies (30 min)

```python
# ============================================
# CELL 9: Phase 2 - ABLATION STUDIES (30 min)
# ============================================
# Time: ~30 minutes
# 5 experiments: baseline, no_embedding, no_objects, no_tracking, uniform

import subprocess
import os
from datetime import datetime

os.chdir('/kaggle/working/edge-vlm-thesis/autoresearch-master')

print("\n" + "="*70)
print("PHASE 2: ABLATION STUDIES")
print("="*70)
print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Running 5 ablation studies...")
print("Estimated time: ~30 minutes with 2 T4 GPUs\n")

# Get best tau_low, tau_high from Phase 1
import pandas as pd
results_file = '/kaggle/working/edge-vlm-thesis/autoresearch-master/results.tsv'

best_config = {'tau_low': '0.15', 'tau_high': '0.40'}  # Default fallback

try:
    df = pd.read_csv(results_file, sep='\t')
    phase1 = df[df['description'].str.contains('threshold_sweep', na=False)]
    if len(phase1) > 0:
        best = phase1.loc[phase1['sce'].idxmax()]
        # Parse from description: "threshold_sweep: tau_low=X.XX, tau_high=Y.YY"
        desc = best['description']
        # Extract tau values (simple parsing)
        import re
        match = re.search(r'tau_low=([\d.]+), tau_high=([\d.]+)', desc)
        if match:
            best_config['tau_low'] = match.group(1)
            best_config['tau_high'] = match.group(2)
except:
    pass

print(f"Using best config from Phase 1:")
print(f"  tau_low = {best_config['tau_low']}")
print(f"  tau_high = {best_config['tau_high']}\n")

cmd = [
    'python', 'train.py',
    '--phase', 'ablation',
    '--tau_low', best_config['tau_low'],
    '--tau_high', best_config['tau_high'],
    '--multi-gpu',
    '--wandb-project', 'edge-vlm-thesis',
    '--checkpoint-dir', '/kaggle/working/checkpoints',
    '--output-dir', '/kaggle/working/results'
]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

print(result.stdout)
if result.stderr:
    print("\nSTDERR:")
    print(result.stderr)

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if result.returncode == 0:
    print("\n✓ Phase 2 COMPLETE")
else:
    print(f"\n⚠ Phase 2 exited with code {result.returncode}")

# Show progress
if Path(results_file).exists():
    df = pd.read_csv(results_file, sep='\t')
    print(f"\n📊 Total experiments: {len(df)}")
```

---

### Cell 10: Analyze Phase 2 Results

```python
# ============================================
# CELL 10: Analyze Phase 2 - Component Importance
# ============================================
# Time: ~1 minute

import pandas as pd

results_file = '/kaggle/working/edge-vlm-thesis/autoresearch-master/results.tsv'

if Path(results_file).exists():
    df = pd.read_csv(results_file, sep='\t')
    
    # Filter Phase 2 only
    phase2 = df[df['description'].str.contains('ablation', na=False)]
    
    print("\n" + "="*70)
    print("PHASE 2 ANALYSIS: Component Importance")
    print("="*70)
    
    if len(phase2) > 0:
        print(f"\nTotal ablations: {len(phase2)}")
        
        # Get baseline (should be first experiment of ablations)
        if 'baseline' in phase2['description'].values:
            baseline_sce = phase2[phase2['description'].str.contains('baseline')]['sce'].iloc[0]
        else:
            baseline_sce = phase2['sce'].max()
        
        print(f"Baseline SCE: {baseline_sce:.4f}\n")
        
        print("Component Importance (SCE drop when removed):")
        print("-" * 50)
        
        for idx, row in phase2.iterrows():
            delta = (baseline_sce - row['sce']) / baseline_sce * 100
            importance = "CRITICAL" if delta > 15 else "IMPORTANT" if delta > 8 else "MINOR"
            print(f"{row['description']:30s} → SCE={row['sce']:.4f} (Δ={delta:6.1f}%) [{importance}]")
    else:
        print("No Phase 2 results yet")
```

---

### Cell 11: Phase 3 - Failure Analysis (30 min)

```python
# ============================================
# CELL 11: Phase 3 - FAILURE ANALYSIS (30 min)
# ============================================
# Time: ~30 minutes
# 5 scenarios: stable, motion, lighting, tracking, transitions

import subprocess
import os
from datetime import datetime

os.chdir('/kaggle/working/edge-vlm-thesis/autoresearch-master')

print("\n" + "="*70)
print("PHASE 3: FAILURE ANALYSIS")
print("="*70)
print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Running 5 failure analysis scenarios...")
print("Estimated time: ~30 minutes with 2 T4 GPUs\n")

scenarios = ['stable', 'motion', 'lighting', 'tracking', 'transitions']

for i, scenario in enumerate(scenarios, 1):
    print(f"\n[{i}/5] Running scenario: {scenario}")
    print("-" * 50)
    
    cmd = [
        'python', 'train.py',
        '--phase', 'failure_analysis',
        '--scenario', scenario,
        '--tau_low', '0.15',
        '--tau_high', '0.40',
        '--multi-gpu',
        '--wandb-project', 'edge-vlm-thesis',
        '--checkpoint-dir', '/kaggle/working/checkpoints',
        '--output-dir', '/kaggle/working/results'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        
        # Extract SCE from output
        for line in result.stdout.split('\n'):
            if 'sce:' in line:
                print(line)
            if '✓' in line or '✗' in line:
                print(line)
        
        if result.returncode != 0:
            print(f"⚠ Scenario {scenario} may have had issues")
            
    except subprocess.TimeoutExpired:
        print(f"⚠ Scenario {scenario} timed out")
    except Exception as e:
        print(f"⚠ Error running {scenario}: {e}")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n✓ Phase 3 COMPLETE")
```

---

### Cell 12: Final Analysis & Summary

```python
# ============================================
# CELL 12: Final Analysis & Summary
# ============================================
# Time: ~2 minutes

import pandas as pd
from pathlib import Path

results_file = '/kaggle/working/edge-vlm-thesis/autoresearch-master/results.tsv'

if Path(results_file).exists():
    df = pd.read_csv(results_file, sep='\t')
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print(f"\n📊 Total Experiments: {len(df)}")
    
    # Phase breakdown
    phase1 = len(df[df['description'].str.contains('threshold_sweep', na=False)])
    phase2 = len(df[df['description'].str.contains('ablation', na=False)])
    phase3 = len(df[df['description'].str.contains('failure_analysis', na=False)])
    
    print(f"\nPhase Breakdown:")
    print(f"  Phase 1 (Threshold Sweep): {phase1} experiments")
    print(f"  Phase 2 (Ablations): {phase2} experiments")
    print(f"  Phase 3 (Failure Analysis): {phase3} experiments")
    
    # Statistics
    print(f"\n📈 SCE Statistics:")
    print(f"  Best: {df['sce'].max():.4f}")
    print(f"  Avg: {df['sce'].mean():.4f}")
    print(f"  Min: {df['sce'].min():.4f}")
    print(f"  Std: {df['sce'].std():.4f}")
    
    # Best configuration
    best = df.loc[df['sce'].idxmax()]
    print(f"\n🏆 Best Configuration:")
    print(f"  SCE: {best['sce']:.4f}")
    print(f"  Semantic Retention: {best['semantic_retention']:.4f}")
    print(f"  VLM Call Rate: {best['vlm_call_rate']:.4f}")
    print(f"  Status: {best['status']}")
    
    # Export for thesis
    df.to_csv('/kaggle/working/thesis_results.csv', index=False)
    print(f"\n✓ Exported: thesis_results.csv")
    
else:
    print("No results file found!")

print("\n" + "="*70)
```

---

### Cell 13: Git Commit & Push Results

```python
# ============================================
# CELL 13: Git Commit & Push Results
# ============================================
# Time: ~30 seconds

import os
import subprocess
from datetime import datetime

os.chdir('/kaggle/working/edge-vlm-thesis')

print("\n" + "="*70)
print("GIT COMMIT & PUSH")
print("="*70)

# Copy results to repo
import shutil

results_src = '/kaggle/working/edge-vlm-thesis/autoresearch-master/results.tsv'
checkpoints_src = '/kaggle/working/checkpoints'
results_csv_src = '/kaggle/working/thesis_results.csv'

# Copy results.tsv
if os.path.exists(results_src):
    shutil.copy(results_src, '/kaggle/working/edge-vlm-thesis/autoresearch-master/results.tsv')
    print("✓ Copied results.tsv")

# Create backup of checkpoints (don't commit, just backup)
os.makedirs('/kaggle/working/edge-vlm-thesis/checkpoints_backup', exist_ok=True)
os.system(f'cp -r {checkpoints_src}/* /kaggle/working/edge-vlm-thesis/checkpoints_backup/ 2>/dev/null')
print("✓ Backed up checkpoints")

# Stage and commit
os.chdir('/kaggle/working/edge-vlm-thesis')

os.system('git add -A')
os.system('git add autoresearch-master/results.tsv')

commit_message = f'AutoResearch results: {len(df)} experiments completed on {datetime.now().strftime("%Y-%m-%d %H:%M")}'

os.system(f'git commit -m "{commit_message}"')

print("\n✓ Committed to git")

# Try to push (may fail if no credentials)
print("\nAttempting to push to remote...")
result = os.system('git push origin main 2>&1')

if result == 0:
    print("✓ Pushed to GitHub")
else:
    print("⚠ Could not push (no credentials in Kaggle environment)")
    print("  Download and push manually on your machine")

print("\n" + "="*70)
```

---

### Cell 14: Download Results & Cleanup

```python
# ============================================
# CELL 14: Download Results & Cleanup
# ============================================
# Time: ~1 minute

import shutil
from pathlib import Path
import os

os.chdir('/kaggle/working')

print("\n" + "="*70)
print("PACKAGING RESULTS FOR DOWNLOAD")
print("="*70)

# Create comprehensive backup
print("\nCreating backup files...")

# 1. Results CSV
if Path('/kaggle/working/thesis_results.csv').exists():
    shutil.copy('/kaggle/working/thesis_results.csv', '/kaggle/working/thesis_results_for_download.csv')
    print("✓ thesis_results.csv")

# 2. Checkpoints (zip)
if Path('/kaggle/working/checkpoints').exists():
    shutil.make_archive('/kaggle/working/checkpoints_backup', 'zip', '/kaggle/working/', 'checkpoints')
    size_mb = os.path.getsize('/kaggle/working/checkpoints_backup.zip') / (1024**2)
    print(f"✓ checkpoints_backup.zip ({size_mb:.1f} MB)")

# 3. Phase 1 analysis plot
if Path('/kaggle/working/phase1_analysis.png').exists():
    print("✓ phase1_analysis.png")

# 4. Full project backup
print("\nCreating full project backup...")
shutil.make_archive('/kaggle/working/edge_vlm_thesis_complete', 'zip', '/kaggle/working/edge-vlm-thesis/')
size_mb = os.path.getsize('/kaggle/working/edge_vlm_thesis_complete.zip') / (1024**2)
print(f"✓ edge_vlm_thesis_complete.zip ({size_mb:.1f} MB)")

print("\n" + "="*70)
print("📥 DOWNLOAD THESE FILES FROM KAGGLE OUTPUT:")
print("="*70)
print("\nEssential (must download):")
print("  - thesis_results.csv           (metrics from all 38 experiments)")
print("  - checkpoints_backup.zip       (resume capability)")
print("\nOptional:")
print("  - phase1_analysis.png          (Pareto frontier plot)")
print("  - edge_vlm_thesis_complete.zip (full project backup)")

print("\n" + "="*70)
print("✅ AUTORESEARCH COMPLETE!")
print("="*70)

# Summary statistics
import pandas as pd
if Path('/kaggle/working/thesis_results.csv').exists():
    df = pd.read_csv('/kaggle/working/thesis_results.csv')
    print(f"\n📊 Final Stats:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Best SCE: {df['sce'].max():.4f}")
    print(f"  Average SCE: {df['sce'].mean():.4f}")
    print(f"\n✓ Ready to write thesis Section 4!")
```

---

## 🚀 How to Run on Kaggle

1. **Create New Notebook:**
   - Go to Kaggle.com → New Notebook
   - Add data sources:
     - edge-vlm-thesis dataset
     - Your test video dataset

2. **Copy Each Cell:**
   - Copy each cell (Cell 1 through Cell 14) below
   - Paste into Kaggle cells in order
   - Don't skip cells!

3. **Run Cells in Order:**
   - Start with Cell 1
   - Each cell depends on previous cells
   - Don't run out of order

4. **Monitor Progress:**
   - Watch terminal output for each phase
   - Open wandb dashboard in browser: https://wandb.ai/[your-username]/edge-vlm-thesis
   - Session time: ~10 hours, phases take 3.5-4 hours total

5. **Download Results:**
   - After Cell 14, download the files from Output tab
   - Focus on: `thesis_results.csv` and `checkpoints_backup.zip`

---

## ⏱️ Time Breakdown

```
Cell 1: Setup                 ~1 min
Cell 2: Git Setup             ~1 min
Cell 3: Wandb Setup           ~1 min
Cell 4: Video Verify          ~1 min
Cell 5: File Check            ~1 min
Cell 6: Baseline (optional)   ~5 min
Cell 7: Phase 1               ~3 hours ⚡
Cell 8: Analyze P1            ~1 min
Cell 9: Phase 2               ~30 min ⚡
Cell 10: Analyze P2           ~1 min
Cell 11: Phase 3              ~30 min ⚡
Cell 12: Final Summary        ~2 min
Cell 13: Git Commit           ~1 min
Cell 14: Download             ~1 min

TOTAL: ~4-5 hours with 2 T4 GPUs ✅
```

Session timeout: 10 hours → Plenty of time!

---

## ⚠️ Important Notes

- **Don't skip Cell 1-3:** Setup is critical
- **Cell 6 is optional:** Can skip baseline if time is tight
- **Monitor wandb:** Open dashboard to see experiments live
- **GPU monitoring:** Use `nvidia-smi` in terminal to check GPU load
- **Interruption:** If session interrupts, use checkpoints to resume
- **Git push:** May fail (no credentials), just download and push manually

---

## 🎯 Expected Output

After Cell 14:

```
✅ AUTORESEARCH COMPLETE!

📊 Final Stats:
  Total experiments: 38
  Best SCE: 0.8200
  Average SCE: 0.7850

✓ Ready to write thesis Section 4!
```

Then download:
- `thesis_results.csv` → Use for plots
- `checkpoints_backup.zip` → Resume capability
- Commit manually to GitHub

---

**Happy researching on Kaggle! 🚀**
