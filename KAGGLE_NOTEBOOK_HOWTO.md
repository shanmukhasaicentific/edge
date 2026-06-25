# How to Run AutoResearch on Kaggle (Complete Guide)

Your project is now ready to run on Kaggle with a complete notebook that handles everything.

---

## 📋 Files You Have

```
edge-vlm-thesis/
├── KAGGLE_AutoResearch_Complete.ipynb  ← Import this into Kaggle
├── KAGGLE_AUTORESEARCH_NOTEBOOK.md     ← Markdown version (reference)
└── KAGGLE_NOTEBOOK_HOWTO.md            ← This file
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Download the Notebook File

**File to download:**
- `KAGGLE_AutoResearch_Complete.ipynb`

This is a ready-to-import Jupyter notebook with all 14 cells configured.

### Step 2: Upload to Kaggle

1. Go to **Kaggle.com** → Sign in
2. Click **New Notebook** → **File** → **Import notebook**
3. Upload: `KAGGLE_AutoResearch_Complete.ipynb`
4. OR: Copy the notebook JSON directly

### Step 3: Add Datasets

In your Kaggle notebook:
1. Click **+ Add** (top right) → **Data**
2. Search and add:
   - Your uploaded `edge-vlm-thesis` dataset
   - Your test video dataset
3. Click **+ Add** → **Internet** (for wandb, if needed)

---

## 📝 Before Running

### Prerequisites

- [ ] Kaggle account (free)
- [ ] Wandb account (free): https://wandb.ai
- [ ] Wandb API key (Settings → API Keys)
- [ ] Test video uploaded to Kaggle dataset
- [ ] edge-vlm-thesis dataset uploaded to Kaggle

### Get Your Wandb API Key

1. Go to https://wandb.ai/settings/keys
2. Click "New key"
3. Copy the key
4. When notebook prompts: paste it

---

## 🏃 Running the Notebook

### 1. Set Up Environment

**Cell 1-5:** Setup & Verification
- Installs dependencies
- Clones your repo
- Logs into wandb
- Checks test video

✅ Should take: ~2 minutes

### 2. Run Phases (Main Work)

**Cell 6:** Phase 0 (Optional Baseline)
- Quick 5-minute test to verify everything works
- **OPTIONAL:** Can skip if confident

✅ Time: ~5 minutes (skip-able)

**Cell 7:** Phase 1 (Threshold Sweep)
- 28 experiments testing τ_low × τ_high combinations
- Takes all available GPU time
- Monitor wandb dashboard in browser

✅ Time: ~3 hours with 2 T4s ⚡

**Cell 8:** Analyze Phase 1
- Creates Pareto frontier plot
- Shows best configuration
- Prepares data for Phase 2

✅ Time: ~1 minute

**Cell 9:** Phase 2 (Ablations)
- 5 experiments testing component importance
- Uses best config from Phase 1

✅ Time: ~30 minutes

**Cell 10:** Analyze Phase 2
- Shows which drift component matters most
- Ranks components by importance

✅ Time: ~1 minute

**Cell 11:** Phase 3 (Failure Analysis)
- 5 scenarios (stable, motion, lighting, tracking, transitions)
- Tests robustness

✅ Time: ~30 minutes

### 3. Finalize & Download

**Cell 12:** Summary
- Total experiments count
- Best SCE score
- Exports results.csv

✅ Time: ~2 minutes

**Cell 13:** Git Commit
- Commits results to git
- Attempts to push (may fail without credentials)

✅ Time: ~30 seconds

**Cell 14:** Download Package
- Creates all downloadable files
- Packages checkpoints
- Ready for download

✅ Time: ~1 minute

---

## ⏱️ Total Timeline

```
Cell 1-5:  ~5 min     [Setup]
Cell 6:    ~5 min     [Baseline - OPTIONAL]
Cell 7:    ~3 hours   [Phase 1 - MAIN]
Cell 8:    ~1 min     [Analyze P1]
Cell 9:    ~30 min    [Phase 2]
Cell 10:   ~1 min     [Analyze P2]
Cell 11:   ~30 min    [Phase 3]
Cell 12:   ~2 min     [Summary]
Cell 13:   ~30 sec    [Git]
Cell 14:   ~1 min     [Package]
          ─────────
TOTAL:     ~4-5 hours ✅

Session time available: 10 hours → plenty of buffer!
```

---

## 👀 Monitoring While Running

### Watch GPU Usage

Keep a terminal open with:
```bash
nvidia-smi
# Or click "GPU" tab in Kaggle notebook
```

Expected with 2 T4s:
```
GPU 0: ~95% utilization
GPU 1: ~95% utilization
```

### Monitor Wandb Dashboard

While Phase 1 runs:
1. Open: https://wandb.ai/[your-username]/edge-vlm-thesis
2. Watch experiments appear in real-time
3. See SCE metric updating live
4. Already logged to wandb before download!

### Check Checkpoints

In Kaggle notebook terminal:
```bash
!ls -lh /kaggle/working/checkpoints/
# Should see: exp_001_*.pt, exp_002_*.pt, ...
```

---

## 📥 Downloading Results

### What to Download (After Cell 14)

**ESSENTIAL:**
1. `thesis_results.csv` — All 38 experiments with metrics
2. `checkpoints_backup.zip` — Resume capability if interrupted

**OPTIONAL:**
3. `phase1_analysis.png` — Pareto frontier plot
4. `edge_vlm_thesis_complete.zip` — Full project backup

### How to Download

1. In Kaggle notebook → **Output** (right panel)
2. Files are listed there
3. Click each file → **Download**

---

## 🔄 Resume If Interrupted

If your Kaggle session times out:

1. **Download checkpoints:**
   ```
   /kaggle/working/checkpoints_backup.zip
   ```

2. **On next Kaggle session:**
   - Upload checkpoints folder
   - Run Cell 3 to set up
   - Check which experiment was last:
     ```bash
     !python train.py --list-checkpoints
     ```
   - Resume from that checkpoint:
     ```bash
     !python train.py --resume-checkpoint checkpoints/exp_025_checkpoint.pt --multi-gpu
     ```

**Advantage:** No data loss, seamless resumption!

---

## ⚠️ Troubleshooting

### Issue: "No test video found"

**Solution:**
- Make sure test video is in your dataset
- Should be at: `experiments/test/test_video.mp4`

### Issue: "Wandb login failed"

**Solution:**
- Have wandb API key ready
- When prompted, paste: `wandb login <YOUR_API_KEY>`
- Or skip with: `--wandb-offline`

### Issue: "CUDA out of memory"

**Solution:**
- Kaggle T4s have 16GB VRAM each
- Should not happen with 2 T4s
- If it does: re-run Cell 7 (may be transient)

### Issue: "Git push failed"

**Solution:**
- Expected in Kaggle (no credentials)
- Download results instead
- Push manually on your local machine
- Not critical for results

### Issue: "Phase X timed out"

**Solution:**
- Kaggle sessions rarely timeout mid-cell (usually 10-12 hours)
- If it does: download checkpoints and resume
- Phone home to check checkpoints

---

## 💡 Tips & Best Practices

✅ **DO:**
- Run cells in order (don't skip Cell 1-3)
- Monitor wandb dashboard live
- Download results immediately after Cell 14
- Keep checkpoints as backup
- Commit results to git when possible

❌ **DON'T:**
- Skip Cell 1-5 setup
- Run multiple phases in parallel (sequential only)
- Forget to download results before session ends
- Close browser during Phase 1 (keep monitoring)
- Skip Cell 6 baseline if unsure about setup

---

## 🎯 Expected Output

After all cells complete:

```
✅ AUTORESEARCH COMPLETE!

📊 Final Stats:
  Total experiments: 38
    - Phase 1: 28
    - Phase 2: 5
    - Phase 3: 5

📈 Results:
  Best SCE: 0.8200
  Average SCE: 0.7850
  Semantic Retention: 94%
  VLM Call Rate: 0.042 calls/sec

✓ Ready to write thesis Section 4!
```

---

## 📊 Using Results for Thesis

### Step 1: Download thesis_results.csv

This CSV has all metrics for 38 experiments.

### Step 2: Create Plots (Python Example)

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('thesis_results.csv')

# Phase 1: Pareto frontier
phase1 = df[df['description'].str.contains('threshold_sweep')]
plt.scatter(phase1['vlm_call_rate'], phase1['semantic_retention'], 
            c=phase1['sce'], cmap='RdYlGn')
plt.xlabel('VLM Call Rate (calls/sec)')
plt.ylabel('Semantic Retention')
plt.title('Pareto Frontier')
plt.colorbar(label='SCE')
plt.show()

# Phase 2: Component importance
phase2 = df[df['description'].str.contains('ablation')]
baseline = phase2['sce'].max()
phase2['importance'] = (baseline - phase2['sce']) / baseline * 100
print(phase2[['description', 'sce', 'importance']])
```

### Step 3: Write Thesis Section 4

Use metrics as evidence for:
- Optimal threshold configuration
- Component importance ranking
- Robustness analysis
- Failure modes identified

---

## 🔗 Resources

- **Wandb Dashboard**: https://wandb.ai/[your-username]/edge-vlm-thesis
- **Kaggle Docs**: https://www.kaggle.com/docs
- **AutoResearch Framework**: https://github.com/karpathy/autoresearch
- **Your Repo**: https://github.com/shanmukhasaicentific/edge-vlm-thesis

---

## 📞 Quick Reference

| Problem | Solution |
|---------|----------|
| Where's the notebook? | File: `KAGGLE_AutoResearch_Complete.ipynb` |
| How do I import? | Upload JSON to Kaggle notebook |
| What datasets? | edge-vlm-thesis + test video |
| How long? | ~4-5 hours with 2 T4s |
| What to download? | thesis_results.csv + checkpoints_backup.zip |
| What if timeout? | Resume from checkpoint |
| How to use results? | Create plots for Section 4 |

---

## ✅ Pre-Flight Checklist

Before clicking "Run all":

- [ ] Notebook imported to Kaggle
- [ ] Datasets added (edge-vlm-thesis + video)
- [ ] Wandb API key ready
- [ ] Session time visible (should be ~10 hours)
- [ ] GPU available (should show 2 T4s)
- [ ] Read through cells (understand what runs)
- [ ] Ready to monitor wandb dashboard

---

## 🚀 Launch

When you're ready:

1. **Upload notebook** to Kaggle
2. **Add datasets** (edge-vlm-thesis + video)
3. **Click "Run All"** or run cells sequentially
4. **Monitor** wandb dashboard
5. **Download** results when complete
6. **Use** results for thesis Section 4

---

**Your thesis experiments run automatically on Kaggle. 10 hours of compute. 38 experiments. One notebook. 🎓**

Go make great research.
