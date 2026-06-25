# Karpathy's AutoResearch Customized for Edge VLM Thesis

This is **Karpathy's actual AutoResearch framework** adapted for your M.Tech thesis on semantic-aware VLM scheduling.

---

## What's This?

Karpathy's AutoResearch is an autonomous research framework that runs experiments in a loop while you sleep. It:

1. **Reads your research protocol** (program.md)
2. **Generates experiments** based on what it learns
3. **Runs them autonomously** (threshold sweeps, ablations, failure analysis)
4. **Logs everything** (results.tsv, metrics.json, research_log.json)
5. **Decides what to test next** based on results

---

## Files

| File | Purpose | Can Modify? |
|------|---------|-------------|
| `program.md` | Your research protocol (3 phases) | ✗ Read-only (defines what to do) |
| `train.py` | Experiment runner (scheduler optimizer) | ✓ Can modify parameters/phases |
| `prepare.py` | Environment setup & validation | ✗ Read-only (fixed metrics) |
| `results.tsv` | Experiment results log | ✗ Auto-generated (do NOT commit) |
| `analysis.ipynb` | Results analysis (from Karpathy) | ✓ Can customize plots |

---

## How It Works

### The Three-Phase Loop

**Phase 1: Threshold Sweep** (Find optimal τ_low, τ_high)
```
tau_low ∈ [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
tau_high ∈ [0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
→ 28 experiments testing the Pareto frontier
Goal: Find configuration with highest SCE
```

**Phase 2: Ablation Studies** (Understand component importance)
```
Baseline:      α=0.5, β=0.3, γ=0.2
No embedding:  α=0.0, β=0.5, γ=0.5
No objects:    α=1.0, β=0.0, γ=0.0
No tracking:   α=0.6, β=0.4, γ=0.0
Uniform:       α=0.33, β=0.33, γ=0.33
→ 5 experiments testing component importance
Goal: Rank which drift component matters most
```

**Phase 3: Failure Analysis** (Test edge cases)
```
Stable scene
Rapid transitions
Camera motion
Lighting changes
Object persistence
→ 5 experiments on diverse scenarios
Goal: Identify failure modes and confounding factors
```

---

## Setup

### 1. Verify Environment

```bash
python prepare.py
```

This checks:
- ✓ Test video exists in ../experiments/test/
- ✓ Project code in ../src/
- ✓ Python dependencies installed
- ✓ Fixed metric definitions (SCE calculation)

### 2. Check Git Status

```bash
cd ..  # Go to parent edge-vlm-thesis directory
git status
git branch  # Should be on main or a feature branch
```

### 3. Create AutoResearch Branch

```bash
git checkout -b autoresearch/jun25  # Use today's date
cd autoresearch-master
```

### 4. Initialize Results File

```bash
# Create empty results.tsv with header
touch results.tsv
```

---

## Running Experiments

### Single Experiment

Run one threshold sweep experiment:

```bash
python train.py --phase threshold_sweep --config exp_001
```

Output:
```
---
sce:                  0.8200
semantic_retention:   0.9400
vlm_call_rate:        0.0420
latency_ms:           45.3
gpu_utilization:      0.42
elapsed_seconds:      127.5
false_positives:      2
false_negatives:      0
```

### Phase 1: Full Threshold Sweep

```bash
# Run all 28 tau_low x tau_high combinations
for tau_low in 0.05 0.10 0.15 0.20 0.25 0.30; do
  for tau_high in 0.25 0.30 0.35 0.40 0.50 0.60; do
    if (( $(echo "$tau_low < $tau_high" | bc -l) )); then
      python train.py \
        --phase threshold_sweep \
        --tau_low $tau_low \
        --tau_high $tau_high
    fi
  done
done
```

Better: Use the autonomous loop (train.py handles this).

### Phase 2: Ablation Studies

```bash
# After Phase 1, identify best tau_low, tau_high from results.tsv
# Then run ablations:

python train.py --phase ablation --alpha 0.0 --beta 0.5 --gamma 0.5
python train.py --phase ablation --alpha 1.0 --beta 0.0 --gamma 0.0
python train.py --phase ablation --alpha 0.6 --beta 0.4 --gamma 0.0
python train.py --phase ablation --alpha 0.33 --beta 0.33 --gamma 0.33
```

### Phase 3: Failure Analysis

```bash
# Test on specific scenarios:

python train.py --phase failure_analysis --scenario stable
python train.py --phase failure_analysis --scenario transitions
python train.py --phase failure_analysis --scenario motion
python train.py --phase failure_analysis --scenario lighting
python train.py --phase failure_analysis --scenario tracking
```

---

## Autonomous Mode (Recommended)

The whole point of AutoResearch is **you don't manually run experiments**. Instead:

1. **Read program.md** to understand the three phases
2. **Commit your changes** to the autoresearch branch
3. **Run train.py in a loop** (or use the autonomous agent)
4. **Let it run overnight** (50-100 experiments)
5. **Wake up to results** the next morning

### Manual Autonomous Loop

```bash
# Keep running experiments and tracking them
while true; do
  python train.py --phase threshold_sweep

  # Check results.tsv to see if improved
  tail -1 results.tsv

  # If you're satisfied, break (Ctrl+C)
done
```

### With Git Workflow (Like Karpathy's)

```bash
# 1. Modify train.py with new hypothesis
vim train.py

# 2. Commit
git commit -am "Test: different tau thresholds"

# 3. Run experiment
python train.py > run.log 2>&1

# 4. Check results
grep "^sce:" run.log

# 5. Log to results.tsv (manually or auto)
# If improved: keep commit, otherwise reset
git reset --hard HEAD~1

# 6. Repeat
```

---

## Interpreting Results

### results.tsv Format

```
commit	sce	semantic_retention	vlm_call_rate	false_pos	false_neg	status	description
a1b2c3d	0.8200	0.9400	0.0420	2	0	keep	threshold_sweep: tau_low=0.15, tau_high=0.40
b2c3d4e	0.7900	0.9200	0.0680	3	1	discard	threshold_sweep: tau_low=0.10, tau_high=0.35
c3d4e5f	0.8150	0.9300	0.0450	1	0	keep	ablation: no_tracking
```

**Key metrics:**
- `sce`: Primary metric (higher is better)
- `semantic_retention`: Fraction of frames with valid state (higher is better)
- `vlm_call_rate`: VLM calls per second (lower is more efficient)
- `false_pos`: Unnecessary VLM calls (lower is better)
- `false_neg`: Missed semantic changes (lower is better)

### Finding the Optimal Configuration

After Phase 1, plot results.csv:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../results.csv')

# Plot Pareto frontier
plt.scatter(df['vlm_call_rate'], df['semantic_retention'], c=df['sce'], cmap='RdYlGn')
plt.xlabel('VLM Call Rate (lower is better)')
plt.ylabel('Semantic Retention (higher is better)')
plt.colorbar(label='SCE')
plt.title('Pareto Frontier')
plt.show()

# Best SCE
best = df.loc[df['sce'].idxmax()]
print(f"Best SCE: {best['sce']:.4f}")
print(f"  tau_low = {best['tau_low']:.2f}")
print(f"  tau_high = {best['tau_high']:.2f}")
```

---

## Key Constraints

**You CAN:**
- Modify `train.py` — add new phases, change which experiments to run
- Change thresholds and weights in program.md
- Modify the logic for what to test next
- Add new ablations or scenarios

**You CANNOT:**
- Modify `prepare.py` — it defines the ground truth metric (SCE calculation)
- Change the definition of semantic_retention or vlm_call_rate
- Modify the test video (must be the same for all runs)
- Use external packages not already imported

**The metric is fixed:**
```python
SCE = semantic_retention / vlm_call_rate
```

You can't change how SCE is calculated. Only improve it by finding better thresholds/weights.

---

## Expected Outcomes

After running all three phases:

✅ **Phase 1 Results**: 28 experiments, Pareto frontier plot, optimal τ_low and τ_high identified
✅ **Phase 2 Results**: 5 experiments, component importance ranking, understanding of what matters
✅ **Phase 3 Results**: 5 experiments, failure modes identified, false positive/negative analysis

**For your thesis:**
- `results.csv` → Figures for Section 4 (Experiments)
- `research_log.json` → Evidence for claims in paper
- `program.md` → Reproducibility statement
- `results.tsv` → Supplementary materials

---

## Troubleshooting

### "No test video found"
```bash
# Make sure you have a video in:
experiments/test/
# File should be .mp4, .avi, or .mov
```

### "prepare.py validation failed"
```bash
# Check that src/ directory exists with semantic_memory and scheduler modules
ls ../src/
```

### "Train.py times out"
```bash
# Increase timeout in train.py or check that pipeline.py exists
ls ../scripts/run_pipeline.py
```

### "SCE is 0 or very low"
```bash
# Check that your test video is reasonable (30+ seconds, actual scene content)
# Check that vlm_call_rate is > 0 (not div by zero)
```

---

## Next Steps

1. **Run prepare.py** to validate environment
2. **Commit to autoresearch branch** (`git checkout -b autoresearch/jun25`)
3. **Run Phase 1** (threshold sweep) — takes ~1-2 hours
4. **Analyze results** — find optimal configuration
5. **Run Phase 2** (ablations) — takes ~30 minutes
6. **Run Phase 3** (failure analysis) — takes ~30 minutes
7. **Write thesis Section 4** using results

---

## References

- Original Karpathy AutoResearch: https://github.com/karpathy/autoresearch
- Your thesis project: ../README.md
- Research protocol: program.md
- Experiment code: train.py

---

**Ready to run autonomous experiments. The framework will handle the rest.**
