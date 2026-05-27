# Experiments Prompt

## Purpose

Use this prompt when: planning experiments, running ablations, debugging results, or interpreting metrics.

## Experiment Matrix

| Experiment | Description | Key Metric |
|------------|-------------|------------|
| baseline_every_frame | VLM called on every frame | VLM calls/sec, GPU%, latency |
| uniform_sampling | VLM called every K frames | Same |
| motion_gating | VLM called on motion events | Same |
| semantic_gating | VLM called on CLIP drift only | Same |
| proposed | Full semantic-state-aware scheduler | Same + SCE |

## Metrics to Record for Every Experiment

```
- vlm_call_rate: VLM invocations per second
- gpu_utilization: % GPU busy (nvidia-smi)
- avg_latency_ms: per-frame wall-clock time
- semantic_retention: cosine similarity of output to every-frame baseline
- memory_mb: peak GPU memory
- throughput_fps: frames processed per second
- sce: semantic_retention / normalized_compute_cost
```

## Ablation Studies

Run ablations removing one module at a time:

1. No temporal memory (M_t = 0)
2. No object drift component (β = 0)
3. No tracking drift component (γ = 0)
4. No cache reuse (always call VLM when drift > τ)
5. Fixed τ vs. swept τ

## Threshold Sweep Protocol

```
τ_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

For each τ:
    run full pipeline on eval_set
    record: semantic_retention, vlm_call_rate, gpu_utilization
    compute SCE

Plot: Pareto frontier of semantic_retention vs. vlm_call_rate
```

## Dataset Protocol

Use short curated video clips (60–300 seconds each) from:
- Office scene (indoor, slow change)
- Traffic scene (outdoor, medium change)
- Busy corridor (indoor, fast change)

For each scene, create:
- `full_run/` — every-frame baseline results
- `proposed_run/` — system results
- `comparison/` — delta metrics

## Reproducibility Requirements

Every experiment must:
1. Set random seed (torch, numpy, random)
2. Save config YAML to `experiments/<name>/config.yaml`
3. Save raw metrics to `experiments/<name>/metrics.json`
4. Save per-frame logs to `experiments/<name>/frame_log.csv`
5. Generate plots to `results/plots/`

## Prompt Template for Running an Experiment

> "I am running [experiment name] on [dataset]. Here are my current results: [paste metrics]. Analyze:
> 1. Are these results reasonable?
> 2. What could cause [anomaly if any]?
> 3. How do these compare to expected baselines?
> 4. What should I fix before finalizing?"
