# Adaptive Semantic Compute Allocation for Edge Robotic Vision-Language Systems

**M.Tech Thesis | IIIT Allahabad**

---

## Research Summary

https://claude.com/cai/oauth/authorize?code=true&client_id=9d1c250a-e61b-44d9-88ed-5944d1962f5e&response_type=code&redirect_uri=https%3A%2F%2Fplatform.claude.com%2Foauth%2Fcode%2Fcallback&scope=org%3Acreate_api_key+user%3Aprofile+user%3Ainference+user%3Asessions%3Aclaude_code+user%3Amcp_servers+user%3Afile_upload&code_challenge=cQYoxPM1XdDv0ovzPylMe2zc3j-Y90IWkdcqrSTzb14&code_challenge_method=S256&state=Wn17qYWUFX_CS5aslJ7nDRoV75jI5Dq9wfu6oCjp-rQ

Existing robotic VLM systems continuously invoke expensive multimodal reasoning even when semantic state changes are minimal. This causes unnecessary GPU usage, excessive latency, and poor edge deployment capability.

We propose a **semantic-aware compute orchestration system** that:
- Monitors **semantic drift** (D_t) across frames using CLIP embeddings, YOLO detections, and ByteTrack identity changes
- Dynamically allocates compute across **three tiers** based on drift magnitude
- **Reuses cached VLM outputs** when the scene is semantically stable
- Defines **Semantic Compute Efficiency (SCE)** as a unified evaluation metric

**Core framing:** Existing systems eliminate visual redundancy. Ours eliminates *semantic reasoning redundancy*.

---

## System Architecture

```
Robot Camera / Video Stream
          ↓
Lightweight Frame Filtering          (Tier 1 — CPU)
          ↓
YOLO Object Detection                (Tier 1 — GPU light)
          ↓
ByteTrack Object Tracking            (Tier 1 — CPU)
          ↓
CLIP Semantic Embedding Extraction   (Tier 1 — GPU)
          ↓
Temporal Semantic Memory             (CPU)
          ↓
Semantic Drift Estimation            (CPU)
   D_t = α·D_embed + β·D_objects + γ·D_track
          ↓
Adaptive Compute Allocation Policy
   D_t < τ_low  → Tier 1 (cache)
   τ_low ≤ D_t < τ_high → Tier 2 (lightweight VLM)
   D_t ≥ τ_high → Tier 3 (full VLM)
          ↓
Selective VLM Invocation (Moondream)  (Tier 2/3 — GPU)
          ↓
Semantic Cache Update
          ↓
Robotic Task Perception Layer
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### 2. Test pipeline (no VLM, fast)

```bash
python scripts/run_pipeline.py \
    --video path/to/video.mp4 \
    --policy threshold \
    --skip_vlm \
    --verbose \
    --experiment_name test_run \
    --output_dir experiments/test/
```

### 3. Run all baselines

```bash
python scripts/run_all_baselines.py \
    --video path/to/video.mp4 \
    --max_frames 300 \
    --skip_vlm
```

### 4. Threshold sweep (Pareto analysis)

```bash
python scripts/sweep_thresholds.py \
    --video path/to/video.mp4 \
    --max_frames 300 \
    --skip_vlm
```

---

## Key Formulations

**Semantic State:**
```
Z_t = f(E_t, O_t, M_t)
```

**Semantic Drift:**
```
D_t = α·D_embed + β·D_objects + γ·D_track
```

**Memory Update:**
```
M_t = λ·M_{t-1} + (1-λ)·E_t
```

**Scheduling Rule:**
```
Invoke(VLM) if D_t > τ
```

**Semantic Compute Efficiency:**
```
SCE = SemanticRetention / NormalizedComputeCost
```

---

## Baselines

| Baseline | Description |
|----------|-------------|
| Every-frame | VLM called on every frame |
| Uniform sampling | VLM every K frames |
| Motion gating | VLM on motion events |
| Embedding threshold | VLM on CLIP drift only |
| **Proposed** | Full semantic drift scheduler |

---

## Stack

| Component | Library |
|-----------|---------|
| Detection | YOLOv8 nano (ultralytics) |
| Tracking | ByteTrack (boxmot) |
| Embeddings | CLIP ViT-B/32 (openai/clip) |
| VLM | Moondream2 (vikhyatk/moondream2) |
| Framework | PyTorch, OpenCV |

---

## Project Structure

```
edge-vlm-thesis/
├── prompts/          # AI advisor prompt library
├── src/
│   ├── detection/    # FrameFilter, YOLODetector
│   ├── tracking/     # ByteTrackWrapper
│   ├── embeddings/   # CLIPExtractor
│   ├── semantic_memory/  # TemporalSemanticMemory, SemanticDriftEstimator
│   ├── scheduler/    # SemanticCache, VLMScheduler
│   ├── vlm/          # MoondreamWrapper
│   ├── policies/     # All 5 compute allocation policies
│   ├── evaluation/   # Evaluator, metrics
│   └── robotics/     # RoboticPerceptionLayer
├── scripts/          # run_pipeline.py, sweep_thresholds.py, run_all_baselines.py
├── configs/          # YAML configs for models and experiments
├── experiments/      # Per-run outputs
├── results/          # Aggregated tables and plots
├── thesis/           # LaTeX chapters
└── research_log.md   # Research journal
```

---

## Timeline

| Month | Goal |
|-------|------|
| 1 | Working prototype, drift estimation, adaptive VLM triggering |
| 2 | Baselines, ablations, threshold sweep, Pareto analysis |
| 3 | Thesis writing, plots, paper submission |
