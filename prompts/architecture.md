# Architecture Prompt

## Purpose

Use this when: making architecture decisions, adding new modules, questioning design trade-offs, or reviewing code structure.

## Architecture Principles

1. **Semantic-first**: Every module exists to serve semantic reasoning, not just visual processing
2. **Tier-aware**: Every component knows which compute tier it belongs to
3. **Modular**: Every component replaceable independently (swap Moondream → SmolVLM with one config change)
4. **Metric-instrumented**: Every component emits timing and cost metrics
5. **Config-driven**: No magic numbers in code — all hyperparameters in YAML configs

## Module Responsibilities

| Module | File | Responsibility |
|--------|------|----------------|
| FrameFilter | src/detection/frame_filter.py | Drop redundant frames early (motion/blur) |
| Detector | src/detection/yolo_detector.py | YOLOv8 nano object detection |
| Tracker | src/tracking/bytetrack_wrapper.py | ByteTrack multi-object tracking |
| EmbeddingExtractor | src/embeddings/clip_extractor.py | CLIP ViT-B/32 semantic embeddings |
| SemanticMemory | src/semantic_memory/memory.py | Temporal memory with exponential decay |
| DriftEstimator | src/semantic_memory/drift.py | Compute D_t from embedding + object + track drift |
| ComputePolicy | src/policies/adaptive_policy.py | Map D_t → compute tier decision |
| VLMScheduler | src/scheduler/vlm_scheduler.py | Gate and queue VLM calls |
| VLMWrapper | src/vlm/moondream_wrapper.py | Moondream inference interface |
| SemanticCache | src/scheduler/semantic_cache.py | Cache VLM outputs for reuse |
| Evaluator | src/evaluation/evaluator.py | Compute all metrics per run |

## Data Flow Contract

Every frame must produce a `FrameState` dict:

```python
FrameState = {
    "frame_id": int,
    "timestamp": float,
    "detections": List[Detection],    # from YOLO
    "tracks": List[Track],            # from ByteTrack
    "embedding": np.ndarray,          # from CLIP (512-dim)
    "drift": float,                   # from DriftEstimator
    "tier": int,                      # 1, 2, or 3
    "vlm_called": bool,
    "vlm_output": Optional[str],
    "cache_hit": bool,
    "compute_cost_ms": float,
}
```

## Config-Driven Design

All modules loaded from `configs/models/default.yaml`. Example:

```yaml
detector:
  model: yolov8n.pt
  conf_threshold: 0.3
  device: cuda

tracker:
  tracker_type: bytetrack

embeddings:
  model: ViT-B/32
  device: cuda
  normalize: true

semantic_memory:
  decay_lambda: 0.9
  memory_size: 10

drift:
  alpha: 0.5   # embedding weight
  beta: 0.3    # object weight
  gamma: 0.2   # tracking weight

policy:
  tau_high: 0.6   # trigger tier 3
  tau_mid: 0.3    # trigger tier 2
  tau_low: 0.1    # cache reuse

vlm:
  model: moondream
  device: cuda
  prompt_template: "Describe the scene and any significant activity."
```

## Architecture Prompt Template

> "I am making an architecture decision about [module]. My options are:
> Option A: [describe]
> Option B: [describe]
> My constraints: Kaggle T4, 3-month timeline, no training.
> Which option better serves semantic reasoning redundancy reduction?
> Justify with compute cost analysis."
