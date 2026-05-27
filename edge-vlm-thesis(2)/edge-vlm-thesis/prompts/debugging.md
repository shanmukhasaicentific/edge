# Debugging Prompt

## Purpose

Use this prompt when: something is broken, results look wrong, GPU OOM, slow inference, or unexpected behavior.

## Debugging Template

Paste this context when asking for help:

```
SYSTEM: edge-vlm-thesis
MODULE: [which module: detection / tracking / embeddings / scheduler / vlm / evaluation]
PROBLEM: [1-2 sentence description]
EXPECTED: [what should happen]
ACTUAL: [what is happening]
ERROR: [paste full traceback if any]
CODE: [paste the relevant function]
HARDWARE: [Kaggle T4 / laptop CPU]
```

## Common Issues & First Checks

### CLIP embedding shape mismatch
- Check: image is PIL or tensor? CLIP expects PIL.
- Check: preprocess applied before encode?

### ByteTrack losing all tracks
- Check: detections are in [x1,y1,x2,y2,score,class] format
- Check: confidence threshold not too high (try 0.3)
- Check: frame size consistent

### Moondream OOM on T4
- Check: batch size = 1
- Check: image resized to 378x378 before passing
- Check: torch.cuda.empty_cache() called between calls

### Semantic drift always zero
- Check: embeddings are L2-normalized before cosine distance
- Check: frame is actually different (not same frame twice)

### VLM never triggered (drift always < τ)
- Check: τ is not too high (try 0.3 first)
- Check: D_embed, D_objects, D_track are all being computed
- Print drift values for 10 frames to verify range

### Results not reproducible
- Set all seeds at start:
```python
import random, numpy as np, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)
```

## Performance Debugging

If FPS is too low:
1. Profile with `torch.profiler` to find bottleneck
2. Check: CLIP running on GPU or CPU?
3. Check: YOLO running in half precision?
4. Check: ByteTrack called every frame or only on detections?

## Sanity Check Protocol

Before running any full experiment:
1. Run pipeline on 10-frame clip
2. Print drift values for each frame
3. Confirm VLM called at least once
4. Confirm cache hit at least once
5. Confirm metrics dict has all required keys
