# Research Log — Edge VLM Thesis
# IIIT Allahabad | M.Tech Thesis

**Thesis Title:** Adaptive Semantic Compute Allocation for Edge Robotic Vision-Language Systems

---

## Log Format

Each entry:
```
### [DATE] — [MODULE/TASK]
**Status:** [Working / Broken / In Progress / Blocked]
**What I did:**
**What worked:**
**What didn't:**
**Next step:**
**Metrics recorded:** (if any)
```

---

## Week 1 Goals

- [ ] Verify YOLO detects correctly on sample video
- [ ] Verify ByteTrack assigns stable track IDs
- [ ] Verify CLIP embeddings are unit-normalized
- [ ] Print drift scores for 10 frames (confirm range is sensible)
- [ ] Run pipeline with `--skip_vlm` end-to-end
- [ ] Commit working skeleton to Git

---

## Week 2 Goals

- [ ] Load Moondream on Kaggle T4
- [ ] Run single VLM call — confirm output is reasonable
- [ ] Run every-frame baseline — record metrics.json
- [ ] Run proposed scheduler — compare VLM call rate

---

## Entries

### [YYYY-MM-DD] — Project Initialized
**Status:** In Progress
**What I did:** Generated full project structure and code skeleton.
**What worked:** Directory structure, all module skeletons created.
**What didn't:** Nothing run yet.
**Next step:** Install requirements, test frame_filter + YOLO on sample video.
**Metrics recorded:** None yet.

---

## Sanity Check Log

Record these values after first working run:

| Metric | Expected | Actual |
|--------|----------|--------|
| D_embed (stable scene) | < 0.05 | |
| D_embed (scene change) | > 0.20 | |
| D_total (stable) | < 0.15 | |
| D_total (change) | > 0.35 | |
| YOLO FPS (T4) | > 50 | |
| CLIP FPS (T4) | > 20 | |
| Moondream latency (T4) | ~800ms | |
| Pipeline FPS (no VLM) | > 15 | |

---

## Known Issues

(Track bugs here)

---

## Ideas / Future Work

- ONNX export for CLIP → faster edge inference
- Test TinyCLIP as drop-in for standard CLIP
- DeepStream integration for Jetson deployment
- ROS2 topic publisher for robotic percepts
