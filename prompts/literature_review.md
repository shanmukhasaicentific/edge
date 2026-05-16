# Literature Review Prompt

## Purpose

Use this when: writing related work, searching for papers, positioning novelty, or checking if a claim is already published.

## Must-Cover Papers

| Paper | Key Idea | How We Differ |
|-------|----------|---------------|
| AdaVFM | Adaptive video frame models | We operate at semantic state level, not frame level |
| NoScope | CNN cascade for fast video queries | We use semantic drift, not query-specific caches |
| Reducto | Frame filtering for video analytics | We filter semantic reasoning, not just frames |
| Video-LLaVA | VLM for video understanding | We schedule VLM calls; they always run |
| StreamingVLM | Streaming multimodal inference | We add semantic memory and drift modeling |
| VideoAgent | Agent-based video understanding | We are lightweight; they require full agents |
| TSN | Temporal segment sampling | We adaptive-sample based on drift, not fixed segments |
| TinyCLIP | Lightweight CLIP | We use standard CLIP; TinyCLIP is complementary |
| Edge Intelligence | Survey of edge AI | Motivation for our edge-oriented design |

## Novelty Checklist

Before claiming novelty, verify:
- [ ] No paper combines semantic drift + temporal memory + compute tier allocation for edge VLMs
- [ ] No paper uses ByteTrack identity change as a semantic drift signal
- [ ] No paper defines Semantic Compute Efficiency (SCE) as a unified metric
- [ ] No paper ablates α, β, γ drift weights for VLM scheduling

## Search Prompt Template

> "I am doing a literature review for my thesis on semantic-aware VLM compute scheduling for edge systems. Search for papers on:
> [topic]
> For each paper, tell me:
> 1. Core idea in one sentence
> 2. How it relates to semantic reasoning redundancy
> 3. Whether it's a baseline, related work, or motivation paper
> Do not suggest papers that require training new foundation models."

## Related Work Structure

Organize related work into sections:

1. **Video Understanding with VLMs** — Video-LLaVA, VideoAgent, StreamingVLM
2. **Efficient Video Analytics** — NoScope, Reducto, AdaVFM
3. **Temporal Sampling Strategies** — TSN, fixed-interval methods
4. **Lightweight Multimodal Models** — TinyCLIP, Moondream, SmolVLM
5. **Edge AI Systems** — Edge Intelligence survey, Jetson deployments
6. **Semantic Caching** — KV-cache work, query-result caching

## Gap Statement Template

> "While [related work] reduces [X], none address the problem of **semantic reasoning redundancy** — the unnecessary re-invocation of expensive VLM inference when semantic state has not meaningfully changed. Our work is the first to define and optimize this form of computational waste through formal semantic drift modeling and adaptive multi-tier compute allocation."
