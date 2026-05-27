# Literature Review Prompt

## Purpose

Use this when: writing related work, searching for papers, positioning novelty, or checking if a claim is already published.

## Must-Cover Papers

| Paper | Key Idea | How We Differ |
|-------|----------|---------------|
| **QueST** (ICLR 2026 Workshop, IIIT Allahabad) | Persistent semantic queries suppress silent drift in long-horizon tracking | We apply semantic monitoring to control VLM compute allocation, not tracking identity |
| AdaVFM (Meta/CMU, May 2026) | NAS + cloud LLM agent selects model subnet size based on task complexity | We gate VLM invocations entirely via drift; they always run the VLM, just smaller |
| NoScope | CNN cascade for fast video queries | We filter semantic reasoning redundancy, not just frames |
| Reducto | Frame filtering for video analytics | We use representation-level drift, not pixel/motion filtering |
| Video-LLaVA | VLM for video understanding | We schedule VLM calls adaptively; they always run |
| StreamingVLM | Streaming multimodal inference | We add semantic memory, drift monitoring, and 3-tier scheduling |
| VideoAgent | Memory-augmented multimodal agent | We are lightweight edge-first; they require full agent infrastructure |
| TSN | Temporal segment sampling | We adaptive-sample based on drift, not fixed segments |
| TinyCLIP | Lightweight CLIP | We use standard CLIP; TinyCLIP is a complementary future direction |
| Edge Intelligence | Survey of edge AI | Motivation for edge-oriented design |
| Cognitive Planning for Object Goal Navigation (IIIT Allahabad) | LLM+LVLM semantic planning for robotic navigation | Robotics framing and embodied AI motivation |

## Critical Notes on Key Papers

### QueST — institution connection
QueST is published at CAO Workshop @ ICLR 2026 by authors from IIIT Allahabad including G.C Nandi. This is your own institution. Your thesis committee will know this paper. You must:
1. Read the full paper carefully (attached in your project documents)
2. Clearly articulate the distinction: QueST monitors drift to preserve tracking identity; you monitor drift to allocate compute
3. Position your work as extending the semantic monitoring philosophy into a new domain (compute allocation), not competing with QueST
4. Acknowledge that your memory.py (exponential decay) is architecturally simpler than QueST's persistent learnable queries, which is a design choice justified by edge feasibility

### AdaVFM — concurrent work warning
AdaVFM was posted May 2026 (essentially concurrent). A reviewer will ask: "How does this differ from AdaVFM?" Your answer: AdaVFM always invokes the VLM, just selects a smaller subnet. Your system can skip VLM invocation entirely via cache reuse when semantic state is stable. That is a fundamentally different strategy — invocation gating vs model size selection.

### Future Work — 3D grounding
QueST additionally suppresses drift via 3D kinematic plausibility constraints. Your system has no geometric grounding. Include this as future work: "Extending the semantic monitoring layer with lightweight 3D geometric consistency checks, inspired by QueST's physical grounding, represents a promising direction for robotic deployment on depth-sensor-equipped platforms."

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
