# Paper Writing Prompt

## Purpose

Use this when: drafting thesis chapters, writing the workshop paper, polishing abstract, or writing related work.

## Paper Structure

### Workshop Paper (6–8 pages)

1. Abstract (150 words)
2. Introduction
3. Related Work
4. Methodology
5. Experiments
6. Results & Analysis
7. Conclusion

### Thesis (60–100 pages)

1. Introduction & Motivation
2. Background & Related Work
3. System Architecture
4. Semantic Drift Modeling
5. Temporal Semantic Memory
6. Adaptive Compute Allocation
7. Multi-Tier Scheduling
8. Experimental Setup
9. Results & Discussion
10. Ablation Studies
11. Conclusion & Future Work
12. References

## Key Narrative to Maintain Throughout

> Existing edge AI systems eliminate visual redundancy.
> Our system eliminates **semantic reasoning redundancy**.
> We show that semantic state transitions can drive multimodal compute allocation while preserving semantic fidelity.

## Abstract Template

```
Edge deployment of Vision-Language Models (VLMs) is constrained by
[problem]. Existing approaches optimize [what they do], but fail to
address [gap]. We propose [system name], a semantic-aware compute
orchestration framework that [core idea]. Our system models semantic
state transitions using [formulation] and dynamically allocates
multimodal compute across three tiers. Experiments on [dataset] show
[X]% reduction in VLM invocations with [Y]% semantic retention
compared to always-on baselines, achieving a Semantic Compute
Efficiency of [Z].
```

## Related Work Coverage

Must cite and position against:
- AdaVFM — adaptive video frame models
- NoScope — fast video query systems
- Reducto — frame filtering for video analytics
- Video-LLaVA — VLM for video understanding
- StreamingVLM — streaming multimodal inference
- VideoAgent — agent-based video understanding
- TinyCLIP — lightweight CLIP for edge
- Edge Intelligence survey

Positioning statement for each:
> "[Paper] optimizes [X]. Our work differs by optimizing [semantic reasoning redundancy], which [X] does not address."

## Writing Prompt Template

> "I am writing [section name] of my M.Tech thesis. Here is my current draft:
> [paste draft]
> Please:
> 1. Improve clarity and flow
> 2. Strengthen the technical argument
> 3. Ensure the edge-computing motivation is clear
> 4. Fix any vague claims — every claim must be backed by experiment or citation
> 5. Suggest one additional point I may have missed"

## Common Writing Mistakes to Avoid

- Vague: "our method is efficient" → Specific: "our method reduces VLM calls by 73% with 91% semantic retention"
- Overclaiming: "state-of-the-art" → "competitive with [baseline] while using [X]% less compute"
- Missing justification: every design choice needs either math or citation
- No ablation: every module must be ablated
