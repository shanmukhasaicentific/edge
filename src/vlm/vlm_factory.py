"""
src/vlm/vlm_factory.py

VLM Factory — single point of VLM instantiation.

Keeps run_pipeline.py and all experiment scripts clean.
Add new VLMs here without touching pipeline code.

Usage:
    vlm = build_vlm("qwen_7b", device="auto", use_4bit=True)
    vlm = build_vlm("qwen_3b", device="cuda:1")
    vlm = build_vlm("moondream", device="cuda")

Registry:
    moondream  → MoondreamWrapper (default, lightweight, 1.9B)
    qwen_3b    → QwenVLWrapper 3B (fast, good reasoning)
    qwen_7b    → QwenVLWrapper 7B (recommended for Kaggle 2×T4)

Thesis note:
    Swapping VLMs via this factory is a zero-code-change operation.
    The entire pipeline (scheduler, cache, monitor, evaluator) is
    VLM-agnostic. This is by design — your contribution is the
    scheduling layer, not the VLM itself.
"""

from typing import Optional


# ─── Registry ─────────────────────────────────────────────────────────────────

VLM_REGISTRY = {
    "moondream": {
        "class": "MoondreamWrapper",
        "module": "src.vlm.moondream_wrapper",
        "model_id": "vikhyatk/moondream2",
        "description": "Moondream2 1.9B — lightweight baseline, ~800ms/call on T4",
    },
    "qwen_3b": {
        "class": "QwenVLWrapper",
        "module": "src.vlm.qwen_wrapper",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "description": "Qwen2.5-VL 3B — fast, strong reasoning, ~600ms/call on T4",
    },
    "qwen_7b": {
        "class": "QwenVLWrapper",
        "module": "src.vlm.qwen_wrapper",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "description": "Qwen2.5-VL 7B — best reasoning, ~1-2s/call on T4 fp16",
    },
    "qwen_7b_4bit": {
        "class": "QwenVLWrapper",
        "module": "src.vlm.qwen_wrapper",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "description": "Qwen2.5-VL 7B 4-bit — good reasoning, ~9GB VRAM, ~2-3s/call",
        "use_4bit": True,
    },
}


def build_vlm(
    name: str = "moondream",
    device: str = "cuda",
    use_4bit: bool = False,
    max_new_tokens: int = 256,
    tier2_prompt: Optional[str] = None,
    tier3_prompt: Optional[str] = None,
):
    """
    Instantiate a VLM by registry name.

    Args:
        name:           Registry key (moondream / qwen_3b / qwen_7b / qwen_7b_4bit)
        device:         'cuda', 'cuda:1', 'auto', or 'cpu'
                        Use 'auto' for device_map across 2× T4 GPUs
        use_4bit:       Force 4-bit quantization (Qwen only, overrides registry default)
        max_new_tokens: Generation length cap
        tier2_prompt:   Custom Tier-2 prompt override
        tier3_prompt:   Custom Tier-3 prompt override

    Returns:
        Loaded VLM wrapper with .infer(), .infer_tier2(), .infer_tier3() interface

    Raises:
        ValueError: if name not in registry
    """
    if name not in VLM_REGISTRY:
        raise ValueError(
            f"Unknown VLM: '{name}'. "
            f"Available: {list(VLM_REGISTRY.keys())}"
        )

    entry = VLM_REGISTRY[name]
    print(f"[VLMFactory] Building VLM: {name}")
    print(f"  Model: {entry['model_id']}")
    print(f"  {entry['description']}")

    # Dynamic import to avoid loading all VLMs at startup
    import importlib
    module = importlib.import_module(entry["module"])
    cls = getattr(module, entry["class"])

    # Build init kwargs
    kwargs = {
        "model_id": entry["model_id"],
        "device": device,
    }

    if entry["class"] == "QwenVLWrapper":
        kwargs["use_4bit"] = use_4bit or entry.get("use_4bit", False)
        kwargs["max_new_tokens"] = max_new_tokens
        if tier2_prompt:
            kwargs["tier2_prompt"] = tier2_prompt
        if tier3_prompt:
            kwargs["tier3_prompt"] = tier3_prompt
    elif entry["class"] == "MoondreamWrapper":
        if tier2_prompt:
            kwargs["tier2_prompt"] = tier2_prompt
        if tier3_prompt:
            kwargs["tier3_prompt"] = tier3_prompt

    return cls(**kwargs)


def list_vlms() -> dict:
    """Return registry entries for display/logging."""
    return {k: v["description"] for k, v in VLM_REGISTRY.items()}
