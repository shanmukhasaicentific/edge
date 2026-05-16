"""
src/vlm/moondream_wrapper.py

Moondream VLM Inference Wrapper.

Moondream2 is a ~1.9B parameter vision-language model.
It is the primary Tier-3 compute module in this system.

Usage pattern:
    - Only called when D_t > τ_high (semantic drift threshold)
    - Input: single frame (PIL image)
    - Output: natural language scene description

Tier 2 lightweight captioning uses a shorter, cheaper prompt.
Tier 3 full reasoning uses a richer, deeper prompt.

Hardware note:
    - T4 (Kaggle): ~800ms per call at fp16
    - CPU (laptop): ~8-12 seconds per call (fallback only)
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class VLMOutput:
    caption: str
    tier: int
    inference_ms: float
    frame_id: int
    prompt_used: str


class MoondreamWrapper:
    """
    Moondream2 VLM wrapper with Tier-2 and Tier-3 prompts.

    Args:
        model_id:    HuggingFace model ID
        device:      'cuda' or 'cpu'
        dtype:       torch.float16 (T4) or torch.float32 (CPU)
        tier2_prompt: short prompt for lightweight captioning (Tier 2)
        tier3_prompt: rich prompt for full semantic reasoning (Tier 3)
    """

    TIER2_PROMPT = "Briefly describe the scene in one sentence."
    TIER3_PROMPT = (
        "Describe the scene in detail. "
        "List the main objects present, any ongoing activities, "
        "and any significant changes from a typical stable scene. "
        "Focus on semantic content relevant to robotic perception."
    )

    def __init__(
        self,
        model_id: str = "vikhyatk/moondream2",
        device: str = "cuda",
        dtype=None,
        tier2_prompt: Optional[str] = None,
        tier3_prompt: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)

        self.tier2_prompt = tier2_prompt or self.TIER2_PROMPT
        self.tier3_prompt = tier3_prompt or self.TIER3_PROMPT

        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self):
        """
        Lazy-load the model. Call once before first inference.
        Separated from __init__ to allow fast unit testing without loading weights.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[MoondreamWrapper] Loading {self.model_id} on {self.device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            revision="2024-08-26",
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            revision="2024-08-26",
            torch_dtype=self.dtype,
        ).to(self.device)
        self._model.eval()
        self._loaded = True
        print(f"[MoondreamWrapper] Model loaded.")

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    def _frame_to_pil(self, frame: np.ndarray):
        """Convert BGR numpy frame to RGB PIL Image."""
        from PIL import Image
        rgb = frame[:, :, ::-1]
        return Image.fromarray(rgb)

    @torch.no_grad()
    def infer(self, frame: np.ndarray, tier: int, frame_id: int = 0) -> VLMOutput:
        """
        Run Moondream inference on a single frame.

        Args:
            frame:    BGR uint8 numpy array
            tier:     2 (lightweight) or 3 (full reasoning)
            frame_id: for logging

        Returns:
            VLMOutput with caption and timing info
        """
        self._ensure_loaded()

        prompt = self.tier2_prompt if tier == 2 else self.tier3_prompt
        pil_image = self._frame_to_pil(frame)

        t0 = time.perf_counter()

        # Moondream2 API: encode_image → answer_question
        enc_image = self._model.encode_image(pil_image)
        caption = self._model.answer_question(enc_image, prompt, self._tokenizer)

        inference_ms = (time.perf_counter() - t0) * 1000

        # Free memory on GPU between calls
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return VLMOutput(
            caption=caption,
            tier=tier,
            inference_ms=inference_ms,
            frame_id=frame_id,
            prompt_used=prompt,
        )

    def infer_tier2(self, frame: np.ndarray, frame_id: int = 0) -> VLMOutput:
        return self.infer(frame, tier=2, frame_id=frame_id)

    def infer_tier3(self, frame: np.ndarray, frame_id: int = 0) -> VLMOutput:
        return self.infer(frame, tier=3, frame_id=frame_id)

    @property
    def is_loaded(self) -> bool:
        return self._loaded
