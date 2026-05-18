"""
src/vlm/qwen_wrapper.py

Qwen2.5-VL Inference Wrapper for Kaggle 2× T4 Setup.

Qwen2.5-VL is a significantly stronger VLM than Moondream2 for:
    - scene reasoning and semantic understanding
    - temporal event interpretation
    - robotics-relevant object descriptions
    - structured output generation

Model options by VRAM budget on 2× T4 (2×16GB = 32GB total):
    ┌─────────────────────────────────┬──────────┬─────────────────────┐
    │ Model                           │ VRAM     │ Fit on 2×T4?        │
    ├─────────────────────────────────┼──────────┼─────────────────────┤
    │ Qwen/Qwen2.5-VL-3B-Instruct     │ ~7GB     │ Yes, single GPU     │
    │ Qwen/Qwen2.5-VL-7B-Instruct     │ ~16GB    │ Yes, fp16 single T4 │
    │ Qwen/Qwen2.5-VL-7B-Instruct     │ ~9GB     │ Yes, 4-bit on 1 T4  │
    │ Qwen/Qwen2.5-VL-72B-Instruct    │ ~144GB   │ NO — too large      │
    └─────────────────────────────────┴──────────┴─────────────────────┘

Recommended Kaggle setup:
    GPU0 (T4 #1): YOLO + ByteTrack + CLIP  (Tier-1 perception)
    GPU1 (T4 #2): Qwen2.5-VL-7B           (Tier-2/3 reasoning)

Thesis framing:
    Using a stronger VLM does NOT change your core contribution.
    Your contribution is the semantic monitoring layer that GATES
    VLM invocations. Whether the VLM is Moondream or Qwen2.5-VL,
    the scheduling/caching/drift machinery is identical.

    Using Qwen2.5-VL strengthens your experiments because:
    - better semantic retention when VLM IS called
    - more meaningful caption consistency metrics
    - stronger robotics-relevant reasoning
    - demonstrates system works with production-grade VLMs

Hardware notes:
    - Qwen2.5-VL-7B in fp16: ~14GB VRAM, ~1-2s per call on T4
    - Qwen2.5-VL-7B in 4-bit: ~9GB VRAM, ~2-3s per call on T4
    - Qwen2.5-VL-3B in fp16: ~7GB VRAM, ~600ms per call on T4
    - Use device_map="auto" to split across 2× T4 if needed

Installation:
    pip install transformers>=4.45.0 accelerate qwen-vl-utils
    pip install bitsandbytes  # for 4-bit quantization
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.vlm.moondream_wrapper import VLMOutput   # reuse same output dataclass


class QwenVLWrapper:
    """
    Qwen2.5-VL wrapper using the HuggingFace transformers API.
    Implements the same interface as MoondreamWrapper.

    Args:
        model_id:       HuggingFace model ID
                        Recommended: "Qwen/Qwen2.5-VL-7B-Instruct"
                        Lighter:     "Qwen/Qwen2.5-VL-3B-Instruct"
        device:         'cuda', 'cuda:1', or 'auto'
                        Use 'auto' for device_map across 2× T4
        use_4bit:       Enable 4-bit quantization (reduces VRAM ~40%)
                        Trade-off: slightly slower, slightly lower quality
        use_flash_attn: Enable Flash Attention 2 (faster, needs compatible GPU)
        max_new_tokens: Max tokens to generate per call
        tier2_prompt:   Override default Tier-2 prompt
        tier3_prompt:   Override default Tier-3 prompt
    """

    TIER2_PROMPT = (
        "Describe this scene in one concise sentence. "
        "Focus on the main objects and any activity visible."
    )

    TIER3_PROMPT = (
        "You are a semantic perception system for a robot. "
        "Analyze this scene and provide: "
        "(1) Main objects present and their approximate locations, "
        "(2) Any ongoing activity or motion, "
        "(3) Semantic changes from a typical stable scene, "
        "(4) Task-relevant observations for robotic navigation or manipulation. "
        "Be specific and concise."
    )

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto",
        use_4bit: bool = False,
        use_flash_attn: bool = False,
        max_new_tokens: int = 256,
        tier2_prompt: Optional[str] = None,
        tier3_prompt: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.use_4bit = use_4bit
        self.use_flash_attn = use_flash_attn
        self.max_new_tokens = max_new_tokens

        self.tier2_prompt = tier2_prompt or self.TIER2_PROMPT
        self.tier3_prompt = tier3_prompt or self.TIER3_PROMPT

        self._model = None
        self._processor = None
        self._loaded = False

    def load(self):
        """
        Lazy-load Qwen2.5-VL model and processor.
        Call once before first inference.

        Loading strategy:
            - 4-bit: use BitsAndBytesConfig, fits 7B on single T4
            - fp16:  device_map="auto" to split across 2× T4 if needed
            - flash_attn: enable if GPU supports it (A100/H100, not T4)
        """
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        print(f"[QwenVLWrapper] Loading {self.model_id}...")
        print(f"  4-bit quantization: {self.use_4bit}")
        print(f"  Flash Attention 2:  {self.use_flash_attn}")

        # Build model kwargs
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": self.device if self.device == "auto" else None,
        }

        if self.use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            # 4-bit handles device placement internally
            model_kwargs.pop("device_map", None)
            model_kwargs["device_map"] = "auto"

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs,
        )

        # If not using device_map=auto, move to device manually
        if self.device not in ("auto",) and not self.use_4bit:
            self._model = self._model.to(self.device)

        self._model.eval()

        # Processor handles image preprocessing + tokenization
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=256 * 28 * 28,   # minimum resolution
            max_pixels=512 * 28 * 28,   # cap resolution for T4 memory
        )

        self._loaded = True
        print(f"[QwenVLWrapper] Model loaded successfully.")
        self._print_memory_usage()

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    def _frame_to_pil(self, frame: np.ndarray):
        """Convert BGR numpy frame to RGB PIL Image."""
        from PIL import Image
        rgb = frame[:, :, ::-1]
        return Image.fromarray(rgb)

    def _print_memory_usage(self):
        """Log GPU memory usage after model load."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                print(f"  GPU {i}: {allocated:.1f}GB allocated / {reserved:.1f}GB reserved")

    @torch.no_grad()
    def infer(self, frame: np.ndarray, tier: int, frame_id: int = 0) -> VLMOutput:
        """
        Run Qwen2.5-VL inference on a single frame.

        Uses the Qwen2.5-VL chat template format:
            messages = [{"role": "user", "content": [image, text]}]

        Args:
            frame:    BGR uint8 numpy array (H, W, 3)
            tier:     2 (Tier-2 lightweight) or 3 (Tier-3 full reasoning)
            frame_id: for logging

        Returns:
            VLMOutput with caption and timing
        """
        self._ensure_loaded()

        prompt_text = self.tier2_prompt if tier == 2 else self.tier3_prompt
        pil_image = self._frame_to_pil(frame)

        # Build Qwen2.5-VL chat message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text",  "text": prompt_text},
                ],
            }
        ]

        t0 = time.perf_counter()

        # Apply chat template
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs (handles image encoding + tokenization)
        inputs = self._processor(
            text=[text],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to the model's device
        if self.device == "auto":
            # With device_map=auto, move to first available CUDA device
            target_device = next(self._model.parameters()).device
        else:
            target_device = self.device

        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        # Generate
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,          # greedy decoding for consistency
            temperature=None,
            top_p=None,
        )

        # Decode only the generated tokens (not the input prompt)
        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs["input_ids"], output_ids)
        ]
        caption = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        inference_ms = (time.perf_counter() - t0) * 1000

        # Free GPU memory between calls
        del inputs, output_ids, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return VLMOutput(
            caption=caption,
            tier=tier,
            inference_ms=inference_ms,
            frame_id=frame_id,
            prompt_used=prompt_text,
        )

    def infer_tier2(self, frame: np.ndarray, frame_id: int = 0) -> VLMOutput:
        return self.infer(frame, tier=2, frame_id=frame_id)

    def infer_tier3(self, frame: np.ndarray, frame_id: int = 0) -> VLMOutput:
        return self.infer(frame, tier=3, frame_id=frame_id)

    @property
    def is_loaded(self) -> bool:
        return self._loaded
