"""PaliGemma backend (Google, open-weight).

PaliGemma exposes a straightforward
``PaliGemmaForConditionalGeneration`` API where the task token is
prepended to the prompt (e.g. ``"caption en\\n"`` or ``"describe\\n"``).
"""

from __future__ import annotations

from typing import Any, Dict

from PIL import Image

from ...utils.logger import get_logger
from .base import VLMBase, preferred_device, resolve_torch_dtype

log = get_logger(__name__)


class PaliGemmaVLM(VLMBase):
    def __init__(self, model_id: str, spec: Dict[str, Any], *,
                 precision: str = "fp16", max_new_tokens: int = 96) -> None:
        super().__init__(model_id, spec, precision=precision,
                         max_new_tokens=max_new_tokens)
        self.model = None
        self.processor = None
        self.device = preferred_device()

    def load(self) -> None:
        if self._loaded:
            return
        from transformers import (  # type: ignore
            AutoProcessor,
            PaliGemmaForConditionalGeneration,
        )

        hf_id = self.spec["hf_id"]
        dtype = resolve_torch_dtype(self.precision)

        log.info("Loading PaliGemma from %s (precision=%s)", hf_id, self.precision)
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            hf_id, dtype=dtype
        ).to(self.device)
        self.model.eval()
        self._loaded = True
        log.info("PaliGemma loaded on %s", self.device)

    def unload(self) -> None:
        self.model = None
        self.processor = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._loaded = False

    def describe(self, image: Image.Image, prompt: str) -> str:
        if not self._loaded:
            self.load()
        import torch
        assert self.model is not None and self.processor is not None

        # PaliGemma treats the prompt as a task instruction. For scene
        # descriptions, prefixing with "caption en" gives strong results.
        text_prompt = prompt.strip()
        if not any(text_prompt.lower().startswith(t)
                   for t in ("caption", "describe", "detect", "ocr", "answer")):
            text_prompt = "caption en\n" + text_prompt

        inputs = self.processor(
            text=text_prompt, images=image, return_tensors="pt"
        ).to(self.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        out = self.processor.decode(gen[0][input_len:], skip_special_tokens=True)
        return out.strip()
