"""HuggingFaceTB SmolVLM backend.

SmolVLM-Instruct (~2.2B parameters) fits in ~5 GB at fp16 and produces
solid scene captions.
"""

from __future__ import annotations

from typing import Any, Dict

from PIL import Image

from ...utils.logger import get_logger
from .base import (
    VLMBase, auto_image_text_model_cls, preferred_device, resolve_torch_dtype,
)

log = get_logger(__name__)


class SmolVLM(VLMBase):
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
        from transformers import AutoProcessor  # type: ignore

        hf_id = self.spec["hf_id"]
        dtype = resolve_torch_dtype(self.precision)
        log.info("Loading SmolVLM from %s (precision=%s)", hf_id, self.precision)

        self.processor = AutoProcessor.from_pretrained(hf_id)
        ModelCls = auto_image_text_model_cls()
        self.model = ModelCls.from_pretrained(hf_id, dtype=dtype).to(self.device)
        self.model.eval()
        self._loaded = True
        log.info("SmolVLM loaded on %s", self.device)

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

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        gen = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        return gen.strip()
