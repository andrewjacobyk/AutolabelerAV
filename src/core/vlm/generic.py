"""Generic HuggingFace VLM fallback (Vision2Seq).

Used when a model family isn't explicitly implemented above but the
weights on the Hub follow the ``AutoModelForVision2Seq`` interface.
"""

from __future__ import annotations

from typing import Any, Dict

from PIL import Image

from ...utils.logger import get_logger
from .base import (
    VLMBase, auto_image_text_model_cls, preferred_device, resolve_torch_dtype,
)

log = get_logger(__name__)


class GenericHFVLM(VLMBase):
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
        trust = bool(self.spec.get("trust_remote_code", False))
        dtype = resolve_torch_dtype(self.precision)
        log.info("Loading generic VLM from %s", hf_id)

        self.processor = AutoProcessor.from_pretrained(
            hf_id, trust_remote_code=trust
        )
        ModelCls = auto_image_text_model_cls()
        self.model = ModelCls.from_pretrained(
            hf_id, dtype=dtype, trust_remote_code=trust,
        ).to(self.device)
        self.model.eval()
        self._loaded = True

    def describe(self, image: Image.Image, prompt: str) -> str:
        if not self._loaded:
            self.load()
        import torch
        assert self.model is not None and self.processor is not None

        inputs = self.processor(images=image, text=prompt,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
