"""Salesforce BLIP-2 backend (image captioning)."""

from __future__ import annotations

from typing import Any, Dict

from PIL import Image

from ...utils.logger import get_logger
from .base import VLMBase, preferred_device, quant_config_for, resolve_torch_dtype

log = get_logger(__name__)


class BLIP2VLM(VLMBase):
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
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        hf_id = self.spec["hf_id"]
        dtype = resolve_torch_dtype(self.precision)
        qcfg = quant_config_for(self.precision)

        log.info("Loading BLIP-2 from %s (precision=%s)", hf_id, self.precision)
        self.processor = Blip2Processor.from_pretrained(hf_id)
        kwargs: Dict[str, Any] = {}
        if qcfg is not None:
            kwargs["quantization_config"] = qcfg
            kwargs["device_map"] = "auto"
        else:
            kwargs["dtype"] = dtype
        self.model = Blip2ForConditionalGeneration.from_pretrained(hf_id, **kwargs)
        if qcfg is None:
            self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        log.info("BLIP-2 loaded on %s", self.device)

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

        # BLIP-2 works best with a short "Question: ... Answer:" prompt.
        text_prompt = f"Question: {prompt} Answer:"
        inputs = self.processor(images=image, text=text_prompt,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
