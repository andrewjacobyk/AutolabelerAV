"""Microsoft Florence-2 backend.

Florence-2 is very small (base=232M, large=770M) and produces detailed
captions with the ``<MORE_DETAILED_CAPTION>`` task token.  It is by far
the fastest local option.
"""

from __future__ import annotations

from typing import Any, Dict

from PIL import Image

from ...utils.logger import get_logger
from .base import VLMBase, preferred_device, resolve_torch_dtype

log = get_logger(__name__)


class FlorenceVLM(VLMBase):
    def __init__(self, model_id: str, spec: Dict[str, Any], *,
                 precision: str = "fp16", max_new_tokens: int = 96) -> None:
        super().__init__(model_id, spec, precision=precision,
                         max_new_tokens=max_new_tokens)
        self.model = None
        self.processor = None
        self.device = preferred_device()
        # Florence supports several task tokens; we default to
        # <MORE_DETAILED_CAPTION> for scene descriptions.
        self.task_token = "<MORE_DETAILED_CAPTION>"

    def load(self) -> None:
        if self._loaded:
            return
        from transformers import AutoModelForCausalLM, AutoProcessor

        hf_id = self.spec["hf_id"]
        trust = bool(self.spec.get("trust_remote_code", True))
        dtype = resolve_torch_dtype(self.precision)

        log.info("Loading Florence-2 from %s (precision=%s)", hf_id, self.precision)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=dtype, trust_remote_code=trust
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=trust)
        self.model.eval()
        self._loaded = True
        log.info("Florence-2 loaded on %s", self.device)

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

        # If user's prompt is a Florence task token, honour it; else use default.
        task = prompt.strip() if prompt.strip().startswith("<") else self.task_token

        inputs = self.processor(text=task, images=image, return_tensors="pt").to(
            self.device, resolve_torch_dtype(self.precision)
        )
        with torch.no_grad():
            gen = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max(64, self.max_new_tokens),
                do_sample=False,
                num_beams=3,
            )
        text = self.processor.batch_decode(gen, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text, task=task, image_size=(image.width, image.height)
        )
        # ``parsed`` looks like {"<MORE_DETAILED_CAPTION>": "the actual caption"}
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return str(parsed).strip()
