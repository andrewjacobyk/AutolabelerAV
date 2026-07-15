"""Qwen2-VL / Qwen2.5-VL backend (Alibaba, open-weight).

Both families share the same chat-template + processor API and are
supported natively in `transformers >= 4.45` through the
``AutoModelForImageTextToText`` factory (with a fallback to the older
``AutoModelForVision2Seq``).
"""

from __future__ import annotations

from typing import Any, Dict

from PIL import Image

from ...utils.logger import get_logger
from .base import (
    VLMBase, auto_image_text_model_cls, preferred_device,
    quant_config_for, resolve_torch_dtype,
)

log = get_logger(__name__)


class QwenVLM(VLMBase):
    """Handles Qwen2-VL-*B-Instruct and Qwen2.5-VL-*B-Instruct."""

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
        qcfg = quant_config_for(self.precision)

        log.info("Loading Qwen-VL from %s (precision=%s)", hf_id, self.precision)
        self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)

        kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if qcfg is not None:
            kwargs["quantization_config"] = qcfg
            kwargs["device_map"] = "auto"
        else:
            kwargs["dtype"] = dtype

        ModelCls = auto_image_text_model_cls()
        self.model = ModelCls.from_pretrained(hf_id, **kwargs)

        if qcfg is None:
            self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        log.info("Qwen-VL loaded on %s", self.device)

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

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        trimmed = gen_ids[:, inputs["input_ids"].shape[1]:]
        out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        return out.strip()
