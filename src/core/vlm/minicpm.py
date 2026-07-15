"""MiniCPM-V backend (OpenBMB / Tsinghua, Chinese).

MiniCPM-V 2.6 is ~8B parameters; run it with ``precision: int4`` on an
8 GB GPU or ``bf16`` on a machine with at least 16 GB VRAM.
"""

from __future__ import annotations

from typing import Any, Dict

from PIL import Image

from ...utils.logger import get_logger
from .base import VLMBase, preferred_device, quant_config_for, resolve_torch_dtype

log = get_logger(__name__)


class MiniCPMVLM(VLMBase):
    def __init__(self, model_id: str, spec: Dict[str, Any], *,
                 precision: str = "fp16", max_new_tokens: int = 96) -> None:
        super().__init__(model_id, spec, precision=precision,
                         max_new_tokens=max_new_tokens)
        self.model = None
        self.tokenizer = None
        self.device = preferred_device()

    def load(self) -> None:
        if self._loaded:
            return
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        hf_id = self.spec["hf_id"]
        dtype = resolve_torch_dtype(self.precision)
        qcfg = quant_config_for(self.precision)

        log.info("Loading MiniCPM-V from %s (precision=%s)", hf_id, self.precision)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

        kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if qcfg is not None:
            kwargs["quantization_config"] = qcfg
            kwargs["device_map"] = "auto"
        else:
            kwargs["dtype"] = dtype

        self.model = AutoModel.from_pretrained(hf_id, **kwargs)
        if qcfg is None:
            self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        log.info("MiniCPM-V loaded on %s", self.device)

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
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
        assert self.model is not None and self.tokenizer is not None

        msgs = [{"role": "user", "content": [image, prompt]}]
        with torch.no_grad():
            answer = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                sampling=False,
            )
        return str(answer).strip()
