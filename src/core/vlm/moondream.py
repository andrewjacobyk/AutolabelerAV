"""Moondream2 backend.

Moondream2 is a small (~1.9B parameter) VLM that runs comfortably in
under 4 GB VRAM at fp16 and gives high-quality scene descriptions.
"""

from __future__ import annotations

from typing import Any, Dict

from PIL import Image

from ...utils.logger import get_logger
from .base import VLMBase, preferred_device, quant_config_for, resolve_torch_dtype

log = get_logger(__name__)


class MoondreamVLM(VLMBase):
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
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_id = self.spec["hf_id"]
        revision = self.spec.get("revision", None)
        trust = bool(self.spec.get("trust_remote_code", True))

        log.info("Loading Moondream from %s (rev=%s, precision=%s)",
                 hf_id, revision, self.precision)

        dtype = resolve_torch_dtype(self.precision)
        qcfg = quant_config_for(self.precision)

        kwargs: Dict[str, Any] = dict(trust_remote_code=trust)
        if revision:
            kwargs["revision"] = revision
        if qcfg is not None:
            kwargs["quantization_config"] = qcfg
            kwargs["device_map"] = "auto"
        else:
            kwargs["dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_id, revision=revision, trust_remote_code=trust
        )
        if qcfg is None:
            self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        log.info("Moondream loaded on %s", self.device)

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
        assert self.model is not None and self.tokenizer is not None

        # Moondream API: encode_image -> answer_question(enc, question, tokenizer)
        try:
            enc = self.model.encode_image(image)
            answer = self.model.answer_question(enc, prompt, self.tokenizer)
            return str(answer).strip()
        except AttributeError:
            # Newer Moondream revisions expose model.query({image, question})
            out = self.model.query(image, prompt)
            if isinstance(out, dict):
                return str(out.get("answer", "")).strip()
            return str(out).strip()
