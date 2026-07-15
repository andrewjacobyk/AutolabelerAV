"""InternVL 2 / 2.5 / 3 backend (OpenGVLab, Shanghai AI Lab).

InternVL ships a custom ``model.chat(tokenizer, pixel_values, question,
generation_config)`` API instead of the standard HF generation loop,
so the input has to be preprocessed manually with a bicubic 448x448
resize + ImageNet normalisation.
"""

from __future__ import annotations

from typing import Any, Dict

from PIL import Image

from ...utils.logger import get_logger
from .base import VLMBase, preferred_device, quant_config_for, resolve_torch_dtype

log = get_logger(__name__)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int = 448):
    import torchvision.transforms as T  # type: ignore
    from torchvision.transforms.functional import InterpolationMode  # type: ignore

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class InternVLM(VLMBase):
    """Handles InternVL2, InternVL2.5 and InternVL3 checkpoints."""

    def __init__(self, model_id: str, spec: Dict[str, Any], *,
                 precision: str = "fp16", max_new_tokens: int = 96) -> None:
        super().__init__(model_id, spec, precision=precision,
                         max_new_tokens=max_new_tokens)
        self.model = None
        self.tokenizer = None
        self._transform = None
        self.device = preferred_device()

    def load(self) -> None:
        if self._loaded:
            return
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        hf_id = self.spec["hf_id"]
        dtype = resolve_torch_dtype(self.precision)
        qcfg = quant_config_for(self.precision)

        log.info("Loading InternVL from %s (precision=%s)", hf_id, self.precision)

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
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=True, use_fast=False,
        )
        self._transform = _build_transform(448)
        self._loaded = True
        log.info("InternVL loaded on %s", self.device)

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self._transform = None
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
        assert self._transform is not None

        pixel_values = self._transform(image).unsqueeze(0)
        target_dtype = next(self.model.parameters()).dtype
        pixel_values = pixel_values.to(self.device, dtype=target_dtype)

        question = "<image>\n" + prompt
        generation_config = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer, pixel_values, question, generation_config
            )
        if isinstance(response, tuple):
            response = response[0]
        return str(response).strip()
