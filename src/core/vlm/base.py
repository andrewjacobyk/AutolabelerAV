"""Abstract base for vision-language model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image


class VLMBase(ABC):
    """Common interface for local & cloud VLMs."""

    def __init__(self, model_id: str, spec: Dict[str, Any], *,
                 precision: str = "fp16", max_new_tokens: int = 96) -> None:
        self.model_id = model_id
        self.spec = spec
        self.precision = precision
        self.max_new_tokens = max_new_tokens
        self._loaded = False

    # ------------------------------------------------------------------
    @abstractmethod
    def load(self) -> None:
        """Load weights / open the client."""

    @abstractmethod
    def describe(self, image: Image.Image, prompt: str) -> str:
        """Return a text description of ``image`` given ``prompt``."""

    # ------------------------------------------------------------------
    def unload(self) -> None:
        """Release resources. Default: no-op."""
        self._loaded = False

    def describe_many(self, images: Iterable[Image.Image], prompt: str) -> List[str]:
        """Simple sequential wrapper; subclasses may override for batching."""
        return [self.describe(img, prompt) for img in images]

    # ------------------------------------------------------------------
    @staticmethod
    def open_image(path: str | Path) -> Image.Image:
        img = Image.open(str(path))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    # ------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# Local-VLM helpers shared by concrete implementations
# ---------------------------------------------------------------------------
def resolve_torch_dtype(precision: str):
    """Map the user-friendly precision string to a ``torch.dtype``."""
    import torch  # local import to keep base module light

    p = (precision or "fp16").lower()
    if p in {"fp32", "float32"}:
        return torch.float32
    if p in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if p in {"fp16", "float16"}:
        return torch.float16
    # int8/int4 handled through bitsandbytes quantization config elsewhere
    return torch.float16


def preferred_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def quant_config_for(precision: str):
    """Return a BitsAndBytesConfig for int8/int4 or None otherwise."""
    p = (precision or "").lower()
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
    except Exception:
        return None
    if p in {"int8", "8bit"}:
        return BitsAndBytesConfig(load_in_8bit=True)
    if p in {"int4", "4bit"}:
        import torch
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return None


def auto_image_text_model_cls():
    """Return the transformers class that instantiates image-text VLMs.

    * transformers >= 4.45 exposes ``AutoModelForImageTextToText``.
    * Older versions (or the legacy path) use ``AutoModelForVision2Seq``.
    * transformers 5.x removed the older name entirely.
    """
    import transformers  # local import
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise ImportError(
        "Neither AutoModelForImageTextToText nor AutoModelForVision2Seq is "
        "available in this transformers version."
    )
