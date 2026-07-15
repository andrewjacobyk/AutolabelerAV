"""Vision-language model backends.

The public entry point is :func:`load_model` which returns an instance
that implements :class:`~.base.VLMBase`.  All backends are local /
open-weight; cloud APIs are intentionally not supported.
"""

from __future__ import annotations

from typing import Any, Dict

from ...utils.logger import get_logger
from .base import VLMBase

log = get_logger(__name__)


def load_model(
    model_id: str,
    spec: Dict[str, Any],
    *,
    precision: str = "fp16",
    max_new_tokens: int = 96,
) -> VLMBase:
    """Instantiate a model given its spec from ``config.models``."""
    family = str(spec.get("family", "")).lower()

    if family == "moondream":
        from .moondream import MoondreamVLM
        return MoondreamVLM(model_id, spec, precision=precision,
                            max_new_tokens=max_new_tokens)
    if family == "florence":
        from .florence import FlorenceVLM
        return FlorenceVLM(model_id, spec, precision=precision,
                           max_new_tokens=max_new_tokens)
    if family == "smolvlm":
        from .smolvlm import SmolVLM
        return SmolVLM(model_id, spec, precision=precision,
                       max_new_tokens=max_new_tokens)
    if family == "blip2":
        from .blip import BLIP2VLM
        return BLIP2VLM(model_id, spec, precision=precision,
                        max_new_tokens=max_new_tokens)
    if family in {"qwenvl", "qwen2vl", "qwen2.5vl", "qwen_vl"}:
        from .qwenvl import QwenVLM
        return QwenVLM(model_id, spec, precision=precision,
                       max_new_tokens=max_new_tokens)
    if family in {"internvl", "internvl2", "internvl2.5", "internvl3"}:
        from .internvl import InternVLM
        return InternVLM(model_id, spec, precision=precision,
                         max_new_tokens=max_new_tokens)
    if family in {"minicpm", "minicpmv", "minicpm-v"}:
        from .minicpm import MiniCPMVLM
        return MiniCPMVLM(model_id, spec, precision=precision,
                          max_new_tokens=max_new_tokens)
    if family in {"paligemma", "paligemma2"}:
        from .paligemma import PaliGemmaVLM
        return PaliGemmaVLM(model_id, spec, precision=precision,
                            max_new_tokens=max_new_tokens)

    from .generic import GenericHFVLM
    log.warning("Family %r not recognized, falling back to GenericHFVLM.", family)
    return GenericHFVLM(model_id, spec, precision=precision,
                        max_new_tokens=max_new_tokens)
