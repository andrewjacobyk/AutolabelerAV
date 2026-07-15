"""Static resource estimates for the shipped VLM catalog.

Numbers are approximations gathered from the model cards on Hugging
Face plus empirical measurements on an RTX 4070 (8 GB).  They are only
used for informational purposes (Inference tab preview + preflight
warning); actual usage will vary a bit with sequence length, batch
size, KV-cache growth and driver overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Per-precision bytes-per-parameter (weights only, no overhead).
# ---------------------------------------------------------------------------
_BYTES_PER_PARAM: Dict[str, float] = {
    "fp32": 4.0,
    "float32": 4.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp16": 2.0,
    "float16": 2.0,
    "int8": 1.0,
    "8bit": 1.0,
    # NF4 double-quant averages ~0.55 bytes/param once the extra scales
    # and CUDA workspace are counted.
    "int4": 0.55,
    "4bit": 0.55,
}

# ---------------------------------------------------------------------------
# Approximate parameter counts (in millions).
# Sourced from the model cards on Hugging Face.
# ---------------------------------------------------------------------------
_PARAMS_M: Dict[str, int] = {
    # Western / global -------------------------------------------------
    "moondream2":              1_930,
    "florence2-base":            232,
    "florence2-large":           770,
    "smolvlm-256m":              256,
    "smolvlm-500m":              507,
    "smolvlm":                 2_240,
    "blip2-opt-2.7b":          3_740,
    "paligemma-3b-mix-448":    3_030,
    # Chinese ----------------------------------------------------------
    "qwen2-vl-2b-instruct":    2_210,
    "qwen2-vl-7b-instruct":    8_290,
    "qwen2_5-vl-3b-instruct":  3_750,
    "qwen2_5-vl-7b-instruct":  8_290,
    "internvl2_5-1b":            940,
    "internvl2_5-2b":          2_210,
    "internvl2_5-4b":          3_710,
    "internvl3-2b":            2_210,
    "minicpm-v-2_6":           8_100,
}

# ---------------------------------------------------------------------------
# Which precisions each backend actually accepts.
# * bitsandbytes int4/int8 needs a compatible loader -- the Florence
#   custom path and PaliGemma stack don't play well with bnb yet.
# ---------------------------------------------------------------------------
_ALLOWED: Dict[str, set[str]] = {
    "moondream": {"fp32", "bf16", "fp16", "int8", "int4"},
    "florence":  {"fp32", "bf16", "fp16"},
    "smolvlm":   {"fp32", "bf16", "fp16"},
    "blip2":     {"fp32", "bf16", "fp16", "int8", "int4"},
    "paligemma": {"fp32", "bf16", "fp16"},
    "qwenvl":    {"fp32", "bf16", "fp16", "int8", "int4"},
    "internvl":  {"fp32", "bf16", "fp16", "int8", "int4"},
    "minicpm":   {"fp32", "bf16", "fp16", "int8", "int4"},
}


@dataclass
class ResourceEstimate:
    """Estimated resource usage for one (model, precision) combination."""

    model_id: str
    precision: str
    params_millions: int
    weights_gb: float          # weights only
    vram_gb: float             # weights + KV cache + CUDA overhead
    disk_gb: float             # download / cache footprint (fp16 wheels)
    ram_gb: float              # peak host RAM during load
    supported: bool
    note: str = ""

    def fits_in(self, vram_gb: float, headroom_gb: float = 0.5) -> bool:
        """Return True if this fits into ``vram_gb`` with some headroom."""
        return self.vram_gb + headroom_gb <= vram_gb


# ---------------------------------------------------------------------------
def _norm(precision: str) -> str:
    p = (precision or "fp16").lower()
    alias = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16",
             "8bit": "int8", "4bit": "int4"}
    return alias.get(p, p)


def estimate(
    model_id: str,
    precision: str,
    family: Optional[str] = None,
) -> ResourceEstimate:
    """Return an estimate for one (model, precision) combination."""
    prec = _norm(precision)
    params_m = _PARAMS_M.get(model_id, 0)
    bpp = _BYTES_PER_PARAM.get(prec, 2.0)

    # Weights, in GB.
    weights_gb = params_m * 1e6 * bpp / (1024 ** 3)

    # VRAM overhead: CUDA context (~500 MB), KV cache & workspace
    # (~10-20% of weights for a batch-1, ~1k token generation) and a
    # small allowance for image encoder activations (~250 MB).
    overhead_gb = 0.75 + 0.15 * weights_gb
    vram_gb = weights_gb + overhead_gb

    # Disk footprint. HF caches store fp16 weights natively; int4/int8
    # models still download the fp16 shards and quantise on load.
    disk_gb = params_m * 1e6 * 2.0 / (1024 ** 3)

    # Peak host RAM during load: shards get staged in RAM before being
    # moved to VRAM, so budget ~1.5x weights_gb (fp16) as CPU load.
    ram_gb = max(1.5, params_m * 1e6 * 2.0 / (1024 ** 3) * 1.5)

    supported = True
    note = ""
    if family:
        allowed = _ALLOWED.get(family.lower())
        if allowed is not None and prec not in allowed:
            supported = False
            note = (
                f"precision {prec!r} isn't wired up for family {family!r}; "
                f"try one of: {', '.join(sorted(allowed))}"
            )

    if params_m == 0:
        note = ("no static param count for this model - "
                "estimate defaults to zero, treat with caution.")

    return ResourceEstimate(
        model_id=model_id,
        precision=prec,
        params_millions=params_m,
        weights_gb=round(weights_gb, 2),
        vram_gb=round(vram_gb, 2),
        disk_gb=round(disk_gb, 2),
        ram_gb=round(ram_gb, 2),
        supported=supported,
        note=note,
    )


def format_short(est: ResourceEstimate) -> str:
    """One-line summary suitable for a GUI status label."""
    if est.params_millions == 0:
        return "unknown size"
    return (
        f"~{est.params_millions/1000:.1f}B params  "
        f"|  VRAM ~{est.vram_gb:.1f} GB  "
        f"|  disk ~{est.disk_gb:.1f} GB  "
        f"|  RAM ~{est.ram_gb:.1f} GB"
    )


def format_table(cfg_models: Dict[str, Dict], precisions=None) -> str:
    """Multi-line table of every configured model at each precision."""
    precisions = precisions or ["fp16", "bf16", "int8", "int4"]
    header = (
        f"{'model':32s} {'family':10s} {'params':>8s}  "
        + "  ".join(f"{p:>8s}" for p in precisions)
    )
    lines = [header, "-" * len(header)]
    for mid, spec in cfg_models.items():
        fam = str(spec.get("family", "-"))
        row_cells = []
        first_params = None
        for p in precisions:
            est = estimate(mid, p, family=fam)
            if first_params is None:
                first_params = est.params_millions
            cell = f"{est.vram_gb:>4.1f}GB" if est.supported else "  n/a "
            row_cells.append(f"{cell:>8s}")
        params_str = f"{(first_params or 0)/1000:.1f}B"
        lines.append(
            f"{mid:32s} {fam:10s} {params_str:>8s}  " + "  ".join(row_cells)
        )
    return "\n".join(lines)
