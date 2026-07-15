"""LoRA-based fine-tuning for local VLMs.

The trainer here is intentionally minimal.  It builds a supervised
``(image, caption)`` dataset from a JSONL and fine-tunes the underlying
model using LoRA (via PEFT).  Vision towers are frozen so the whole
run fits inside 8 GB VRAM.

**Supported families:**

* ``florence`` -- fully supported.  Florence-2 exposes a clean
  encoder-decoder training API through Transformers, so this is our
  reference local fine-tune target.

Other families (e.g. Moondream) require model-specific plumbing that
changes between revisions.  For those, run inference locally and
fine-tune on a cloud GPU using the model author's official recipe.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from ..utils.logger import get_logger
from ..utils.paths import ensure_dir

log = get_logger(__name__)


SUPPORTED_FAMILIES = {"florence"}


@dataclass
class FineTuneConfig:
    base_model: str
    model_spec: Dict[str, Any]
    dataset_jsonl: Path
    output_dir: Path
    method: str = "lora"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    epochs: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation: int = 8
    eval_ratio: float = 0.1
    seed: int = 42


ProgressCb = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
def _load_pairs(jsonl: Path) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    with jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs


def _split(pairs: List[Dict[str, Any]], eval_ratio: float, seed: int):
    r = random.Random(seed)
    idx = list(range(len(pairs)))
    r.shuffle(idx)
    n_eval = max(1, int(math.floor(len(pairs) * eval_ratio))) if pairs else 0
    eval_idx = set(idx[:n_eval])
    train, evl = [], []
    for i, p in enumerate(pairs):
        (evl if i in eval_idx else train).append(p)
    return train, evl


# ---------------------------------------------------------------------------
def check_support(model_spec: Dict[str, Any]) -> Optional[str]:
    """Return an error message if this model can't be fine-tuned here."""
    family = str(model_spec.get("family", "")).lower()
    if model_spec.get("kind") != "local":
        return "Cloud models cannot be fine-tuned locally."
    if family not in SUPPORTED_FAMILIES:
        return (
            f"Local fine-tuning of family '{family}' is not implemented in "
            "this build. Supported families: "
            + ", ".join(sorted(SUPPORTED_FAMILIES))
            + ". For other families, run inference locally and fine-tune the "
              "checkpoint on a cloud GPU using the model author's recipe."
        )
    return None


# ---------------------------------------------------------------------------
def run_finetune(
    cfg: FineTuneConfig,
    progress: Optional[ProgressCb] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Path:
    """Execute a LoRA fine-tune and return the output directory."""
    err = check_support(cfg.model_spec)
    if err:
        raise NotImplementedError(err)

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoProcessor

    hf_id = cfg.model_spec["hf_id"]
    trust = bool(cfg.model_spec.get("trust_remote_code", True))
    out_dir = ensure_dir(cfg.output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    log.info("Loading %s for fine-tune (device=%s, dtype=%s) ...",
             hf_id, device, dtype)
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=trust)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, dtype=dtype, trust_remote_code=trust
    ).to(device)

    # Freeze the vision tower so we only tune the text side.
    for name, p in model.named_parameters():
        lname = name.lower()
        if any(tag in lname for tag in ("vision", "visual", "image_encoder")):
            p.requires_grad = False

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,   # Florence-2 is encoder-decoder
    )
    model = get_peft_model(model, lora_cfg)
    try:
        model.print_trainable_parameters()
    except Exception:  # pragma: no cover
        pass

    pairs = _load_pairs(cfg.dataset_jsonl)
    if not pairs:
        raise RuntimeError(f"Dataset {cfg.dataset_jsonl} is empty")
    train_pairs, eval_pairs = _split(pairs, cfg.eval_ratio, cfg.seed)
    log.info("Fine-tune: %d train / %d eval samples",
             len(train_pairs), len(eval_pairs))

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )

    steps_per_epoch = max(
        1, math.ceil(len(train_pairs) / cfg.gradient_accumulation)
    )
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    step = 0
    accum_loss = 0.0

    task = "<MORE_DETAILED_CAPTION>"

    def _forward_loss(pair: Dict[str, Any]) -> "torch.Tensor":
        img = Image.open(pair["image_path"]).convert("RGB")
        text = pair["text"]
        inputs = processor(text=task, images=img, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        pixel_values = inputs["pixel_values"].to(device, dtype)
        labels = processor.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True,
            max_length=128,
        ).input_ids.to(device)
        # BART-style: internal shifting handles decoder inputs.
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=labels,
        )
        return outputs.loss

    model.train()
    for epoch in range(cfg.epochs):
        random.Random(cfg.seed + epoch).shuffle(train_pairs)
        for i, pair in enumerate(train_pairs):
            if should_stop and should_stop():
                log.warning("Fine-tune cancelled by user at step %d", step)
                model.save_pretrained(out_dir / "checkpoint-cancelled")
                return out_dir / "checkpoint-cancelled"

            try:
                loss = _forward_loss(pair)
            except Exception as exc:  # noqa: BLE001
                log.exception("Sample %d failed: %s", i, exc)
                continue

            loss = loss / cfg.gradient_accumulation
            loss.backward()
            accum_loss += float(loss.detach().cpu())

            if (i + 1) % cfg.gradient_accumulation == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if progress:
                    progress(
                        step, total_steps,
                        f"epoch {epoch + 1}/{cfg.epochs} loss={accum_loss:.3f}",
                    )
                accum_loss = 0.0

        ckpt = out_dir / f"checkpoint-epoch{epoch + 1}"
        model.save_pretrained(ckpt)
        log.info("Saved checkpoint -> %s", ckpt)

    final = out_dir / "final"
    model.save_pretrained(final)
    processor.save_pretrained(final)
    log.info("Fine-tune complete -> %s", final)
    return final
