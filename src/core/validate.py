"""Validation metrics for scene descriptions.

Compares a hypothesis output document (produced by the inference step)
against a reference document with the same frame layout.  Metrics:

* **BLEU-4**: n-gram overlap with the reference.
* **ROUGE-L**: longest common subsequence.
* **Cosine**: sentence embedding cosine similarity (MiniLM by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger
from . import dataset as ds

log = get_logger(__name__)


@dataclass
class ValidationResult:
    n_pairs: int
    bleu: Optional[float] = None
    rougeL: Optional[float] = None
    cosine: Optional[float] = None
    per_frame: List[Dict[str, Any]] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_pairs": self.n_pairs,
            "bleu": self.bleu,
            "rougeL": self.rougeL,
            "cosine": self.cosine,
            "per_frame": self.per_frame or [],
        }


# ---------------------------------------------------------------------------
def _bleu(refs: List[str], hyps: List[str]) -> float:
    try:
        import nltk
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:  # pragma: no cover
            nltk.download("punkt", quiet=True)
    except Exception as e:  # pragma: no cover
        log.warning("BLEU unavailable: %s", e)
        return float("nan")
    smooth = SmoothingFunction().method1
    ref_toks = [[r.split()] for r in refs]
    hyp_toks = [h.split() for h in hyps]
    return float(corpus_bleu(ref_toks, hyp_toks, smoothing_function=smooth))


def _rouge_l(refs: List[str], hyps: List[str]) -> float:
    try:
        from rouge_score import rouge_scorer
    except Exception as e:  # pragma: no cover
        log.warning("ROUGE unavailable: %s", e)
        return float("nan")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    if not refs:
        return float("nan")
    scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]
    return sum(scores) / len(scores)


def _cosine(refs: List[str], hyps: List[str], model_name: str) -> float:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except Exception as e:  # pragma: no cover
        log.warning("SentenceTransformer unavailable: %s", e)
        return float("nan")
    model = SentenceTransformer(model_name)
    a = model.encode(refs, convert_to_numpy=True, normalize_embeddings=True)
    b = model.encode(hyps, convert_to_numpy=True, normalize_embeddings=True)
    sims = (a * b).sum(axis=1)
    return float(sims.mean())


# ---------------------------------------------------------------------------
def _align(ref_doc: Dict[str, Any], hyp_doc: Dict[str, Any]):
    """Match frames by file name; fall back to positional order."""
    refs = ref_doc.get("frames", [])
    hyps = hyp_doc.get("frames", [])
    ref_by_file = {f["file"]: f for f in refs if "file" in f}
    aligned = []
    for h in hyps:
        r = ref_by_file.get(h.get("file"))
        if r is None and len(refs) == len(hyps):
            r = refs[h["index"]]
        if r is not None:
            aligned.append((r, h))
    return aligned


def validate(
    ref_path: Path | str,
    hyp_path: Path | str,
    metrics: List[str],
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> ValidationResult:
    ref_doc = ds.load(ref_path)
    hyp_doc = ds.load(hyp_path)
    pairs = _align(ref_doc, hyp_doc)
    refs = [(r.get("description") or "").strip() for r, _ in pairs]
    hyps = [(h.get("description") or "").strip() for _, h in pairs]

    result = ValidationResult(n_pairs=len(pairs), per_frame=[])
    if not pairs:
        return result

    if "bleu" in metrics:
        result.bleu = _bleu(refs, hyps)
    if "rougeL" in metrics or "rouge" in metrics:
        result.rougeL = _rouge_l(refs, hyps)
    if "cosine" in metrics:
        result.cosine = _cosine(refs, hyps, embed_model)

    for (r, h) in pairs:
        result.per_frame.append({
            "file": h.get("file"),
            "timestamp_sec": h.get("timestamp_sec"),
            "reference": r.get("description"),
            "hypothesis": h.get("description"),
        })
    return result
