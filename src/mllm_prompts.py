"""
Prompt templates for the MLLM experiments (fixed before any held-out evaluation).

Configurations
- M1  clinical text only
- M2  image panel + clinical text
- M3  image panel + clinical text + physics/heterogeneity features, XGBoost
      probability and per-case SHAP contributions (grounded)
- M3f image panel + clinical text + physics/heterogeneity features only
      (no reference-model probability, no SHAP): tests reasoning from the features
- M4  M3 + k reference cases with known outcome drawn from the training folds

All configurations request the same JSON object, so prediction and report
faithfulness are evaluated identically.
"""

from __future__ import annotations

import math
from typing import Any

CLINICAL_KEYS = ["age", "HR", "HER2", "TNBC", "MammaPrint"]

FEATURE_INFO: dict[str, tuple[str, str]] = {
    "core15_Ktrans": ("1/min", "Tofts-PINN volume transfer constant, inner 15% of the tumour"),
    "core15_kep": ("1/min", "Tofts-PINN efflux rate constant, inner 15% of the tumour"),
    "safe_rim_Ktrans": ("1/min", "Tofts-PINN volume transfer constant, intratumoral rim"),
    "safe_rim_kep": ("1/min", "Tofts-PINN efflux rate constant, intratumoral rim"),
    "tumor_volume_mm3": ("mm^3", "segmented tumour volume"),
    "core15_volume_mm3": ("mm^3", "core region volume"),
    "safe_rim_volume_mm3": ("mm^3", "rim region volume"),
    "safe_rim_to_core15_volume_ratio": ("ratio", "rim volume / core volume"),
    "het_pe_mean": ("%", "mean early percent enhancement in the tumour"),
    "het_pe_sd": ("%", "SD of early percent enhancement"),
    "het_pe_cv": ("ratio", "coefficient of variation of percent enhancement"),
    "het_pe_p10": ("%", "10th percentile of percent enhancement"),
    "het_pe_p25": ("%", "25th percentile of percent enhancement"),
    "het_pe_p50": ("%", "median percent enhancement"),
    "het_pe_p75": ("%", "75th percentile of percent enhancement"),
    "het_pe_p90": ("%", "90th percentile of percent enhancement"),
    "het_pe_skew": ("unitless", "skewness of percent enhancement"),
    "het_pe_kurtosis": ("unitless", "excess kurtosis of percent enhancement"),
    "het_pe_entropy": ("bits", "histogram entropy of percent enhancement"),
    "het_ser_median": ("ratio", "median signal enhancement ratio (SER) of enhancing voxels"),
    "het_ser_p90": ("ratio", "90th percentile SER"),
    "het_washout_fraction": ("fraction", "enhancing voxels with washout kinetics (SER > 1.1)"),
    "het_plateau_fraction": ("fraction", "enhancing voxels with plateau kinetics (0.9-1.1)"),
    "het_persistent_fraction": ("fraction", "enhancing voxels with persistent kinetics (SER < 0.9)"),
    "het_washout_volume_mm3": ("mm^3", "volume of washout voxels"),
    "het_rim_minus_core_pe_median": ("%", "rim minus core median percent enhancement"),
    "het_peri_ring1_pe_mean": ("%", "mean percent enhancement 0-3 mm outside the tumour"),
    "het_peri_ring1_pe_median": ("%", "median percent enhancement 0-3 mm outside the tumour"),
    "het_peri_ring1_ser_median": ("ratio", "median SER 0-3 mm outside the tumour"),
    "het_peri_ring1_enh_fraction": ("fraction", "enhancing fraction 0-3 mm outside the tumour"),
    "het_peri_ring1_to_tumor_pe_ratio": ("ratio", "peritumoral (0-3 mm) / tumour median enhancement"),
    "het_peri_ring2_pe_mean": ("%", "mean percent enhancement 3-6 mm outside the tumour"),
    "het_peri_ring2_pe_median": ("%", "median percent enhancement 3-6 mm outside the tumour"),
    "het_peri_ring2_ser_median": ("ratio", "median SER 3-6 mm outside the tumour"),
    "het_peri_ring2_enh_fraction": ("fraction", "enhancing fraction 3-6 mm outside the tumour"),
    "het_peri_ring2_to_tumor_pe_ratio": ("ratio", "peritumoral (3-6 mm) / tumour median enhancement"),
}

SYSTEM_PROMPT = (
    "You are an experienced breast radiologist and breast oncologist supporting a research study. "
    "You estimate the probability of pathologic complete response (pCR: no residual invasive disease in "
    "breast and lymph nodes) after neoadjuvant systemic therapy, using ONLY pretreatment information. "
    "You never invent measurements. You answer with a single JSON object and nothing else."
)

IMAGE_NOTE = (
    "IMAGE: one 2x2 panel from the pretreatment dynamic contrast-enhanced breast MRI, axial slice with the "
    "largest tumour cross-section, cropped around the tumour. Top-left: pre-contrast. Top-right: early "
    "post-contrast. Bottom-left: late post-contrast. Bottom-right: early post-contrast with the tumour outline "
    "in red. The first three panels share one intensity window, so brightness differences reflect enhancement."
)

OUTPUT_SPEC = """OUTPUT: return exactly one JSON object with these fields:
{
  "pcr_probability": <number between 0 and 1>,
  "conclusion": "likely_pcr" | "unlikely_pcr" | "uncertain",
  "evidence": [ {"feature": "<key>", "value": <number or null>, "direction": "supports_pcr" | "against_pcr" | "neutral"} ],
  "report": "<at most 120 words for the treating clinician>"
}
RULES:
- "feature" must be one of the keys listed above (copy it exactly) or "image" for a visual observation.
- "value" must be copied from the listed value for that key; use null for "image". Never state a number that is not listed above.
- List 2 to 6 evidence items, most important first.
- The report must not mention any number that is not listed above."""


def fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "missing"
    v = float(v)
    if v == 0:
        return "0"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.3g}"


def clinical_block(case: dict[str, Any]) -> str:
    c = case["clinical"]
    lines = ["CLINICAL (pretreatment):"]
    lines.append(f"- age = {fmt(c.get('age'))} (years)")
    lines.append(f"- HR = {fmt(c.get('HR'))} (hormone receptor; 1 = positive, 0 = negative)")
    lines.append(f"- HER2 = {fmt(c.get('HER2'))} (1 = positive, 0 = negative)")
    lines.append(f"- TNBC = {fmt(c.get('TNBC'))} (triple-negative; 1 = yes, 0 = no)")
    if c.get("MammaPrint") is not None:
        lines.append(f"- MammaPrint = {fmt(c.get('MammaPrint'))} (binary MammaPrint risk category as coded in the I-SPY2 data; 1 = higher-risk category)")
    return "\n".join(lines)


def grounding_block(case: dict[str, Any], max_shap: int = 8, include_model: bool = True) -> str:
    lines = ["QUANTITATIVE FEATURES from a validated image-analysis pipeline (cite only these values):"]
    for k, v in case["features"].items():
        unit, desc = FEATURE_INFO.get(k, ("", k))
        lines.append(f"- {k} = {fmt(v)} ({unit}; {desc})")
    if not include_model:
        return "\n".join(lines)
    lines.append("")
    lines.append(f"REFERENCE MODEL: gradient-boosted tree model trained on other patients, predicted pCR probability = {fmt(case['xgb_probability'])}")
    lines.append("Per-feature contributions to that prediction (SHAP, log-odds; positive pushes toward pCR):")
    shap = sorted(case["shap"].items(), key=lambda kv: -abs(kv[1]))[:max_shap]
    for k, v in shap:
        lines.append(f"  {k}: {'+' if v >= 0 else ''}{v:.3f}")
    return "\n".join(lines)


def example_block(examples: list[dict[str, Any]]) -> str:
    out = ["REFERENCE CASES with known outcome (other patients from the training set):"]
    for i, ex in enumerate(examples, 1):
        c = ex["clinical"]
        feats = ", ".join(f"{k}={fmt(v)}" for k, v in list(ex["features"].items())[:12])
        out.append(f"Case {i}: age={fmt(c.get('age'))}, HR={fmt(c.get('HR'))}, HER2={fmt(c.get('HER2'))}, "
                   f"MammaPrint={fmt(c.get('MammaPrint'))}; {feats}; outcome: {'pCR' if ex['label'] == 1 else 'no pCR'}")
    return "\n".join(out)


def build_messages(config: str, case: dict[str, Any], image_data_url: str | None,
                   examples: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if config not in {"M1", "M2", "M3f", "M3", "M4"}:
        raise ValueError(config)
    parts: list[str] = [
        "TASK: estimate the probability that this patient achieves pCR after neoadjuvant therapy.",
    ]
    if config == "M4" and examples:
        parts += ["", example_block(examples)]
    parts += ["", "CASE TO ASSESS", ""]
    if config != "M1":
        parts += [IMAGE_NOTE, ""]
    parts.append(clinical_block(case))
    if config in {"M3", "M4"}:
        parts += ["", grounding_block(case)]
    elif config == "M3f":
        parts += ["", grounding_block(case, include_model=False)]
    allowed = CLINICAL_KEYS + (list(case["features"].keys()) if config in {"M3f", "M3", "M4"} else [])
    parts += ["", "ALLOWED evidence keys: " + ", ".join(allowed) + (", image" if config != "M1" else ""), "", OUTPUT_SPEC]
    text = "\n".join(parts)
    content: list[dict[str, Any]] | str
    if config == "M1" or image_data_url is None:
        content = text
    else:
        content = [{"type": "image_url", "image_url": {"url": image_data_url}}, {"type": "text", "text": text}]
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def pick_examples(case: dict[str, Any], dev_cases: list[dict[str, Any]], k: int, seed: int) -> list[dict[str, Any]]:
    """Balanced few-shot references from the case's training folds (all development cases for a held-out case)."""
    import random

    pool = [c for c in dev_cases if c["pid"] != case["pid"]
            and (case["split"] == "heldout_test" or c["fold_index"] != case["fold_index"])]
    rng = random.Random(f"{seed}-{case['pid']}")
    pos = [c for c in pool if c["label"] == 1]
    neg = [c for c in pool if c["label"] == 0]
    ex = rng.sample(pos, min(k // 2, len(pos))) + rng.sample(neg, min(k - k // 2, len(neg)))
    rng.shuffle(ex)
    return ex


def provided_values(config: str, case: dict[str, Any]) -> dict[str, float]:
    """Numbers shown to the model in this configuration (for faithfulness checking)."""
    vals: dict[str, float] = {}
    for k in CLINICAL_KEYS:
        v = case["clinical"].get(k)
        if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v)):
            vals[k] = float(v)
    if config in {"M3f", "M3", "M4"}:
        for k, v in case["features"].items():
            if v is not None and math.isfinite(float(v)):
                vals[k] = float(v)
    if config in {"M3", "M4"}:
        vals["xgb_probability"] = float(case["xgb_probability"])
        for k, v in case["shap"].items():
            vals[f"shap::{k}"] = float(v)
    return vals


__all__ = ["CLINICAL_KEYS", "FEATURE_INFO", "SYSTEM_PROMPT", "build_messages", "pick_examples", "provided_values", "fmt"]
