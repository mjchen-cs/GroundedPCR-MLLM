#!/usr/bin/env python3
"""
Step 21: counterfactual test of reliance on the reference model (paid API calls).

The M3 prompt of every case of the chosen split is re-sent with a deliberately WRONG
reference model output, everything else unchanged (image, clinical values, features):

  flip_prob   B1 probability replaced by 1 - p; SHAP values unchanged
              (the reference output now contradicts its own attributions)
  flip_all    B1 probability replaced by 1 - p AND every SHAP value negated
              (a coherent but wrong reference model)

If the replies follow the perturbed reference, the explanations are driven by the
supplied model output rather than by the evidence or prior knowledge.

Output: <mllm_dir>/runs_cf/<provider>__<model>/M3-<variant>_<split>.csv
(one reply per case; same columns as step 13). Evaluated by step 23.

Example (held-out set, 198 cases x 2 variants = 396 requests per provider):
  python scripts/21_counterfactual.py --mllm_dir results/mllm --provider zhipu --model glm-4.5v --disable_thinking
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mllm_client import image_to_data_url  # noqa: E402
from mllm_prompts import build_messages  # noqa: E402
from mllm_runner import add_client_args, make_client, run_jobs, run_tag  # noqa: E402

VARIANTS = ("flip_prob", "flip_all")


def perturb(case: dict, variant: str) -> dict:
    c = copy.deepcopy(case)
    c["xgb_probability"] = 1.0 - float(case["xgb_probability"])
    if variant == "flip_all":
        c["shap"] = {k: -float(v) for k, v in case["shap"].items()}
    return c


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_client_args(p)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--split", choices=["development", "heldout_test"], default="heldout_test")
    a = p.parse_args()
    cases = [json.loads(l) for l in (a.mllm_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    todo = [c for c in cases if c["split"] == a.split][: a.limit or None]
    client = make_client(a)
    out_dir = a.mllm_dir / "runs_cf" / run_tag(a)
    out_dir.mkdir(parents=True, exist_ok=True)
    for v in a.variants:
        jobs = []
        for c in todo:
            img = image_to_data_url(a.mllm_dir / "images" / f"{c['pid']}.png")
            cf = perturb(c, v)
            jobs.append(({"pid": c["pid"], "split": c["split"], "variant": v,
                          "b1_original": c["xgb_probability"], "b1_shown": cf["xgb_probability"]},
                         build_messages("M3", cf, img)))
        rows = run_jobs(client, jobs, f"M3-{v}", a.workers)
        df = pd.DataFrame(rows).sort_values("pid")
        f = out_dir / f"M3-{v}_{a.split}.csv"
        df.to_csv(f, index=False)
        print(f"[M3-{v}] wrote {f} | rows={len(df)} parse_ok={df.parse_ok.mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
