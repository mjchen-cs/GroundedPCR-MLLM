#!/usr/bin/env python3
"""
Step 22: verify-and-revise loop (paid API calls, only for replies that fail the audit).

Every first-repeat reply of the chosen configurations is checked with the step-14 rules.
A reply fails if it contains a number that is not in the prompt, an evidence direction
that contradicts the B1 SHAP sign (only in configurations that show SHAP values), or a
conclusion that contradicts its own probability. For each failing reply the original
conversation is continued with ONE feedback message listing the concrete violations, and
the model is asked to return a corrected JSON object. The revision is checked again in
step 23.

Output: <mllm_dir>/runs_revise/<provider>__<model>/<config>_<split>.csv with the
violations found, the original reply and the revised reply.

Example:
  python scripts/22_verify_revise.py --mllm_dir results/mllm --provider zhipu --model glm-4.5v --disable_thinking
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
from mllm_client import image_to_data_url  # noqa: E402
from mllm_prompts import build_messages, pick_examples, provided_values  # noqa: E402
from mllm_runner import add_client_args, make_client, run_jobs, run_tag  # noqa: E402

_s = importlib.util.spec_from_file_location("ev14", _HERE / "14_evaluate_mllm.py")
ev14 = importlib.util.module_from_spec(_s); _s.loader.exec_module(ev14)
_s = importlib.util.spec_from_file_location("ex17", _HERE / "17_extra_analyses.py")
ex17 = importlib.util.module_from_spec(_s); _s.loader.exec_module(ex17)

SHAP_SHOWN = {"M3", "M4"}


def violations(config: str, case: dict, rec: pd.Series, prompt_text: str) -> list[str]:
    """Concrete, human-readable list of audit failures for one reply."""
    out = []
    prov = provided_values(config, case)
    try:
        ev = json.loads(rec.evidence) if isinstance(rec.evidence, str) else []
    except json.JSONDecodeError:
        ev = []
    for e in ev:
        if not isinstance(e, dict):
            continue
        k, v = str(e.get("feature", "")), e.get("value")
        if isinstance(v, (int, float)) and k != "image":
            ref = prov.get(k)
            if ref is None:
                out.append(f'Evidence item "{k}" has value {v}, but "{k}" is not one of the listed inputs.')
            elif not ev14._supported(float(v), [ref], False):
                out.append(f'Evidence item "{k}" has value {v}, but the value provided for {k} is {ev14_fmt(ref)}.')
        sv = case["shap"].get(k)
        d = e.get("direction")
        if config in SHAP_SHOWN and sv is not None and abs(sv) >= 0.02 and d in ("supports_pcr", "against_pcr") \
                and (d == "supports_pcr") != (sv > 0):
            want = "supports_pcr" if sv > 0 else "against_pcr"
            out.append(f'Evidence item "{k}" is marked "{d}", but its contribution in the reference model is '
                       f'{sv:+.3f} (log-odds), i.e. "{want}".')
    bad = [tok for _, tok in ex17.classify_unsupported(rec.report, prov, prompt_text, rec.probability)]
    if bad:
        out.append("The report contains number(s) that do not appear in the information provided: "
                   + ", ".join(sorted(set(bad))) + ". Do not state numbers that were not provided "
                   "(no population statistics, ranges or estimates other than your own probability).")
    p, c = rec.probability, rec.conclusion
    if p == p and p is not None and c in ("likely_pcr", "unlikely_pcr") and (c == "likely_pcr") != (float(p) >= 0.5):
        out.append(f'The conclusion "{c}" is inconsistent with your probability {p:g}.')
    return out


def ev14_fmt(v: float) -> str:
    from mllm_prompts import fmt
    return fmt(v)


FEEDBACK_HEAD = "An automatic check of your answer against the information provided found these problems:\n"
FEEDBACK_TAIL = ("\nPlease correct them. Keep everything that was correct, use only the values provided above, "
                 "and return the complete corrected JSON object in the same format and nothing else.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_client_args(p)
    p.add_argument("--configs", nargs="+", default=["M1", "M2", "M3f", "M3", "M4"])
    p.add_argument("--splits", nargs="+", default=["development", "heldout_test"])
    p.add_argument("--n_shots", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true", help="only count failing replies (no API calls, no key needed)")
    a = p.parse_args()
    cases = [json.loads(l) for l in (a.mllm_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {c["pid"]: c for c in cases}
    dev_cases = [c for c in cases if c["split"] == "development"]
    client = None if a.dry_run else make_client(a)
    tag = run_tag(a)
    out_dir = a.mllm_dir / "runs_revise" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for cfg in a.configs:
        for sp in a.splits:
            src = a.mllm_dir / "runs" / tag / f"{cfg}_{sp}.csv"
            orig = pd.read_csv(src)
            orig = orig[orig.repeat == 0].sort_values("pid")
            if a.limit:
                orig = orig.head(a.limit)
            jobs, audit = [], []
            for _, rec in orig.iterrows():
                case = by[rec.pid]
                ex = pick_examples(case, dev_cases, a.n_shots, a.seed) if cfg == "M4" else None
                text_msgs = build_messages(cfg, case, None, ex)
                prompt_text = text_msgs[-1]["content"]
                viol = violations(cfg, case, rec, prompt_text if isinstance(prompt_text, str) else "")
                audit.append({"pid": rec.pid, "n_violations": len(viol)})
                if not viol:
                    continue
                img = None if (cfg == "M1" or a.dry_run) else image_to_data_url(a.mllm_dir / "images" / f"{rec.pid}.png")
                msgs = build_messages(cfg, case, img, ex) + [
                    {"role": "assistant", "content": str(rec.raw_text)},
                    {"role": "user", "content": FEEDBACK_HEAD + "\n".join(f"- {v}" for v in viol) + FEEDBACK_TAIL}]
                meta = {"pid": rec.pid, "split": sp, "config": cfg, "violations": json.dumps(viol, ensure_ascii=False),
                        "orig_probability": rec.probability, "orig_conclusion": rec.conclusion,
                        "orig_evidence": rec.evidence, "orig_report": rec.report}
                jobs.append((meta, msgs))
            print(f"[{cfg} {sp}] {len(jobs)} of {len(orig)} replies fail the audit and are sent back", flush=True)
            if a.dry_run:
                continue
            rows = run_jobs(client, jobs, f"revise {cfg} {sp}", a.workers) if jobs else []
            pd.DataFrame(rows).to_csv(out_dir / f"{cfg}_{sp}.csv", index=False)
            pd.DataFrame(audit).to_csv(out_dir / f"{cfg}_{sp}_audit.csv", index=False)


if __name__ == "__main__":
    main()
