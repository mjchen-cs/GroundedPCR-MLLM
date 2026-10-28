#!/usr/bin/env python3
"""
Step 13: Query a commercial multimodal LLM for every case and configuration.

    set ZHIPU_API_KEY=...            (Windows)   /  export ZHIPU_API_KEY=...
    python scripts/13_run_mllm.py --mllm_dir out/mllm --provider zhipu --model glm-4.5v \
        --configs M1 M2 M3f M3 M4 --split development --repeats 2

Outputs ``<mllm_dir>/runs/<provider>__<model>/<config>_<split>.csv`` with one
row per case x repeat: parsed probability, conclusion, evidence (JSON),
report, parse status, token usage, latency. Raw replies are cached under
``<mllm_dir>/cache/`` and are reused on re-runs.

Few-shot references (M4) for a development case are drawn only from the
OTHER folds (its training folds); for a held-out case from the whole
development set. They are balanced by outcome and fixed by the random seed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mllm_client import MLLMClient, extract_json, image_to_data_url
from mllm_prompts import build_messages, pick_examples


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run MLLM configurations over exported cases.")
    p.add_argument("--mllm_dir", type=Path, required=True, help="Output folder of step 12 (cases.jsonl, images/).")
    p.add_argument("--provider", choices=["zhipu", "siliconflow", "mock"], required=True)
    p.add_argument("--model", type=str, required=True, help="Model id on that platform, e.g. glm-4.5v or Qwen/Qwen3-VL-32B-Instruct.")
    p.add_argument("--configs", nargs="+", default=["M1", "M2", "M3f", "M3", "M4"])
    p.add_argument("--split", choices=["development", "heldout_test"], default="development")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--n_shots", type=int, default=6)
    p.add_argument("--limit", type=int, default=0, help="Only the first N cases (pilot run / cost check).")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--max_tokens", type=int, default=1200)
    p.add_argument("--extra_body", type=str, default="",
                   help='JSON merged into the request, e.g. \'{"thinking": {"type": "disabled"}}\' for GLM-4.5V.')
    p.add_argument("--disable_thinking", action="store_true",
                   help="Turn off reasoning mode (zhipu: thinking=disabled; siliconflow: enable_thinking=false).")
    p.add_argument("--seed", type=int, default=42)
    return p


def parse_reply(text: str) -> dict:
    obj = extract_json(text)
    out = {"parse_ok": 0, "probability": None, "conclusion": None, "evidence": "[]", "report": ""}
    if not obj:
        return out
    try:
        p = float(obj.get("pcr_probability"))
        if p > 1.0 and p <= 100.0:  # a percentage was returned
            p = p / 100.0
        if 0.0 <= p <= 1.0:
            out["probability"] = p
            out["parse_ok"] = 1
    except (TypeError, ValueError):
        pass
    out["conclusion"] = obj.get("conclusion")
    ev = obj.get("evidence") if isinstance(obj.get("evidence"), list) else []
    out["evidence"] = json.dumps(ev, ensure_ascii=False)
    out["report"] = str(obj.get("report", ""))
    return out


def main() -> None:
    args = build_argparser().parse_args()
    cases = [json.loads(l) for l in (args.mllm_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    dev_cases = [c for c in cases if c["split"] == "development"]
    todo = [c for c in cases if c["split"] == args.split]
    if args.limit:
        todo = todo[: args.limit]
    extra = json.loads(args.extra_body) if args.extra_body else {}
    if args.disable_thinking:
        extra.update({"thinking": {"type": "disabled"}} if args.provider == "zhipu" else
                     {"enable_thinking": False} if args.provider == "siliconflow" else {})
    client = MLLMClient(args.provider, args.model, args.mllm_dir / "cache", temperature=args.temperature,
                        max_tokens=args.max_tokens, extra_body=extra or None)
    tag = f"{args.provider}__{args.model.replace('/', '_')}"
    run_dir = args.mllm_dir / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    for config in args.configs:
        jobs = []
        for case in todo:
            img = None
            if config != "M1":
                img_path = args.mllm_dir / "images" / f"{case['pid']}.png"
                img = image_to_data_url(img_path)
            examples = pick_examples(case, dev_cases, args.n_shots, args.seed) if config == "M4" else None
            msgs = build_messages(config, case, img, examples)
            for r in range(args.repeats):
                jobs.append((case, r, msgs))
        rows, t0, errors = [], time.time(), 0

        def work(job):
            case, r, msgs = job
            res = client.chat(msgs, repeat=r)
            parsed = parse_reply(res.text)
            return {"pid": case["pid"], "split": case["split"], "fold_index": case["fold_index"], "repeat": r,
                    **parsed, "raw_text": res.text, "prompt_tokens": res.usage.get("prompt_tokens"),
                    "completion_tokens": res.usage.get("completion_tokens"), "latency_s": round(res.latency_s, 2),
                    "cached": int(res.cached)}

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = [ex.submit(work, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    rows.append(fut.result())
                except Exception as exc:
                    errors += 1
                    print(f"[{config}] request failed: {exc}", flush=True)
                if i % 10 == 0 or i == len(futs):
                    print(f"[{config}] {i}/{len(futs)} done ({time.time() - t0:.0f}s, {errors} errors)", flush=True)
        df = pd.DataFrame(rows).sort_values(["pid", "repeat"])
        out_csv = run_dir / f"{config}_{args.split}.csv"
        df.to_csv(out_csv, index=False)
        ok = df["parse_ok"].mean() if len(df) else 0
        tok = (df["prompt_tokens"].fillna(0).sum() + df["completion_tokens"].fillna(0).sum()) if len(df) else 0
        print(f"[{config}] wrote {out_csv} | rows={len(df)} parse_ok={ok:.1%} tokens={int(tok)} failed={errors}", flush=True)


if __name__ == "__main__":
    main()
