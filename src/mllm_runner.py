"""
Shared helpers for the follow-up MLLM experiments (steps 21 and 22): client set-up from
command-line options, a parallel request runner, and reply parsing identical to step 13.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from mllm_client import MLLMClient, extract_json


def add_client_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mllm_dir", type=Path, required=True, help="Output folder of step 12/13 (cases.jsonl, images/, runs/).")
    p.add_argument("--provider", choices=["zhipu", "siliconflow", "mock"], required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--max_tokens", type=int, default=1200)
    p.add_argument("--disable_thinking", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Only the first N cases (cost check).")


def make_client(a: argparse.Namespace) -> MLLMClient:
    extra: dict[str, Any] = {}
    if a.disable_thinking:
        extra = {"thinking": {"type": "disabled"}} if a.provider == "zhipu" else \
                {"enable_thinking": False} if a.provider == "siliconflow" else {}
    return MLLMClient(a.provider, a.model, a.mllm_dir / "cache", temperature=a.temperature,
                      max_tokens=a.max_tokens, extra_body=extra or None)


def run_tag(a: argparse.Namespace) -> str:
    return f"{a.provider}__{a.model.replace('/', '_')}"


def parse_reply(text: str) -> dict:
    """Same rules as scripts/13_run_mllm.py::parse_reply."""
    obj = extract_json(text)
    out = {"parse_ok": 0, "probability": None, "conclusion": None, "evidence": "[]", "report": ""}
    if not obj:
        return out
    try:
        p = float(obj.get("pcr_probability"))
        if 1.0 < p <= 100.0:
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


def run_jobs(client: MLLMClient, jobs: list[tuple[dict, list]], label: str, workers: int,
             extra: Callable[[dict], dict] | None = None) -> list[dict]:
    """jobs: (meta, messages). Returns one row per job with the parsed reply and usage."""
    rows, t0, errors = [], time.time(), 0

    def work(job):
        meta, msgs = job
        res = client.chat(msgs, repeat=0)
        return {**meta, **parse_reply(res.text), "raw_text": res.text,
                "prompt_tokens": res.usage.get("prompt_tokens"), "completion_tokens": res.usage.get("completion_tokens"),
                "latency_s": round(res.latency_s, 2), "cached": int(res.cached)}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"[{label}] request failed: {exc}", flush=True)
            if i % 10 == 0 or i == len(futs):
                print(f"[{label}] {i}/{len(futs)} done ({time.time() - t0:.0f}s, {errors} errors)", flush=True)
    return rows
