#!/usr/bin/env python3
"""Quick connectivity check for the MLLM providers (one tiny request each, no cache).

Usage:  python scripts\\check_api.py --provider zhipu --model glm-4.5v
        python scripts\\check_api.py --provider siliconflow --model Qwen/Qwen2.5-VL-72B-Instruct
Prints HTTP status, latency and the first part of the reply or error body. The key is never printed.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mllm_client import PROVIDERS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["zhipu", "siliconflow"])
    ap.add_argument("--model", default="")
    ap.add_argument("--list_models", action="store_true",
                    help="List the models this key can use (vision-language models first) and exit.")
    ap.add_argument("--disable_thinking", action="store_true")
    ap.add_argument("--timeout", type=float, default=60)
    a = ap.parse_args()
    import requests

    cfg = PROVIDERS[a.provider]
    key = os.environ.get(cfg["key_env"], "")
    print(f"{cfg['key_env']} set: {bool(key)} (length {len(key)})")
    if a.list_models:
        r = requests.get(f"{cfg['base_url']}/models", timeout=a.timeout, headers={"Authorization": f"Bearer {key}"})
        print(f"HTTP {r.status_code}")
        if r.status_code != 200:
            print(r.text[:600])
            return
        ids = sorted(m.get("id", "") for m in r.json().get("data", []))
        vl = [i for i in ids if any(t in i.lower() for t in ("vl", "4.5v", "4.1v", "vision", "omni"))]
        print("Vision-language candidates:")
        for i in vl:
            print("  ", i)
        print(f"({len(ids)} models in total; use --model <id> to test one)")
        return
    if not a.model:
        ap.error("--model is required unless --list_models is given")
    body = {"model": a.model, "messages": [{"role": "user", "content": 'Reply with the JSON {"ok": 1} only.'}],
            "temperature": 0.01, "max_tokens": 50, "stream": False}
    if a.disable_thinking:
        body.update({"thinking": {"type": "disabled"}} if a.provider == "zhipu" else {"enable_thinking": False})
    t0 = time.time()
    try:
        r = requests.post(f"{cfg['base_url']}/chat/completions", json=body, timeout=a.timeout,
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    except Exception as exc:
        print(f"NETWORK ERROR after {time.time() - t0:.1f}s: {exc!r}")
        print("Check internet / proxy settings (HTTPS_PROXY) on this computer.")
        return
    print(f"HTTP {r.status_code} after {time.time() - t0:.1f}s")
    try:
        data = r.json()
        if r.status_code == 200:
            print("reply:", data["choices"][0]["message"].get("content"))
            print("usage:", json.dumps(data.get("usage")))
        else:
            print("error body:", json.dumps(data, ensure_ascii=False)[:600])
    except ValueError:
        print("body:", r.text[:600])


if __name__ == "__main__":
    main()
