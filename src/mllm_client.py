"""
Minimal OpenAI-compatible chat client for commercial multimodal LLM APIs.

Providers (both expose an OpenAI-compatible ``/chat/completions`` endpoint):

- ``zhipu``       Zhipu AI open platform   base URL https://open.bigmodel.cn/api/paas/v4   key: ZHIPU_API_KEY
- ``siliconflow`` SiliconFlow             base URL https://api.siliconflow.cn/v1         key: SILICONFLOW_API_KEY
- ``mock``        offline deterministic stand-in used by the tests (no network)

Every request/response pair is cached on disk (key = SHA-256 of provider,
model, messages and parameters), so re-running a script never pays twice and
the exact model outputs analysed in the paper can be released with the code.
API keys are read from environment variables only and are never written to
the cache.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROVIDERS: dict[str, dict[str, str]] = {
    "zhipu": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "key_env": "ZHIPU_API_KEY"},
    "siliconflow": {"base_url": "https://api.siliconflow.cn/v1", "key_env": "SILICONFLOW_API_KEY"},
    "mock": {"base_url": "", "key_env": ""},
}


def image_to_data_url(path: Path) -> str:
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _strip_wrappers(text: str) -> str:
    # GLM vision models may wrap the final answer in <|begin_of_box|> ... <|end_of_box|>;
    # reasoning models may prepend <think> ... </think>.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "").strip()


def extract_json(text: str) -> dict[str, Any] | None:
    """Return the first JSON object found in a model reply (handles ```json fences)."""
    if not text:
        return None
    text = _strip_wrappers(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = [fenced.group(1)] if fenced else []
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


@dataclass
class ChatResult:
    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


class MLLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        cache_dir: Path,
        temperature: float = 0.01,
        max_tokens: int = 1200,
        timeout_s: float = 120.0,
        max_retries: int = 5,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
        self.provider = provider
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.extra_body = dict(extra_body or {})
        cfg = PROVIDERS[provider]
        self.base_url = cfg["base_url"]
        self.api_key = os.environ.get(cfg["key_env"], "") if cfg["key_env"] else ""
        if provider != "mock" and not self.api_key:
            raise RuntimeError(f"Set the environment variable {cfg['key_env']} with your {provider} API key.")

    def _cache_key(self, messages: list[dict[str, Any]], repeat: int) -> str:
        payload = json.dumps(
            {"provider": self.provider, "model": self.model, "messages": messages,
             "temperature": self.temperature, "max_tokens": self.max_tokens,
             "extra_body": self.extra_body, "repeat": int(repeat)},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def chat(self, messages: list[dict[str, Any]], repeat: int = 0) -> ChatResult:
        key = self._cache_key(messages, repeat)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
                return ChatResult(text=rec["text"], usage=rec.get("usage", {}), latency_s=rec.get("latency_s", 0.0),
                                  cached=True, raw=rec.get("raw", {}))
            except (json.JSONDecodeError, KeyError):
                pass  # unreadable cache entry: query again
        t0 = time.time()
        if self.provider == "mock":
            text, usage, raw = _mock_reply(messages, repeat), {"prompt_tokens": 0, "completion_tokens": 0}, {}
        else:
            text, usage, raw = self._post(messages)
        res = ChatResult(text=text, usage=usage, latency_s=time.time() - t0, cached=False, raw=raw)
        record = {"provider": self.provider, "model": self.model, "repeat": repeat,
                  "temperature": self.temperature, "text": text, "usage": usage,
                  "latency_s": res.latency_s, "raw": raw,
                  "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        # atomic write: identical prompts (e.g. same clinical profile in M1) may be answered concurrently
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
        return res

    def _post(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any], dict[str, Any]]:
        import requests

        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "stream": False, **self.extra_body}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_err = ""
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=body, timeout=self.timeout_s)
                if r.status_code == 200:
                    data = r.json()
                    msg = data["choices"][0]["message"]
                    text = msg.get("content") or ""
                    if isinstance(text, list):  # some providers return content parts
                        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
                    raw = {"id": data.get("id"), "model": data.get("model"),
                           "finish_reason": data["choices"][0].get("finish_reason")}
                    return text, data.get("usage", {}) or {}, raw
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in (408, 409, 429, 500, 502, 503, 504):
                    break
            except Exception as exc:  # network errors are retried
                last_err = repr(exc)
            time.sleep(min(60.0, (2 ** attempt) + random.random()))
        raise RuntimeError(f"{self.provider}/{self.model} request failed: {last_err}")


def _mock_reply(messages: list[dict[str, Any]], repeat: int) -> str:
    """Deterministic fake reply that exercises parsing and evaluation (cites real and invented values)."""
    text = ""
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            text += c
        elif isinstance(c, list):
            text += "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    case_block = text.split("CASE TO ASSESS")[-1]
    h = int(hashlib.md5((case_block + str(repeat)).encode()).hexdigest(), 16)
    rng = random.Random(h)
    rows = re.findall(r"^- ([a-zA-Z0-9_]+) = (-?[0-9.]+)", case_block, flags=re.M)
    rows = rows[5:] if len(rows) > 5 else rows  # cite quantitative features when they were provided
    evidence = []
    for name, val in rows[:3]:
        v = float(val) * (1.0 if rng.random() < 0.8 else 1.5)  # 20% of cited values are wrong
        evidence.append({"feature": name, "value": round(v, 4),
                         "direction": "supports_pcr" if rng.random() < 0.5 else "against_pcr"})
    hr = re.search(r"- HR = (\S+)", case_block)
    p = rng.uniform(0.15, 0.75) - (0.15 if hr and hr.group(1) == "1" else 0.0)
    p = round(min(max(p, 0.02), 0.98), 3)
    concl = "likely_pcr" if p >= 0.5 else "unlikely_pcr"
    report = f"Estimated pCR probability {p:.2f}. " + " ".join(f"{e['feature']} is {e['value']}." for e in evidence)
    return "```json\n" + json.dumps({"pcr_probability": p, "conclusion": concl, "evidence": evidence,
                                     "report": report}) + "\n```"


__all__ = ["PROVIDERS", "MLLMClient", "ChatResult", "extract_json", "image_to_data_url"]
