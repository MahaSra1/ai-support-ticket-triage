"""
llm_client.py - a reusable, production-style Ollama client.

Why a class?  In real companies nobody sprinkles `requests.post(...)` all over
the code. You wrap the model in ONE client so that:
  * the URL / model / timeouts live in one place
  * errors become clear messages ("run: ollama pull mistral")
  * every call returns the same result object (text + metrics)
  * swapping Ollama for OpenAI/Claude later means changing ONE file

Endpoints used (Ollama REST API on http://localhost:11434):
  GET  /api/tags      -> which models are installed
  POST /api/generate  -> one prompt in, one answer out (no chat history)
  POST /api/chat      -> list of messages in (system/user/assistant), answer out
  POST /api/embed     -> text in, vector out (embedding models only)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Iterator

import requests

# Base URL only (no /api/... path). Accepts the old full-path style too.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").split("/api/")[0]
NS = 1_000_000_000  # Ollama reports durations in nanoseconds


class OllamaError(Exception):
    """A friendly, human-readable error for the UI."""


@dataclass
class LLMResult:
    text: str
    model: str = ""
    prompt_tokens: int = 0      # tokens the model READ (system + history + question)
    output_tokens: int = 0      # tokens the model WROTE
    total_sec: float = 0.0      # wall time inside Ollama
    load_sec: float = 0.0       # time to load the model into RAM (cold start)
    prompt_sec: float = 0.0     # time to read the prompt
    gen_sec: float = 0.0        # time to write the answer
    ttft_sec: float = 0.0       # time to first token (streaming only, measured by us)
    done_reason: str = ""       # "stop" = finished naturally, "length" = hit max tokens
    raw: dict = field(default_factory=dict, repr=False)  # full JSON from Ollama

    @property
    def tokens_per_sec(self) -> float:
        return self.output_tokens / self.gen_sec if self.gen_sec else 0.0

    @classmethod
    def from_response(cls, text: str, data: dict, ttft: float = 0.0) -> "LLMResult":
        return cls(
            text=text,
            model=data.get("model", ""),
            prompt_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            total_sec=data.get("total_duration", 0) / NS,
            load_sec=data.get("load_duration", 0) / NS,
            prompt_sec=data.get("prompt_eval_duration", 0) / NS,
            gen_sec=data.get("eval_duration", 0) / NS,
            ttft_sec=ttft,
            done_reason=data.get("done_reason", ""),
            raw=data,
        )

    def stats(self) -> dict:
        return {
            "prompt tokens": self.prompt_tokens,
            "output tokens": self.output_tokens,
            "tokens/sec": round(self.tokens_per_sec, 1),
            "time to first token (s)": round(self.ttft_sec, 2) if self.ttft_sec else "-",
            "model load (s)": round(self.load_sec, 2),
            "total (s)": round(self.total_sec, 2),
            "finished because": self.done_reason or "-",
        }


class LLMClient:
    def __init__(self, model: str = "mistral", base_url: str = OLLAMA_URL,
                 timeout: int = 300, retries: int = 2, keep_alive: str = "10m"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout          # seconds to wait for the answer (CPU = slow!)
        self.retries = retries          # retry only on network / server (5xx) errors
        self.keep_alive = keep_alive    # keep model in RAM -> no cold start next call
        self.last_result: LLMResult | None = None
        self.last_payload: dict | None = None   # for the "show me the raw request" tab

    # ---------- low level ----------
    def _post(self, path: str, payload: dict, stream: bool = False) -> requests.Response:
        self.last_payload = payload
        url = f"{self.base_url}{path}"
        last_err: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                # timeout=(connect, read): fail fast if Ollama is off, wait long for answers
                r = requests.post(url, json=payload, stream=stream, timeout=(5, self.timeout))
            except requests.ConnectionError:
                last_err = OllamaError(f"Cannot reach Ollama at {self.base_url}. "
                                       "Is the Ollama app running?")
            except requests.Timeout:
                last_err = OllamaError(f"No answer within {self.timeout}s. "
                                       "Try fewer max tokens or a smaller model.")
            else:
                if r.status_code == 404:
                    raise OllamaError(f"Model '{payload.get('model')}' not found. "
                                      f"Run: ollama pull {payload.get('model')}")
                if r.status_code >= 500:
                    last_err = OllamaError(f"Ollama server error {r.status_code}: {r.text[:200]}")
                elif r.status_code >= 400:   # our fault (bad payload) -> retrying won't help
                    raise OllamaError(f"Bad request {r.status_code}: {r.text[:200]}")
                else:
                    return r

            if attempt < self.retries:
                time.sleep(2 ** attempt)    # exponential backoff: 1s, 2s, 4s...
        raise last_err  # type: ignore[misc]

    def _options(self, temperature: float, max_tokens: int) -> dict:
        # These MUST sit inside "options", or Ollama silently ignores them
        return {"temperature": temperature, "num_predict": max_tokens}

    # ---------- public API ----------
    def is_up(self) -> bool:
        try:
            return requests.get(self.base_url, timeout=3).ok
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except requests.RequestException:
            return []

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.7,
                 max_tokens: int = 256, json_mode: bool = False) -> LLMResult:
        """Single-turn: /api/generate. No memory of previous calls."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": self._options(temperature, max_tokens),
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"
        data = self._post("/api/generate", payload).json()
        self.last_result = LLMResult.from_response(data.get("response", ""), data)
        return self.last_result

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 256,
             json_mode: bool | dict = False) -> LLMResult:
        """Multi-turn: /api/chat. YOU send the whole history every time."""
        payload = self._chat_payload(messages, temperature, max_tokens, json_mode, stream=False)
        data = self._post("/api/chat", payload).json()
        self.last_result = LLMResult.from_response(data.get("message", {}).get("content", ""), data)
        return self.last_result

    def chat_stream(self, messages: list[dict], temperature: float = 0.7,
                    max_tokens: int = 256) -> Iterator[str]:
        """Streaming chat: yields text pieces as they arrive (NDJSON, one JSON per line).
        After the loop ends, read client.last_result for the stats."""
        payload = self._chat_payload(messages, temperature, max_tokens, False, stream=True)
        start = time.perf_counter()
        ttft = 0.0
        parts: list[str] = []
        with self._post("/api/chat", payload, stream=True) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise OllamaError(chunk["error"])
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    if not ttft:
                        ttft = time.perf_counter() - start
                    parts.append(piece)
                    yield piece
                if chunk.get("done"):
                    self.last_result = LLMResult.from_response("".join(parts), chunk, ttft)

    def _chat_payload(self, messages, temperature, max_tokens, json_mode, stream) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": self._options(temperature, max_tokens),
        }
        if json_mode:
            # "json" = any valid JSON;  a dict = a JSON schema the output MUST follow
            payload["format"] = "json" if json_mode is True else json_mode
        return payload


if __name__ == "__main__":
    # Quick smoke test:  python llm_client.py
    client = LLMClient()
    print("Ollama up:", client.is_up())
    print("Models:", client.list_models())
    res = client.chat([{"role": "user", "content": "Say hi in 5 words."}], max_tokens=30)
    print(res.text)
    print(res.stats())
