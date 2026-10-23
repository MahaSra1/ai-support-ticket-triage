"""
test_llm_client.py - tests for the Ollama client WITHOUT a running Ollama
(requests is replaced with fakes via monkeypatch).
Run:  python -m pytest -v
"""
import pytest
import requests

import llm_client
from llm_client import LLMClient, LLMResult, OllamaError

DONE = {"model": "mistral", "message": {"content": '{"ok": true}'}, "done": True, "done_reason": "stop",
        "prompt_eval_count": 40, "eval_count": 20, "total_duration": 3_000_000_000,
        "load_duration": 10_000_000, "prompt_eval_duration": 500_000_000, "eval_duration": 2_000_000_000}


class FakeResponse:
    def __init__(self, status=200, data=None):
        self.status_code, self._data, self.text = status, data or {}, str(data)

    def json(self):
        return self._data


def test_metrics_from_ollama_response():
    r = LLMResult.from_response("hi", DONE)
    assert r.tokens_per_sec == 10.0          # 20 tokens / 2 s
    assert r.total_sec == 3.0 and r.prompt_tokens == 40


def test_chat_sends_options_and_schema(monkeypatch):
    sent = {}

    def fake_post(url, json, **kw):
        sent.update(json)
        return FakeResponse(200, DONE)

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    res = LLMClient("mistral", retries=0).chat([{"role": "user", "content": "x"}],
                                               temperature=0.0, max_tokens=50, json_mode={"type": "object"})
    assert res.text == '{"ok": true}'
    assert sent["options"] == {"temperature": 0.0, "num_predict": 50}   # inside "options"!
    assert sent["format"] == {"type": "object"} and sent["stream"] is False


def test_missing_model_gives_pull_hint(monkeypatch):
    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: FakeResponse(404))
    with pytest.raises(OllamaError, match="ollama pull"):
        LLMClient("nope", retries=0).chat([])


def test_retries_server_errors_then_succeeds(monkeypatch):
    answers = iter([FakeResponse(500), FakeResponse(200, DONE)])
    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: next(answers))
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)       # no real waiting
    assert LLMClient(retries=1).chat([]).text == '{"ok": true}'


def test_does_not_retry_bad_request(monkeypatch):
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return FakeResponse(400)

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    with pytest.raises(OllamaError, match="Bad request"):
        LLMClient(retries=3).chat([])
    assert len(calls) == 1                     # 4xx = our fault, retrying won't help


def test_ollama_down_gives_friendly_error(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError()

    monkeypatch.setattr(llm_client.requests, "post", boom)
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    with pytest.raises(OllamaError, match="Is the Ollama app running"):
        LLMClient(retries=1).chat([])
