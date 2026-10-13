"""Offline test of the DeepSeek provider with a fake `openai` module (no network, no key).

Usage (repo root, PYTHONPATH=.): python tests/test_deepseek_client.py
"""
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.getcwd())

calls = []
behaviour = {"mode": "ok"}


class _Err(Exception):
    def __init__(self, message="x", status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


fake = types.ModuleType("openai")
for name in ("AuthenticationError", "PermissionDeniedError", "BadRequestError", "RateLimitError",
             "APIStatusError", "APIConnectionError"):
    setattr(fake, name, type(name, (_Err,), {}))


def _resp(content, finish="stop"):
    msg = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=msg, finish_reason=finish)
    usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=50, prompt_cache_hit_tokens=0,
                                  completion_tokens_details=types.SimpleNamespace(reasoning_tokens=30))
    return types.SimpleNamespace(choices=[choice], usage=usage, model="deepseek-flash", id="req-1")


class _Completions:
    def create(self, **kw):
        calls.append(kw)
        m = behaviour["mode"]
        if m == "auth":
            raise fake.AuthenticationError("bad key", 401)
        if m == "402":
            raise fake.APIStatusError("no balance", 402)
        if m == "empty":
            return _resp("")
        if m == "length":
            return _resp('{"a":', "length")
        return _resp(json.dumps({"architecture": {"attention": "eca"}, "rationale": "r"}))


class OpenAI:
    def __init__(self, **kw):
        self.kw = kw
        self.chat = types.SimpleNamespace(completions=_Completions())


fake.OpenAI = OpenAI
sys.modules["openai"] = fake

from visionevoagent.agent_client import (AgentClient, AgentClientConfig, AgentClientError,  # noqa: E402
                                         AgentClientFatal)

os.environ["DEEPSEEK_API_KEY"] = "test-key-not-real"
log = os.path.join(tempfile.mkdtemp(), "llm_calls.jsonl")

# thinking on (default): reasoning_effort mapped, temperature not sent
c = AgentClient(AgentClientConfig(), log_path=log)
assert not c.temperature_effective
out = c.complete_json("design", "SYS", "hello", {"type": "object"}, mock_fn=lambda: {}, effort="medium",
                      temperature=0.0)
kw = calls[-1]
assert out["rationale"] == "r"
assert kw["model"] == "deepseek-flash" and kw["response_format"] == {"type": "json_object"}
assert kw["extra_body"] == {"thinking": {"type": "enabled"}} and kw["reasoning_effort"] == "high"
assert "temperature" not in kw and "json" in kw["messages"][0]["content"]
assert kw["messages"][0]["role"] == "system" and kw["messages"][1] == {"role": "user", "content": "hello"}

# thinking off: temperature applied, no reasoning_effort
c2 = AgentClient(AgentClientConfig(thinking=False), log_path=log)
assert c2.temperature_effective
c2.complete_json("design", "SYS", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                   {"role": "user", "content": "c"}], {}, mock_fn=lambda: {}, temperature=0.6)
kw = calls[-1]
assert kw["temperature"] == 0.6 and "reasoning_effort" not in kw
assert kw["extra_body"] == {"thinking": {"type": "disabled"}} and len(kw["messages"]) == 4

# error mapping
for mode, exc in (("empty", AgentClientError), ("length", AgentClientError), ("auth", AgentClientFatal),
                  ("402", AgentClientFatal)):
    behaviour["mode"] = mode
    try:
        c.complete_json("design", "S", "u", {}, mock_fn=lambda: {})
        raise AssertionError(f"{mode} did not raise")
    except exc:
        pass
    except AgentClientError as e:
        assert exc is AgentClientError, (mode, e)

# missing key -> fatal at construction
del os.environ["DEEPSEEK_API_KEY"]
try:
    AgentClient(AgentClientConfig())
    raise AssertionError("missing key accepted")
except AgentClientFatal:
    pass

recs = [json.loads(line) for line in open(log, encoding="utf-8")]
assert recs[0]["reasoning_effort"] == "high" and recs[0]["temperature_applied"] is False
assert recs[0]["input_tokens"] == 100 and recs[0]["reasoning_tokens"] == 30
assert recs[1]["temperature_applied"] is True
assert [r["ok"] for r in recs] == [True, True, False, False, False, False]
assert all("test-key" not in json.dumps(r) for r in recs)  # the key never reaches the log
print("deepseek client: all checks passed")
