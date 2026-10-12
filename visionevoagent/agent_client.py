"""Single entry point for all LLM calls.

provider="anthropic": Claude via the official Anthropic SDK (structured JSON output).
provider="mock":      no network; each agent supplies a deterministic heuristic payload.
                      Only for dry runs / tests -- never for reported results.
Every call is appended to <run_dir>/llm_calls.jsonl (tokens, latency, errors) for cost reporting.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)


class AgentClientError(RuntimeError):
    """Any failure to obtain a usable JSON answer (network, refusal, truncation, bad JSON)."""


class AgentClientFatal(AgentClientError):
    """Misconfiguration (bad key, invalid request): retrying or falling back would only waste GPU time."""


@dataclass
class AgentClientConfig:
    provider: str = "anthropic"          # anthropic | mock
    model: str = "claude-opus-5-5"
    effort: str = "medium"               # low | medium | high | xhigh | max
    max_tokens: int = 16000
    timeout_s: float = 600.0
    max_retries: int = 3                 # SDK retries 408/409/429/5xx and connection errors
    use_fallbacks: bool = True           # server-side refusal fallback (beta)


class AgentClient:
    def __init__(self, cfg: AgentClientConfig, log_path: Optional[str] = None):
        self.cfg = cfg
        self.log_path = log_path
        self._client = None
        if cfg.provider == "anthropic":
            import anthropic  # imported lazily so dry runs need no SDK / key
            self._anthropic = anthropic
            self._client = anthropic.Anthropic(max_retries=cfg.max_retries, timeout=cfg.timeout_s)
        elif cfg.provider != "mock":
            raise ValueError(f"unknown provider {cfg.provider!r}")

    def complete_json(self, purpose: str, system: str, user: str, schema: Dict[str, Any],
                      mock_fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()
        record: Dict[str, Any] = {"purpose": purpose, "provider": self.cfg.provider, "model": self.cfg.model}
        try:
            if self.cfg.provider == "mock":
                out = mock_fn()
            else:
                out = self._anthropic_json(system, user, schema, record)
            record["ok"] = True
            return out
        except AgentClientError as e:
            record.update(ok=False, error=str(e))
            raise
        finally:
            record["latency_s"] = round(time.time() - t0, 2)
            self._log(record)

    # ------------------------------------------------------------------
    def _anthropic_json(self, system: str, user: str, schema: Dict[str, Any],
                        record: Dict[str, Any]) -> Dict[str, Any]:
        anthropic = self._anthropic
        kwargs: Dict[str, Any] = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.cfg.effort,
                           "format": {"type": "json_schema", "schema": schema}},
        )
        try:
            if self.cfg.use_fallbacks:
                resp = self._client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                resp = self._client.messages.create(**kwargs)
        except anthropic.BadRequestError as e:
            raise AgentClientFatal(f"bad request: {e.message}") from e
        except anthropic.AuthenticationError as e:
            raise AgentClientFatal("authentication failed (check ANTHROPIC_API_KEY)") from e
        except anthropic.RateLimitError as e:
            raise AgentClientError("rate limited after retries") from e
        except anthropic.APIStatusError as e:
            raise AgentClientError(f"API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise AgentClientError("network error") from e

        usage = getattr(resp, "usage", None)
        if usage is not None:
            record["input_tokens"] = getattr(usage, "input_tokens", None)
            record["output_tokens"] = getattr(usage, "output_tokens", None)
        record["served_model"] = getattr(resp, "model", None)
        record["request_id"] = getattr(resp, "_request_id", None)
        record["stop_reason"] = resp.stop_reason
        if resp.stop_reason == "refusal":
            raise AgentClientError("model refused the request")
        if resp.stop_reason == "max_tokens":
            raise AgentClientError("output truncated at max_tokens")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise AgentClientError(f"invalid JSON: {text[:200]!r}") from e

    def _log(self, record: Dict[str, Any]) -> None:
        if not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
