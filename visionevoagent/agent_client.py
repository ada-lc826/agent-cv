"""Single entry point for all LLM calls.

provider="deepseek":  DeepSeek V4.1 Flash (`deepseek-flash`) through DeepSeek's OpenAI-compatible API
                      (official `openai` SDK, base_url https://api.deepseek.com), JSON output mode.
                      Key: DEEPSEEK_API_KEY.
provider="anthropic": Claude via the official Anthropic SDK (structured JSON output). Key: ANTHROPIC_API_KEY.
provider="mock":      no network; each agent supplies a deterministic heuristic payload.
                      Only for dry runs / tests -- never for reported results.
Every call is appended to <run_dir>/llm_calls.jsonl (tokens, latency, errors) for cost reporting.

Differences between the providers that matter for the experiments:
- Schema enforcement: Claude enforces the JSON schema server-side; DeepSeek's JSON mode only guarantees
  valid JSON, so the schema is put in the system prompt and every answer is validated by the caller
  (invalid answers are retried with feedback, then replaced by a legal fallback and counted).
- Temperature: Claude Opus 5.5 rejects it; DeepSeek ignores it in thinking mode and honours it only with
  `thinking: false`. `temperature_effective` tells callers whether a requested temperature is applied.
- Effort: Claude effort low..max; DeepSeek reasoning_effort low/high/max (medium -> high, xhigh -> max).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

log = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEEPSEEK_EFFORT = {"low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max"}


class AgentClientError(RuntimeError):
    """Any failure to obtain a usable JSON answer (network, refusal, truncation, bad JSON)."""


class AgentClientFatal(AgentClientError):
    """Misconfiguration (bad key, invalid request): retrying or falling back would only waste GPU time."""


@dataclass
class AgentClientConfig:
    provider: str = "deepseek"           # deepseek | anthropic | mock
    model: str = "deepseek-flash"        # DeepSeek V4.1 Flash; for anthropic use claude-opus-5-5
    effort: str = "medium"               # low | medium | high | xhigh | max
    thinking: bool = True                # DeepSeek only: thinking mode (its default). False -> temperature applies
    max_tokens: int = 16000
    timeout_s: float = 600.0
    max_retries: int = 3                 # SDK retries 408/409/429/5xx and connection errors
    use_fallbacks: bool = True           # Anthropic only: server-side refusal fallback (beta)
    base_url: str = ""                   # DeepSeek only: override the API endpoint


class AgentClient:
    def __init__(self, cfg: AgentClientConfig, log_path: Optional[str] = None):
        self.cfg = cfg
        self.log_path = log_path
        self._client = None
        if cfg.provider == "anthropic":
            import anthropic  # imported lazily so dry runs need no SDK / key
            self._anthropic = anthropic
            self._client = anthropic.Anthropic(max_retries=cfg.max_retries, timeout=cfg.timeout_s)
        elif cfg.provider == "deepseek":
            import openai  # DeepSeek's API is OpenAI-compatible
            key = os.environ.get("DEEPSEEK_API_KEY")
            if not key:
                raise AgentClientFatal("DEEPSEEK_API_KEY is not set")
            self._openai = openai
            self._client = openai.OpenAI(api_key=key, base_url=cfg.base_url or DEEPSEEK_BASE_URL,
                                         max_retries=cfg.max_retries, timeout=cfg.timeout_s)
        elif cfg.provider != "mock":
            raise ValueError(f"unknown provider {cfg.provider!r}")

    @property
    def temperature_effective(self) -> bool:
        """True when a `temperature` passed to complete_json actually changes sampling."""
        return self.cfg.provider == "deepseek" and not self.cfg.thinking

    def complete_json(self, purpose: str, system: str, user: Union[str, List[Dict[str, str]]],
                      schema: Dict[str, Any], mock_fn: Callable[[], Dict[str, Any]],
                      effort: Optional[str] = None, temperature: Optional[float] = None) -> Dict[str, Any]:
        """`user` is either one user message or a whole conversation (alternating user/assistant
        text turns, ending with a user turn). `effort` overrides the configured effort for this call.
        `temperature` is applied only when `temperature_effective`; otherwise it is logged as ignored."""
        t0 = time.time()
        effort = effort or self.cfg.effort
        record: Dict[str, Any] = {"purpose": purpose, "provider": self.cfg.provider, "model": self.cfg.model,
                                  "effort": effort}
        if temperature is not None:
            record["temperature"] = temperature
            record["temperature_applied"] = self.temperature_effective
        messages = [{"role": "user", "content": user}] if isinstance(user, str) else list(user)
        try:
            if self.cfg.provider == "mock":
                out = mock_fn()
            elif self.cfg.provider == "deepseek":
                out = self._deepseek_json(system, messages, schema, record, effort, temperature)
            else:
                out = self._anthropic_json(system, messages, schema, record, effort)
            record["ok"] = True
            return out
        except AgentClientError as e:
            record.update(ok=False, error=str(e))
            raise
        finally:
            record["latency_s"] = round(time.time() - t0, 2)
            self._log(record)

    # ------------------------------------------------------------------ DeepSeek
    def _deepseek_json(self, system: str, messages: List[Dict[str, str]], schema: Dict[str, Any],
                       record: Dict[str, Any], effort: str, temperature: Optional[float]) -> Dict[str, Any]:
        openai = self._openai
        # JSON mode needs the word "json" in the prompt and benefits from the expected format.
        system_json = (f"{system}\n\nRespond with a single json object that conforms to this JSON schema "
                       f"(no other text):\n{json.dumps(schema, ensure_ascii=False)}")
        kwargs: Dict[str, Any] = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            messages=[{"role": "system", "content": system_json}] + messages,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "enabled" if self.cfg.thinking else "disabled"}},
        )
        if self.cfg.thinking:
            kwargs["reasoning_effort"] = _DEEPSEEK_EFFORT.get(effort, "high")
            record["reasoning_effort"] = kwargs["reasoning_effort"]
        elif temperature is not None:
            kwargs["temperature"] = temperature
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as e:
            raise AgentClientFatal("authentication failed (check DEEPSEEK_API_KEY)") from e
        except openai.PermissionDeniedError as e:
            raise AgentClientFatal(f"permission denied: {e.message}") from e
        except openai.BadRequestError as e:
            raise AgentClientFatal(f"bad request: {e.message}") from e
        except openai.RateLimitError as e:  # also raised for an exhausted balance (HTTP 402 is APIStatusError)
            raise AgentClientError("rate limited after retries") from e
        except openai.APIStatusError as e:
            if e.status_code == 402:
                raise AgentClientFatal("insufficient DeepSeek balance") from e
            raise AgentClientError(f"API error {e.status_code}: {e.message}") from e
        except openai.APIConnectionError as e:
            raise AgentClientError("network error") from e

        usage = getattr(resp, "usage", None)
        if usage is not None:
            record["input_tokens"] = getattr(usage, "prompt_tokens", None)
            record["output_tokens"] = getattr(usage, "completion_tokens", None)
            record["cache_hit_tokens"] = getattr(usage, "prompt_cache_hit_tokens", None)
            details = getattr(usage, "completion_tokens_details", None)
            record["reasoning_tokens"] = getattr(details, "reasoning_tokens", None) if details else None
        record["served_model"] = getattr(resp, "model", None)
        record["request_id"] = getattr(resp, "id", None)
        if not resp.choices:
            raise AgentClientError("empty response (no choices)")
        choice = resp.choices[0]
        record["stop_reason"] = choice.finish_reason
        if choice.finish_reason == "length":
            raise AgentClientError("output truncated at max_tokens")
        if choice.finish_reason == "content_filter":
            raise AgentClientError("response blocked by content filter")
        text = (choice.message.content or "").strip()
        if not text:  # documented occasional behaviour of JSON mode
            raise AgentClientError("empty content in JSON mode")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise AgentClientError(f"invalid JSON: {text[:200]!r}") from e

    # ------------------------------------------------------------------ Anthropic
    def _anthropic_json(self, system: str, messages: List[Dict[str, str]], schema: Dict[str, Any],
                        record: Dict[str, Any], effort: str) -> Dict[str, Any]:
        anthropic = self._anthropic
        kwargs: Dict[str, Any] = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": effort,
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
