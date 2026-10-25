"""Provider-agnostic LLM wrapper with a disk cache.

Every call is keyed by (model, system, prompt, max_tokens, temperature) and cached as
JSON under data/cache/llm/. The cache is committed, so `python -m evaluation.run`
reproduces the reported numbers with no API key and no spend. Delete the directory to
force a live re-run. The cache key includes the provider, so a response produced by one
backend is never reused for another.

temperature=0 everywhere: this is an evaluation harness, and sampling noise would show
up as model improvement.

Provider selection is entirely environment-driven - no code change is needed to switch:

    GENERATION_PROVIDER=openai      # reply drafting   (default: openai)
    JUDGE_PROVIDER=openai           # reply scoring    (default: openai)
    LABEL_PROVIDER=openai           # bulk labelling   (default: openai)
    OPENAI_API_KEY=...              # key for the selected provider only

Supported providers: openai (official API), openrouter, gemini, anthropic.
Each role's model can be overridden independently, e.g. GENERATION_MODEL=gpt-5.6-luna.
The provider for a given call is inferred from its model ID; pass `provider=` only for a
custom or proxied model name the prefix rules cannot classify.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache" / "llm"

try:  # optional: read ANTHROPIC_API_KEY from a local .env
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

DEFAULT_PROVIDER = "openai"

# Generation uses the strongest model in the family: drafting a grounded reply is the
# task being measured, so it should not be handicapped. The judge and the bulk labeller
# are the two places a cheaper model is appropriate - a judge scoring against a fixed
# rubric and a bulk classifier are both narrow, well-specified tasks.
#
# The Gemini defaults are the current non-preview tiers. gemini-3.1-pro-preview is more
# capable still, but a preview endpoint is a poor foundation for a harness whose whole
# point is reproducibility; set GENERATION_MODEL to opt into it.
#
# OpenRouter has no defaults on purpose: its catalogue (especially the ":free" tier)
# changes without notice, so a hardcoded id would rot. Its models must be named in the
# environment, and a missing one produces a clear error at first use rather than a
# confusing 404 from the API.
_MODELS: dict[str, dict[str, str]] = {
    "openai": {
        "generation": "gpt-5.6-luna",
        "judge": "gpt-5.6-luna",
        "label": "gpt-5.6-luna",
    },
    "gemini": {
        "generation": "gemini-3.8-flash",
        "judge": "gemini-3.7-flash",
        "label": "gemini-3.5-flash-lite",
    },
    "anthropic": {
        "generation": "claude-opus-5",
        "judge": "claude-sonnet-5",
        "label": "claude-haiku-4-5",
    },
    "openrouter": {},
}

_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Filled in by _resolve so a model id chosen via <ROLE>_MODEL keeps its provider.
# OpenRouter ids ("deepseek/deepseek-chat:free") share no prefix with anything, so
# they cannot be classified by pattern alone.
_MODEL_PROVIDER: dict[str, str] = {}


def _resolve(role: str) -> str | None:
    """Model for a role: explicit <ROLE>_MODEL wins, else the provider's default.

    Returns None when the provider has no default and none was configured; the error
    is raised by LLMClient at construction, so importing this module never fails just
    because an unused role is unconfigured.
    """
    provider = os.environ.get(f"{role.upper()}_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider not in _MODELS:
        raise ValueError(
            f"{role.upper()}_PROVIDER={provider!r} is not supported. "
            f"Choose one of: {', '.join(sorted(_MODELS))}."
        )
    model = os.environ.get(f"{role.upper()}_MODEL") or _MODELS[provider].get(role)
    if model:
        _MODEL_PROVIDER[model] = provider
    return model


GENERATION_MODEL = _resolve("generation")
JUDGE_MODEL = _resolve("judge")
LABEL_MODEL = _resolve("label")


def infer_provider(model: str) -> str:
    if model in _MODEL_PROVIDER:
        return _MODEL_PROVIDER[model]
    if model.startswith("gemini-"):
        return "gemini"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if "/" in model:  # every OpenRouter id is vendor/model
        return "openrouter"
    raise ValueError(
        f"Cannot infer a provider from model {model!r}. "
        "Pass provider= explicitly to LLMClient."
    )


# Both generation and judging are short, tightly specified tasks, so both providers run
# at their lowest reasoning setting. Anthropic: `effort: low`. Gemini: thinking_level
# LOW. Haiku 4.5 rejects `effort`, so it is only sent for models that accept it.
_EFFORT_UNSUPPORTED = ("claude-haiku-4-5",)

# Gemini counts thinking tokens against max_output_tokens and returns EMPTY output if it
# exhausts the budget while reasoning. Callers here pass tight caps (the judge asks for
# 400), which would silently yield empty strings, so Gemini requests get headroom on top
# of the caller's cap. The prompts constrain reply length, not this number.
_GEMINI_THINKING_HEADROOM = 2048

_MAX_ATTEMPTS = 6
_MAX_BACKOFF = 30
# Retry capacity and server faults; fail fast on anything the request itself caused
# (bad key, unknown model, malformed request) so a typo does not burn six backoffs.
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS
    return True  # connection/timeout errors carry no status


class LLMUnavailable(RuntimeError):
    """Raised when a live call is needed but no API key is configured."""


class LLMClient:
    def __init__(
        self,
        model: str = GENERATION_MODEL,
        cache_dir: Path = CACHE_DIR,
        allow_live: bool = True,
        provider: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        if not model:
            raise LLMUnavailable(
                "No model configured for this role. Set GENERATION_MODEL / JUDGE_MODEL "
                "in .env (required when the provider is openrouter, which has no "
                "defaults - see .env.example)."
            )
        self.model = model
        self.provider = (provider or infer_provider(model)).lower()
        self.cache_dir = cache_dir
        self.allow_live = allow_live
        # OpenAI-compatible providers only. None sends nothing, as the evaluation did;
        # the chat app passes "low", which cut reply latency ~3.5x with the same output.
        self.reasoning_effort = reasoning_effort
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self.hits = 0
        self.misses = 0

    @property
    def _supports_effort(self) -> bool:
        return not self.model.startswith(_EFFORT_UNSUPPORTED)

    def _key(self, system: str, prompt: str, max_tokens: int, temperature: float) -> str:
        # Provider is part of the key so switching providers can never reuse a response
        # produced by a different backend - the same model id is reachable both directly
        # and through OpenRouter, and the two are not interchangeable.
        h = hashlib.sha256()
        parts = [self.provider, self.model, system, prompt, str(max_tokens), str(temperature)]
        if self.reasoning_effort:  # only when set, so existing cache keys are unchanged
            parts.append(self.reasoning_effort)
        for part in parts:
            h.update(part.encode("utf-8", "replace"))
            h.update(b"\x00")
        return h.hexdigest()[:24]

    def _api_key(self) -> str:
        env = _API_KEY_ENV[self.provider]
        key = os.environ.get(env)
        if not key:
            raise LLMUnavailable(
                f"{env} is not set and this call is not cached.\n"
                "Set it in .env (see .env.example), or run with the committed cache."
            )
        return key

    def _sdk(self):
        if self._client is None:
            if self.provider == "gemini":
                from google import genai

                self._client = genai.Client(api_key=self._api_key())
            elif self.provider == "openai":
                from openai import OpenAI

                # Official OpenAI API: no base_url override, so the SDK's default
                # endpoint is used.
                self._client = OpenAI(api_key=self._api_key())
            elif self.provider == "openrouter":
                from openai import OpenAI

                # OpenRouter speaks the OpenAI wire format, so the same SDK is the
                # client - only the base URL differs.
                self._client = OpenAI(
                    api_key=self._api_key(), base_url=OPENROUTER_BASE_URL
                )
            else:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self._api_key())
        return self._client

    def _call_openai_compatible(self, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        # The two OpenAI-compatible backends disagree on parameter names. OpenRouter
        # takes the legacy `max_tokens` and an explicit temperature; the official API
        # rejects both for its reasoning-family models, which require
        # `max_completion_tokens` and only accept the default temperature.
        if self.provider == "openai":
            params: dict = {"max_completion_tokens": max_tokens}
        else:
            params = {"max_tokens": max_tokens, "temperature": temperature}
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        resp = self._sdk().chat.completions.create(
            model=self.model,
            messages=messages,
            **params,
        )
        if not resp.choices:
            # OpenRouter surfaces upstream provider failures in the body, not the status.
            raise RuntimeError(
                f"{self.model} returned no choices: {getattr(resp, 'error', None)}"
            )
        return (resp.choices[0].message.content or "").strip()

    def _call_gemini(self, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            # Headroom so reasoning cannot consume the whole budget - see the constant.
            max_output_tokens=max_tokens + _GEMINI_THINKING_HEADROOM,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            # No tools are ever passed; without this the SDK logs an automatic-function-
            # calling advisory on every single call.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        resp = self._sdk().models.generate_content(
            model=self.model, contents=prompt, config=cfg
        )
        text = _gemini_text(resp)
        if not text:
            reason = getattr(resp.candidates[0], "finish_reason", None) if resp.candidates else None
            raise RuntimeError(
                f"{self.model} returned no text (finish_reason={reason}). "
                "If this is MAX_TOKENS, raise max_tokens or _GEMINI_THINKING_HEADROOM."
            )
        return text

    def _call_anthropic(self, prompt: str, system: str, max_tokens: int, temperature: float) -> str:
        kwargs: dict = {}
        if self._supports_effort:
            kwargs["output_config"] = {"effort": "low"}
        else:
            # Only pre-4.6-era models accept sampling params; current ones reject them.
            kwargs["temperature"] = temperature
        resp = self._sdk().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 700,
        temperature: float = 0.0,
    ) -> str:
        key = self._key(system, prompt, max_tokens, temperature)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]

        if not self.allow_live:
            raise LLMUnavailable(f"cache miss for {key} and live calls are disabled")

        call = {
            "gemini": self._call_gemini,
            "openai": self._call_openai_compatible,
            "openrouter": self._call_openai_compatible,
            "anthropic": self._call_anthropic,
        }[self.provider]
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                text = call(prompt, system, max_tokens, temperature)
                break
            except Exception as exc:
                last_error = exc
                if attempt == _MAX_ATTEMPTS - 1 or not _is_retryable(exc):
                    raise
                # Capacity errors (503 "high demand", 429) need real backoff, not the
                # ~1s that a transient network blip needs; a full evaluation run makes
                # hundreds of calls and must survive a busy model.
                delay = min(2 ** (attempt + 1), _MAX_BACKOFF)
                time.sleep(delay + random.uniform(0, 0.5))
        else:  # pragma: no cover
            raise last_error  # type: ignore[misc]

        self.misses += 1
        path.write_text(
            json.dumps({"model": self.model, "system": system, "prompt": prompt, "text": text}),
            encoding="utf-8",
        )
        return text


def _gemini_text(resp) -> str:
    """Concatenate answer text, skipping thought parts.

    `resp.text` normally does this already, but it returns None when the only parts are
    thoughts, so the parts are walked explicitly.
    """
    if not resp.candidates:
        return ""
    out = []
    for part in resp.candidates[0].content.parts or []:
        if getattr(part, "thought", False):
            continue
        if getattr(part, "text", None):
            out.append(part.text)
    return "".join(out).strip() or (resp.text or "").strip()


_JSON_NUDGE = (
    "\n\nReturn ONLY the raw JSON object. No explanation, no reasoning, no markdown "
    "fences. Start your response with { and end it with }."
)


def complete_json(
    client: "LLMClient", prompt: str, system: str = "", max_tokens: int = 700
) -> dict:
    """`complete` + `parse_json`, with one strict retry if the model did not emit JSON.

    Needed because not every model honours a JSON instruction. Reasoning models served
    over OpenRouter are the common case: they emit their chain-of-thought as ordinary
    content and can exhaust max_tokens before reaching the JSON at all. The retry
    appends a blunter instruction and doubles the budget, which is enough for models
    that merely wrapped or prefaced the object. Models that still fail raise, rather
    than silently scoring an empty reply.
    """
    raw = client.complete(prompt, system=system, max_tokens=max_tokens)
    try:
        return parse_json(raw)
    except ValueError:
        retry = client.complete(
            prompt + _JSON_NUDGE, system=system, max_tokens=max_tokens * 2
        )
        return parse_json(retry)


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model response.

    Models sometimes wrap JSON in prose or a fenced block despite instructions; failing
    the whole evaluation over that would be silly.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, depth = text.find("{"), 0
    if start == -1:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"unbalanced JSON in response: {text[:200]!r}")
