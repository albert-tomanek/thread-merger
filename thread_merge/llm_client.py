"""Thin wrapper over the OpenAI-compatible API: JSON-validated chat, embeddings, a request
throttle and an on-disk response cache (re-runs cost no quota).

Gemini (OpenAI-compatible endpoint), Mistral, Groq and Ollama all accept this request format,
so switching provider is a change of ``base_url``, key and model names.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar
from zoneinfo import ZoneInfo

import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from .config import ApiSettings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

_SERVER_PROBLEMS = (openai.APIConnectionError, openai.InternalServerError)  # incl. timeouts, 5xx
_MAX_BACKOFF_S = 120.0


class LLMError(RuntimeError):
    """A request failed after retries, or the model never produced valid output."""


class QuotaExhausted(RuntimeError):
    """The provider's daily request quota for a model is used up.

    Deliberately not an ``LLMError``: stages catch ``LLMError`` and fall back to defaults,
    whereas this must stop the whole run (retrying would only wait until the quota resets).
    """

    def __init__(self, model: str, detail: str) -> None:
        super().__init__(f"daily request quota for {model} is used up: {detail}")
        self.model = model


def next_pacific_midnight(now: datetime | None = None) -> datetime:
    """When Gemini daily quotas reset (midnight Pacific time), as an aware datetime."""
    pacific = ZoneInfo("America/Los_Angeles")
    now = (now or datetime.now(pacific)).astimezone(pacific)
    return datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=pacific)


def _error_text(exc: openai.APIStatusError) -> str:
    return f"{exc} {json.dumps(exc.body, default=str) if exc.body is not None else ''}"


def _is_daily_quota(exc: openai.RateLimitError) -> bool:
    """Gemini names the violated quota, e.g. "GenerateRequestsPerDayPerProjectPerModel-FreeTier"."""
    return "perday" in re.sub(r"[^a-z]", "", _error_text(exc).lower())


def _retry_delay(exc: openai.APIError, attempt: int) -> float:
    """Server-suggested delay (Retry-After header or Gemini's retryDelay), else exponential."""
    response = getattr(exc, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header:
        try:
            return min(float(header), _MAX_BACKOFF_S)
        except ValueError:
            pass
    if isinstance(exc, openai.APIStatusError):
        match = re.search(r"retry_?delay\W+(\d+(?:\.\d+)?)s", _error_text(exc), re.IGNORECASE)
        if match:
            return min(float(match.group(1)) + 1.0, _MAX_BACKOFF_S)
    return min(5.0 * 2 ** attempt, _MAX_BACKOFF_S)


class DiskCache:
    def __init__(self, directory: Path | None) -> None:
        self._dir = directory
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(payload: Mapping[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path | None:
        return None if self._dir is None else self._dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def put(self, key: str, value: Any) -> None:
        path = self._path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


class RateLimiter:
    """Spaces requests at least ``60 / requests_per_minute`` seconds apart (0 disables)."""

    def __init__(self, requests_per_minute: float) -> None:
        self._interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._last = float("-inf")

    def wait(self) -> None:
        delay = self._last + self._interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        requests_per_minute: float,
        cache: DiskCache,
        max_retries: int = 6,
        timeout_s: float = 120.0,
    ) -> None:
        # Retries are done in ``_request`` rather than by the SDK, so every wait is logged.
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout_s)
        self._limiter = RateLimiter(requests_per_minute)
        self.max_retries = max_retries
        self.cache = cache
        self.requests_made = 0
        self.seconds_waited = 0.0

    @classmethod
    def from_settings(cls, api: ApiSettings, cache_dir: Path | None) -> LLMClient:
        key = os.environ.get(api.api_key_env)
        if not key:
            raise LLMError(f"environment variable {api.api_key_env} is not set")
        return cls(
            base_url=api.base_url,
            api_key=key,
            requests_per_minute=api.requests_per_minute,
            cache=DiskCache(cache_dir),
            max_retries=api.max_retries,
            timeout_s=api.timeout_s,
        )

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[T],
        temperature: float = 0.0,
        retries_on_invalid: int = 1,
    ) -> T:
        """One chat completion in JSON mode, validated against ``schema``.

        On invalid output the model is shown the error and asked again, up to
        ``retries_on_invalid`` times. Valid results are cached by request content.
        """
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        key = DiskCache.key({"op": "chat", "model": model, "temperature": temperature,
                             "schema": schema.__name__, "messages": messages})
        cached = self.cache.get(key)
        if cached is not None:
            try:
                return schema.model_validate(cached)
            except ValidationError:
                pass

        last_error: Exception | None = None
        for _ in range(retries_on_invalid + 1):
            response = self._request(model, lambda: self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            ))
            content = response.choices[0].message.content or ""
            try:
                parsed = schema.model_validate_json(_strip_code_fences(content))
            except ValidationError as exc:
                last_error = exc
                log.debug("invalid JSON from %s: %s", model, exc)
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "That reply did not match the required JSON schema "
                     f"({_short_error(exc)}). Reply again with only the corrected JSON object."},
                ]
                continue
            self.cache.put(key, parsed.model_dump(mode="json"))
            return parsed
        raise LLMError(f"model {model} returned invalid JSON: {last_error}")

    def embed(self, *, model: str, texts: Sequence[str]) -> list[list[float]]:
        """Embeddings for ``texts`` in input order (one request; batch upstream)."""
        key = DiskCache.key({"op": "embed", "model": model, "input": list(texts)})
        cached = self.cache.get(key)
        if cached is not None:
            return [list(map(float, row)) for row in cached]
        response = self._request(model, lambda: self._client.embeddings.create(
            model=model, input=list(texts), encoding_format="float"
        ))
        data = list(response.data)
        # Gemini's OpenAI-compatible endpoint leaves ``index`` unset; its rows are in input order.
        if all(d.index is not None for d in data):
            data.sort(key=lambda d: d.index)
        rows = [list(d.embedding) for d in data]
        if len(rows) != len(texts):
            raise LLMError(f"expected {len(texts)} embeddings, got {len(rows)}")
        self.cache.put(key, rows)
        return rows


    def _request(self, model: str, send: Callable[[], R]) -> R:
        """Send one request with throttling and logged retries.

        Per-minute rate limits and server/connection problems are retried up to
        ``max_retries`` times, each wait logged with its cause. A used-up daily quota raises
        ``QuotaExhausted`` at once; anything else non-retryable propagates.
        """
        for attempt in range(self.max_retries + 1):
            self._limiter.wait()
            try:
                result = send()
            except openai.RateLimitError as exc:
                if _is_daily_quota(exc):
                    raise QuotaExhausted(model, str(exc)[:300]) from exc
                cause, error = "rate limit reached (per-minute quota)", exc
            except _SERVER_PROBLEMS as exc:
                cause, error = f"server or connection problem ({type(exc).__name__})", exc
            else:
                self.requests_made += 1
                return result
            if attempt == self.max_retries:
                raise LLMError(f"{model}: {cause}; gave up after {self.max_retries} retries: {error}") from error
            delay = _retry_delay(error, attempt)
            log.warning("%s: %s; waiting %.0fs before retry %d/%d",
                        model, cause, delay, attempt + 1, self.max_retries)
            time.sleep(delay)
            self.seconds_waited += delay
        raise AssertionError("unreachable")


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _short_error(exc: ValidationError) -> str:
    errors = exc.errors()
    parts = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in errors[:3]]
    return "; ".join(parts) + (" ..." if len(errors) > 3 else "")
