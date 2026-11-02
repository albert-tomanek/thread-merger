from __future__ import annotations

import logging
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import httpx2
import openai
import pytest

from thread_merge import llm_client
from thread_merge.llm_client import DiskCache, LLMClient, LLMError, QuotaExhausted, next_pacific_midnight
from thread_merge.progress import Progress, format_duration


def _client(max_retries: int = 3) -> LLMClient:
    return LLMClient(base_url="https://example.invalid/v1", api_key="x", requests_per_minute=0,
                     cache=DiskCache(None), max_retries=max_retries)


def _rate_limit(quota_id: str, retry_delay: str | None = None, retry_after: str | None = None) -> openai.RateLimitError:
    """A 429 shaped like Gemini's: the violated quota id and an optional retryDelay."""
    details: list[dict[str, Any]] = [{"violations": [{"quotaId": quota_id}]}]
    if retry_delay:
        details.append({"retryDelay": retry_delay})
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": details}}
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx2.Response(429, request=httpx2.Request("POST", "https://example.invalid/v1/chat"),
                               headers=headers, json=body)
    return openai.RateLimitError("You exceeded your current quota", response=response, body=body)


class _Flaky:
    def __init__(self, *errors: Exception) -> None:
        self.errors = list(errors)
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(llm_client.time, "sleep", slept.append)
    return slept


def test_per_minute_limit_is_retried_and_logged(sleeps: list[float], caplog: pytest.LogCaptureFixture) -> None:
    client = _client()
    send = _Flaky(_rate_limit("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", retry_delay="37s"),
                  _rate_limit("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", retry_after="4"))
    with caplog.at_level(logging.WARNING):
        assert client._request("m", send) == "ok"
    assert send.calls == 3 and client.requests_made == 1
    assert sleeps == [38.0, 4.0]  # Gemini's retryDelay (+1s margin), then the Retry-After header
    assert client.seconds_waited == 42.0
    assert "m: rate limit reached (per-minute quota); waiting 38s before retry 1/3" in caplog.text


def test_daily_quota_stops_without_retrying(sleeps: list[float]) -> None:
    send = _Flaky(_rate_limit("GenerateRequestsPerDayPerProjectPerModel-FreeTier", retry_delay="20s"))
    with pytest.raises(QuotaExhausted) as info:
        _client()._request("gemini-x", send)
    assert info.value.model == "gemini-x"
    assert not isinstance(info.value, LLMError)  # must not be swallowed by the stages' fallbacks
    assert send.calls == 1 and sleeps == []


def test_gives_up_after_max_retries(sleeps: list[float]) -> None:
    request = httpx2.Request("POST", "https://example.invalid/v1/chat")
    send = _Flaky(*[openai.APIConnectionError(request=request) for _ in range(3)])
    with pytest.raises(LLMError, match="gave up after 2 retries"):
        _client(max_retries=2)._request("m", send)
    assert sleeps == [5.0, 10.0]  # exponential backoff without a server hint


def test_next_pacific_midnight() -> None:
    pacific = ZoneInfo("America/Los_Angeles")
    assert next_pacific_midnight(datetime(2026, 11, 2, 23, 30, tzinfo=pacific)) == \
        datetime(2026, 11, 3, 0, 0, tzinfo=pacific)


def test_progress_logs_eta(caplog: pytest.LogCaptureFixture) -> None:
    progress = Progress("judge", 4)
    with caplog.at_level(logging.INFO):
        progress.step()
        progress.step(3)
    assert "judge: 1/4 batches (25%), ~" in caplog.text and "judge: 4/4 batches done in" in caplog.text
    assert [format_duration(s) for s in (7, 125, 3720)] == ["7s", "2m05s", "1h02m"]


@pytest.mark.parametrize("indices", [[None, None], [1, 0]])
def test_embed_handles_missing_or_shuffled_indices(monkeypatch: pytest.MonkeyPatch, indices: list[int | None]) -> None:
    # Gemini's OpenAI-compatible endpoint returns rows in input order with index=None.
    vectors = [[1.0, 0.0], [0.0, 1.0]] if indices[0] is None else [[0.0, 1.0], [1.0, 0.0]]
    data = [SimpleNamespace(embedding=v, index=i) for v, i in zip(vectors, indices)]
    client = _client()
    monkeypatch.setattr(client._client.embeddings, "create", lambda **_: SimpleNamespace(data=data))
    assert client.embed(model="m", texts=["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
