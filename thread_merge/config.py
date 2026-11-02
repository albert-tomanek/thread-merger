"""Settings and the Option A / Option B presets.

Each pipeline stage has its own backend switch, so the options can be mixed
(e.g. API claim extraction + local embeddings + threshold matching):

* ``extractor``: "api" (Option B) or "local" (Option A, Ollama).
* ``embedder``:  "api" (Option B, gemini-embedding-001) or "local" (Option A, sentence-transformers).

The "api" backends talk to one OpenAI-compatible provider (``PROVIDERS``: Gemini by default,
or Mistral).
* ``matcher``:   "llm_judge" (Option B) or "threshold" (Option A).
* ``reply_fit``: "api" (Option B) or "local" (Option A, Ollama). Checks that replies moved
  under a merged node still fit its text; skipped with text policy "concat".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .tree import TextPolicy

ExtractorKind = Literal["api", "local"]
EmbedderKind = Literal["api", "local"]
MatcherKind = Literal["llm_judge", "threshold"]
ReplyFitKind = Literal["api", "local"]
OptionName = Literal["A", "B"]


@dataclass
class ApiSettings:
    """Any OpenAI-compatible endpoint works; defaults target the Google Gemini API free tier.

    Gemini free-tier limits are per project and per model, reset at midnight Pacific time, and
    are only shown in AI Studio (https://aistudio.google.com/rate-limit). Reported figures
    (late 2026): about 500 requests/day and 10/minute for each Flash-Lite model, about
    1,000/day for embeddings, about 20/day for Flash. Extraction and judging therefore use two
    different Flash-Lite models, so each gets its own daily quota. Lower
    ``requests_per_minute`` if 429 errors persist.
    """

    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key_env: str = "GEMINI_API_KEY"
    extract_model: str = "gemini-3.1-flash-lite"
    judge_model: str = "gemini-3.5-flash-lite"
    embed_model: str = "gemini-embedding-001"
    requests_per_minute: float = 9.0
    max_retries: int = 6
    timeout_s: float = 180.0


def mistral_api() -> ApiSettings:
    """Mistral La Plateforme. New accounts no longer get a free API tier (late 2026), so this
    needs pay-as-you-go credits. Limits: https://admin.mistral.ai/plateforme/limits."""
    return ApiSettings(
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        extract_model="mistral-small-latest",
        judge_model="mistral-medium-latest",
        embed_model="mistral-embed",
        requests_per_minute=20.0,
        timeout_s=120.0,
    )


ProviderName = Literal["gemini", "mistral"]
PROVIDERS: dict[ProviderName, Callable[[], ApiSettings]] = {"gemini": ApiSettings, "mistral": mistral_api}


@dataclass
class LocalSettings:
    """Option A backends (not yet implemented, see extraction.py / embeddings.py / matching.py)."""

    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen3:14b"
    sentence_transformer_model: str = "BAAI/bge-large-en-v1.5"
    extract_batch_size: int = 8


@dataclass
class Settings:
    extractor: ExtractorKind = "api"
    embedder: EmbedderKind = "api"
    matcher: MatcherKind = "llm_judge"
    reply_fit: ReplyFitKind = "api"
    api: ApiSettings = field(default_factory=ApiSettings)
    local: LocalSettings = field(default_factory=LocalSettings)

    # Bundling: comments per extraction request, with a character budget per request. Large
    # bundles keep a ~5,000-comment pair of threads within one day of free-tier requests.
    extract_batch_size: int = 80
    extract_batch_chars: int = 48_000
    # Candidate pairs per judge request (also replies per reply-fit request).
    judge_batch_size: int = 40
    # Truncation of texts placed in prompts.
    max_text_chars: int = 1_500
    max_parent_chars: int = 400
    embed_batch_size: int = 64

    # Candidate retrieval (cosine similarity of claim embeddings). Loose on purpose: the judge
    # filters. Values depend on the embedding model and should be tuned on labelled pairs.
    candidate_top_k: int = 3
    candidate_min_sim: float = 0.70
    sibling_min_sim: float = 0.75

    # Option B: judge verdicts below this confidence are discarded.
    judge_min_confidence: float = 0.6
    # Option A: similarity at or above which the threshold matcher accepts a pair.
    threshold_accept_sim: float = 0.88

    # Comments with at most this many words are junk: never embedded or merged.
    junk_max_words: int = 2

    text_policy: TextPolicy = "best"
    cache_dir: Path | None = Path(".cache")


def preset(option: OptionName) -> Settings:
    if option == "B":
        return Settings()
    return Settings(extractor="local", embedder="local", matcher="threshold", reply_fit="local")
