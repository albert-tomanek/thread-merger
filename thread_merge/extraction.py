"""Claim extraction: rewrite each comment into a normalised, context-resolved claim.

Contract shared by all extractors: ``extract(nodes, index)`` returns a dict mapping EVERY
input ``FlatNode.id`` to a ``Claim`` (fallback claims for comments the model failed on).
``index`` maps ids to FlatNodes of both threads and is used to look up parent texts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol, TypeVar, get_args

from pydantic import BaseModel, field_validator

from . import prompts
from .config import LocalSettings
from .llm_client import LLMClient, LLMError
from .progress import Progress
from .tree import FlatNode

log = logging.getLogger(__name__)

Stance = Literal["pro", "anti", "neutral", "unclear"]
Relation = Literal["top_level", "agrees", "disagrees", "asks", "answers", "elaborates", "tangent"]
Kind = Literal["argument", "evidence", "question", "anecdote", "agreement", "joke", "meta", "other"]

LOW_CONTENT_KINDS: frozenset[str] = frozenset({"agreement", "joke", "meta"})
"""Kinds that always count as junk ("This.", "lol", comments about the thread itself)."""


def _coerce(value: object, allowed: tuple[str, ...], default: str) -> str:
    text = str(value).strip().lower()
    return text if text in allowed else default


class Claim(BaseModel):
    id: str
    stance: Stance = "unclear"
    relation: Relation = "tangent"
    kind: Kind = "other"
    claim: str
    junk: bool = False

    def embedding_text(self) -> str:
        """Text that gets embedded; the stance prefix separates opposite positions."""
        return f"[{self.stance}] {self.claim}"


class _ExtractedItem(BaseModel):
    ref: int
    stance: Stance = "unclear"
    relation: Relation = "tangent"
    kind: Kind = "other"
    claim: str
    junk: bool = False

    @field_validator("stance", mode="before")
    @classmethod
    def _stance(cls, v: object) -> str:
        return _coerce(v, get_args(Stance), "unclear")

    @field_validator("relation", mode="before")
    @classmethod
    def _relation(cls, v: object) -> str:
        return _coerce(v, get_args(Relation), "tangent")

    @field_validator("kind", mode="before")
    @classmethod
    def _kind(cls, v: object) -> str:
        return _coerce(v, get_args(Kind), "other")


class _ExtractionBatch(BaseModel):
    items: list[_ExtractedItem]


class ClaimExtractor(Protocol):
    def extract(self, nodes: Sequence[FlatNode], index: Mapping[str, FlatNode]) -> dict[str, Claim]: ...


def is_junk(claim: Claim, text: str, max_words: int) -> bool:
    """Junk comments are never embedded or merged; they stay in the tree, marked as junk.

    A comment is junk if the extraction model flagged it (short filler, non-sequitur, or no
    contribution to the discussion), if its kind is in LOW_CONTENT_KINDS, or if it has at most
    ``max_words`` words.
    """
    return claim.junk or claim.kind in LOW_CONTENT_KINDS or len(text.split()) <= max_words


def fallback_claim(node_id: str, text: str) -> Claim:
    return Claim(id=node_id, claim=" ".join(text.split())[:300] or "(empty)")


def truncate(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    return text if len(text) <= limit else text[: limit - 1] + "…"


I = TypeVar("I")


def chunk_by_budget(items: Sequence[I], max_items: int, max_chars: int, cost: Callable[[I], int]) -> list[list[I]]:
    """Split into consecutive chunks of at most ``max_items`` items and about ``max_chars`` cost."""
    chunks: list[list[I]] = []
    current: list[I] = []
    used = 0
    for item in items:
        c = cost(item)
        if current and (len(current) >= max_items or used + c > max_chars):
            chunks.append(current)
            current, used = [], 0
        current.append(item)
        used += c
    if current:
        chunks.append(current)
    return chunks


class ApiClaimExtractor:
    """Option B: bundled extraction through an OpenAI-compatible API (Gemini by default)."""

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        topic: str,
        batch_size: int,
        batch_chars: int,
        max_text_chars: int,
        max_parent_chars: int,
    ) -> None:
        self.client = client
        self.model = model
        self.topic = topic
        self.batch_size = batch_size
        self.batch_chars = batch_chars
        self.max_text_chars = max_text_chars
        self.max_parent_chars = max_parent_chars

    def extract(self, nodes: Sequence[FlatNode], index: Mapping[str, FlatNode]) -> dict[str, Claim]:
        batches = chunk_by_budget(
            list(nodes), self.batch_size, self.batch_chars,
            cost=lambda n: min(len(n.text), self.max_text_chars) + self.max_parent_chars // 2,
        )
        out: dict[str, Claim] = {}
        progress = Progress("claim extraction", len(batches))
        for batch in batches:
            out.update(self._run(batch, index, attempts_left=2))
            progress.step()
        return out

    def _run(self, batch: Sequence[FlatNode], index: Mapping[str, FlatNode], attempts_left: int) -> dict[str, Claim]:
        payload: list[dict[str, Any]] = []
        for ref, node in enumerate(batch, 1):
            parent = index.get(node.parent_id) if node.parent_id else None
            # Replies to the post get null: the post text is the topic itself.
            parent_text = parent.text if parent is not None and parent.depth > 0 else None
            payload.append({
                "ref": ref,
                "replying_to": truncate(parent_text, self.max_parent_chars),
                "text": truncate(node.text, self.max_text_chars),
            })

        got: dict[int, _ExtractedItem] = {}
        try:
            result = self.client.chat_json(
                model=self.model,
                system=prompts.extraction_system(self.topic),
                user=prompts.extraction_user(payload),
                schema=_ExtractionBatch,
            )
            got = {item.ref: item for item in result.items}
        except LLMError as exc:
            log.warning("claim extraction failed for %d comments: %s", len(batch), exc)

        out: dict[str, Claim] = {}
        missing: list[FlatNode] = []
        for ref, node in enumerate(batch, 1):
            item = got.get(ref)
            if item is None:
                missing.append(node)
            else:
                out[node.id] = Claim(id=node.id, stance=item.stance, relation=item.relation,
                                     kind=item.kind, claim=item.claim, junk=item.junk)

        if missing:
            if attempts_left > 0:
                half = max(1, len(missing) // 2)
                for part in (missing[:half], missing[half:]):
                    if part:
                        out.update(self._run(part, index, attempts_left - 1))
            else:
                log.warning("using fallback claims for %d comments", len(missing))
                out.update({n.id: fallback_claim(n.id, n.text) for n in missing})
        return out


class LocalClaimExtractor:
    """Option A: claim extraction with a local model served by Ollama. NOT YET IMPLEMENTED.

    Implementation notes:

    * Same contract as ``ApiClaimExtractor.extract`` (see module docstring); same prompts
      (``prompts.extraction_system`` / ``prompts.extraction_user``) and output schema
      (``_ExtractionBatch``; items keyed by the 1-based ``ref`` of each comment in the batch).
    * Ollama exposes an OpenAI-compatible API at ``settings.ollama_base_url``
      (default http://localhost:11434/v1). The simplest implementation therefore builds
      ``LLMClient(base_url=settings.ollama_base_url, api_key="ollama", requests_per_minute=0,
      cache=DiskCache(cache_dir))`` and delegates to an ``ApiClaimExtractor`` with
      ``model=settings.ollama_model`` and ``batch_size=settings.extract_batch_size``.
    * Local 7-14B models follow long multi-item JSON instructions less reliably than hosted
      models: keep batches small (5-10 comments), temperature 0, and keep the
      split-and-retry / fallback-claim behaviour of ``ApiClaimExtractor._run``.
    * Before the first request, check the server is reachable (GET ``/models`` on the base URL)
      and raise a clear error naming ``ollama serve`` and ``ollama pull <model>`` if not.
    """

    def __init__(self, settings: LocalSettings, *, topic: str, cache_dir: Any = None,
                 max_text_chars: int = 1_500, max_parent_chars: int = 400) -> None:
        self.settings = settings
        self.topic = topic
        self.cache_dir = cache_dir
        self.max_text_chars = max_text_chars
        self.max_parent_chars = max_parent_chars

    def extract(self, nodes: Sequence[FlatNode], index: Mapping[str, FlatNode]) -> dict[str, Claim]:
        raise NotImplementedError("LocalClaimExtractor (Option A) is not implemented yet; see its docstring.")
