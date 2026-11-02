"""Candidate retrieval (embeddings) and merge deciders (LLM judge or similarity threshold).

Also the reply-fit check that runs after merging (``LLMReplyFitter``, ``LocalReplyFitter``).

Contract shared by all matchers: ``decide(pairs, ctx)`` returns ``MergeDecision``s only for
pairs it accepts, and appends one dict per judged pair to ``self.log`` (used for debugging
and for tuning thresholds). Keys of a log entry: ``a``, ``b``, ``similarity``, ``verdict``,
``confidence``, ``accepted``, ``reason`` (plus matcher-specific extras).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, field_validator

from . import prompts
from .embeddings import Vector
from .extraction import Claim, truncate
from .llm_client import LLMClient, LLMError
from .progress import Progress
from .config import LocalSettings
from .tree import MergeDecision, ReplyFitCheck, Verdict

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- retrieval


@dataclass(frozen=True)
class CandidatePair:
    a: str
    b: str
    similarity: float


def _stack(ids: Sequence[str], vectors: Mapping[str, Vector]) -> Vector:
    return np.stack([vectors[i] for i in ids]).astype(np.float32)


def _top_k(sim: Vector, k: int) -> np.ndarray:
    """Column indices of the k largest values in each row (unordered)."""
    return np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]


def cross_candidates(
    ids_a: Sequence[str], ids_b: Sequence[str], vectors: Mapping[str, Vector], *, top_k: int, min_sim: float
) -> list[CandidatePair]:
    """Pairs (a in A, b in B) where b is among a's top-k neighbours or vice versa, with
    cosine >= min_sim. Sorted by decreasing similarity."""
    if not ids_a or not ids_b:
        return []
    sim = _stack(ids_a, vectors) @ _stack(ids_b, vectors).T
    chosen: dict[tuple[int, int], float] = {}
    for i, cols in enumerate(_top_k(sim, min(top_k, len(ids_b)))):
        for j in cols:
            if sim[i, j] >= min_sim:
                chosen[(i, int(j))] = float(sim[i, j])
    for j, rows in enumerate(_top_k(sim.T, min(top_k, len(ids_a)))):
        for i in rows:
            if sim[i, j] >= min_sim:
                chosen[(int(i), j)] = float(sim[i, j])
    pairs = [CandidatePair(ids_a[i], ids_b[j], s) for (i, j), s in chosen.items()]
    return sorted(pairs, key=lambda p: -p.similarity)


def sibling_candidates(
    ids: Sequence[str], vectors: Mapping[str, Vector], *, top_k: int, min_sim: float
) -> list[CandidatePair]:
    """Unordered pairs within one set of siblings, top-k per node, cosine >= min_sim."""
    ids = [i for i in ids if i in vectors]
    if len(ids) < 2:
        return []
    m = _stack(ids, vectors)
    sim = m @ m.T
    np.fill_diagonal(sim, -np.inf)
    chosen: dict[tuple[int, int], float] = {}
    for i, cols in enumerate(_top_k(sim, min(top_k, len(ids) - 1))):
        for j in cols:
            if sim[i, j] >= min_sim:
                chosen[(min(i, int(j)), max(i, int(j)))] = float(sim[i, j])
    pairs = [CandidatePair(ids[i], ids[j], s) for (i, j), s in chosen.items()]
    return sorted(pairs, key=lambda p: -p.similarity)


def select_one_to_one(decisions: Sequence[MergeDecision]) -> list[MergeDecision]:
    """Greedy one-to-one selection by confidence: every node joins at most one accepted pair.
    Used for the cross-thread pass so that transitive chains cannot glue distinct points."""
    used: set[str] = set()
    out: list[MergeDecision] = []
    for d in sorted(decisions, key=lambda d: -d.confidence):
        if d.a not in used and d.b not in used:
            out.append(d)
            used.update((d.a, d.b))
    return out


# --------------------------------------------------------------------------- deciders


@dataclass(frozen=True)
class MatchContext:
    claims: Mapping[str, Claim]
    texts: Mapping[str, str]
    parent_texts: Mapping[str, str | None]  # None for replies to the post


class Matcher(Protocol):
    log: list[dict[str, Any]]

    def decide(self, pairs: Sequence[CandidatePair], ctx: MatchContext) -> list[MergeDecision]: ...


class _PairJudgement(BaseModel):
    pair: int
    stance_same: bool
    function_same: bool
    core_same: bool
    verdict: Literal["same", "a_subsumes_b", "b_subsumes_a", "distinct"]
    confidence: float = 0.5
    reason: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def _verdict(cls, v: object) -> str:
        text = str(v).strip().lower()
        return text if text in ("same", "a_subsumes_b", "b_subsumes_a") else "distinct"

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, v: object) -> float:
        try:
            return min(1.0, max(0.0, float(v)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.5


class _JudgementBatch(BaseModel):
    items: list[_PairJudgement]


class LLMJudgeMatcher:
    """Option B: bundled LLM judgement of candidate pairs against a fixed rubric.

    The three yes/no tests gate the verdict: if any is false the pair is treated as
    distinct, whatever verdict the model wrote.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        topic: str,
        batch_size: int,
        min_confidence: float,
        max_text_chars: int,
        max_parent_chars: int,
    ) -> None:
        self.client = client
        self.model = model
        self.topic = topic
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        self.max_text_chars = max_text_chars
        self.max_parent_chars = max_parent_chars
        self.log: list[dict[str, Any]] = []

    def decide(self, pairs: Sequence[CandidatePair], ctx: MatchContext) -> list[MergeDecision]:
        out: list[MergeDecision] = []
        progress = Progress("judge", (len(pairs) + self.batch_size - 1) // self.batch_size)
        for start in range(0, len(pairs), self.batch_size):
            out.extend(self._judge(pairs[start:start + self.batch_size], ctx, retry_missing=True))
            progress.step()
        return out

    def _side(self, nid: str, ctx: MatchContext) -> dict[str, Any]:
        claim = ctx.claims.get(nid)
        return {
            "claim": claim.claim if claim else None,
            "stance": claim.stance if claim else "unclear",
            "kind": claim.kind if claim else "other",
            "text": truncate(ctx.texts.get(nid, ""), self.max_text_chars),
            "replying_to": truncate(ctx.parent_texts.get(nid), self.max_parent_chars),
        }

    def _judge(self, batch: Sequence[CandidatePair], ctx: MatchContext, retry_missing: bool) -> list[MergeDecision]:
        payload = [{"pair": k, "a": self._side(p.a, ctx), "b": self._side(p.b, ctx)}
                   for k, p in enumerate(batch, 1)]
        try:
            result = self.client.chat_json(
                model=self.model,
                system=prompts.judge_system(self.topic),
                user=prompts.judge_user(payload),
                schema=_JudgementBatch,
            )
            by_pair = {j.pair: j for j in result.items}
        except LLMError as exc:
            log.warning("judge request failed for %d pairs: %s", len(batch), exc)
            by_pair = {}

        out: list[MergeDecision] = []
        missing: list[CandidatePair] = []
        for k, p in enumerate(batch, 1):
            j = by_pair.get(k)
            if j is None:
                missing.append(p)
                continue
            verdict = j.verdict if (j.stance_same and j.function_same and j.core_same) else "distinct"
            accepted = verdict != "distinct" and j.confidence >= self.min_confidence
            self.log.append({
                "a": p.a, "b": p.b, "similarity": round(p.similarity, 4), "verdict": verdict,
                "confidence": j.confidence, "accepted": accepted, "reason": j.reason,
                "tests": {"stance_same": j.stance_same, "function_same": j.function_same,
                          "core_same": j.core_same},
            })
            if accepted:
                out.append(MergeDecision(p.a, p.b, Verdict(verdict), j.confidence, j.reason))

        if missing:
            if retry_missing:
                out.extend(self._judge(missing, ctx, retry_missing=False))
            else:
                for p in missing:
                    self.log.append({"a": p.a, "b": p.b, "similarity": round(p.similarity, 4),
                                     "verdict": None, "confidence": None, "accepted": False,
                                     "reason": "no judgement returned"})
        return out


class ThresholdMatcher:
    """Option A: accept pairs from similarity plus claim metadata, without an LLM.
    NOT YET IMPLEMENTED.

    Implementation notes for ``decide(pairs, ctx)``:

    * For each ``CandidatePair`` look up ``ctx.claims[pair.a]`` and ``ctx.claims[pair.b]``.
    * Reject when the stances are opposite ("pro" vs "anti"; "neutral" and "unclear" are
      compatible with anything).
    * Reject when exactly one of the two claims has kind "question" (function mismatch).
    * Reject when the relations are opposite ("agrees" vs "disagrees").
    * Accept when ``pair.similarity >= self.accept_sim``; verdict ``Verdict.SAME``;
      ``confidence = pair.similarity``; reason e.g. ``"cos=0.91, stances compatible"``.
    * Subsumption heuristic: when accepted and one original text (``ctx.texts``) is more than
      twice as long as the other, use ``A_SUBSUMES_B`` / ``B_SUBSUMES_A`` for the longer one so
      the "best" text policy keeps the fuller comment.
    * Append one log dict per pair with the keys listed in this module's docstring.
    * The cross-thread one-to-one constraint is applied by the caller
      (``select_one_to_one``). Optional improvement: offer ``assignment="hungarian"`` using
      ``scipy.optimize.linear_sum_assignment`` on a matrix of accepted similarities (rejected
      pairs set to 0) to maximise total similarity instead of greedy selection.
    * Tuning: run once, export the logged pairs with similarities to CSV, hand-label ~30-50
      pairs as merge / no-merge and choose the ``accept_sim`` with the best F1.
    """

    def __init__(self, accept_sim: float) -> None:
        self.accept_sim = accept_sim
        self.log: list[dict[str, Any]] = []

    def decide(self, pairs: Sequence[CandidatePair], ctx: MatchContext) -> list[MergeDecision]:
        raise NotImplementedError("ThresholdMatcher (Option A) is not implemented yet; see its docstring.")


# --------------------------------------------------------------------------- reply fit


class ReplyFitter(Protocol):
    """Decides which replies no longer make sense under a merged node's retained text
    (a ``tree.ReplyFitDecider``). Appends one dict per check to ``self.log``."""

    log: list[dict[str, Any]]

    def __call__(self, checks: Sequence[ReplyFitCheck]) -> set[str]: ...


class _ReplyFit(BaseModel):
    check: int
    fits: bool
    reason: str = ""


class _ReplyFitBatch(BaseModel):
    items: list[_ReplyFit]


class LLMReplyFitter:
    """Option B: bundled LLM check of whether moved replies fit their new parent's text.

    Checks the model does not answer are treated as fitting (the merge stays as it is) and
    logged with reason "no judgement returned".
    """

    def __init__(self, client: LLMClient, *, model: str, topic: str, batch_size: int,
                 max_text_chars: int, max_parent_chars: int) -> None:
        self.client = client
        self.model = model
        self.topic = topic
        self.batch_size = batch_size
        self.max_text_chars = max_text_chars
        self.max_parent_chars = max_parent_chars
        self.log: list[dict[str, Any]] = []

    def __call__(self, checks: Sequence[ReplyFitCheck]) -> set[str]:
        failed: set[str] = set()
        progress = Progress("reply fit", (len(checks) + self.batch_size - 1) // self.batch_size)
        for start in range(0, len(checks), self.batch_size):
            batch = checks[start:start + self.batch_size]
            payload = [{"check": k,
                        "retained": truncate(c.retained_text, self.max_text_chars),
                        "discarded": truncate(c.discarded_text, self.max_text_chars),
                        "reply": truncate(c.reply_text, self.max_parent_chars)}
                       for k, c in enumerate(batch, 1)]
            try:
                result = self.client.chat_json(
                    model=self.model,
                    system=prompts.reply_fit_system(self.topic),
                    user=prompts.reply_fit_user(payload),
                    schema=_ReplyFitBatch,
                )
                by_check = {item.check: item for item in result.items}
            except LLMError as exc:
                log.warning("reply fit request failed for %d replies: %s", len(batch), exc)
                by_check = {}
            for k, c in enumerate(batch, 1):
                item = by_check.get(k)
                fits = item.fits if item is not None else True
                if not fits:
                    failed.add(c.reply_id)
                self.log.append({"merged": c.merged_id, "discarded": c.discarded_id, "reply": c.reply_id,
                                 "fits": fits, "reason": item.reason if item else "no judgement returned"})
            progress.step()
        return failed


class LocalReplyFitter:
    """Option A: reply-fit check with a local model served by Ollama. NOT YET IMPLEMENTED.

    Implementation notes:

    * Same contract as ``LLMReplyFitter.__call__``: receives ``tree.ReplyFitCheck``s and
      returns the ``reply_id``s that do NOT fit; appends one log dict per check with keys
      ``merged``, ``discarded``, ``reply``, ``fits``, ``reason``.
    * Simplest implementation: build ``LLMClient(base_url=settings.ollama_base_url,
      api_key="ollama", requests_per_minute=0, cache=DiskCache(cache_dir))`` and delegate to
      ``LLMReplyFitter`` with ``model=settings.ollama_model``; same prompts
      (``prompts.reply_fit_system`` / ``prompts.reply_fit_user``).
    * Keep batches small (5-10 checks) for 7-14B models, as in ``LocalClaimExtractor``.
    * There is no reliable non-LLM version of this test (it depends on what the reply refers
      to), so an embedding-only Option A should instead run with ``--text-policy concat``,
      which keeps every text and skips this check.
    """

    def __init__(self, settings: LocalSettings, *, topic: str, cache_dir: Any = None) -> None:
        self.settings = settings
        self.topic = topic
        self.cache_dir = cache_dir
        self.log: list[dict[str, Any]] = []

    def __call__(self, checks: Sequence[ReplyFitCheck]) -> set[str]:
        raise NotImplementedError("LocalReplyFitter (Option A) is not implemented yet; see its docstring.")
