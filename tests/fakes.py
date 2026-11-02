"""Offline stand-ins for the LLM-backed components (used by the tests)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from thread_merge.embeddings import Vector, l2_normalize
from thread_merge.extraction import Claim
from thread_merge.matching import CandidatePair, MatchContext
from thread_merge.tree import FlatNode, MergeDecision, ReplyFitCheck, Verdict

_WORD = re.compile(r"[a-z']+")


class FakeExtractor:
    def extract(self, nodes: Sequence[FlatNode], index: Mapping[str, FlatNode]) -> dict[str, Claim]:
        out: dict[str, Claim] = {}
        for n in nodes:
            short = len(n.text.split()) <= 2
            out[n.id] = Claim(id=n.id, kind="agreement" if short else "argument", claim=n.text)
        return out


class FakeEmbedder:
    """Hashed bag of words."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> Vector:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in _WORD.findall(t.lower()):
                m[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        return l2_normalize(m)


class FakeMatcher:
    def __init__(self, accept_sim: float, allow: set[frozenset[str]] | None = None) -> None:
        self.accept_sim = accept_sim
        self.allow = allow  # if given, only these pairs can be accepted
        self.log: list[dict[str, Any]] = []

    def decide(self, pairs: Sequence[CandidatePair], ctx: MatchContext) -> list[MergeDecision]:
        out = []
        for p in pairs:
            ok = p.similarity >= self.accept_sim and (self.allow is None or frozenset((p.a, p.b)) in self.allow)
            self.log.append({"a": p.a, "b": p.b, "similarity": p.similarity, "accepted": ok})
            if ok:
                out.append(MergeDecision(p.a, p.b, Verdict.SAME, p.similarity))
        return out


class RecordingEmbedder(FakeEmbedder):
    """FakeEmbedder that remembers every text it was asked to embed."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__(dim)
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> Vector:
        self.texts.extend(texts)
        return super().embed(texts)


class FakeReplyFitter:
    """Fails exactly the reply ids in ``fail``; records every check it receives."""

    def __init__(self, fail: set[str] | None = None) -> None:
        self.fail = fail or set()
        self.checks: list[ReplyFitCheck] = []
        self.log: list[dict[str, Any]] = []

    def __call__(self, checks: Sequence[ReplyFitCheck]) -> set[str]:
        self.checks.extend(checks)
        return {c.reply_id for c in checks if c.reply_id in self.fail}
