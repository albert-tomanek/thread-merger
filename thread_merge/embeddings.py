"""Embedding backends. All return L2-normalised float32 matrices, so dot product = cosine."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .llm_client import LLMClient
from .progress import Progress

log = logging.getLogger(__name__)

Vector = NDArray[np.float32]


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> Vector:
        """Return an (n, d) L2-normalised float32 matrix, row i for texts[i]."""
        ...


def l2_normalize(matrix: NDArray[np.floating]) -> Vector:
    m = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class ApiEmbedder:
    """Option B: embeddings via the OpenAI-compatible API (gemini-embedding-001 by default)."""

    def __init__(self, client: LLMClient, *, model: str, batch_size: int = 64) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> Vector:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        rows: list[list[float]] = []
        progress = Progress("embedding", (len(texts) + self.batch_size - 1) // self.batch_size)
        for start in range(0, len(texts), self.batch_size):
            rows.extend(self.client.embed(model=self.model, texts=texts[start:start + self.batch_size]))
            progress.step()
        return l2_normalize(np.asarray(rows, dtype=np.float32))


class LocalEmbedder:
    """Option A: local embeddings with sentence-transformers. NOT YET IMPLEMENTED.

    Implementation notes:

    * Dependency: ``pip install sentence-transformers`` (pulls in PyTorch; install the CUDA
      build of torch first if a GPU is available).
    * In ``__init__``: ``self.model = SentenceTransformer(model_name, device=None)``; ``None``
      selects CUDA/MPS automatically. Import lazily inside ``__init__`` so Option B users do
      not need the package installed.
    * ``embed``: ``self.model.encode(list(texts), batch_size=32, normalize_embeddings=True,
      convert_to_numpy=True, show_progress_bar=len(texts) > 200)``, then
      ``.astype(np.float32)``. Empty input must return ``np.zeros((0, 0), np.float32)``.
    * Model-specific prefixes: BGE v1.5 models need none for symmetric similarity; E5 models
      require ``"query: "`` before every text. Make the prefix a constructor argument.
    * Similarity scales differ from the API embedding model, so ``Settings.candidate_min_sim``,
      ``sibling_min_sim`` and ``threshold_accept_sim`` must be re-tuned for the local model.
    * Optional: cache vectors on disk keyed by sha256(model_name + text), mirroring
      ``llm_client.DiskCache``.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed(self, texts: Sequence[str]) -> Vector:
        raise NotImplementedError("LocalEmbedder (Option A) is not implemented yet; see its docstring.")


def embed_mapping(embedder: Embedder, texts: Mapping[str, str]) -> dict[str, Vector]:
    """Embed ``{id: text}`` and return ``{id: vector}``."""
    ids = list(texts)
    if not ids:
        return {}
    matrix = embedder.embed([texts[i] for i in ids])
    if matrix.shape[0] != len(ids):
        raise ValueError(f"embedder returned {matrix.shape[0]} rows for {len(ids)} texts")
    return {nid: matrix[row] for row, nid in enumerate(ids)}
