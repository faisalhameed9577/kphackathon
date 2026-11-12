"""
Embedding generation.

Primary backend : Gemini embeddings (semantic, needs GEMINI_API_KEY).
Fallback backend: a local, offline "hashing" embedder. It is purely lexical (it matches
shared words, not meaning), but it needs no API, so search/RAG retrieval and all tests
work without keys.

Each backend writes to its OWN ChromaDB collection (the collection name includes the
backend and dimension), so vectors of different sizes are never mixed.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List, Optional, Sequence

from config.settings import Settings
from services.gemini_service import GeminiError, GeminiService
from utils.helpers import get_logger

log = get_logger(__name__)

_LOCAL_STOPWORDS = {
    "the", "and", "of", "in", "to", "a", "an", "is", "are", "was", "were", "for", "on", "with",
    "as", "by", "at", "or", "this", "that", "from", "be", "it", "its", "than", "these", "those",
}


class LocalHashingEmbedder:
    """Deterministic bag-of-words (+ bigrams) embedding using the hashing trick."""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension

    @staticmethod
    def _tokens(text: str) -> List[str]:
        words = re.findall(r"[a-z0-9]+", (text or "").lower())
        out = []
        for w in words:
            if w in _LOCAL_STOPWORDS or len(w) < 2:
                continue
            if len(w) > 3 and w.endswith("s"):
                w = w[:-1]  # crude plural folding
            out.append(w)
        return out

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        tokens = self._tokens(text)
        features = [(t, 1.0) for t in tokens] + [(f"{a}_{b}", 0.5) for a, b in zip(tokens, tokens[1:])]
        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "big")
            vec[h % self.dimension] += weight if (h >> 63) & 1 else -weight
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


class EmbeddingService:
    """Chooses a backend and exposes ``embed_documents`` / ``embed_query``."""

    GEMINI_MIN_RELEVANCE = 0.35  # cosine similarity below this = "not relevant enough"
    LOCAL_MIN_RELEVANCE = 0.08

    def __init__(self, settings: Settings, gemini: Optional[GeminiService] = None):
        self.settings = settings
        self.gemini = gemini
        self.local = LocalHashingEmbedder()
        self.fallback_reason: Optional[str] = None
        use_gemini = bool(gemini and gemini.available and not settings.force_local_embeddings)
        self._backend = "gemini" if use_gemini else "local"
        if not use_gemini:
            self.fallback_reason = ("FORCE_LOCAL_EMBEDDINGS is on" if settings.force_local_embeddings
                                    else "no Gemini API key")

    # ---------------------------------------------------------------- backend
    @property
    def backend(self) -> str:
        return self._backend

    @property
    def dimension(self) -> int:
        return self.settings.embedding_dimension if self._backend == "gemini" else self.local.dimension

    @property
    def collection_name(self) -> str:
        return f"papers_{self._backend}_{self.dimension}"

    @property
    def min_relevance(self) -> float:
        return self.GEMINI_MIN_RELEVANCE if self._backend == "gemini" else self.LOCAL_MIN_RELEVANCE

    def _switch_to_local(self, exc: Exception) -> None:
        """Permanently (for this session) fall back to local embeddings after a Gemini failure."""
        self._backend = "local"
        self.fallback_reason = f"Gemini embeddings failed: {exc}"
        log.warning("Switching to local embeddings. %s", self.fallback_reason)

    # ------------------------------------------------------------------ embed
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._backend == "gemini":
            try:
                return self.gemini.embed(texts, "RETRIEVAL_DOCUMENT")  # type: ignore[union-attr]
            except GeminiError as exc:
                self._switch_to_local(exc)
        return [self.local.embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if self._backend == "gemini":
            try:
                return self.gemini.embed([text], "RETRIEVAL_QUERY")[0]  # type: ignore[union-attr]
            except GeminiError as exc:
                self._switch_to_local(exc)
        return self.local.embed(text)
