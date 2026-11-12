"""ChromaDB vector store: chunk text + embeddings + metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from langchain_core.documents import Document

from config.settings import Settings
from services.embedding_service import EmbeddingService
from utils.helpers import chunked, get_logger

log = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised when ChromaDB cannot be used (message is safe to show)."""


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: float  # cosine similarity, higher = more relevant


class VectorStore:
    def __init__(self, settings: Settings, embedder: EmbeddingService):
        self.settings = settings
        self.embedder = embedder
        self._client = None

    # ----------------------------------------------------------------- client
    def _get_client(self):
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=str(self.settings.chroma_dir),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"Could not open the ChromaDB vector database: {exc}") from exc
        return self._client

    def _collection(self):
        """Collection for the CURRENT embedding backend (cosine distance)."""
        try:
            return self._get_client().get_or_create_collection(
                name=self.embedder.collection_name, metadata={"hnsw:space": "cosine"})
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"ChromaDB error: {exc}") from exc

    # ---------------------------------------------------------------- writing
    def existing_ids(self, ids: Sequence[str]) -> Set[str]:
        """Which of ``ids`` are already stored (used to skip re-embedding)."""
        found: Set[str] = set()
        collection = self._collection()
        for batch in chunked(list(ids), 500):
            result = collection.get(ids=list(batch), include=[])
            found.update(result.get("ids", []))
        return found

    def add_documents(self, docs: Sequence[Document]) -> Dict[str, int]:
        """
        Embed and store documents that are not stored yet.
        Each document needs ``metadata["chunk_id"]``.
        Returns ``{"new": n, "reused": n}``.
        """
        if not docs:
            return {"new": 0, "reused": 0}
        ids = [d.metadata["chunk_id"] for d in docs]
        existing = self.existing_ids(ids)
        todo = [d for d in docs if d.metadata["chunk_id"] not in existing]
        if todo:
            vectors = self.embedder.embed_documents([d.page_content for d in todo])
            collection = self._collection()  # re-resolve: backend may have switched to local
            for batch_docs, batch_vecs in zip(chunked(todo, 100), chunked(vectors, 100)):
                collection.upsert(
                    ids=[d.metadata["chunk_id"] for d in batch_docs],
                    documents=[d.page_content for d in batch_docs],
                    embeddings=[list(v) for v in batch_vecs],
                    metadatas=[self._clean_metadata(d.metadata) for d in batch_docs],
                )
        return {"new": len(todo), "reused": len(docs) - len(todo)}

    @staticmethod
    def _clean_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Chroma metadata may only hold str/int/float/bool, so drop None values."""
        return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}

    # ---------------------------------------------------------------- reading
    def count(self) -> int:
        return int(self._collection().count())

    def query(self, question: str, k: int = 6, paper_ids: Optional[Sequence[int]] = None) -> List[RetrievedChunk]:
        """Return the ``k`` chunks most similar to ``question`` (optionally within some papers)."""
        collection = self._collection()
        total = collection.count()
        if total == 0:
            return []
        kwargs: Dict[str, Any] = {}
        if paper_ids is not None:
            if not paper_ids:
                return []
            kwargs["where"] = {"paper_id": {"$in": [int(p) for p in paper_ids]}}
        query_vec = self.embedder.embed_query(question)
        collection = self._collection()  # backend may have switched during embed_query
        if collection.count() == 0:
            return []
        try:
            result = collection.query(
                query_embeddings=[query_vec], n_results=min(k, collection.count()),
                include=["documents", "metadatas", "distances"], **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"ChromaDB search failed: {exc}") from exc
        chunks = []
        for cid, doc, meta, dist in zip(result["ids"][0], result["documents"][0],
                                        result["metadatas"][0], result["distances"][0]):
            chunks.append(RetrievedChunk(chunk_id=cid, text=doc, metadata=meta or {}, score=1.0 - float(dist)))
        return chunks

    def count_for_paper(self, paper_id: int) -> int:
        result = self._collection().get(where={"paper_id": int(paper_id)}, include=[])
        return len(result.get("ids", []))

    def delete_paper(self, paper_id: int) -> None:
        self._collection().delete(where={"paper_id": int(paper_id)})
