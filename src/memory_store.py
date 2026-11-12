"""
Long-term Memory Store

Two-tier design:
- ChromaDB vector store: semantic recall of historical schemas and
  strategies (e.g., "high-missing-rate columns -> drop or model-fill").
  Uses a deterministic feature-hashing embedder — zero model download,
  fully offline. A real embedding model can be swapped in later.
- JSON file fallback: if chromadb is unavailable, persist memories to
  output/memory.json so the pipeline still works offline.

Memories recorded:
  schema_<hash>   : column schema + data-quality notes of a dataset
  strategy_<hash> : cleaning/segmentation strategies that worked well
  insight_<hash>  : business conclusions from previous analyses
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

try:
    import chromadb
    _CHROMA_OK = True
except ImportError:  # pragma: no cover
    chromadb = None
    _CHROMA_OK = False


class _HashEmbedder:
    """Deterministic feature-hashing embedder.

    ChromaDB's default embedder downloads a ~80MB ONNX model on first
    use (unacceptable offline / slow networks). This embedder hashes
    tokens into a fixed-dim vector — no download, deterministic, and
    good enough for keyword-level memory recall. Swap in a real
    embedding function (e.g., sentence-transformers) when needed.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def name(self) -> str:
        return "hash_embedder_v1"

    def __call__(self, input):  # chromadb calls with a list of strings
        out = []
        for text in input:
            vec = np.zeros(self.dim, dtype=np.float32)
            tokens = re.findall(r"[\w\u4e00-\u9fff]+", str(text).lower())
            for tok in tokens:
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            out.append(vec.tolist())
        return out


class MemoryStore:
    """Semantic long-term memory with ChromaDB; JSON fallback."""

    def __init__(self, persist_dir: str = "output/memory"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None

        if _CHROMA_OK:
            try:
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_dir))
                self._collection = self._client.get_or_create_collection(
                    name="agent_memory",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=_HashEmbedder(),
                )
                print("  [Memory] ChromaDB ready (agent_memory collection)")
            except Exception as e:
                print(f"  [Memory] ChromaDB init failed ({e}); JSON fallback")
                self._client = None
        else:
            print("  [Memory] chromadb not installed; JSON fallback active")

    @property
    def available(self) -> bool:
        return self._collection is not None

    # ── Write ─────────────────────────────────────────────────

    def remember(self, kind: str, content: str, metadata: dict = None) -> str:
        """Store a memory. kind: 'schema' | 'strategy' | 'insight'."""
        doc_id = f"{kind}_{hashlib.md5(content.encode()).hexdigest()[:12]}"
        meta = {"kind": kind}
        meta.update(metadata or {})

        if self._collection is not None:
            try:
                self._collection.upsert(
                    ids=[doc_id], documents=[content], metadatas=[meta],
                )
                return doc_id
            except Exception:
                pass  # fall through to JSON

        self._json_write(doc_id, kind, content, meta)
        return doc_id

    # ── Recall ────────────────────────────────────────────────

    def recall(self, query: str, kind: str = None, k: int = 3) -> list[str]:
        """Return top-k related memory texts for a query."""
        if self._collection is not None:
            try:
                where = {"kind": kind} if kind else None
                res = self._collection.query(
                    query_texts=[query], n_results=k, where=where,
                )
                docs = (res.get("documents") or [[]])[0]
                if docs:
                    return docs
            except Exception:
                pass
        return self._json_search(query, kind, k)

    def context_for(self, query: str, kind: str = None, k: int = 3) -> str:
        """Return recalled memories joined as prompt context."""
        docs = self.recall(query, kind=kind, k=k)
        if not docs:
            return ""
        lines = [f"- {d[:400]}" for d in docs if d]
        return "\n".join(lines)

    # ── JSON fallback ─────────────────────────────────────────

    def _json_path(self) -> Path:
        return self.persist_dir / "memory.json"

    def _json_write(self, doc_id, kind, content, meta):
        path = self._json_path()
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data[doc_id] = {"kind": kind, "content": content, "metadata": meta}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    def _json_search(self, query, kind, k) -> list[str]:
        path = self._json_path()
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        # Lightweight keyword overlap scoring (fallback, no embeddings)
        q_words = set(query.lower().split())
        scored = []
        for doc in data.values():
            if kind and doc.get("kind") != kind:
                continue
            text = doc.get("content", "")
            overlap = len(q_words & set(text.lower().split()))
            scored.append((overlap, text))
        scored.sort(key=lambda x: -x[0])
        return [t for s, t in scored[:k] if s > 0 or not q_words]


# ── Convenience singleton ─────────────────────────────────────

_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store
