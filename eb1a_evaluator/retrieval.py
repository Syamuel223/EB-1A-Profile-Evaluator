"""BM25 + persistent Chroma/HNSW vector retrieval, fusion, and optional reranking."""
from collections import Counter, defaultdict
import json
import math
import os
import re
from pathlib import Path
from .config import KNOWLEDGE_DIR
from .llm import embed_query
from .knowledge import Chunk

_hnsw_cache = {}
_bm25_cache = {}
_cross_encoder_cache = {}
_chroma_cache = {}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{2,}", text.lower())


class BM25Index:
    """Inverted-index BM25 implementation; avoids scoring every corpus chunk per query."""
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = .75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.lengths = []
        self.postings = defaultdict(list)
        self.df = Counter()
        for doc_id, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk.text))
            length = sum(counts.values())
            self.lengths.append(length)
            for term, freq in counts.items():
                self.postings[term].append((doc_id, freq))
                self.df[term] += 1
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunk_ids": [c.chunk_id for c in self.chunks], "k1": self.k1, "b": self.b,
                   "lengths": self.lengths, "avgdl": self.avgdl,
                   "df": dict(self.df), "postings": {term: rows for term, rows in self.postings.items()}}
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, chunks: list[Chunk], path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("chunk_ids") != [c.chunk_id for c in chunks]:
            return None
        index = cls.__new__(cls)
        index.chunks = chunks
        index.k1, index.b = payload["k1"], payload["b"]
        index.lengths, index.avgdl = payload["lengths"], payload["avgdl"]
        index.df = Counter(payload["df"])
        index.postings = defaultdict(list, {term: [tuple(row) for row in rows] for term, rows in payload["postings"].items()})
        return index

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        scores = defaultdict(float)
        n = len(self.chunks)
        for term in tokenize(query):
            idf = math.log(1 + (n - self.df[term] + .5) / (self.df[term] + .5))
            for doc_id, freq in self.postings.get(term, []):
                norm = freq + self.k1 * (1 - self.b + self.b * self.lengths[doc_id] / max(1, self.avgdl))
                scores[doc_id] += idf * freq * (self.k1 + 1) / norm
        return sorted(scores.items(), key=lambda row: row[1], reverse=True)[:limit]


def build_vector_index(chunks: list[Chunk], path: Path | None = None) -> bool:
    """Persist HNSW if hnswlib is installed and all chunks have embeddings."""
    if not chunks or any(not chunk.embedding for chunk in chunks):
        return False
    try:
        import hnswlib
        dimension = len(chunks[0].embedding)
        index = hnswlib.Index(space="cosine", dim=dimension)
        index.init_index(max_elements=len(chunks), ef_construction=160, M=24)
        index.add_items([c.embedding for c in chunks], list(range(len(chunks))))
        index.set_ef(min(max(40, 2 * 10), len(chunks)))
        path = path or KNOWLEDGE_DIR / "hnsw.index"
        path.parent.mkdir(parents=True, exist_ok=True)
        index.save_index(str(path))
        path.with_suffix(".meta.json").write_text(json.dumps({"ids": [c.chunk_id for c in chunks], "dimension": dimension}), encoding="utf-8")
        _hnsw_cache.clear()
        return True
    except ImportError:
        return False


def build_chroma_index(chunks: list[Chunk], path: Path | None = None) -> bool:
    """Persist chunk embeddings and searchable metadata in a local Chroma collection."""
    if not chunks or any(not chunk.embedding for chunk in chunks):
        return False
    try:
        import chromadb
        from chromadb.config import Settings

        path = path or KNOWLEDGE_DIR / "chroma_db"
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        name = os.getenv("VECTOR_COLLECTION_NAME", "uscis_policy")
        try:
            client.delete_collection(name)
        except Exception:
            pass
        collection = client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[{
                "source": chunk.source,
                "section": chunk.section,
                "page": chunk.page if chunk.page is not None else -1,
                "criterion_id": chunk.criterion_id or "",
            } for chunk in chunks],
        )
        _chroma_cache.clear()
        return True
    except ImportError:
        return False


def _query_embedding(query: str, chunks: list[Chunk]):
    if not chunks or not chunks[0].embedding:
        return None
    return embed_query(query)


def _vector_search(query_vector: list[float], chunks: list[Chunk], limit: int) -> list[tuple[int, float]]:
    # Chroma is the primary persistent vector store. Keep HNSW/exact cosine as fallbacks.
    try:
        import chromadb
        from chromadb.config import Settings

        path = KNOWLEDGE_DIR / "chroma_db"
        name = os.getenv("VECTOR_COLLECTION_NAME", "uscis_policy")
        key = (str(path), name)
        if path.exists():
            if key not in _chroma_cache:
                client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
                _chroma_cache.clear()
                _chroma_cache[key] = client.get_collection(name=name, embedding_function=None)
            collection = _chroma_cache[key]
            if collection.count() == len(chunks):
                result = collection.query(query_embeddings=[query_vector], n_results=min(limit, len(chunks)), include=["distances"])
                by_chunk = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}
                ids = result.get("ids", [[]])[0]
                distances = result.get("distances", [[]])[0]
                rows = [(by_chunk[chunk_id], 1.0 - float(distance))
                        for chunk_id, distance in zip(ids, distances) if chunk_id in by_chunk]
                if rows:
                    return rows
    except Exception:
        # A missing/stale local collection must not disable the existing retrieval path.
        pass

    index_path = KNOWLEDGE_DIR / "hnsw.index"
    meta_path = index_path.with_suffix(".meta.json")
    key = (str(index_path), index_path.stat().st_mtime if index_path.exists() else None)
    try:
        import hnswlib
        if index_path.exists() and meta_path.exists():
            if key not in _hnsw_cache:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                index = hnswlib.Index(space="cosine", dim=meta["dimension"])
                index.load_index(str(index_path))
                index.set_ef(max(40, limit * 4))
                _hnsw_cache.clear()
                _hnsw_cache[key] = (index, meta["ids"])
            index, ids = _hnsw_cache[key]
            labels, distances = index.knn_query([query_vector], k=min(limit, len(ids)))
            by_chunk = {c.chunk_id: i for i, c in enumerate(chunks)}
            return [(by_chunk[ids[int(label)]], 1 - float(distance)) for label, distance in zip(labels[0], distances[0]) if ids[int(label)] in by_chunk]
    except ImportError:
        pass
    # Exact cosine fallback keeps local installs usable; HNSW is used when installed.
    output = []
    qnorm = math.sqrt(sum(x*x for x in query_vector))
    for i, chunk in enumerate(chunks):
        if not chunk.embedding:
            continue
        denom = qnorm * math.sqrt(sum(x*x for x in chunk.embedding))
        score = sum(a*b for a, b in zip(query_vector, chunk.embedding)) / denom if denom else 0
        output.append((i, score))
    return sorted(output, key=lambda row: row[1], reverse=True)[:limit]


def _cross_encoder_rerank(query: str, candidates: list[tuple[int, float]], chunks: list[Chunk]) -> list[tuple[int, float]]:
    model_name = os.getenv("RERANKER_MODEL", "")
    if not model_name or not candidates:
        return candidates
    try:
        from sentence_transformers import CrossEncoder
        if model_name not in _cross_encoder_cache:
            _cross_encoder_cache[model_name] = CrossEncoder(model_name)
        model = _cross_encoder_cache[model_name]
        pairs = [(query, chunks[index].text) for index, _ in candidates]
        scores = model.predict(pairs, show_progress_bar=False)
        return sorted(((candidates[i][0], float(scores[i])) for i in range(len(candidates))), key=lambda row: row[1], reverse=True)
    except ImportError:
        pass
    # Fast metadata-aware fallback reranker; criterion heading matches are valuable.
    bm25 = _get_bm25(chunks)
    raw = dict(bm25.search(query, len(chunks)))
    max_score = max(raw.values(), default=1.0) or 1.0
    q = query.casefold()
    reranked = []
    for idx, original in candidates:
        c = chunks[idx]
        heading_match = bool(c.section.casefold() in q or (c.criterion_id and c.criterion_id.casefold() in q))
        score = .75 * (raw.get(idx, 0.0) / max_score) + .20 * float(heading_match) + .05 * original
        reranked.append((idx, score))
    return sorted(reranked, key=lambda row: row[1], reverse=True)


def retrieve(query: str, chunks: list[Chunk], k: int = 5, mode: str = "hybrid") -> list[Chunk]:
    if not chunks:
        return []
    bm25 = _get_bm25(chunks)
    candidate_n = min(len(chunks), max(k * 5, 20))
    lexical = bm25.search(query, candidate_n)
    qvector = _query_embedding(query, chunks)
    vector = _vector_search(qvector, chunks, candidate_n) if qvector else []
    if mode == "bm25" or not vector:
        ranked = lexical
    elif mode == "vector":
        ranked = vector
    else:
        # Reciprocal-rank fusion is robust to incomparable BM25 and cosine scales.
        scores = defaultdict(float)
        for rank, (idx, _) in enumerate(lexical, start=1):
            scores[idx] += 1 / (60 + rank)
        for rank, (idx, _) in enumerate(vector, start=1):
            scores[idx] += 1 / (60 + rank)
        ranked = sorted(scores.items(), key=lambda row: row[1], reverse=True)
    ranked = _cross_encoder_rerank(query, ranked[:candidate_n], chunks)
    return [chunks[idx] for idx, score in ranked[:k] if score > 0]


def _get_bm25(chunks: list[Chunk]) -> BM25Index:
    cache_key = id(chunks)
    if cache_key not in _bm25_cache:
        _bm25_cache.clear()
        path = KNOWLEDGE_DIR / "bm25.index.json"
        index = BM25Index.load(chunks, path) if path.exists() else None
        _bm25_cache[cache_key] = (chunks, index or BM25Index(chunks))
    return _bm25_cache[cache_key][1]
