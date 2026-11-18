"""Offline before/after retrieval evaluation: old token overlap vs BM25/hybrid retrieval."""
import json
import importlib.util
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eb1a_evaluator.knowledge import load_chunks
from eb1a_evaluator.retrieval import retrieve, tokenize
from eb1a_evaluator.schemas import CRITERIA

K = 5


def old_overlap(query, chunks):
    terms = set(tokenize(query))
    scored = []
    for chunk in chunks:
        doc = set(tokenize(chunk.text))
        score = len(terms & doc) / max(1, len(terms))
        scored.append((score, chunk))
    return [chunk for score, chunk in sorted(scored, key=lambda x: x[0], reverse=True)[:K] if score > 0]


def metrics(ranked, relevant):
    ids = [chunk.chunk_id for chunk in ranked]
    hits = [i for i, chunk_id in enumerate(ids, 1) if chunk_id in relevant]
    recall = len(set(ids) & relevant) / max(1, len(relevant))
    rr = 1 / hits[0] if hits else 0
    dcg = sum(1 / math.log2(rank + 1) for rank in hits)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), K) + 1))
    return {"recall_at_5": recall, "reciprocal_rank": rr, "ndcg_at_5": dcg / ideal if ideal else 0}


def main():
    chunks = load_chunks()
    rows = []
    for cid, name in CRITERIA:
        query = f"{cid} {name} USCIS evidence requirements"
        relevant = {chunk.chunk_id for chunk in chunks if chunk.criterion_id == cid}
        if not relevant:
            continue
        before = metrics(old_overlap(query, chunks), relevant)
        after = metrics(retrieve(query, chunks, K, "hybrid"), relevant)
        rows.append({"criterion_id": cid, "relevant_chunks": len(relevant), "before_token_overlap": before, "after_bm25_vector_hybrid": after})
    if not rows:
        raise SystemExit("No criterion-tagged chunks found. Re-index the USCIS PDF with `python -m eb1a_evaluator.cli ingest --local`.")
    metrics_names = ("recall_at_5", "reciprocal_rank", "ndcg_at_5")
    summary = {}
    for name in metrics_names:
        before = sum(row["before_token_overlap"][name] for row in rows) / len(rows)
        after = sum(row["after_bm25_vector_hybrid"][name] for row in rows) / len(rows)
        summary[name] = {"before": round(before, 4), "after": round(after, 4), "delta": round(after - before, 4)}
    hnsw_file = Path(__file__).resolve().parents[1] / "data/knowledge_base/hnsw.index"
    has_vectors = sum(bool(c.embedding) for c in chunks)
    vector_index = "HNSW" if importlib.util.find_spec("hnswlib") and hnsw_file.exists() else "exact cosine"
    limitations = ["Criterion tags come from detected section headings, so these are weak labels, not human relevance judgments.", "Queries include criterion IDs and the reranker uses criterion metadata; the measured gains are optimistic by construction."]
    if not has_vectors:
        limitations.append("No stored embeddings were available; the after method was BM25 with metadata-aware reranking.")
    elif vector_index != "HNSW":
        limitations.append("Gemini vectors were used with exact cosine search; HNSW was unavailable in this Windows environment.")
    report = {"method": "Criterion metadata is used as weak relevance labels; compare legacy set-overlap baseline to current BM25 + optional vector retrieval + reranking.", "vector_chunks_available": has_vectors, "vector_index": vector_index if has_vectors else "not active", "queries_scored": len(rows), "k": K, "summary": summary, "limitations": limitations}
    target = Path(__file__).resolve().parents[1] / "data/retrieval_eval_results.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
