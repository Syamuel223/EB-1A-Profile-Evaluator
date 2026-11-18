"""Run model evaluation on synthetic profiles. Requires a configured API key and indexed corpus."""
import json
import re
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eb1a_evaluator.pipeline import assess_profile
from eb1a_evaluator.knowledge import load_chunks
from eb1a_evaluator.config import provider_api_key


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def add_metrics(case, output, chunks, totals):
    by_id = {item["criterion_id"]: item for item in output["criteria"]}
    for cid, expected in case["expected"].items():
        totals["rated"] += 1
        totals["correct"] += by_id[cid]["rating"] == expected
    chunk_map = {chunk.chunk_id: chunk.text for chunk in chunks}
    resume = norm(case["profile"])
    for item in output["criteria"]:
        for citation in item["citations"]:
            totals["citations"] += 1
            source = chunk_map.get(citation["chunk_id"], "")
            quote = norm(citation["quote"])
            totals["valid_citations"] += bool(source and quote and quote in norm(source))
        for evidence in item["evidence"]:
            totals["evidence"] += 1
            normalized = norm(evidence)
            totals["grounded_evidence"] += bool(normalized and normalized in resume)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the synthetic EB-1A profiles.")
    parser.add_argument("--restart", action="store_true", help="Ignore any partial checkpoint and restart all profiles.")
    args = parser.parse_args()
    cases = json.loads((ROOT / "data/profiles/synthetic_profiles.json").read_text(encoding="utf-8"))
    chunks = load_chunks()
    totals = {"correct": 0, "rated": 0, "citations": 0, "valid_citations": 0, "evidence": 0, "grounded_evidence": 0}
    partial_path = ROOT / "data/eval_results.partial.json"
    rows = []
    if partial_path.exists() and not args.restart:
        checkpoint = json.loads(partial_path.read_text(encoding="utf-8"))
        rows = checkpoint.get("completed_profiles", [])
        case_map = {case["id"]: case for case in cases}
        usable_rows = []
        for row in rows:
            assessment = row.get("assessment", {})
            criteria = assessment.get("criteria", [])
            all_fallback = bool(criteria) and all(
                item.get("reasoning") == "The model did not return a valid structured result for this criterion."
                for item in criteria
            )
            if row.get("profile_id") in case_map and len(criteria) == 10 and not all_fallback:
                usable_rows.append(row)
        discarded = len(rows) - len(usable_rows)
        rows = usable_rows
        if discarded:
            print(f"Discarding {discarded} incomplete checkpoint result(s); those profiles will be retried.", flush=True)
        for row in rows:
            add_metrics(case_map[row["profile_id"]], row["assessment"], chunks, totals)
        if rows:
            print(f"Resuming after {len(rows)} completed profiles; remaining quota will be used on the rest.", flush=True)
    completed_ids = {row["profile_id"] for row in rows}
    for case in cases:
        if case["id"] in completed_ids:
            continue
        try:
            assessment = assess_profile(case["profile"])
        except Exception as exc:
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            detail = str(exc).replace(provider_api_key(), "[redacted]")[:500]
            checkpoint = {"completed_profiles": rows, "failed_profile": case["id"], "error_type": type(exc).__name__, "status": status, "detail": detail, "note": "Check provider rate/quota details, then retry. No credentials are recorded."}
            (ROOT / "data/eval_results.partial.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
            raise SystemExit(f"Evaluation stopped at {case['id']} ({type(exc).__name__}, status={status}). Details and partial progress saved to data/eval_results.partial.json.") from None
        output = assessment.model_dump(mode="json")
        add_metrics(case, output, chunks, totals)
        rows.append({"profile_id": case["id"], "assessment": output})
        print(f"Completed {case['id']}", flush=True)
        partial_path.write_text(json.dumps({"completed_profiles": rows, "next_profile": "checkpointed"}, indent=2), encoding="utf-8")
    metrics = {
        "rating_accuracy": totals["correct"] / max(1, totals["rated"]),
        "citation_validity": totals["valid_citations"] / max(1, totals["citations"]),
        "hallucinated_evidence_rate": 1 - totals["grounded_evidence"] / max(1, totals["evidence"]),
        "counts": totals,
        "note": "Small synthetic set; metrics are diagnostic, not a general quality guarantee. Citation matching is exact after basic normalization.",
    }
    output_path = ROOT / "data/eval_results.json"
    output_path.write_text(json.dumps({"metrics": metrics, "cases": rows}, indent=2), encoding="utf-8")
    if partial_path.exists():
        partial_path.unlink()
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
