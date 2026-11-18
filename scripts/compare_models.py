"""Compare two model/prompt configurations on the labeled synthetic profiles."""
import argparse
import json
import re
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eb1a_evaluator.config import provider_model
from eb1a_evaluator.knowledge import load_chunks
from eb1a_evaluator.pipeline import assess_profile

ROOT = Path(__file__).resolve().parents[1]


def norm(value):
    return re.sub(r"\W+", " ", value.casefold()).strip()


def score_cases(cases, outputs, chunks):
    chunk_text = {c.chunk_id: norm(c.text) for c in chunks}
    total = correct = citations = citation_valid = evidence = evidence_valid = 0
    for case, result in zip(cases, outputs):
        rows = {row["criterion_id"]: row for row in result["criteria"]}
        for cid, expected in case["expected"].items():
            total += 1
            correct += rows.get(cid, {}).get("rating") == expected
        resume = norm(case["profile"])
        for row in result["criteria"]:
            for cite in row["citations"]:
                citations += 1
                quote = norm(cite.get("quote", ""))
                citation_valid += bool(quote and quote in chunk_text.get(cite.get("chunk_id"), ""))
            for item in row["evidence"]:
                evidence += 1
                grounded = norm(item)
                evidence_valid += bool(grounded and grounded in resume)
    metas = [result.get("run_metadata", {}) for result in outputs]
    costs = [m.get("estimated_cost_usd") for m in metas if m.get("estimated_cost_usd") is not None]
    model_latencies = [m["model_latency_seconds"] for m in metas if m.get("model_latency_seconds") is not None]
    return {
        "rating_accuracy": correct / max(1, total),
        "citation_validity": citation_valid / max(1, citations),
        "hallucinated_evidence_rate": 1 - evidence_valid / max(1, evidence),
        "avg_model_latency_seconds_per_profile": sum(model_latencies) / max(1, len(model_latencies)) if model_latencies else None,
        "avg_end_to_end_seconds_per_profile": sum(m.get("elapsed_seconds", 0) for m in metas) / max(1, len(metas)),
        "total_input_tokens": sum(m.get("input_tokens", 0) for m in metas),
        "total_output_tokens": sum(m.get("output_tokens", 0) for m in metas),
        "estimated_total_cost_usd": sum(costs) if costs else None,
        "counts": {"ratings": total, "citations": citations, "evidence_items": evidence},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", default=provider_model())
    # Provider adapters route both runs through the active provider. Default to
    # the same model so the out-of-box comparison measures prompt strategy only.
    parser.add_argument("--model-b", default=provider_model())
    parser.add_argument("--prompt-a", choices=["balanced", "evidence_first"], default="balanced")
    parser.add_argument("--prompt-b", choices=["balanced", "evidence_first"], default="evidence_first")
    parser.add_argument("--limit", type=int, default=5, help="Number of synthetic profiles to run (1-5)")
    args = parser.parse_args()
    cases = json.loads((ROOT / "data/profiles/synthetic_profiles.json").read_text(encoding="utf-8"))[:max(1, min(5, args.limit))]
    chunks = load_chunks()
    variants = [("A", args.model_a, args.prompt_a), ("B", args.model_b, args.prompt_b)]
    report = {"profiles": [c["id"] for c in cases], "runs": {}}
    for label, model, prompt in variants:
        outputs = []
        for case in cases:
            started = time.perf_counter()
            result = assess_profile(case["profile"], model=model, prompt_strategy=prompt)
            outputs.append(result.model_dump(mode="json"))
            print(f"{label}: finished {case['id']} in {time.perf_counter()-started:.1f}s")
        report["runs"][label] = {"model": model, "prompt_strategy": prompt, "metrics": score_cases(cases, outputs, chunks), "cases": [{"profile_id": case["id"], "assessment": out} for case, out in zip(cases, outputs)]}
    a, b = (report["runs"][x]["metrics"] for x in ("A", "B"))
    report["delta_B_minus_A"] = {key: round(b[key] - a[key], 4) for key in ("rating_accuracy", "citation_validity", "hallucinated_evidence_rate")}
    report["delta_B_minus_A"]["avg_model_latency_seconds_per_profile"] = round(b["avg_model_latency_seconds_per_profile"] - a["avg_model_latency_seconds_per_profile"], 4) if a["avg_model_latency_seconds_per_profile"] is not None and b["avg_model_latency_seconds_per_profile"] is not None else None
    report["limitations"] = ["Tiny synthetic set; differences are directional only.", "Model latency measures the API request only; end-to-end latency includes retrieval and configured provider pacing.", "Cost is calculated only when MODEL_INPUT_COST_PER_1M and MODEL_OUTPUT_COST_PER_1M are set to current provider rates."]
    path = ROOT / "data/model_comparison_results.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"A": report["runs"]["A"]["metrics"], "B": report["runs"]["B"]["metrics"], "delta_B_minus_A": report["delta_B_minus_A"]}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
