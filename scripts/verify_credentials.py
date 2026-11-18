"""Minimal credential smoke check. Prints status only, never credentials or responses."""
import argparse
import os
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from eb1a_evaluator.config import LLM_PROVIDER, provider_api_key, provider_model
from eb1a_evaluator.llm import generate_json


def main():
    parser = argparse.ArgumentParser(description="Check model and search credentials without printing secrets.")
    parser.add_argument("--skip-tavily", action="store_true", help="Skip search verification to avoid spending a Tavily credit.")
    args = parser.parse_args()
    failures = 0
    if not provider_api_key():
        key_name = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY"}.get(LLM_PROVIDER, "provider key")
        print(f"{LLM_PROVIDER}: MISSING_KEY ({key_name})")
        failures += 1
    else:
        try:
            result = generate_json("Reply only with a JSON object containing ok=true.", "{\"ok\": true}", model=provider_model())
            actual = result.model or provider_model()
            label = "primary" if actual == provider_model() else "fallback"
            print(f"{LLM_PROVIDER}: AUTHENTICATED ({label} model: {actual})")
        except Exception as exc:
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            detail = str(exc).replace(provider_api_key(), "[redacted]")[:300]
            detail = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[redacted]", detail)
            print(f"{LLM_PROVIDER}: FAILED ({type(exc).__name__}, status={status}; {detail})")
            failures += 1
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if args.skip_tavily:
        print("Tavily: SKIPPED")
    elif not tavily_key:
        print("Tavily: MISSING_KEY")
        failures += 1
    else:
        try:
            response = requests.post("https://api.tavily.com/search", json={"api_key": tavily_key, "query": "USCIS EB-1A policy manual", "search_depth": "basic", "max_results": 1, "include_answer": False}, timeout=20)
            response.raise_for_status()
            print("Tavily: AUTHENTICATED (one minimal search request)")
        except Exception as exc:
            print(f"Tavily: FAILED ({type(exc).__name__})")
            failures += 1
    if failures:
        raise SystemExit(1)
    print("Credential checks passed. No key values or provider response content were printed.")


if __name__ == "__main__":
    main()
