"""Opt-in tool-using opportunity research agent backed by Tavily web search."""
import json
import os
import re
import requests
from datetime import date, datetime, timezone
from urllib.parse import urlparse
from .config import LLM_PROVIDER, provider_api_key, provider_model
from .llm import generate_json
from .pipeline import SYSTEM
from .schemas import CRITERIA


def _web_search(query: str) -> dict:
    key = os.getenv("TAVILY_API_KEY", "")
    if not key:
        raise RuntimeError("Set TAVILY_API_KEY in .env to enable live opportunity research.")
    response = requests.post("https://api.tavily.com/search", json={"api_key": key, "query": query, "search_depth": "basic", "max_results": 3, "include_answer": False}, timeout=25)
    response.raise_for_status()
    data = response.json()
    return {"query": query, "results": [{"title": r.get("title", "")[:200], "url": r.get("url", ""), "published_date": r.get("published_date"), "content": r.get("content", "")[:700]} for r in data.get("results", [])]}


def _category_matches(opportunity: dict) -> bool:
    """Reject obvious mismatches between an opportunity type and EB-1A criterion."""
    criterion_id = str(opportunity.get("criterion_id", "")).upper()
    text = " ".join(str(opportunity.get(key, "")) for key in ("title", "why_relevant", "next_step")).casefold()
    if criterion_id == "C1":
        return bool(re.search(r"\b(award|prize|medal)\b", text))
    if criterion_id == "C2":
        return bool(re.search(r"\b(membership|member|fellowship|fellow|elected|association)\b", text)) and not bool(re.search(r"\b(award|prize|medal)\b", text))
    if criterion_id == "C4":
        return bool(re.search(r"\b(judge|judging|reviewer|reviewing|program committee|editorial board)\b", text))
    return True


def _contains_past_date(opportunity: dict, today: date) -> bool:
    text = " ".join(str(opportunity.get(key, "")) for key in ("title", "why_relevant", "next_step", "deadline_or_status"))
    for year_text in re.findall(r"\b20\d{2}\b", text):
        if int(year_text) < today.year:
            return True
    month_names = "January February March April May June July August September October November December"
    month_pattern = "|".join(name[:3] + r"(?:[a-z]+)?" for name in month_names.split())
    for match in re.finditer(rf"\b({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(20\d{{2}})\b", text, re.IGNORECASE):
        try:
            if date(int(match.group(3)), datetime.strptime(match.group(1)[:3].title(), "%b").month, int(match.group(2))) < today:
                return True
        except ValueError:
            continue
    for match in re.finditer(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_pattern})[,]?\s+(20\d{{2}})\b", text, re.IGNORECASE):
        try:
            if date(int(match.group(3)), datetime.strptime(match.group(2)[:3].title(), "%b").month, int(match.group(1))) < today:
                return True
        except ValueError:
            continue
    for match in re.finditer(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text):
        try:
            if date(int(match.group(1)), int(match.group(2)), int(match.group(3))) < today:
                return True
        except ValueError:
            continue
    return False


def _normalized_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    return f"{parsed.scheme.casefold()}://{parsed.netloc.casefold()}{parsed.path.rstrip('/')}" + (f"?{parsed.query}" if parsed.query else "")


def _explicit_dates(text: str) -> set[date]:
    dates = set()
    month_names = "January February March April May June July August September October November December"
    month_pattern = "|".join(name[:3] + r"(?:[a-z]+)?" for name in month_names.split())
    patterns = [
        (rf"\b({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(20\d{{2}})\b", (3, 1, 2)),
        (rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_pattern})[,]?\s+(20\d{{2}})\b", (3, 2, 1)),
        (r"\b(20\d{2})-(\d{2})-(\d{2})\b", (1, 2, 3)),
    ]
    for pattern, order in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                values = [match.group(index) for index in order]
                year = int(values[0])
                if pattern.startswith(r"\b(20"):
                    month, day_number = int(values[1]), int(values[2])
                else:
                    month_value, day_value = values[1], values[2]
                    month = int(month_value) if month_value.isdigit() else datetime.strptime(month_value[:3].title(), "%b").month
                    day_number = int(day_value)
                dates.add(date(year, month, day_number))
            except (ValueError, IndexError):
                continue
    return dates


def _dates_supported_by_result(opportunity: dict, result: dict) -> bool:
    claimed = _explicit_dates(" ".join(str(opportunity.get(key, "")) for key in ("title", "why_relevant", "next_step", "deadline_or_status")))
    if not claimed:
        return True
    source_text = " ".join(str(result.get(key, "")) for key in ("title", "published_date", "content"))
    source_dates = _explicit_dates(source_text)
    return claimed.issubset(source_dates)


def suggest_opportunities(assessment: dict, field: str, max_searches: int = 2) -> dict:
    """Plan live web searches, call Tavily as a tool, then synthesize linked leads.

    Only the declared field and criterion labels are sent to search; no resume text
    or personal identifiers are included in web search requests.
    """
    if not provider_api_key():
        raise RuntimeError(f"Set {LLM_PROVIDER.upper()} provider API key in .env for the opportunity agent.")
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Set TAVILY_API_KEY in .env to enable live opportunity research.")
    criterion_names = dict(CRITERIA)
    rows = assessment.get("criteria", []) if isinstance(assessment, dict) else assessment.model_dump(mode="json").get("criteria", [])
    weak = [row for row in rows if row.get("rating") != "Strong"]
    if not weak:
        return {"opportunities": [], "sources": [], "note": "No non-Strong criteria were identified for targeted opportunity research."}
    weak_summary = [{"criterion_id": row["criterion_id"], "criterion": criterion_names.get(row["criterion_id"], row.get("criterion", "")), "rating": row.get("rating")} for row in weak]
    today = datetime.now(timezone.utc).date()
    system = "You are an opportunity research planner. Find only currently open or future opportunities as of the supplied date. Exclude events, award cycles, or deadlines already in the past; do not recycle old search results. Prefer official organization pages. C1 means prizes/awards; C2 means selective membership or fellowship, never an award; C4 means judging, peer review, or committee service. Do not claim the candidate qualifies. Return compact JSON only."
    search_limit = min(max(1, max_searches), 2)
    plan_prompt = {"as_of_date": today.isoformat(), "field": field[:200], "weak_criteria": weak_summary, "task": f"Return JSON {{queries:[string]}} with at most {search_limit} specific searches for currently open or upcoming opportunities only. Search current cycles or future dates; avoid closed 2024/2025/2026 calls. Keep C1 awards, C2 selective membership, and C4 judging opportunities distinct. Include official organizations when known. No personal data."}
    plan = generate_json(system, json.dumps(plan_prompt, ensure_ascii=False), model=provider_model(), max_output_tokens=800)
    queries = json.loads(plan.text).get("queries", [])[:search_limit]
    search_results = [_web_search(str(query)[:300]) for query in queries if str(query).strip()]
    if not search_results:
        return {"opportunities": [], "sources": [], "note": "No useful search queries were produced. Try a more specific field description."}
    synthesis_prompt = {
        "field": field[:200],
        "weak_criteria": weak_summary,
        "search_results": search_results,
        "as_of_date": today.isoformat(),
        "task": "Return compact JSON with at most four opportunities (title, criterion_id, source_url copied exactly from one supplied result, why_relevant under 35 words, next_step under 25 words, deadline_or_status). Include only open or future opportunities as of as_of_date; exclude any event or deadline already past, and do not repeat historical award winners or closed calls. Never invent a deadline: any exact date must appear verbatim in the linked result title, published_date, or content. If current status/deadline is not stated by the results, say 'status not stated; verify on official page' and prefer omitting it. C1 is for awards/prizes; C2 is for selective membership/fellowship (never label an award as C2); C4 is for judging/reviewing. Only use weak criterion IDs supplied. Every suggestion and source must be grounded in these results. Give practical leads, not claims of eligibility or evidence already achieved.",
    }
    final = generate_json("You synthesize web research into cautious leads. Never claim a search result says more than its text. Return compact JSON only.", json.dumps(synthesis_prompt, ensure_ascii=False), model=provider_model(), max_output_tokens=1800)
    output = json.loads(final.text)
    result_by_url = {_normalized_url(result.get("url", "")): result for batch in search_results for result in batch["results"] if _normalized_url(result.get("url", ""))}
    weak_ids = {row["criterion_id"] for row in weak_summary}
    candidates = output.get("opportunities", []) if isinstance(output.get("opportunities", []), list) else []
    retained, source_rows = [], []
    for item in candidates:
        if not isinstance(item, dict) or item.get("criterion_id") not in weak_ids or not _category_matches(item) or _contains_past_date(item, today):
            continue
        source_url = _normalized_url(str(item.get("source_url", "")))
        source_result = result_by_url.get(source_url)
        if source_result is None or not _dates_supported_by_result(item, source_result):
            continue
        item["source_url"] = source_result["url"]
        retained.append(item)
        source_rows.append({"title": source_result.get("title", "Search result"), "url": source_result["url"], "criterion_id": item["criterion_id"]})
    output["opportunities"] = retained
    output["sources"] = source_rows
    if candidates and not output["opportunities"]:
        output["caveats"] = "No suggestions passed current-date and criterion-type checks. Try again later or verify leads directly on official pages."
    output["filtered_suggestion_count"] = len(candidates) - len(output["opportunities"])
    output["weak_criteria_researched"] = weak_summary
    output["note"] = "Opportunities are unverified leads. Confirm deadlines, eligibility, and requirements on official pages. A suggestion does not count as achieved EB-1A evidence."
    output["run_metadata"] = {"provider": LLM_PROVIDER, "model": provider_model(), "searches": len(search_results)}
    return output
