"""Input validation and evidence/citation validation for untrusted uploads."""
import re
import unicodedata

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"system prompt",
    r"developer message",
    r"reveal (the )?(prompt|secret|api key)",
    r"act as (the )?(system|developer)",
    r"output only",
]


class ProfileInputError(ValueError):
    pass


def inspect_profile(text: str, max_chars: int = 50_000) -> tuple[str, list[str]]:
    if not isinstance(text, str):
        raise ProfileInputError("Profile must be text.")
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        raise ProfileInputError(f"Profile is too large ({len(text)} characters); maximum is {max_chars}.")
    if len(text) < 40:
        raise ProfileInputError("Profile text is too short to assess. Provide at least a short résumé summary.")
    if text.count("\ufffd") / max(1, len(text)) > .01:
        raise ProfileInputError("The document appears garbled (many replacement characters). Re-export or OCR it first.")
    visible = sum(ch.isalnum() for ch in text) / max(1, len(text))
    if visible < .12:
        raise ProfileInputError("The document does not contain enough readable text. Try a text-based PDF or OCR.")
    alerts = ["Possible prompt-injection text detected in the uploaded profile; treated as untrusted evidence."] if any(re.search(p, text, re.I) for p in INJECTION_PATTERNS) else []
    return text, alerts


def normalized(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold(), flags=re.UNICODE).strip()


def validate_output(profile: str, result: dict, retrieved_by_criterion: dict, chunks_by_id: dict) -> tuple[dict, list[str]]:
    warnings = []
    ids = {f"C{i}" for i in range(1, 11)}
    rows = result.get("criteria", [])
    seen = set()
    accepted = []
    for row in rows:
        cid = row.get("criterion_id")
        if cid not in ids or cid in seen:
            warnings.append(f"Dropped invalid or duplicate criterion result: {cid}.")
            continue
        seen.add(cid)
        allowed_chunks = {c.chunk_id for c in retrieved_by_criterion.get(cid, [])}
        checked_citations = []
        for citation in row.get("citations", []):
            chunk_id = citation.get("chunk_id", "")
            source = chunks_by_id.get(chunk_id)
            quote = normalized(citation.get("quote", ""))
            if chunk_id in allowed_chunks and source and citation.get("source") == source.source and quote and quote in normalized(source.text) and citation.get("page") in (None, source.page):
                checked_citations.append(citation)
            else:
                warnings.append(f"Dropped unsupported citation for {cid} ({chunk_id}).")
        row["citations"] = checked_citations
        evidence = []
        for item in row.get("evidence", []):
            item_norm = normalized(item)
            if item_norm and item_norm in normalized(profile):
                evidence.append(item)
            else:
                warnings.append(f"Dropped résumé evidence not found in the uploaded text for {cid}.")
        row["evidence"] = evidence
        accepted.append(row)
    missing = ids - seen
    for cid in sorted(missing):
        warnings.append(f"No valid result returned for {cid}.")
    result["criteria"] = accepted
    result.setdefault("limitations", [])
    result["limitations"] = list(dict.fromkeys(result["limitations"] + warnings))
    return result, warnings
