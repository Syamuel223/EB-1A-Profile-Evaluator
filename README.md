# EB-1A Profile Evaluator

Educational prototype for résumé-based EB-1A screening. It retrieves USCIS Policy Manual evidence, produces structured cited criterion assessments, and includes retrieval, model, and judge evaluations.

> Not legal advice, an eligibility decision, or a prediction of USCIS adjudication. USCIS separately reviews the totality of the petition. A résumé alone is not a full petition record.

## Setup

- Python 3.10+
- Groq API key for assessment, model comparison, and LLM judging (this experiment's active provider)
- Optional Gemini API key for semantic query embeddings; BM25 remains available without it
- Tavily API key for live opportunity research

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` in `.env`; `LLM_PROVIDER=groq` and `GROQ_MODEL=openai/gpt-oss-20b` are the defaults in this experiment. Add `GEMINI_API_KEY` if you want semantic vector retrieval, and `TAVILY_API_KEY` for live opportunity research. Groq generation calls are spaced by `GROQ_MIN_REQUEST_INTERVAL_SECONDS` to respect the configured tier. Never commit `.env`. The USCIS chapter PDF is included in `data/knowledge_base/`. Index it with:

The Groq assessment prompt sends one best policy passage per criterion (up to 800 characters) and uses concise JSON output to fit the on-demand token budget. Groq is not used for embeddings.

Check the active model provider and Tavily with minimal requests (the Tavily check may consume one search credit):

```powershell
python scripts/verify_credentials.py
# Check Groq only, without consuming a Tavily search credit:
python scripts/verify_credentials.py --skip-tavily
```

```powershell
python -m eb1a_evaluator.cli ingest --local
```

The indexer applies section-aware chunking, preserves PDF page numbers, uses 360-word target chunks with 60-word overlap, creates embeddings when an API key is configured, and writes a persistent BM25 inverted index plus a local Chroma vector collection. Chroma stores vectors, chunk text, and source/section/page metadata under `data/knowledge_base/chroma_db/`. Exact cosine search remains available if the local Chroma collection is missing or stale; an optional HNSW index is also supported with `python -m pip install -e ".[retrieval]"`. A lightweight BM25 path works without embeddings.

USCIS may block automated page requests. For updates, download the latest chapter from [USCIS Policy Manual, Volume 6, Part F, Chapter 2](https://www.uscis.gov/policy-manual/volume-6-part-f-chapter-2), place the PDF or text under `data/knowledge_base/`, and rerun local ingestion.

### Optional neural reranking

Default retrieval uses metadata-aware reranking after BM25/vector reciprocal-rank fusion. For a neural cross-encoder, install `python -m pip install -e ".[rerank]"` and set `RERANKER_MODEL` (for example, a compatible cross-encoder model name). The model may download on first use. Empty setting uses the deterministic lightweight reranker.

## Assessment interfaces

```powershell
streamlit run streamlit_app.py --server.address 127.0.0.1
python -m eb1a_evaluator.cli evaluate path\to\resume.pdf
python -m eb1a_evaluator.cli evaluate path\to\resume.txt
uvicorn eb1a_evaluator.api:app --reload
```

The Streamlit app at `http://localhost:8501` lets users upload PDF/TXT/Markdown profiles, view the overall and criterion-level assessment, and download JSON. The FastAPI API provides `POST /evaluate` for profile uploads and `POST /opportunities` for an assessment plus a non-identifying field description. The opportunity agent chooses up to two searches and returns leads only when their URLs match Tavily results; explicit dates must also appear in linked result text. Criterion/date filters can leave the result empty. It does not apply, contact anyone, or treat a lead as achieved evidence. The CLI also supports:

```powershell
python -m eb1a_evaluator.cli opportunities data\assessment.json --field "computational biology"
```

## Retrieval design

1. **Hybrid chunking:** detect USCIS criterion/section headings, keep sections separate, pack paragraph-sized units to a target size, and overlap adjacent chunks. Store source, heading, page, criterion tag, and chunk ID.
2. **BM25:** an inverted-index BM25 implementation provides lexical retrieval without an API key.
3. **Vector retrieval:** Gemini embeddings are stored with chunk records and indexed in persistent local Chroma. Set `GEMINI_API_KEY` to compute query vectors; without it, retrieval falls back to BM25. HNSW and exact cosine are available as vector-search fallbacks.
4. **Hybrid ranking:** BM25 and vector rankings are joined with reciprocal-rank fusion, then reranked with section/criterion-aware scoring or an optional cross-encoder.
5. **Citations:** every citation must refer to a chunk retrieved for that criterion, use the matching source name, and quote text present in that chunk. Unsupported citations and resume evidence are dropped and recorded as warnings.

## Evaluation scripts

### Local checks

Run the unit checks without model API calls:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q eb1a_evaluator scripts streamlit_app.py
python -m pip check
```

### Retrieval before/after

Run without model credentials:

```powershell
python scripts/eval_retrieval.py
```

It compares the former token-overlap baseline with current BM25 plus optional vector retrieval/reranking. Results go to `data/retrieval_eval_results.json`, with recall@5, MRR, and nDCG@5. Criterion-heading tags are weak relevance labels, so report these as diagnostic numbers and review samples manually. If chunks have no stored embeddings, the hybrid run falls back to BM25 and says so in the report.

### Rating, citation, and evidence metrics

```powershell
python scripts/run_eval.py
```

Runs the five synthetic labeled profiles and writes `data/eval_results.json`. Requires the configured provider key and an indexed corpus.
The evaluator makes one structured generation request per profile, with separate retrieved passages for each of the ten criteria. If the provider rejects a request, it writes diagnostic details and completed profiles to `data/eval_results.partial.json` without recording credentials. Rerunning the command automatically resumes completed profiles; use `--restart` only when you deliberately want to spend quota rerunning all five.

### Observed results and failure analysis

Latest five-profile Groq run (`data/eval_results.json`):

| Metric | Result |
| --- | ---: |
| Rating accuracy | 39/50 (78%) |
| Citation validity | 23/23 (100%) |
| Evidence grounded in the supplied profile | 24/24 (100%) |

The system missed 11 of 50 hand-labeled ratings. Errors were most frequent in C2 (membership), C5 (major contributions), and C8 (leading/critical role), with two mismatches each; C7 and C10 had no mismatches in this small set. These cases often turn on whether a profile gives enough detail about selectivity, significance, or organizational impact. The evaluator receives only one short retrieved passage per criterion to stay within the Groq on-demand token budget, which can omit nuance. Treat these errors as areas for review, not as a stable estimate of real-world performance.

Other recorded experiments:

- **Prompt comparison:** balanced scored 40/50 (80%); evidence-first scored 35/50 (70%). Both had 100% citation validity and 0% ungrounded evidence on this dataset. API-call latency averaged 2.35s and 2.83s respectively. The roughly 48–60s end-to-end times include deliberate request pacing and are not model-only latency.
- **LLM judge:** 39/50 label accuracy, matching the system's 39/50. The judge repeated the system's ratings on all 50 criteria, so this run did not provide independent confirmation.
- **Retrieval:** recall@5 stayed at 1.00; MRR improved from 0.5167 to 1.00 and nDCG@5 from 0.6393 to 1.00. These scores are optimistic because criterion IDs and section metadata are used in the weak labels and reranking. Review `data/retrieval_eval_results.json` for the caveat.

These measurements use five synthetic profiles and hand-written labels. Citation checks confirm that quotes occur in indexed chunks; they do not prove that a citation or evidence is legally sufficient.

### Design decisions and trade-offs

- Groq handles generation; Gemini embeddings are optional and separate. Without a Gemini key, retrieval degrades to BM25 rather than blocking the assessment.
- Chroma persists embeddings and source metadata locally, while BM25 stays a separate lexical index. The small corpus can fall back to exact cosine if Chroma is unavailable; HNSW remains an optional approximate-search path.
- The assessor sends one best policy passage per criterion and uses low GPT-OSS reasoning effort to fit Groq's token budget. This trades context breadth for reliable request sizes.
- Opportunity suggestions are treated as leads. The agent filters them unless their source URL came from search results and stated dates are supported by those results; it may return no lead.

### Improvements with more time

- Evaluate on a larger, independently reviewed profile set, with adjudicated labels and examples for ambiguous criteria.
- Measure the Chroma-backed retrieval against a larger set of human relevance labels and compare persistent index latency as the corpus grows.
- Retrieve multiple non-duplicate passages when the provider budget allows, and compare retrieval depth and evidence quality.
- Add model/provider fallback, calibrated confidence, privacy controls, and authenticated deployment before handling real client profiles.
- Track current provider pricing and rerun cost comparisons; the checked-in comparison has `estimated_total_cost_usd: null` because rates were not configured.

### LLM-as-judge

```powershell
python scripts/run_llm_judge.py
```

Requires `data/eval_results.json`. The judge rates outputs without seeing the expected labels; the script then compares judge ratings and system ratings with the manual labels. Results go to `data/llm_judge_results.json`. The judge is not ground truth.

### Model/prompt comparison

```powershell
python scripts/compare_models.py --limit 5
```

By default this compares the active model under `balanced` and `evidence_first` prompt strategies. It reports rating accuracy, citation validity, unsupported-evidence rate, API-only latency, end-to-end latency, token counts, and optional cost estimates to `data/model_comparison_results.json`. End-to-end time includes provider pacing. For cost estimates set current provider rates in `.env` using `MODEL_INPUT_COST_PER_1M` and `MODEL_OUTPUT_COST_PER_1M`. This can make many model calls; `--limit 1` is useful for a small smoke run.

## Guardrails

- Prompt-injection patterns in uploaded text are flagged; résumé contents are sent as JSON data under a system instruction to ignore embedded commands.
- Input is Unicode-normalized, bounded to 20,000 characters in the evaluator and 10 MB at the API, and checked for too little text, garbling, or poor readability.
- API errors guide users to OCR scanned PDFs or re-save malformed text. No OCR is bundled.
- Model output is schema-validated; unsupported citations and evidence are filtered. Missing criterion outputs receive explicit warnings.
- Opportunity research sends only the declared field and weak criterion names to web search, not the résumé or personal identifiers.

## Limitations

- The attached USCIS PDF notes that some example and consideration lists were omitted from its retrieved copy. The starter corpus is not exhaustive; refresh against current USCIS guidance before use.
- The evaluation set has five synthetic profiles and hand-written reference labels. Retrieval labels are inferred from section headings. These are small diagnostic datasets, not statistical guarantees.
- Cross-encoder quality depends on the selected model; its first use may require internet access and significant memory.
- HNSW only helps when embeddings exist and the corpus is large enough to justify approximate search.
- The demo API has no authentication, persistence, or production privacy controls. Avoid sensitive client profiles.
