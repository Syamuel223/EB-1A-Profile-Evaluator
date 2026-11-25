"""Simple upload-and-assess interface for the EB-1A screening prototype."""
from __future__ import annotations

import json
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eb1a_evaluator.config import LLM_PROVIDER, provider_model
from eb1a_evaluator.guardrails import ProfileInputError
from eb1a_evaluator.opportunities import suggest_opportunities
from eb1a_evaluator.pipeline import assess_profile

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

st.set_page_config(
    page_title="EB-1A Profile Evaluator",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
      .hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg,#102a43,#1e4e67); color: #fff; margin-bottom: 1.2rem;}
      .hero h1 {margin: 0 0 .35rem 0; font-size: 2rem;}
      .hero p {margin: 0; color: #d8e8f0; font-size: 1rem;}
      .eyebrow {letter-spacing: .12em; text-transform: uppercase; color: #a6dcef; font-size: .72rem; font-weight: 700;}
      .notice {padding: .85rem 1rem; border-left: 4px solid #d39b27; background: #fff8e8; border-radius: 6px; color: #463814;}
      div[data-testid="stMetric"] {background: #f5f8fa; border: 1px solid #e3ebef; padding: .8rem 1rem; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<section class="hero"><div class="eyebrow">Evidence screening prototype</div>'
    '<h1>EB-1A Profile Evaluator</h1>'
    '<p>Upload a profile to review evidence across the ten extraordinary ability criteria.</p></section>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Assessment setup")
    st.caption(f"Text model: **{LLM_PROVIDER.upper()}** · `{provider_model()}`")
    st.caption("The profile is assessed in memory and is not saved to a project file.")
    st.divider()
    st.markdown(
        "**Supported files**\n\nPDF, TXT, or Markdown · up to 10 MB\n\n"
        "Scanned PDFs need OCR before upload."
    )

st.markdown(
    '<div class="notice"><strong>Important:</strong> This is an educational screening aid, '
    'not legal advice, an eligibility decision, or a prediction of USCIS adjudication.</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Upload a resume or profile",
    type=["pdf", "txt", "md"],
    help="Upload a text-based PDF or UTF-8 text/Markdown file. Maximum size: 10 MB.",
)


def extract_profile(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.casefold()
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("This file is larger than the 10 MB limit.")
    if suffix == ".pdf":
        try:
            pages = PdfReader(BytesIO(content)).pages
            text = "\n".join(page.extract_text() or "" for page in pages)
        except Exception as exc:
            raise ValueError("Could not read this PDF. Re-export it or run OCR, then upload a text-based PDF.") from exc
        if not text.strip():
            raise ValueError("No readable text was found. Scanned PDFs need OCR before upload.")
        return text
    if suffix in {".txt", ".md"}:
        try:
            return content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("This text file is not UTF-8. Re-save it as UTF-8 and try again.") from exc
    raise ValueError("Upload a PDF, TXT, or Markdown file.")


if uploaded is not None:
    st.caption(f"Selected: **{uploaded.name}** · {uploaded.size / 1024:.1f} KB")

if st.button("Assess profile", type="primary", disabled=uploaded is None, use_container_width=True):
    try:
        profile_text = extract_profile(uploaded.name, uploaded.getvalue())
        with st.spinner("Retrieving USCIS guidance and assessing the profile… this may take a minute on the current model tier."):
            result = assess_profile(profile_text).model_dump(mode="json")
        st.session_state["eb1a_last_assessment"] = result
        st.session_state["eb1a_last_filename"] = uploaded.name
        st.session_state.pop("eb1a_opportunities", None)
        st.success("Assessment complete.")
    except ProfileInputError as exc:
        st.error(str(exc))
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Assessment failed ({type(exc).__name__}). Check the configured model key, quota, and terminal logs, then retry.")

assessment = st.session_state.get("eb1a_last_assessment")
if assessment:
    overall = assessment.get("overall", {})
    criteria = assessment.get("criteria", [])
    st.divider()
    st.header("Assessment overview")
    st.caption(f"Profile: {st.session_state.get('eb1a_last_filename', 'Uploaded profile')}")

    ratings = Counter(item.get("rating", "Insufficient Info") for item in criteria)
    strongest = overall.get("strongest_criteria", [])
    gaps = overall.get("main_gaps", [])
    score_col, strong_col, partial_col, gap_col = st.columns(4)
    score_col.metric("Screening indicator", f"{overall.get('readiness_score', 0)}/100")
    strong_col.metric("Strong", ratings.get("Strong", 0))
    partial_col.metric("Partial", ratings.get("Partial", 0))
    gap_col.metric("Missing / unclear", ratings.get("Missing", 0) + ratings.get("Insufficient Info", 0))

    st.info(overall.get("summary", "Preliminary screening only."))
    left, right = st.columns(2)
    with left:
        st.markdown("#### Strongest criteria")
        st.write(", ".join(strongest) if strongest else "None identified")
    with right:
        st.markdown("#### Main gaps")
        st.write(", ".join(gaps) if gaps else "None identified")

    st.header("Opportunity research (optional)")
    st.caption(
        "Search for possible awards, selective memberships, or judging opportunities related to weaker criteria. "
        "This sends the field and criterion names to the search service, not the uploaded profile. "
        "Leads are unverified and do not count as achieved evidence."
    )
    field = st.text_input(
        "Field of expertise",
        placeholder="For example: software engineering",
        key="eb1a_opportunity_field",
    )
    if st.button("Find opportunities", disabled=not field.strip()):
        try:
            with st.spinner("Searching for current opportunity leads…"):
                opportunities = suggest_opportunities(assessment, field.strip())
            st.session_state["eb1a_opportunities"] = {
                "field": field.strip(),
                "result": opportunities,
            }
        except Exception as exc:
            st.error(f"Opportunity search failed ({type(exc).__name__}). Check the Groq and Tavily keys, then retry.")

    saved_opportunities = st.session_state.get("eb1a_opportunities")
    if saved_opportunities and saved_opportunities.get("field") == field.strip():
        opportunity_result = saved_opportunities.get("result", {})
        leads = opportunity_result.get("opportunities", [])
        if leads:
            for lead in leads:
                criterion_id = lead.get("criterion_id", "")
                st.markdown(f"**{lead.get('title', 'Opportunity')}** · {criterion_id}")
                st.write(lead.get("why_relevant", ""))
                if lead.get("deadline_or_status"):
                    st.caption(lead["deadline_or_status"])
                st.write(f"**Possible next step:** {lead.get('next_step', 'Verify details on the source page.')}")
                if lead.get("source_url", "").startswith(("https://", "http://")):
                    st.link_button("Open source", lead["source_url"])
        else:
            st.info(
                opportunity_result.get("caveats")
                or opportunity_result.get("note")
                or "No verified leads were returned. Try again later or use a more specific field."
            )
        if opportunity_result.get("note") and leads:
            st.caption(opportunity_result["note"])

    st.header("Criterion-by-criterion assessment")
    for item in criteria:
        label = f"{item.get('criterion_id', '')} · {item.get('criterion', '')} — {item.get('rating', 'Insufficient Info')}"
        with st.expander(label):
            st.markdown(f"**Rating:** {item.get('rating', 'Insufficient Info')}")
            st.markdown("**Reasoning**")
            st.write(item.get("reasoning", ""))
            evidence = item.get("evidence", [])
            st.markdown("**Profile evidence**")
            if evidence:
                for excerpt in evidence:
                    st.write(f"• {excerpt}")
            else:
                st.caption("No direct profile excerpt was identified.")
            citations = item.get("citations", [])
            st.markdown("**USCIS source citations**")
            if citations:
                for citation in citations:
                    page = citation.get("page")
                    page_label = f" · page {page}" if page is not None else ""
                    st.caption(f"{citation.get('source', 'USCIS source')}{page_label} · {citation.get('chunk_id', '')}")
                    st.write(f"> {citation.get('quote', '')}")
            else:
                st.caption("No source citation was returned for this criterion.")
            next_steps = item.get("next_steps", [])
            if next_steps:
                st.markdown("**Possible evidence to gather**")
                for step in next_steps:
                    st.write(f"• {step}")

    with st.expander("Limitations and run details"):
        for limitation in assessment.get("limitations", []):
            st.write(f"• {limitation}")
        metadata = assessment.get("run_metadata", {})
        if metadata:
            st.json(metadata)

    st.download_button(
        "Download assessment JSON",
        data=json.dumps(assessment, ensure_ascii=False, indent=2),
        file_name="eb1a_assessment.json",
        mime="application/json",
    )
