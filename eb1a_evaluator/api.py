from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from pypdf import PdfReader
import io
from .pipeline import assess_profile
from .guardrails import ProfileInputError
from .opportunities import suggest_opportunities
from .schemas import ProfileAssessment

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
app = FastAPI(title="EB-1A Profile Evaluator", version="0.2.0", description="Educational USCIS-cited screening prototype")


class OpportunityRequest(BaseModel):
    assessment: ProfileAssessment
    field: str = Field(min_length=2, max_length=300)


@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    try:
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Upload is larger than the 10 MB limit.")
        if name.endswith(".pdf"):
            try:
                profile = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
            except Exception as exc:
                raise HTTPException(422, "Could not read this PDF. Re-export it or OCR it, then upload a text-based PDF.") from exc
        elif name.endswith((".txt", ".md")):
            try:
                profile = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HTTPException(422, "Text file is not valid UTF-8. Re-save it as UTF-8.") from exc
        else:
            raise HTTPException(400, "Upload a PDF, TXT, or Markdown profile.")
        if not profile.strip():
            raise HTTPException(422, "No readable text found. Scanned PDFs require OCR before upload.")
        return assess_profile(profile).model_dump(mode="json")
    except ProfileInputError as exc:
        raise HTTPException(422, str(exc)) from exc
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Assessment failed: {type(exc).__name__}") from exc


@app.post("/opportunities")
def opportunities(request: OpportunityRequest):
    try:
        return suggest_opportunities(request.assessment.model_dump(mode="json"), request.field)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Opportunity search failed: {type(exc).__name__}") from exc
