from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class Rating(str, Enum):
    strong = "Strong"
    partial = "Partial"
    missing = "Missing"
    insufficient_info = "Insufficient Info"


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(description="Source title and USCIS Policy Manual section")
    chunk_id: str
    page: int | None = None
    quote: str = Field(description="Short supporting excerpt from the retrieved source chunk")


class CriterionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criterion_id: str
    criterion: str
    rating: Rating
    evidence: list[str] = Field(description="Exact resume excerpts or faithful references; [] if none")
    reasoning: str
    citations: list[Citation]
    next_steps: list[str] = Field(description="One or two practical actions")


class OverallAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strongest_criteria: list[str]
    main_gaps: list[str]
    readiness_score: int = Field(ge=0, le=100, description="Screening estimate, not a legal probability")
    summary: str


class ProfileAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criteria: list[CriterionAssessment]
    overall: OverallAssessment
    limitations: list[str] = Field(default_factory=list)
    run_metadata: dict = Field(default_factory=dict)


CRITERIA = [
    ("C1", "Lesser nationally or internationally recognized prizes or awards for excellence"),
    ("C2", "Membership in associations requiring outstanding achievements, judged by recognized experts"),
    ("C3", "Published material about the person in professional or major trade publications or other major media"),
    ("C4", "Participation as a judge of the work of others in the same or an allied field"),
    ("C5", "Original scientific, scholarly, artistic, athletic, or business-related contributions of major significance"),
    ("C6", "Authorship of scholarly articles in the field in professional or major trade publications or other major media"),
    ("C7", "Display of the person's work at artistic exhibitions or showcases"),
    ("C8", "Leading or critical role for organizations or establishments with distinguished reputations"),
    ("C9", "High salary or other significantly high remuneration relative to others in the field"),
    ("C10", "Commercial successes in the performing arts, shown by box office receipts or sales"),
]
