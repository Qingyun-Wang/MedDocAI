"""
API request/response models — api/schemas.py

The HTTP contract, kept separate from the pipeline's internal models
(models/schemas.py). Two reasons for the split:

  - The internal `PipelineState` is a working scratchpad — it carries retry
    counters, routing flags, raw evidence, reviewer feedback. Publishing that
    shape would leak implementation detail into a public contract and freeze it.
  - A response model lets FastAPI generate accurate OpenAPI docs, so the service
    is self-describing at /docs.

Field descriptions here are the API documentation — they end up in the schema.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

USER_ROLES = Literal["anonymous", "patient", "care_manager"]


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    """One prior conversation turn, for follow-up resolution."""
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=2000,
        description="The user's question about a medication, condition, or Medicaid policy.",
        examples=["What are the main warnings for warfarin?"],
    )
    patient_id: Optional[str] = Field(
        None,
        description="Optional patient UUID (from GET /patients). When supplied, the "
                    "answer is personalised to that patient's conditions and medications.",
    )
    user_role: USER_ROLES = Field(
        "anonymous",
        description="Shapes the answer's tone and which evidence is injected.",
    )
    conversation_history: list[Turn] = Field(
        default_factory=list,
        description="Recent prior turns, oldest first, so follow-ups like "
                    "'what about its side effects?' resolve correctly.",
    )
    max_iterations: int = Field(
        2, ge=0, le=4,
        description="Corrective-RAG retry budget. 0 disables self-correction.",
    )


class FeedbackRequest(BaseModel):
    query_id: str = Field(..., description="The query_id returned by POST /query.")
    rating: Literal["up", "down"]
    comment: str = Field(
        "", max_length=2000,
        description="Optional note on a thumbs-down — the highest-value signal, "
                    "since it is what turns a bad answer into an eval case.",
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    source: str = Field(description="fda_label | medlineplus | openfda | sqlite | patient_record")
    title: str
    text: str
    score: Optional[float] = Field(
        None,
        description="Cross-encoder relevance (0-1). Null for pinned items such as "
                    "the patient record, which bypass reranking.",
    )
    citation: str


class QueryMetrics(BaseModel):
    llm_calls: int = 0
    total_tokens: int = 0
    cost_usd: Optional[float] = Field(
        None, description="Null when a model in the run has no published price — "
                          "never a fake 0.00.",
    )
    total_latency_ms: float = Field(
        0.0,
        description="Wall-clock ms for the whole query. A float: the pipeline records "
                    "0.1 ms precision (round(..., 1)), so declaring this int would "
                    "reject every real measurement.",
    )


class QueryResponse(BaseModel):
    query_id: str = Field(description="Use this to submit feedback on this answer.")
    answer: str = Field(description="The answer WITHOUT safety disclaimers.")
    final_answer: str = Field(description="Answer + sources + safety disclaimers, as shown to users.")
    citations: list[str]
    evidence: list[EvidenceItem]
    disclaimers: list[str]
    intent: str = Field(description="Routing decision, e.g. medication_info.")
    iterations: int = Field(description="Corrective-RAG passes used (0 = no retry needed).")
    review_passed: bool = Field(
        description="False means the Reviewer could not fully verify the answer; a "
                    "transparency note is included in final_answer.",
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Non-empty when fan-out retrieval fired for a diffuse multi-drug question.",
    )
    trace: list[str] = Field(description="Per-agent execution log.")
    metrics: QueryMetrics


class PatientSummary(BaseModel):
    """Roster entry. Deliberately thin — it mirrors MedDocDB.list_patients(),
    which selects only these columns so the dropdown does not deserialise every
    patient's full clinical JSON. Counts live on PatientDetail."""
    patient_id: str
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    has_summary: bool = Field(
        False, description="Whether a pre-computed care summary exists for this patient.",
    )


class PatientDetail(BaseModel):
    patient_id: str
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    n_conditions: int = 0
    n_active_medications: int = 0
    n_abnormal_labs: int = 0
    conditions: list[dict] = Field(default_factory=list)
    medications: list[dict] = Field(default_factory=list, description="ACTIVE medications only.")
    abnormal_labs: list[dict] = Field(default_factory=list, description="Abnormal labs only.")
    summary_md: Optional[str] = Field(
        None, description="Pre-computed care summary (built offline by the batch pipeline).",
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict = Field(description="Per-dependency status (database, vector store, keys).")


class FeedbackResponse(BaseModel):
    recorded: bool
    query_id: str
    rating: str


class ErrorResponse(BaseModel):
    detail: str
