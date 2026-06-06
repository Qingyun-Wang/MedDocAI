"""
Pipeline State — agents/state.py

The shared state object that flows through the LangGraph pipeline.
Each agent reads from it and returns a partial update.

Uses a TypedDict (LangGraph-idiomatic) so nodes can return partial dict updates.
Helper Pydantic models (RouterDecision, ReviewResult) structure the LLM outputs.
"""

from __future__ import annotations

from typing import Optional, TypedDict

from pydantic import BaseModel, Field

from models.schemas import Evidence


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

INTENTS = [
    "medication_info",      # drug questions: what is X, side effects, warnings, dosing
    "drug_recall",          # recall / shortage / safety-alert checks
    "policy_eligibility",   # Medicaid eligibility, drug pricing, policy
    "condition_education",  # what is condition X, patient education
    "patient_summary",      # summarise this patient's situation (care manager)
    "general",              # fallback / out-of-scope
]

ROLES = ["anonymous", "patient", "care_manager"]


# ---------------------------------------------------------------------------
# Structured LLM outputs
# ---------------------------------------------------------------------------

class RouterDecision(BaseModel):
    """Structured output of the Router agent."""
    intent: str = Field(description="One of the INTENTS values")
    shaped_query: str = Field(
        description="The query reshaped for semantic retrieval, enriched with "
                    "patient conditions if context is present"
    )
    drug_name: Optional[str] = Field(
        None, description="Drug name extracted from the query, if any"
    )
    state_name: Optional[str] = Field(
        None, description="US state name extracted from the query, if any"
    )
    reasoning: str = Field(description="Brief explanation of the routing decision")


class ReviewResult(BaseModel):
    """Structured output of the Reviewer agent."""
    relevant: bool = Field(description="Does the answer address the user's query?")
    faithful: bool = Field(description="Is every claim grounded in the evidence?")
    passed: bool = Field(description="True only if both relevant AND faithful")
    route_back_to: Optional[str] = Field(
        None,
        description="'router' if evidence was insufficient, 'answer_generator' if "
                    "evidence was good but the answer misused it, None if passed",
    )
    feedback: str = Field(
        description="Actionable feedback for the next attempt (empty if passed)"
    )


# ---------------------------------------------------------------------------
# Pipeline state (flows through the graph)
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    """Shared state for the LangGraph pipeline.

    total=False so each node can return a partial update.
    """
    # --- Inputs ---
    query: str
    patient_context: Optional[dict]   # pre-computed summary dict (or None)
    user_role: str                    # one of ROLES

    # --- Router outputs ---
    intent: str
    shaped_query: str
    drug_name: Optional[str]
    state_name: Optional[str]
    tools_to_call: list[str]

    # --- Retrieval outputs ---
    raw_evidence: list[Evidence]

    # --- Evidence Filter outputs ---
    filtered_evidence: list[Evidence]

    # --- Answer Generator outputs ---
    answer: str
    citations: list[str]

    # --- Reviewer outputs ---
    review_passed: bool
    reviewer_feedback: Optional[str]
    route_back_to: Optional[str]      # "router" | "answer_generator" | None

    # --- Loop control ---
    iteration: int
    max_iterations: int

    # --- Safety outputs (final) ---
    final_answer: str
    disclaimers: list[str]

    # --- Diagnostics (for tracing/debugging) ---
    trace: list[str]                  # human-readable log of what each node did


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def new_state(
    query: str,
    patient_context: Optional[dict] = None,
    user_role: str = "anonymous",
    max_iterations: int = 2,
) -> PipelineState:
    """Create a fresh PipelineState with sensible defaults."""
    return PipelineState(
        query=query,
        patient_context=patient_context,
        user_role=user_role,
        intent="",
        shaped_query=query,
        drug_name=None,
        state_name=None,
        tools_to_call=[],
        raw_evidence=[],
        filtered_evidence=[],
        answer="",
        citations=[],
        review_passed=False,
        reviewer_feedback=None,
        route_back_to=None,
        iteration=0,
        max_iterations=max_iterations,
        final_answer="",
        disclaimers=[],
        trace=[],
    )
