"""Core domain objects shared across every layer.

Applicant fields are deliberately coarse (age *band* rather than birthdate, region
rather than address) — data minimization is a GDPR requirement, not an afterthought,
see `credit_risk.governance.gdpr`.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class EmploymentStatus(StrEnum):
    EMPLOYED = "employed"
    SELF_EMPLOYED = "self_employed"
    UNEMPLOYED = "unemployed"
    RETIRED = "retired"


class Applicant(BaseModel):
    applicant_id: str
    income_band: int = Field(description="Annual income bucketed to nearest 5k, GBP")
    debt_to_income_ratio: float
    credit_history_years: float
    num_delinquencies_last_2y: int
    loan_amount_requested: int
    employment_status: EmploymentStatus
    age_band: str = Field(description="e.g. '25-34' — never raw date of birth")
    region: str = Field(description="NUTS-1 region, never full address")


class Decision(StrEnum):
    APPROVE = "approve"
    DENY = "deny"
    REFER_TO_HUMAN = "refer_to_human"


class ScoreSource(StrEnum):
    CHAMPION = "champion"
    SHADOW = "shadow"


class RiskScore(BaseModel):
    applicant_id: str
    model_name: str
    model_version: str
    probability_of_default: float
    decision: Decision
    score_source: ScoreSource
    endpoint_name: str
    scored_at: datetime


class FactorContribution(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases_risk" | "decreases_risk"


class Explanation(BaseModel):
    applicant_id: str
    narrative: str
    top_factors: list[FactorContribution]
    cited_policy_clauses: list[str]
    guardrail_passed: bool
    human_review_required: bool
    prompt_name: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    generated_at: datetime


class AuditEvent(BaseModel):
    event_id: str
    event_type: str
    applicant_id: str | None = None
    actor: str
    payload: dict
    occurred_at: datetime
