"""GDPR compliance helpers.

Data minimization is enforced structurally by `FeatureStore.read_table_minimized`
(quasi-identifiers never leave the feature store boundary) — this module covers
the two remaining GDPR concerns that are specific to an automated-decision system:
pseudonymization for anything that leaves the trust boundary (e.g. gets logged or
sent to a third-party LLM provider), and Article 15/22 subject-access responses —
"give me a copy of my data and an explanation of any automated decision about me."
"""

import hashlib
import hmac
from datetime import UTC, datetime, timedelta

from credit_risk.config import settings
from credit_risk.domain.schemas import Explanation, RiskScore

# Article 5(1)(e) storage limitation: how long an application record and its
# automated-decision trail are retained before they must be purged/anonymised.
RETENTION_PERIOD_DAYS = 395


def pseudonymize(applicant_id: str) -> str:
    """One-way, deterministic pseudonym for anything crossing a trust boundary
    (external LLM provider logs, third-party monitoring). Deterministic so the
    same applicant maps to the same pseudonym across calls without storing a
    reversible mapping outside the system of record.

    Keyed with `settings.pseudonymization_secret` rather than a bare hash:
    applicant IDs are sequential and low-entropy (`APP-000000`, `APP-000001`, ...),
    so an unkeyed hash would let anyone who obtains a pseudonym reverse it by
    hashing the entire ID space."""
    digest = hmac.new(
        settings.pseudonymization_secret.encode("utf-8"),
        applicant_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"PSN-{digest[:16]}"


def is_past_retention(collected_at: datetime, as_of: datetime | None = None) -> bool:
    as_of = as_of or datetime.now(UTC)
    return as_of - collected_at > timedelta(days=RETENTION_PERIOD_DAYS)


def subject_access_response(risk_score: RiskScore, explanation: Explanation) -> dict:
    """Article 15 (right of access) + Article 22 (right to an explanation of an
    automated decision) response bundle — everything a data subject is entitled
    to receive about a decision made about them."""
    return {
        "applicant_id": risk_score.applicant_id,
        "decision": risk_score.decision.value,
        "probability_of_default": risk_score.probability_of_default,
        "model_name": risk_score.model_name,
        "model_version": risk_score.model_version,
        "decided_at": risk_score.scored_at.isoformat(),
        "explanation": explanation.narrative,
        "contributing_factors": [
            {"feature": f.feature, "direction": f.direction} for f in explanation.top_factors
        ],
        "policy_basis": explanation.cited_policy_clauses,
        "right_to_human_review": True,
        "retention_period_days": RETENTION_PERIOD_DAYS,
    }
