"""EU AI Act compliance surface.

Credit scoring / creditworthiness assessment is explicitly listed as a
**high-risk** AI system use case under Annex III(5)(b) of the EU AI Act. That
classification carries concrete obligations this module makes operational
rather than leaving as a policy document nobody reads:

- Article 12 (record-keeping): every decision and every explanation must be
  logged with enough detail to reconstruct what happened — `AuditLog`.
- Article 13 (transparency): the person affected must be told the decision was
  automated and given a meaningful explanation — enforced upstream by
  `credit_risk.agent.guardrails` (mandatory human-review disclosure) and
  `credit_risk.governance.gdpr.subject_access_response`.
- Article 14 (human oversight): a human must be able to review and override any
  decision that isn't a clear-cut approval — `requires_human_oversight`.
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from credit_risk.domain.schemas import AuditEvent, Decision, Explanation, RiskScore


@dataclass(frozen=True)
class RiskClassification:
    category: str
    annex_reference: str
    obligations: list[str]


SYSTEM_CLASSIFICATION = RiskClassification(
    category="high_risk",
    annex_reference="EU AI Act Annex III(5)(b) — creditworthiness assessment",
    obligations=[
        "record_keeping (Article 12)",
        "transparency_to_affected_persons (Article 13)",
        "human_oversight (Article 14)",
        "accuracy_robustness_monitoring (Article 15)",
    ],
)


def human_oversight_required(decision: Decision, guardrail_passed: bool) -> bool:
    """Article 14 human-oversight predicate — the single source of truth for
    whether a decision needs human review. Takes primitives rather than a
    `RiskScore`/`Explanation` pair so `ExplainabilityAgent.explain()` can call it
    while still building the `Explanation` (whose `human_review_required` field
    is derived from this same rule), not just after the fact from `AuditLog`."""
    return decision != Decision.APPROVE or not guardrail_passed


def requires_human_oversight(risk_score: RiskScore, explanation: Explanation) -> bool:
    return human_oversight_required(risk_score.decision, explanation.guardrail_passed)


@dataclass
class AuditLog:
    """In-memory stand-in for the Unity Catalog audit/log table a real deployment
    would write to (e.g. via Lakehouse Monitoring's inference tables or a
    dedicated Delta table). Same shape, same append-only semantics."""

    events: list[AuditEvent] = field(default_factory=list)

    def record(
        self, event_type: str, actor: str, payload: dict, applicant_id: str | None = None
    ) -> AuditEvent:
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            applicant_id=applicant_id,
            actor=actor,
            payload=payload,
            occurred_at=datetime.now(UTC),
        )
        self.events.append(event)
        return event

    def record_decision(self, risk_score: RiskScore, explanation: Explanation) -> AuditEvent:
        oversight_needed = requires_human_oversight(risk_score, explanation)
        return self.record(
            event_type="automated_credit_decision",
            actor="system",
            applicant_id=risk_score.applicant_id,
            payload={
                "decision": risk_score.decision.value,
                "probability_of_default": risk_score.probability_of_default,
                "model_version": risk_score.model_version,
                "score_source": risk_score.score_source.value,
                "guardrail_passed": explanation.guardrail_passed,
                "human_oversight_required": oversight_needed,
                "prompt_version": explanation.prompt_version,
                "risk_classification": SYSTEM_CLASSIFICATION.category,
            },
        )

    def record_human_override(
        self, applicant_id: str, reviewer: str, final_decision: Decision, rationale: str
    ) -> AuditEvent:
        return self.record(
            event_type="human_oversight_override",
            actor=f"human:{reviewer}",
            applicant_id=applicant_id,
            payload={"final_decision": final_decision.value, "rationale": rationale},
        )
