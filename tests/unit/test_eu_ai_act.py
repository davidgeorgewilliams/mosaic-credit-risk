from datetime import UTC, datetime

from credit_risk.domain.schemas import Decision, Explanation, RiskScore, ScoreSource
from credit_risk.governance.eu_ai_act import (
    SYSTEM_CLASSIFICATION,
    AuditLog,
    requires_human_oversight,
)


def _risk_score(decision: Decision) -> RiskScore:
    return RiskScore(
        applicant_id="APP-1",
        model_name="credit_risk_classifier",
        model_version="1",
        probability_of_default=0.6 if decision == Decision.DENY else 0.1,
        decision=decision,
        score_source=ScoreSource.CHAMPION,
        endpoint_name="credit-risk-endpoint",
        scored_at=datetime.now(UTC),
    )


def _explanation(guardrail_passed: bool) -> Explanation:
    return Explanation(
        applicant_id="APP-1",
        narrative="Explanation text.",
        top_factors=[],
        cited_policy_clauses=[],
        guardrail_passed=guardrail_passed,
        human_review_required=False,
        prompt_name="credit_risk_explanation",
        prompt_version="1",
        input_tokens=10,
        output_tokens=10,
        generated_at=datetime.now(UTC),
    )


def test_system_is_classified_high_risk():
    assert SYSTEM_CLASSIFICATION.category == "high_risk"
    assert "Annex III" in SYSTEM_CLASSIFICATION.annex_reference


def test_oversight_required_on_non_approve_decision():
    assert requires_human_oversight(_risk_score(Decision.DENY), _explanation(True)) is True


def test_oversight_required_on_guardrail_failure():
    assert requires_human_oversight(_risk_score(Decision.APPROVE), _explanation(False)) is True


def test_oversight_not_required_on_clean_approve():
    assert requires_human_oversight(_risk_score(Decision.APPROVE), _explanation(True)) is False


def test_audit_log_records_decision():
    log = AuditLog()
    event = log.record_decision(_risk_score(Decision.DENY), _explanation(True))

    assert event.event_type == "automated_credit_decision"
    assert event.applicant_id == "APP-1"
    assert event.payload["risk_classification"] == "high_risk"
    assert log.events == [event]
