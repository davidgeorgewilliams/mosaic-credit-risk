from fastapi.testclient import TestClient

from credit_risk.serving.api import app

SAFE_APPLICANT = {
    "applicant_id": "APP-SAFE-001",
    "income_band": 80000,
    "debt_to_income_ratio": 0.05,
    "credit_history_years": 15.0,
    "num_delinquencies_last_2y": 0,
    "loan_amount_requested": 2000,
    "employment_status": "employed",
    "age_band": "35-44",
    "region": "UKC",
}

RISKY_APPLICANT = {
    "applicant_id": "APP-RISKY-001",
    "income_band": 15000,
    "debt_to_income_ratio": 0.95,
    "credit_history_years": 0.5,
    "num_delinquencies_last_2y": 5,
    "loan_amount_requested": 40000,
    "employment_status": "unemployed",
    "age_band": "18-24",
    "region": "UKG",
}


def test_health_reports_champion_and_shadow():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    aliases = {m["alias"] for m in response.json()["served_models"]}
    assert aliases == {"champion", "shadow"}


def test_safe_applicant_is_approved_without_review():
    with TestClient(app) as client:
        score = client.post("/score", json=SAFE_APPLICANT).json()
        explanation = client.post("/explain", json=SAFE_APPLICANT).json()

    assert score["decision"] == "approve"
    assert explanation["guardrail_passed"] is True
    assert explanation["human_review_required"] is False


def test_risky_applicant_is_denied_and_flagged_for_review():
    with TestClient(app) as client:
        score = client.post("/score", json=RISKY_APPLICANT).json()
        explanation = client.post("/explain", json=RISKY_APPLICANT).json()

    assert score["decision"] == "deny"
    assert explanation["human_review_required"] is True
    # quasi-identifiers must never surface in the explanation's factor list
    factor_names = " ".join(f["feature"] for f in explanation["top_factors"])
    assert "region" not in factor_names
    assert "age_band" not in factor_names


def test_explanation_cites_policy_and_records_audit_event():
    with TestClient(app) as client:
        client.post("/explain", json=RISKY_APPLICANT).json()
        audit = client.get("/governance/audit").json()
        classification = client.get("/governance/classification").json()

    assert len(audit) == 1
    assert audit[0]["event_type"] == "automated_credit_decision"
    assert audit[0]["payload"]["risk_classification"] == "high_risk"
    assert classification["category"] == "high_risk"


def test_ai_gateway_usage_accumulates_across_requests():
    with TestClient(app) as client:
        client.post("/explain", json=SAFE_APPLICANT)
        client.post("/explain", json=RISKY_APPLICANT)
        usage = client.get("/governance/ai_gateway_usage").json()

    assert usage["calls"] == 2
    assert usage["total_input_tokens"] > 0
