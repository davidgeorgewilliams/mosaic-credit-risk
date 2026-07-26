"""One-command, narrated walkthrough of the whole system.

Trains + registers champion/shadow -> scores a batch of applicants through the
shadow-mirrored serving layer -> runs the full explainability agent pipeline on
one applicant -> simulates population drift and shows the retrain trigger fire
-> prints GDPR/EU AI Act governance artifacts. Meant to be screen-share-able:
`python scripts/demo.py` (or `make demo`).
"""

import pandas as pd

from credit_risk.adapters.ai_gateway import InMemoryAIGateway
from credit_risk.adapters.ai_search import InMemoryVectorSearch
from credit_risk.adapters.llm_client import FakeLLMClient
from credit_risk.adapters.model_serving import InMemoryModelServing
from credit_risk.agent.explainability_agent import ExplainabilityAgent
from credit_risk.data.synthetic_data import FEATURE_COLUMNS, generate_dataset
from credit_risk.governance import gdpr
from credit_risk.governance.eu_ai_act import SYSTEM_CLASSIFICATION, AuditLog
from credit_risk.models import train as train_module
from credit_risk.monitoring import retrain_trigger
from credit_risk.monitoring.lakehouse_monitoring import detect_drift


def _header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    _header("1. Train + register champion/shadow (local MLflow tracking + registry)")
    train_module.main()

    _header("2. Score a batch of applicants (shadow-mirrored serving)")
    serving = InMemoryModelServing()
    print("Served models:", serving.served_models)
    batch = generate_dataset(n=200, seed=7)
    for i in range(len(batch)):
        row = batch.iloc[[i]][["applicant_id", *FEATURE_COLUMNS]]
        serving.score(row)
    comparison = serving.shadow_comparison()
    print(
        f"Shadow comparison over {comparison.n_mirrored} mirrored requests: "
        f"mean |Δp|={comparison.mean_abs_probability_diff:.4f}, "
        f"decision agreement={comparison.decision_agreement_rate:.1%}"
    )
    print(f"Last inference latency: {serving.last_latency_ms:.3f} ms (real, not simulated)")

    _header("3. Explain one decision end-to-end (SHAP -> AI Search -> LLM -> guardrails)")
    ai_gateway = InMemoryAIGateway()
    agent = ExplainabilityAgent(serving, InMemoryVectorSearch(), ai_gateway, FakeLLMClient())
    audit_log = AuditLog()

    applicant_row = pd.DataFrame(
        [
            {
                "applicant_id": "APP-DEMO-001",
                "income_band": 18000,
                "debt_to_income_ratio": 0.72,
                "credit_history_years": 0.8,
                "num_delinquencies_last_2y": 3,
                "loan_amount_requested": 22000,
                "employment_status": "unemployed",
                "age_band": "18-24",
                "region": "UKG",
            }
        ]
    )
    risk_score = serving.score(applicant_row)
    explanation = agent.explain(applicant_row, risk_score)
    audit_log.record_decision(risk_score, explanation)

    print(f"Decision: {risk_score.decision.value} (p={risk_score.probability_of_default:.1%})")
    print(f"Explanation: {explanation.narrative}")
    print(f"Guardrail passed: {explanation.guardrail_passed}")
    print(f"Human review required: {explanation.human_review_required}")
    print(f"Cited policy clauses: {explanation.cited_policy_clauses}")

    _header("4. Simulate population drift and evaluate the retrain trigger")
    reference = generate_dataset(n=3000, seed=1, drift_shift=0.0)
    drifted_current = generate_dataset(n=1000, seed=99, drift_shift=0.3)
    drift_report = detect_drift(reference, drifted_current)
    for fd in drift_report.feature_drifts:
        print(f"  {fd.feature:<28} PSI={fd.psi:6.3f}  status={fd.status.value}")
    decision = retrain_trigger.evaluate(drift_report)
    print(
        f"Overall status: {drift_report.overall_status.value} -> "
        f"should_retrain={decision.should_retrain}"
    )
    if decision.job_payload:
        print(f"Would submit retrain job: {decision.job_payload}")

    _header("5. Governance artifacts")
    print("EU AI Act classification:", SYSTEM_CLASSIFICATION)
    print("AI Gateway usage:", ai_gateway.usage_report())
    print("GDPR subject access response:")
    for key, value in gdpr.subject_access_response(risk_score, explanation).items():
        print(f"  {key}: {value}")

    _header("Done")
    print(f"{len(audit_log.events)} audit event(s) recorded this run.")


if __name__ == "__main__":
    main()
