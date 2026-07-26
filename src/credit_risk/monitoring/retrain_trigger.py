"""Retrain trigger policy.

When `DriftReport.overall_status` reaches CRITICAL, this is what would, in
production, call the Databricks Jobs API to kick off the training job defined in
`databricks/resources/training_job.yml`. Locally it just returns the trigger
decision and the job payload it would submit, so the demo can show the decision
being made without needing a live workspace to fire it at.
"""

from dataclasses import dataclass

from credit_risk.monitoring.lakehouse_monitoring import DriftReport, DriftStatus


@dataclass
class RetrainDecision:
    should_retrain: bool
    reason: str
    job_payload: dict | None


def evaluate(drift_report: DriftReport) -> RetrainDecision:
    if drift_report.overall_status != DriftStatus.CRITICAL:
        return RetrainDecision(
            should_retrain=False,
            reason=f"drift status '{drift_report.overall_status.value}' is below retrain threshold",
            job_payload=None,
        )

    worst = drift_report.max_psi_feature
    return RetrainDecision(
        should_retrain=True,
        reason=f"PSI for '{worst.feature}' is {worst.psi:.3f}, at or above the critical threshold",
        job_payload={
            "job_name": "credit-risk-retrain",
            "trigger": "drift_detected",
            "triggering_feature": worst.feature,
            "psi": worst.psi,
        },
    )
