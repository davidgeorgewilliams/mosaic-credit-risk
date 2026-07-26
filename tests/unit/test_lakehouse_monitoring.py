from credit_risk.data.synthetic_data import generate_dataset
from credit_risk.monitoring import retrain_trigger
from credit_risk.monitoring.lakehouse_monitoring import DriftStatus, detect_drift


def test_stable_population_reports_ok():
    reference = generate_dataset(n=3000, seed=1, drift_shift=0.0)
    current = generate_dataset(n=1000, seed=2, drift_shift=0.0)

    report = detect_drift(reference, current)

    assert report.overall_status == DriftStatus.OK
    decision = retrain_trigger.evaluate(report)
    assert decision.should_retrain is False
    assert decision.job_payload is None


def test_drifted_population_triggers_retrain():
    reference = generate_dataset(n=3000, seed=1, drift_shift=0.0)
    current = generate_dataset(n=1000, seed=3, drift_shift=0.35)

    report = detect_drift(reference, current)

    assert report.overall_status == DriftStatus.CRITICAL
    decision = retrain_trigger.evaluate(report)
    assert decision.should_retrain is True
    assert decision.job_payload["job_name"] == "credit-risk-retrain"
