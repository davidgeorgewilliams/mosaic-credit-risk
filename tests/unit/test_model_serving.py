from credit_risk.adapters.model_serving import InMemoryModelServing
from credit_risk.data.synthetic_data import FEATURE_COLUMNS, generate_dataset


def test_served_models_reports_champion_and_shadow():
    serving = InMemoryModelServing()
    aliases = {m["alias"] for m in serving.served_models}
    assert aliases == {"champion", "shadow"}


def test_score_records_real_latency():
    serving = InMemoryModelServing()
    assert serving.last_latency_ms is None

    df = generate_dataset(n=1, seed=1)
    row = df.iloc[[0]][["applicant_id", *FEATURE_COLUMNS]]
    serving.score(row)

    assert serving.last_latency_ms is not None
    assert serving.last_latency_ms >= 0


def test_explain_inputs_returns_top_shap_factors():
    serving = InMemoryModelServing()
    df = generate_dataset(n=1, seed=1)
    row = df.iloc[[0]][["applicant_id", *FEATURE_COLUMNS]]

    factors = serving.explain_inputs(row)

    assert 1 <= len(factors) <= 5
    assert {"feature", "shap_value", "direction"} <= factors[0].keys()
    # quasi-identifiers must never appear — the model isn't trained on them
    feature_names = {f["feature"] for f in factors}
    assert not any("region" in name or "age_band" in name for name in feature_names)
