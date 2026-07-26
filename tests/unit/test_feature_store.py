from credit_risk.data.feature_store import InMemoryFeatureStore
from credit_risk.data.synthetic_data import generate_dataset


def test_write_and_read_round_trip():
    store = InMemoryFeatureStore()
    df = generate_dataset(n=5, seed=1)
    store.write_table("applicants", df, primary_key="applicant_id")

    result = store.read_table("applicants")
    assert len(result) == 5
    assert set(result.columns) == set(df.columns)


def test_read_by_applicant_ids_filters():
    store = InMemoryFeatureStore()
    df = generate_dataset(n=10, seed=1)
    store.write_table("applicants", df, primary_key="applicant_id")

    subset = store.read_table("applicants", applicant_ids=["APP-000000", "APP-000001"])
    assert sorted(subset["applicant_id"]) == ["APP-000000", "APP-000001"]


def test_minimized_read_drops_quasi_identifiers():
    store = InMemoryFeatureStore()
    df = generate_dataset(n=5, seed=1)
    store.write_table("applicants", df, primary_key="applicant_id")

    minimized = store.read_table_minimized("applicants")
    assert "age_band" not in minimized.columns
    assert "region" not in minimized.columns
    assert "applicant_id" in minimized.columns
