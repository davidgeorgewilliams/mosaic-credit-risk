"""Synthetic applicant data generator.

No real applicant data ever touches this repo — data minimization / GDPR by
design applies during development and demoing, not just in production.
"""

import numpy as np
import pandas as pd

EMPLOYMENT_STATUSES = ["employed", "self_employed", "unemployed", "retired"]
EMPLOYMENT_WEIGHTS = [0.68, 0.15, 0.07, 0.10]
AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
REGIONS = ["UKC", "UKD", "UKE", "UKF", "UKG", "UKH", "UKI", "UKJ", "UKK", "UKL", "UKM", "UKN"]

FEATURE_COLUMNS = [
    "income_band",
    "debt_to_income_ratio",
    "credit_history_years",
    "num_delinquencies_last_2y",
    "loan_amount_requested",
    "employment_status",
    "age_band",
    "region",
]

# Quasi-identifiers (age_band, region) are collected and stored — they're needed
# for population-level fairness/disparate-impact monitoring — but deliberately
# EXCLUDED from the classifier's inputs. This is enforced at training time, not
# just filtered out of explanations after the fact: EU AI Act Article 10 requires
# bias-aware governance of a high-risk system's training data, and letting a
# protected/quasi-identifying attribute drive the score (and then show up in a
# SHAP-based explanation) is exactly the failure mode that creates fair-lending
# and Article 10 exposure. See `credit_risk.data.feature_store.COLUMN_SENSITIVITY`.
MODEL_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in ("age_band", "region")]


def _default_probability(df: pd.DataFrame) -> np.ndarray:
    employment_risk = df["employment_status"].map(
        {"employed": 0.0, "self_employed": 0.08, "unemployed": 0.35, "retired": 0.02}
    )
    logit = (
        -3.0
        + 5.5 * df["debt_to_income_ratio"]
        + 0.55 * df["num_delinquencies_last_2y"]
        - 0.14 * df["credit_history_years"]
        - 0.00002 * df["income_band"]
        + 2.2 * employment_risk
        + 0.00006 * df["loan_amount_requested"]
    )
    return 1 / (1 + np.exp(-logit))


def generate_dataset(n: int = 2000, seed: int = 42, drift_shift: float = 0.0) -> pd.DataFrame:
    """Generate a synthetic applicant population.

    `drift_shift` > 0 nudges debt-to-income and delinquency distributions upward to
    simulate a macroeconomic downturn — used by the monitoring demo to trigger drift
    detection without needing a second real dataset.
    """
    rng = np.random.default_rng(seed)

    income_band = rng.integers(15_000, 120_000, size=n) // 5000 * 5000
    debt_to_income_ratio = np.clip(
        rng.normal(0.28 + drift_shift, 0.12, size=n), 0.0, 1.2
    )
    credit_history_years = np.clip(rng.exponential(6.0, size=n), 0.0, 40.0)
    num_delinquencies_last_2y = rng.poisson(0.4 + drift_shift * 3, size=n)
    loan_amount_requested = rng.integers(1000, 45000, size=n) // 500 * 500
    employment_status = rng.choice(EMPLOYMENT_STATUSES, size=n, p=EMPLOYMENT_WEIGHTS)
    age_band = rng.choice(AGE_BANDS, size=n)
    region = rng.choice(REGIONS, size=n)
    applicant_id = [f"APP-{i:06d}" for i in range(n)]

    df = pd.DataFrame(
        {
            "applicant_id": applicant_id,
            "income_band": income_band,
            "debt_to_income_ratio": debt_to_income_ratio,
            "credit_history_years": credit_history_years,
            "num_delinquencies_last_2y": num_delinquencies_last_2y,
            "loan_amount_requested": loan_amount_requested,
            "employment_status": employment_status,
            "age_band": age_band,
            "region": region,
        }
    )
    prob_default = _default_probability(df)
    df["defaulted"] = rng.binomial(1, prob_default)
    return df
