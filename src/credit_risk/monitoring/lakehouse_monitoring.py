"""Model monitoring — drift detection.

Stands in for Databricks Lakehouse Monitoring's drift metrics, which compute
these same statistics automatically against a designated baseline table/profile.
PSI (population stability index) is bucketed per numeric feature for the primary
signal; a Kolmogorov-Smirnov test backs it up. `psi_warning_threshold` /
`psi_critical_threshold` mirror the alert thresholds you'd configure on a real
Lakehouse Monitor. `overall_status` is a simple multivariate rollup — the worst
status across tracked features — good enough to decide "should this page
someone," which is the only question a retraining trigger actually needs
answered.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from scipy import stats

from credit_risk.config import settings

NUMERIC_DRIFT_COLUMNS = [
    "debt_to_income_ratio",
    "credit_history_years",
    "num_delinquencies_last_2y",
    "income_band",
    "loan_amount_requested",
]


class DriftStatus(StrEnum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


_SEVERITY_ORDER = [DriftStatus.OK, DriftStatus.WARNING, DriftStatus.CRITICAL]


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    status: DriftStatus


@dataclass
class DriftReport:
    feature_drifts: list[FeatureDrift]
    overall_status: DriftStatus

    @property
    def max_psi_feature(self) -> FeatureDrift:
        return max(self.feature_drifts, key=lambda f: f.psi)


def _population_stability_index(
    reference: np.ndarray, current: np.ndarray, buckets: int = 10
) -> float:
    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(np.quantile(reference, quantiles))
    if len(breakpoints) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)
    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _status_for_psi(psi: float) -> DriftStatus:
    if psi >= settings.psi_critical_threshold:
        return DriftStatus.CRITICAL
    if psi >= settings.psi_warning_threshold:
        return DriftStatus.WARNING
    return DriftStatus.OK


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame) -> DriftReport:
    drifts = []
    for column in NUMERIC_DRIFT_COLUMNS:
        psi = _population_stability_index(reference[column].to_numpy(), current[column].to_numpy())
        ks_statistic, ks_pvalue = stats.ks_2samp(reference[column], current[column])
        drifts.append(
            FeatureDrift(
                feature=column,
                psi=psi,
                ks_statistic=float(ks_statistic),
                ks_pvalue=float(ks_pvalue),
                status=_status_for_psi(psi),
            )
        )
    overall = max((d.status for d in drifts), key=_SEVERITY_ORDER.index)
    return DriftReport(feature_drifts=drifts, overall_status=overall)
