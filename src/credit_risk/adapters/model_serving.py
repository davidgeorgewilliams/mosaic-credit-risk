"""Model serving adapter.

`InMemoryModelServing` stands in for Databricks Mosaic AI Model Serving. In
production this maps to a Mosaic AI Model Serving endpoint with two served
entities — `champion` and `shadow` — and a traffic-split rule (see
`databricks/resources/model_serving_endpoint.yml` for the equivalent DAB
resource). The shadow entity receives a mirrored fraction of production
requests, scores them, and its output is logged for comparison but never
returned to the caller: that's exactly what `score()` does here, and it's how a
candidate model earns production traffic — reviewed shadow performance first,
promoted to `champion` alias second, never the other way round.
"""

import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import mlflow
import pandas as pd

from credit_risk.config import settings
from credit_risk.domain.schemas import Decision, RiskScore, ScoreSource

# Decision thresholds on P(default): >=DENY -> deny, >=REFER -> human review, else approve.
DENY_THRESHOLD = 0.5
REFER_THRESHOLD = 0.3


def _decide(probability: float) -> Decision:
    if probability >= DENY_THRESHOLD:
        return Decision.DENY
    if probability >= REFER_THRESHOLD:
        return Decision.REFER_TO_HUMAN
    return Decision.APPROVE


@dataclass
class ServedModel:
    alias: str
    version: str
    pyfunc_model: Any  # mlflow.pyfunc.PyFuncModel — untyped in mlflow's public API


@dataclass
class ShadowComparison:
    n_mirrored: int
    mean_abs_probability_diff: float
    decision_agreement_rate: float


class ModelServingClient(Protocol):
    def score(self, applicant_row: pd.DataFrame) -> RiskScore: ...

    def explain_inputs(self, applicant_row: pd.DataFrame) -> list[dict]:
        """SHAP contributions from the champion model, for the agent layer."""
        ...

    def shadow_comparison(self) -> ShadowComparison: ...


@dataclass
class _RequestLogEntry:
    applicant_id: str
    champion_probability: float
    shadow_probability: float | None
    champion_decision: Decision
    shadow_decision: Decision | None


class InMemoryModelServing:
    def __init__(self, endpoint_name: str = "credit-risk-endpoint") -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.endpoint_name = endpoint_name
        self.champion = self._load("champion")
        self.shadow = self._load("shadow")
        self._log: list[_RequestLogEntry] = []
        self._last_latency_ms: float | None = None

    def _load(self, alias: str) -> ServedModel:
        client = mlflow.MlflowClient()
        model_version = client.get_model_version_by_alias(settings.registered_model_name, alias)
        model = mlflow.pyfunc.load_model(f"models:/{settings.registered_model_name}@{alias}")
        return ServedModel(alias=alias, version=str(model_version.version), pyfunc_model=model)

    @property
    def served_models(self) -> list[dict]:
        """Multi-model serving snapshot — what's live behind this one endpoint."""
        return [
            {"alias": m.alias, "version": m.version, "model_name": settings.registered_model_name}
            for m in (self.champion, self.shadow)
        ]

    def score(self, applicant_row: pd.DataFrame) -> RiskScore:
        applicant_id = applicant_row["applicant_id"].iloc[0]

        champion_proba = self._predict(self.champion, applicant_row)
        result = RiskScore(
            applicant_id=applicant_id,
            model_name=settings.registered_model_name,
            model_version=self.champion.version,
            probability_of_default=champion_proba,
            decision=_decide(champion_proba),
            score_source=ScoreSource.CHAMPION,
            endpoint_name=self.endpoint_name,
            scored_at=datetime.now(UTC),
        )

        shadow_proba = None
        shadow_decision = None
        if random.random() < settings.shadow_traffic_fraction:
            shadow_proba = self._predict(self.shadow, applicant_row)
            shadow_decision = _decide(shadow_proba)

        self._log.append(
            _RequestLogEntry(
                applicant_id=applicant_id,
                champion_probability=champion_proba,
                shadow_probability=shadow_proba,
                champion_decision=result.decision,
                shadow_decision=shadow_decision,
            )
        )
        return result

    @property
    def last_latency_ms(self) -> float | None:
        """Wall-clock inference latency of the most recent `score()` call. Real
        numbers, not simulated — at this dataset/model scale it's sub-millisecond,
        which is exactly why batching/quantization/autoscaling only start to
        matter at real production throughput, not in this demo."""
        return self._last_latency_ms

    def _predict(self, served: ServedModel, applicant_row: pd.DataFrame) -> float:
        start = time.perf_counter()
        proba = float(served.pyfunc_model.predict(applicant_row)[0])
        self._last_latency_ms = (time.perf_counter() - start) * 1000
        return proba

    def explain_inputs(self, applicant_row: pd.DataFrame) -> list[dict]:
        inner = self.champion.pyfunc_model.unwrap_python_model()
        return inner.explain(applicant_row)[0]

    def shadow_comparison(self) -> ShadowComparison:
        mirrored = [entry for entry in self._log if entry.shadow_probability is not None]
        if not mirrored:
            return ShadowComparison(0, 0.0, 0.0)
        diffs = [
            abs(e.champion_probability - e.shadow_probability)
            for e in mirrored
            if e.shadow_probability is not None
        ]
        agreements = [e.champion_decision == e.shadow_decision for e in mirrored]
        return ShadowComparison(
            n_mirrored=len(mirrored),
            mean_abs_probability_diff=sum(diffs) / len(diffs),
            decision_agreement_rate=sum(agreements) / len(agreements),
        )
