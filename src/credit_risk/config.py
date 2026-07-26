"""Central settings. In production these adapters would point at real Databricks
services (Unity Catalog, Mosaic AI Model Serving, AI Gateway, Vector Search); here
they default to local, in-memory implementations so the whole system runs offline.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CREDIT_RISK_")

    # MLflow 3.x deprecated the plain filesystem store for new features (prompt
    # registry, model registry aliases); sqlite is the lightweight local equivalent
    # of the Postgres-backed tracking server Databricks runs in production.
    mlflow_tracking_uri: str = f"sqlite:///{MLFLOW_DB_PATH}"
    mlflow_experiment_name: str = "credit-risk-scoring"
    registered_model_name: str = "credit_risk_classifier"

    # LLM: falls back to a deterministic offline client unless a real key is set.
    openai_api_key: str | None = None
    llm_model_name: str = "gpt-4o-mini"

    # Shadow deployment traffic split for the candidate model, 0.0-1.0.
    shadow_traffic_fraction: float = 0.1

    # Drift detection thresholds (population stability index).
    psi_warning_threshold: float = 0.1
    psi_critical_threshold: float = 0.25

    # AI Gateway governance.
    ai_gateway_rate_limit_per_minute: int = 60
    ai_gateway_token_budget_per_day: int = 200_000

    # HMAC key for pseudonymizing identifiers that cross a trust boundary (see
    # credit_risk.governance.gdpr.pseudonymize). The default below is fine for
    # local/demo use only — since it ships in this public repo, a real deployment
    # MUST override it via CREDIT_RISK_PSEUDONYMIZATION_SECRET, otherwise a
    # bare/known-key hash is exactly as reversible as no key at all.
    pseudonymization_secret: str = "dev-only-insecure-default-change-me"


settings = Settings()
