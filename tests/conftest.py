import mlflow
import pytest

from credit_risk.config import settings
from credit_risk.models import train as train_module


@pytest.fixture(scope="session", autouse=True)
def ensure_models_trained():
    """Trains and registers champion/shadow once per test session if they don't
    already exist in the local MLflow store — keeps the suite runnable from a
    clean checkout with zero setup while staying fast on repeat runs."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    try:
        client.get_model_version_by_alias(settings.registered_model_name, "champion")
        client.get_model_version_by_alias(settings.registered_model_name, "shadow")
        return
    except Exception:
        train_module.main()
