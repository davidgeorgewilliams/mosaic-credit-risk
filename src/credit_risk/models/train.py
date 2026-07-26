"""Trains the credit-risk classifier against local MLflow tracking + registry.

Trains two versions — one registered under the `champion` alias, one under
`shadow` — mirroring a real retrain-and-validate cycle: a candidate model earns
production traffic only after its shadow performance is reviewed
(see `credit_risk.adapters.model_serving`).
"""

import mlflow
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from credit_risk.config import settings
from credit_risk.data.synthetic_data import (
    FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    generate_dataset,
)
from credit_risk.models.pyfunc_model import CATEGORICAL_COLUMNS, CreditRiskModel

NUMERIC_COLUMNS = [c for c in MODEL_FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]


def _build_preprocessor() -> ColumnTransformer:
    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("cat", categorical_encoder, CATEGORICAL_COLUMNS),
            ("num", "passthrough", NUMERIC_COLUMNS),
        ]
    )


def _train_one(df, n_estimators: int, max_depth: int, seed: int) -> tuple[CreditRiskModel, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        df[MODEL_FEATURE_COLUMNS],
        df["defaulted"],
        test_size=0.2,
        random_state=seed,
        stratify=df["defaulted"],
    )
    preprocessor = _build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed
    )
    clf.fit(X_train_t, y_train)

    auc = roc_auc_score(y_test, clf.predict_proba(X_test_t)[:, 1])
    model = CreditRiskModel(
        classifier=clf, preprocessor=preprocessor, feature_columns=MODEL_FEATURE_COLUMNS
    )
    return model, auc


def train_and_register(alias: str, n_estimators: int, max_depth: int, seed: int) -> str:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    df = generate_dataset(n=4000, seed=seed)
    model, auc = _train_one(df, n_estimators, max_depth, seed)

    with mlflow.start_run(run_name=f"train-{alias}"):
        mlflow.log_params(
            {"n_estimators": n_estimators, "max_depth": max_depth, "seed": seed, "alias": alias}
        )
        mlflow.log_metric("test_auc", auc)
        model_info = mlflow.pyfunc.log_model(
            python_model=model,
            artifact_path="model",
            registered_model_name=settings.registered_model_name,
            input_example=df[FEATURE_COLUMNS].head(3),
        )

    client = MlflowClient()
    version = model_info.registered_model_version
    client.set_registered_model_alias(settings.registered_model_name, alias, version)
    print(f"Registered {settings.registered_model_name} v{version} as '{alias}' (AUC={auc:.4f})")
    return version


def main() -> None:
    train_and_register(alias="champion", n_estimators=150, max_depth=3, seed=42)
    train_and_register(alias="shadow", n_estimators=250, max_depth=4, seed=43)


if __name__ == "__main__":
    main()
