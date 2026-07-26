"""Custom MLflow pyfunc model.

Bundling the classifier, its preprocessing pipeline, and a SHAP explainer into one
`mlflow.pyfunc.PythonModel` means a single versioned artifact in the MLflow Model
Registry (later, Unity Catalog) is both the thing that serves predictions and the
thing that explains them — no drift between "the model in prod" and "the model we
explain," which is a real failure mode in regulated ML systems.
"""

import numpy as np
import pandas as pd
import shap
from mlflow.pyfunc import PythonModel

# age_band/region are intentionally NOT model features — see
# credit_risk.data.synthetic_data.MODEL_FEATURE_COLUMNS for why.
CATEGORICAL_COLUMNS = ["employment_status"]


class CreditRiskModel(PythonModel):
    def __init__(self, classifier, preprocessor, feature_columns: list[str]) -> None:
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.feature_columns = feature_columns
        self._explainer: shap.TreeExplainer | None = None

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.preprocessor.transform(df[self.feature_columns]), dtype=float)

    def predict(self, context, model_input: pd.DataFrame, params=None):
        proba = self.classifier.predict_proba(self._prepare(model_input))[:, 1]
        return proba

    def explain(self, df: pd.DataFrame) -> list[list[dict]]:
        """Return top SHAP feature contributions per row. Only reachable via
        `unwrap_python_model()` — not part of the standard pyfunc serving surface,
        deliberately, since it's more expensive than a plain prediction and the
        agent layer calls it explicitly."""
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.classifier)
        X = self._prepare(df)
        feature_names = self.preprocessor.get_feature_names_out(self.feature_columns)
        shap_values = self._explainer.shap_values(X)

        results = []
        for row in shap_values:
            contributions = sorted(
                zip(feature_names, row, strict=True), key=lambda t: abs(t[1]), reverse=True
            )
            results.append(
                [
                    {
                        "feature": _clean_feature_name(name),
                        "shap_value": float(value),
                        "direction": "increases_risk" if value > 0 else "decreases_risk",
                    }
                    for name, value in contributions[:5]
                ]
            )
        return results


def _clean_feature_name(raw: str) -> str:
    # ColumnTransformer prefixes one-hot columns like "cat__employment_status_unemployed".
    return raw.split("__", 1)[-1]
