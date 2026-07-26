"""Feature store abstraction.

`InMemoryFeatureStore` stands in for Databricks' Unity Catalog Feature Store
(`databricks.feature_engineering.FeatureEngineeringClient`). The `Protocol` boundary
is the point: production code depends on `FeatureStore`, never on the concrete
in-memory class, so a `UnityCatalogFeatureStore(FeatureStore)` adapter (calling
`fe_client.create_table` / `write_table` / `read_table` against a real workspace)
can be swapped in without touching any caller.

Column sensitivity tagging mirrors Unity Catalog column tags/masks and is what lets
`credit_risk.governance.gdpr` enforce data minimization when features are pulled for
the LLM explanation agent — quasi-identifiers never leave the feature store boundary.
"""

from enum import StrEnum
from typing import Protocol

import pandas as pd


class Sensitivity(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PSEUDONYMOUS_ID = "pseudonymous_id"
    QUASI_IDENTIFIER = "quasi_identifier"


COLUMN_SENSITIVITY: dict[str, Sensitivity] = {
    "applicant_id": Sensitivity.PSEUDONYMOUS_ID,
    "income_band": Sensitivity.INTERNAL,
    "debt_to_income_ratio": Sensitivity.INTERNAL,
    "credit_history_years": Sensitivity.INTERNAL,
    "num_delinquencies_last_2y": Sensitivity.INTERNAL,
    "loan_amount_requested": Sensitivity.INTERNAL,
    "employment_status": Sensitivity.INTERNAL,
    "age_band": Sensitivity.QUASI_IDENTIFIER,
    "region": Sensitivity.QUASI_IDENTIFIER,
    "defaulted": Sensitivity.INTERNAL,
}


class FeatureStore(Protocol):
    def write_table(self, table_name: str, df: pd.DataFrame, primary_key: str) -> None: ...

    def read_table(
        self, table_name: str, applicant_ids: list[str] | None = None
    ) -> pd.DataFrame: ...

    def read_table_minimized(
        self, table_name: str, applicant_ids: list[str] | None = None
    ) -> pd.DataFrame:
        """Read with quasi-identifier columns stripped — the only read path the
        LLM explanation agent is allowed to use."""
        ...


class InMemoryFeatureStore:
    """Local stand-in for Unity Catalog Feature Store. Same table/read semantics,
    zero external dependencies."""

    def __init__(self) -> None:
        self._tables: dict[str, pd.DataFrame] = {}
        self._primary_keys: dict[str, str] = {}

    def write_table(self, table_name: str, df: pd.DataFrame, primary_key: str) -> None:
        self._tables[table_name] = df.set_index(primary_key, drop=False)
        self._primary_keys[table_name] = primary_key

    def read_table(
        self, table_name: str, applicant_ids: list[str] | None = None
    ) -> pd.DataFrame:
        df = self._tables[table_name]
        if applicant_ids is None:
            return df.reset_index(drop=True).copy()
        return df.loc[df.index.intersection(applicant_ids)].reset_index(drop=True).copy()

    def read_table_minimized(
        self, table_name: str, applicant_ids: list[str] | None = None
    ) -> pd.DataFrame:
        df = self.read_table(table_name, applicant_ids)
        drop_cols = [
            c for c in df.columns if COLUMN_SENSITIVITY.get(c) == Sensitivity.QUASI_IDENTIFIER
        ]
        return df.drop(columns=drop_cols)
