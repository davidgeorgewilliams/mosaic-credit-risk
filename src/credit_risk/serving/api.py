"""FastAPI serving layer.

Stands in for what a Databricks App (or any service in front of a Mosaic AI
endpoint) would call. `/score` hits the champion model, mirroring a fraction of
traffic to the shadow model per `shadow_traffic_fraction`; `/explain` runs the
full explainability agent pipeline; every explained decision is written to the
in-process `AuditLog` — the local equivalent of a Unity Catalog audit table —
and surfaced at `/governance/audit`.
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from credit_risk.adapters.ai_gateway import InMemoryAIGateway
from credit_risk.adapters.ai_search import InMemoryVectorSearch
from credit_risk.adapters.llm_client import get_default_llm_client
from credit_risk.adapters.model_serving import InMemoryModelServing
from credit_risk.agent.explainability_agent import ExplainabilityAgent
from credit_risk.domain.schemas import Applicant, Explanation, RiskScore
from credit_risk.governance.eu_ai_act import SYSTEM_CLASSIFICATION, AuditLog


class AppState:
    model_serving: InMemoryModelServing
    agent: ExplainabilityAgent
    ai_gateway: InMemoryAIGateway
    audit_log: AuditLog


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.model_serving = InMemoryModelServing()
    state.ai_gateway = InMemoryAIGateway()
    state.agent = ExplainabilityAgent(
        model_serving=state.model_serving,
        vector_search=InMemoryVectorSearch(),
        ai_gateway=state.ai_gateway,
        llm_client=get_default_llm_client(),
    )
    state.audit_log = AuditLog()
    yield


app = FastAPI(title="Credit Risk Decisioning API", lifespan=lifespan)


def _applicant_row(applicant: Applicant) -> pd.DataFrame:
    return pd.DataFrame([applicant.model_dump(mode="json")])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "served_models": state.model_serving.served_models}


@app.post("/score", response_model=RiskScore)
def score(applicant: Applicant) -> RiskScore:
    return state.model_serving.score(_applicant_row(applicant))


@app.post("/explain", response_model=Explanation)
def explain(applicant: Applicant) -> Explanation:
    row = _applicant_row(applicant)
    risk_score = state.model_serving.score(row)
    explanation = state.agent.explain(row, risk_score)
    state.audit_log.record_decision(risk_score, explanation)
    return explanation


@app.get("/governance/audit")
def audit_events() -> list[dict]:
    return [event.model_dump(mode="json") for event in state.audit_log.events]


@app.get("/governance/classification")
def classification() -> dict:
    return {
        "category": SYSTEM_CLASSIFICATION.category,
        "annex_reference": SYSTEM_CLASSIFICATION.annex_reference,
        "obligations": SYSTEM_CLASSIFICATION.obligations,
    }


@app.get("/governance/ai_gateway_usage")
def ai_gateway_usage() -> dict:
    return state.ai_gateway.usage_report()
