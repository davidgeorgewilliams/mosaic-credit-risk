# Credit Risk Decisioning — Databricks / Mosaic AI Demo

A self-contained demo of a **regulated credit-risk decisioning system**: a
classical ML model scores applications and is shadow-deployed and monitored for
drift; an LLM agent behind an AI Gateway explains each decision in natural
language, grounded by retrieval over a lending-policy corpus, satisfying EU AI
Act transparency and human-oversight obligations for a "high-risk" AI system.
Built to demonstrate production MLOps + AI-app engineering patterns on
Databricks/Mosaic AI.

No live Databricks workspace is required. Every Databricks-managed service
(Unity Catalog Feature Store, Mosaic AI Model Serving, AI Gateway, Vector
Search, Lakehouse Monitoring) sits behind a small `Protocol` interface with a
local, in-memory adapter — see `docs/ARCHITECTURE.md`. Everything that *can*
run for real, does: MLflow (tracking, model registry, prompt registry),
scikit-learn, FastAPI, pytest.

## Quickstart

```bash
make setup   # create venv, install deps
make demo    # narrated end-to-end walkthrough (train -> serve -> explain -> drift -> retrain)
make test    # pytest — unit + integration
make lint    # ruff
make typecheck  # mypy

make serve   # FastAPI app on :8000
make app     # Streamlit "loan officer" UI, in a second terminal, once `make serve` is running
```

`make demo` is the fastest way to see the whole system work — it trains and
registers both model versions, scores a batch of applicants through the
shadow-mirrored serving layer, runs the full explainability agent pipeline on
one applicant, simulates population drift and shows the retrain trigger fire,
then prints the GDPR/EU AI Act governance artifacts for that decision.

By default the LLM explanation step uses a deterministic offline double
(`FakeLLMClient`) so the whole demo runs with zero network access and no API
key. Set `CREDIT_RISK_OPENAI_API_KEY` to route explanations through a real
OpenAI-compatible endpoint instead (see `credit_risk.adapters.llm_client`).

## Project layout

```
src/credit_risk/
  config.py                  Settings (env-driven)
  domain/schemas.py          Applicant, RiskScore, Explanation, AuditEvent
  data/                      synthetic data generator + feature store abstraction
  models/                    training + the MLflow pyfunc model (classifier + SHAP)
  adapters/                  model_serving, ai_gateway, ai_search, llm_client
  agent/                     prompts (MLflow Prompt Registry), explainability agent, guardrails
  governance/                gdpr.py, eu_ai_act.py
  monitoring/                drift detection, retrain trigger
  serving/                   FastAPI app, illustrative SQL AI functions
  app/                       Streamlit loan-officer UI
databricks/                  illustrative DAB bundle (not deployed by this demo)
tests/                       unit + integration (pytest)
scripts/demo.py              the narrated end-to-end walkthrough
```

## What maps to what

| JD requirement | Where |
|---|---|
| MLflow — pipelines, custom environments, registry automation | `models/train.py`, `models/pyfunc_model.py` |
| Databricks Mosaic AI Model Serving, multi-model, shadow/A-B | `adapters/model_serving.py` |
| AI Gateway governance and rate management | `adapters/ai_gateway.py` |
| Mosaic AI Agent Framework | `agent/explainability_agent.py` |
| Lakehouse Monitoring, drift detection | `monitoring/lakehouse_monitoring.py` |
| Feature Store governance within Unity Catalog | `data/feature_store.py` |
| AI Search / Vector Search | `adapters/ai_search.py` |
| Prompt management | `agent/prompts.py` (MLflow Prompt Registry) |
| Databricks Apps | `app/streamlit_app.py` |
| AI functions (SQL) | `serving/ai_functions_example.sql` |
| DAB / CI/CD | `databricks/`, `.github/workflows/ci.yml` |
| Azure OpenAI / LLM productionisation, cost governance | `adapters/llm_client.py`, `adapters/ai_gateway.py` |
| GDPR compliance | `governance/gdpr.py`, `data/feature_store.py` (minimization) |
| EU AI Act — high-risk classification, transparency, oversight | `governance/eu_ai_act.py`, `agent/guardrails.py` |
| Software engineering discipline — clean architecture, testing | `Protocol`-based ports/adapters throughout, `tests/` |

## A design decision worth knowing cold

The classifier is trained on `MODEL_FEATURE_COLUMNS`, which **excludes**
`age_band` and `region` even though they're collected and stored (for
population-level fairness monitoring). Early in building this, the model *did*
train on those fields, and a SHAP-based explanation leaked `region_UKG` — a
quasi-identifier — straight into an LLM narrative. The fix wasn't a filter at
the output boundary; it was removing the quasi-identifiers from the model's
input features entirely, because the EU AI Act's Article 10 data-governance
obligations for high-risk systems apply to training data, not just outputs. See
`credit_risk/data/synthetic_data.py` and `credit_risk/agent/guardrails.py`
(which still checks for this pattern as defense in depth).
