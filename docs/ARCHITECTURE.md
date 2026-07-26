# Architecture

## Ports and adapters

Every Databricks-managed service this system depends on is expressed as a
`typing.Protocol` in `credit_risk/adapters/` and `credit_risk/data/`, with a
local, in-memory implementation used everywhere in this demo:

| Port (Protocol) | Local adapter | Real adapter (production) |
|---|---|---|
| `FeatureStore` | `InMemoryFeatureStore` | `UnityCatalogFeatureStore` via `databricks.feature_engineering.FeatureEngineeringClient` |
| `ModelServingClient` | `InMemoryModelServing` | Mosaic AI Model Serving endpoint via `databricks-sdk` / `mlflow.deployments` |
| `AIGatewayClient` | `InMemoryAIGateway` | Mosaic AI Gateway route |
| `VectorSearchClient` | `InMemoryVectorSearch` | Mosaic AI Vector Search index over a Delta table |
| `LLMClient` | `FakeLLMClient` / `OpenAILLMClient` | Azure OpenAI Service (same `OpenAILLMClient`, different base URL) |

Application code (`credit_risk.agent.explainability_agent.ExplainabilityAgent`,
`credit_risk.serving.api`) depends only on the `Protocol`, never the concrete
class. Swapping a local adapter for a real Databricks-backed one is a
constructor change at the composition root (`serving/api.py`'s `lifespan`), not
a rewrite of business logic — that's the entire point of the boundary, and it's
what makes "clean architecture" in the JD concrete rather than a buzzword.

## Data flow for one decision

```
Applicant (API request)
  -> InMemoryFeatureStore              (would write/read Unity Catalog features)
  -> InMemoryModelServing.score()      champion model scores; shadow model mirrors
       - GradientBoostingClassifier trained on MODEL_FEATURE_COLUMNS only
       - decision thresholds -> approve / deny / refer_to_human
  -> ExplainabilityAgent.explain()
       1. InMemoryModelServing.explain_inputs()  SHAP top factors
       2. InMemoryVectorSearch.search()          relevant lending-policy clauses (RAG)
       3. agent.prompts.render()                 versioned prompt from MLflow Prompt Registry
       4. InMemoryAIGateway.route()               rate limit + token budget + cost tracking
       5. LLMClient.complete()                    FakeLLMClient (offline) or OpenAILLMClient
       6. agent.guardrails.validate()              block on forbidden-claim/PII, else sanitize
  -> AuditLog.record_decision()          Article 12 record-keeping
```

## Why the model excludes quasi-identifiers

`credit_risk.data.synthetic_data.FEATURE_COLUMNS` is the full applicant record
(what's collected and stored). `MODEL_FEATURE_COLUMNS` is the subset the
classifier actually trains and scores on — it excludes `age_band` and `region`.

This was found, not designed up front: an early version trained on the full
feature set, and a SHAP explanation surfaced `region_UKG` directly in an LLM
narrative. Filtering it out of the *explanation* would have treated the
symptom. The actual fix is upstream — the model never sees the quasi-identifier
in the first place, which is what EU AI Act Article 10 (data governance for
high-risk systems) actually requires, and it also removes an entire class of
disparate-impact risk. `credit_risk.agent.guardrails` still checks for this
pattern in every generated narrative — defense in depth, not the primary
control.

## What's real vs. simulated

- **Real**: MLflow tracking, model registry (with aliases), prompt registry;
  scikit-learn training; SHAP explanations; FastAPI; the guardrail, drift
  (PSI/KS), and governance logic; the full test suite.
- **Simulated locally, real-shaped**: Unity Catalog Feature Store, Mosaic AI
  Model Serving, AI Gateway, Vector Search — same interface a real Databricks
  adapter would implement, backed by in-memory Python instead of a workspace.
- **Illustrative only, not executed**: `databricks/*.yml` (DAB bundle),
  `serving/ai_functions_example.sql` — these show the production-deployment
  shape without requiring a live workspace to run this repo.
