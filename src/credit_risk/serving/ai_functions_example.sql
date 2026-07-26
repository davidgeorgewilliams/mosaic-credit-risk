-- Illustrative only — Databricks SQL AI functions require a live Unity Catalog
-- workspace with a model-serving endpoint configured, so this is not executed by
-- the demo. It shows the pattern this project's Python components (model
-- serving, the explainability agent) map to for teams that prefer a SQL-native
-- surface, e.g. batch-scoring a table of applicants or gating on the same
-- guardrail logic from `credit_risk.agent.guardrails` at the warehouse layer.

-- Batch score every open application using the served champion model, the SQL
-- equivalent of `credit_risk.adapters.model_serving.InMemoryModelServing.score`.
SELECT
  applicant_id,
  ai_query(
    'credit-risk-endpoint',
    named_struct(
      'income_band', income_band,
      'debt_to_income_ratio', debt_to_income_ratio,
      'credit_history_years', credit_history_years,
      'num_delinquencies_last_2y', num_delinquencies_last_2y,
      'loan_amount_requested', loan_amount_requested,
      'employment_status', employment_status
    )
  ) AS probability_of_default
FROM main.credit_risk.applications
WHERE status = 'pending';

-- Cheap pre-filter classification before the full explainability agent runs —
-- route only borderline cases to the (more expensive) LLM explanation path,
-- the SQL-native analogue of the REFER_THRESHOLD / DENY_THRESHOLD split in
-- credit_risk.adapters.model_serving.
SELECT
  applicant_id,
  ai_classify(
    concat('Default probability: ', probability_of_default),
    ARRAY('clear_approve', 'needs_explanation', 'clear_deny')
  ) AS triage_bucket
FROM main.credit_risk.scored_applications;
