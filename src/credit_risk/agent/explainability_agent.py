"""Explainability agent — a Mosaic AI Agent Framework-style component.

Retrieval-grounded, guardrailed LLM explanation of one automated credit
decision: (1) pull the champion model's SHAP contributions for this applicant
from `ModelServingClient`, (2) retrieve the lending-policy clauses most relevant
to those factors via `VectorSearchClient` (AI Search), (3) render the versioned
prompt from the MLflow Prompt Registry, (4) call the LLM through the AI Gateway
(rate limit + cost tracking), (5) validate the output through guardrails before
it reaches a human. Every dependency is injected as a `Protocol`, so this class
is agnostic to whether it's driving in-memory doubles or real Databricks
services — that's the whole point of the ports-and-adapters boundary.
"""

from datetime import UTC, datetime

import pandas as pd

from credit_risk.adapters.ai_gateway import AIGatewayClient
from credit_risk.adapters.ai_search import VectorSearchClient
from credit_risk.adapters.llm_client import LLMClient
from credit_risk.adapters.model_serving import ModelServingClient
from credit_risk.agent import guardrails, prompts
from credit_risk.domain.schemas import Explanation, FactorContribution, RiskScore
from credit_risk.governance.eu_ai_act import human_oversight_required


class ExplainabilityAgent:
    def __init__(
        self,
        model_serving: ModelServingClient,
        vector_search: VectorSearchClient,
        ai_gateway: AIGatewayClient,
        llm_client: LLMClient,
    ) -> None:
        self._model_serving = model_serving
        self._vector_search = vector_search
        self._ai_gateway = ai_gateway
        self._llm_client = llm_client

    def explain(self, applicant_row: pd.DataFrame, risk_score: RiskScore) -> Explanation:
        shap_factors = self._model_serving.explain_inputs(applicant_row)
        top_factors = [
            FactorContribution(
                feature=f["feature"], shap_value=f["shap_value"], direction=f["direction"]
            )
            for f in shap_factors
        ]

        query_terms = [
            f["feature"].replace("_", " ")
            for f in shap_factors
            if f["direction"] == "increases_risk"
        ]
        query = " ".join(query_terms) or "credit risk policy"
        policy_matches = self._vector_search.search(query, k=2)

        system_prompt, user_prompt, prompt_version = prompts.render(
            decision=risk_score.decision.value,
            probability=f"{risk_score.probability_of_default:.0%}",
            top_factors="\n".join(
                f"- {f.feature} {f.direction.replace('_', ' ')}" for f in top_factors
            ),
            policy_clauses="\n".join(f"- {m.text}" for m in policy_matches),
        )

        response = self._ai_gateway.route(
            "credit-risk-explanation",
            lambda: self._llm_client.complete(system_prompt, user_prompt),
        )

        result = guardrails.validate(response.text)

        return Explanation(
            applicant_id=risk_score.applicant_id,
            narrative=result.sanitized_text,
            top_factors=top_factors,
            cited_policy_clauses=[m.clause_id for m in policy_matches],
            guardrail_passed=result.passed,
            human_review_required=human_oversight_required(risk_score.decision, result.passed),
            prompt_name=prompts.PROMPT_NAME,
            prompt_version=prompt_version,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            generated_at=datetime.now(UTC),
        )
