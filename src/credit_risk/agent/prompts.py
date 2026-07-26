"""Prompt templates, versioned via MLflow's Prompt Registry.

Every prompt used in production is registered here rather than inlined at the
call site — that's what gives you a diffable history of prompt changes, the
ability to roll back a regression, and a stable `@production` alias that serving
code depends on instead of a hardcoded version number. Same alias pattern as the
model registry in `credit_risk.adapters.model_serving`.
"""

import mlflow
import mlflow.exceptions
import mlflow.genai as genai

from credit_risk.config import settings

PROMPT_NAME = "credit_risk_explanation"

SYSTEM_MESSAGE = (
    "You are a lending assistant that explains automated credit decisions to loan "
    "officers and applicants in plain language. Be factual, cite only the policy "
    "clauses provided, never invent a reason that is not grounded in the given "
    "factors, and always mention the applicant's right to request a human review."
)

USER_MESSAGE = (
    "DECISION: {{decision}}\n"
    "PROBABILITY: {{probability}}\n"
    "TOP_FACTORS:\n{{top_factors}}\n"
    "POLICY_CLAUSES:\n{{policy_clauses}}\n"
)


def ensure_registered() -> None:
    """Registers version 1 and aliases it `production` the first time this runs.
    In a real deployment this is a one-off CI/release step (see `.github/workflows
    /ci.yml`), not something invoked from request-serving code — it's inlined here
    only so the demo is runnable from a clean checkout with one command."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        genai.load_prompt(f"prompts:/{PROMPT_NAME}@production")
        return
    except mlflow.exceptions.MlflowException:
        pass

    genai.register_prompt(
        name=PROMPT_NAME,
        template=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_MESSAGE},
        ],
        commit_message="Initial explanation prompt.",
        tags={"component": "explainability_agent"},
    )
    genai.set_prompt_alias(PROMPT_NAME, alias="production", version=1)


def render(
    decision: str, probability: str, top_factors: str, policy_clauses: str
) -> tuple[str, str, str]:
    """Returns (system_prompt, user_prompt, prompt_version) for the `production`
    alias of the registered explanation prompt."""
    ensure_registered()
    prompt = genai.load_prompt(f"prompts:/{PROMPT_NAME}@production")
    messages = prompt.format(
        decision=decision,
        probability=probability,
        top_factors=top_factors,
        policy_clauses=policy_clauses,
    )
    system = next(m["content"] for m in messages if m["role"] == "system")
    user = next(m["content"] for m in messages if m["role"] == "user")
    return system, user, str(prompt.version)
