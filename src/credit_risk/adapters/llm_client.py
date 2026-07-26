"""LLM client abstraction.

Two implementations behind one `Protocol`, selected in `get_default_llm_client()`:
`FakeLLMClient` is a deterministic, fully offline double used by default (and in
CI/tests, so the suite never needs network access or an API key). `OpenAILLMClient`
calls a real OpenAI-compatible endpoint — in production this would target Azure
OpenAI Service — when `CREDIT_RISK_OPENAI_API_KEY` is set. This mirrors how the
Mosaic AI Agent Framework lets you swap the underlying foundation model behind a
stable interface without touching agent logic.
"""

import re
from dataclasses import dataclass
from typing import Protocol

from credit_risk.config import settings


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse: ...


def _approx_token_count(text: str) -> int:
    # Rough offline approximation (~4 chars/token); the real client reports exact usage.
    return max(1, len(text) // 4)


def _extract_field(text: str, key: str) -> str:
    match = re.search(rf"{key}:\s*(.+)", text)
    return match.group(1).strip() if match else ""


def _extract_bulleted_list(text: str, key: str) -> list[str]:
    block = re.search(rf"{key}:\n((?:- .+\n?)+)", text)
    if not block:
        return []
    return [line[2:].strip() for line in block.group(1).splitlines() if line.startswith("- ")]


class FakeLLMClient:
    """Deterministic template-based double. Understands the fixed section headers
    emitted by `credit_risk.agent.prompts` (DECISION:, PROBABILITY:, TOP_FACTORS:,
    POLICY_CLAUSES:) — no network, no API key, identical output every run, which is
    what makes the guardrail and integration tests reliable."""

    model = "fake-offline-model-v1"

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        decision = _extract_field(user_prompt, "DECISION") or "reviewed"
        probability = _extract_field(user_prompt, "PROBABILITY") or "an estimated risk level"
        factors = _extract_bulleted_list(user_prompt, "TOP_FACTORS")
        clauses = _extract_bulleted_list(user_prompt, "POLICY_CLAUSES")

        factor_sentence = "; ".join(factors[:3]) if factors else "no single dominant factor"
        clause_sentence = " ".join(clauses[:2])

        text = (
            f"This application was {decision.lower()}, with an estimated default "
            f"probability of {probability}. The main contributing factors were: "
            f"{factor_sentence}. {clause_sentence} "
            "You may request a human review of this decision at any time."
        ).strip()

        return LLMResponse(
            text=text,
            input_tokens=_approx_token_count(system_prompt + user_prompt),
            output_tokens=_approx_token_count(text),
            model=self.model,
        )


class OpenAILLMClient:
    """Real provider adapter, used only when an API key is configured. Targets any
    OpenAI-compatible chat completions endpoint, so the same client code points at
    Azure OpenAI Service in production without modification — only the base URL
    and auth change."""

    def __init__(self, api_key: str, model: str | None = None) -> None:
        from openai import OpenAI  # imported lazily so `openai` stays an optional dep

        self._client = OpenAI(api_key=api_key)
        self.model = model or settings.llm_model_name

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=self.model,
        )


def get_default_llm_client() -> LLMClient:
    if settings.openai_api_key:
        return OpenAILLMClient(api_key=settings.openai_api_key)
    return FakeLLMClient()
