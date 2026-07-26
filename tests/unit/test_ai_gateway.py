import pytest

from credit_risk.adapters.ai_gateway import (
    InMemoryAIGateway,
    RateLimitExceeded,
    TokenBudgetExceeded,
)
from credit_risk.adapters.llm_client import LLMResponse
from credit_risk.config import settings


def _response(input_tokens: int = 10, output_tokens: int = 10) -> LLMResponse:
    return LLMResponse(
        text="ok", input_tokens=input_tokens, output_tokens=output_tokens, model="fake"
    )


def test_rate_limit_enforced(monkeypatch):
    monkeypatch.setattr(settings, "ai_gateway_rate_limit_per_minute", 2)
    gateway = InMemoryAIGateway()

    gateway.route("explain", lambda: _response())
    gateway.route("explain", lambda: _response())
    with pytest.raises(RateLimitExceeded):
        gateway.route("explain", lambda: _response())


def test_token_budget_enforced(monkeypatch):
    monkeypatch.setattr(settings, "ai_gateway_token_budget_per_day", 15)
    gateway = InMemoryAIGateway()

    gateway.route("explain", lambda: _response(input_tokens=10, output_tokens=5))
    with pytest.raises(TokenBudgetExceeded):
        gateway.route("explain", lambda: _response(input_tokens=1, output_tokens=1))


def test_usage_report_aggregates_cost_and_tokens():
    gateway = InMemoryAIGateway()
    gateway.route("explain", lambda: _response(input_tokens=100, output_tokens=50))

    report = gateway.usage_report()
    assert report["calls"] == 1
    assert report["total_input_tokens"] == 100
    assert report["total_output_tokens"] == 50
    assert report["estimated_cost_usd"] > 0
