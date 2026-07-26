"""AI Gateway adapter.

Mosaic AI Gateway sits in front of every model-serving endpoint and external LLM
provider, giving unified rate limiting, usage/cost tracking, and payload logging
regardless of which model actually serves a request. `InMemoryAIGateway`
reproduces that governance surface locally: `route()` enforces
`ai_gateway_rate_limit_per_minute` and `ai_gateway_token_budget_per_day` before a
call is allowed through, and every call lands in `usage_log`, the local
equivalent of the inference tables Databricks writes usage to in production.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from credit_risk.adapters.llm_client import LLMResponse
from credit_risk.config import settings

# Illustrative pricing, per 1K tokens — not tied to any specific model's real rate card.
COST_PER_1K_INPUT_TOKENS = 0.00015
COST_PER_1K_OUTPUT_TOKENS = 0.0006


class RateLimitExceeded(Exception):
    pass


class TokenBudgetExceeded(Exception):
    pass


@dataclass
class UsageRecord:
    route: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float


class AIGatewayClient(Protocol):
    def route(self, route_name: str, call_fn) -> LLMResponse: ...

    def usage_report(self) -> dict: ...


class InMemoryAIGateway:
    def __init__(self) -> None:
        self._recent_call_times: deque[float] = deque()
        self.usage_log: list[UsageRecord] = []

    def _enforce_rate_limit(self) -> None:
        now = time.monotonic()
        window_start = now - 60
        while self._recent_call_times and self._recent_call_times[0] < window_start:
            self._recent_call_times.popleft()
        if len(self._recent_call_times) >= settings.ai_gateway_rate_limit_per_minute:
            raise RateLimitExceeded(
                f"AI Gateway rate limit exceeded: "
                f"{settings.ai_gateway_rate_limit_per_minute} calls/min"
            )
        self._recent_call_times.append(now)

    def _enforce_token_budget(self) -> None:
        used_today = sum(r.input_tokens + r.output_tokens for r in self.usage_log)
        if used_today >= settings.ai_gateway_token_budget_per_day:
            raise TokenBudgetExceeded(
                f"AI Gateway daily token budget exceeded: "
                f"{settings.ai_gateway_token_budget_per_day} tokens"
            )

    def route(self, route_name: str, call_fn) -> LLMResponse:
        """Governs a single LLM call: rate limit, then budget, then execute and record."""
        self._enforce_rate_limit()
        self._enforce_token_budget()

        response: LLMResponse = call_fn()

        cost = (
            response.input_tokens / 1000 * COST_PER_1K_INPUT_TOKENS
            + response.output_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS
        )
        self.usage_log.append(
            UsageRecord(
                route=route_name,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=cost,
                timestamp=time.time(),
            )
        )
        return response

    def usage_report(self) -> dict:
        return {
            "calls": len(self.usage_log),
            "total_input_tokens": sum(r.input_tokens for r in self.usage_log),
            "total_output_tokens": sum(r.output_tokens for r in self.usage_log),
            "estimated_cost_usd": round(sum(r.cost_usd for r in self.usage_log), 6),
        }
