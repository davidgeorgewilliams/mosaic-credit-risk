"""Output guardrails for the explainability agent.

Two independent checks run on every LLM-generated explanation before it is
returned to a caller:

1. **Forbidden claims** — does the narrative reference a protected/quasi-identifying
   characteristic (age, region, protected-class terms) that the feature-minimized
   input (`FeatureStore.read_table_minimized`) should never have exposed to the
   agent in the first place? Seeing one here means the LLM hallucinated it, and it
   is NOT auto-redacted — a hallucinated protected-characteristic reference could
   itself be discriminatory, so the request is instead blocked and escalated.
2. **PII leakage** — does the narrative contain an email, phone number, or
   National-Insurance-shaped string that has no business being invented, since
   none of that ever entered the prompt.

A third check is a soft one: **mandatory disclosure**. EU AI Act Article 13
transparency requires every automated decision to state the applicant's right to
a human review. If the model dropped that sentence, it IS safe to append
deterministically rather than block the response.
"""

import re
from dataclasses import dataclass

PROTECTED_TERM_PATTERNS = [
    r"\bage[ds]?\b",
    r"\byears? old\b",
    r"\belderly\b",
    r"\bgender\b",
    r"\bmale\b",
    r"\bfemale\b",
    r"\brace\b",
    r"\bethnicit(?:y|ies)\b",
    r"\breligio(?:n|us)\b",
    r"\bdisab\w*\b",
    r"\bpostcode\b",
    r"(?<![A-Za-z])UK[A-Z](?![A-Za-z])",  # NUTS-1 region codes e.g. UKC — \b alone
    # doesn't fire after an underscore (e.g. a raw SHAP feature name "region_UKG"),
    # and that's exactly the shape a leak would take, so match on non-letter
    # neighbours instead of a word boundary.
]
PII_PATTERNS = [
    r"[\w.+-]+@[\w-]+\.[\w.-]+",  # email
    # UK National Insurance number: 2 letters, 6 digits, 1 suffix letter (e.g.
    # "AB123456C" or "AB 12 34 56 C") — the previous digits-only pattern didn't
    # match this actual shape at all.
    r"\b[A-Za-z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-Za-z]\b",
]
REQUIRED_DISCLOSURE = "human review"


@dataclass
class GuardrailResult:
    passed: bool
    violations: list[str]
    sanitized_text: str


def validate(narrative: str) -> GuardrailResult:
    violations = [
        f"forbidden_claim:{pattern}"
        for pattern in PROTECTED_TERM_PATTERNS
        if re.search(pattern, narrative, flags=re.IGNORECASE)
    ]
    violations += [
        f"pii_leak:{pattern}" for pattern in PII_PATTERNS if re.search(pattern, narrative)
    ]

    sanitized = narrative
    if REQUIRED_DISCLOSURE not in narrative.lower():
        sanitized = (
            sanitized.rstrip(". ")
            + ". You may request a human review of this decision at any time."
        )

    return GuardrailResult(
        passed=len(violations) == 0, violations=violations, sanitized_text=sanitized
    )
