from credit_risk.agent import guardrails


def test_clean_narrative_passes():
    result = guardrails.validate(
        "This application was denied due to a high debt-to-income ratio. "
        "You may request a human review of this decision at any time."
    )
    assert result.passed
    assert result.violations == []


def test_missing_disclosure_is_appended_not_blocked():
    result = guardrails.validate("This application was approved.")
    assert result.passed
    assert "human review" in result.sanitized_text.lower()


def test_protected_term_blocks():
    result = guardrails.validate(
        "This application was denied because the applicant is elderly. "
        "You may request a human review of this decision at any time."
    )
    assert not result.passed
    assert any(v.startswith("forbidden_claim") for v in result.violations)


def test_pii_email_blocks():
    result = guardrails.validate(
        "Contact the applicant at jane.doe@example.com for a human review."
    )
    assert not result.passed
    assert any(v.startswith("pii_leak") for v in result.violations)


def test_underscored_region_code_is_caught():
    """Regression test: a raw SHAP feature name like 'region_UKG' leaking into a
    narrative must be caught even though 'UKG' is preceded by an underscore, not
    whitespace — a naive \\b-based regex misses this."""
    result = guardrails.validate(
        "Risk was driven primarily by region_UKG and a high loan amount. "
        "You may request a human review of this decision at any time."
    )
    assert not result.passed
    assert any("UK" in v for v in result.violations)
