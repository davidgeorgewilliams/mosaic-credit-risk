from credit_risk.agent import prompts


def test_render_fills_all_sections():
    system_prompt, user_prompt, version = prompts.render(
        decision="deny",
        probability="72%",
        top_factors="- debt_to_income_ratio increases risk",
        policy_clauses="- Applicants with high DTI require additional checks.",
    )

    assert "lending assistant" in system_prompt.lower()
    assert "DECISION: deny" in user_prompt
    assert "PROBABILITY: 72%" in user_prompt
    assert "debt_to_income_ratio increases risk" in user_prompt
    assert version


def test_render_is_stable_across_calls():
    """Registration should be idempotent for the demo — repeated calls reuse the
    same `production` alias rather than minting a new version each time."""
    _, _, version_1 = prompts.render(
        decision="approve", probability="10%", top_factors="- none", policy_clauses="- none"
    )
    _, _, version_2 = prompts.render(
        decision="deny", probability="80%", top_factors="- none", policy_clauses="- none"
    )
    assert version_1 == version_2
