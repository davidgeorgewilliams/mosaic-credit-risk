"""Loan-officer UI over the FastAPI backend.

Stands in for a Databricks App — same shape (a lightweight Python web UI calling
a governed backend endpoint, deployed alongside the rest of the workspace) using
Streamlit locally instead of Databricks' hosted app runtime. Run the API first
(`make serve`), then `make app`.
"""

import os

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("CREDIT_RISK_API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Credit Risk Decisioning", layout="centered")
st.title("Credit Risk Decisioning — Loan Officer View")
st.caption(
    "Demo UI over the credit-risk FastAPI service. Backed by a champion/shadow "
    "model pair on a Mosaic-AI-style serving layer and an LLM explainability "
    "agent behind an AI Gateway."
)

with st.form("applicant_form"):
    col1, col2 = st.columns(2)
    with col1:
        applicant_id = st.text_input("Applicant ID", value="APP-DEMO-001")
        income_band = st.number_input("Annual income (GBP)", min_value=0, value=32000, step=1000)
        debt_to_income_ratio = st.slider("Debt-to-income ratio", 0.0, 1.2, 0.35)
        credit_history_years = st.number_input("Credit history (years)", min_value=0.0, value=4.0)
        num_delinquencies_last_2y = st.number_input(
            "Delinquencies (last 2y)", min_value=0, value=0, step=1
        )
    with col2:
        loan_amount_requested = st.number_input(
            "Loan amount requested (GBP)", min_value=0, value=8000, step=500
        )
        employment_status = st.selectbox(
            "Employment status", ["employed", "self_employed", "unemployed", "retired"]
        )
        age_band = st.selectbox("Age band", ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"])
        region = st.selectbox("Region (NUTS-1)", ["UKC", "UKD", "UKE", "UKF", "UKG", "UKH", "UKI"])

    submitted = st.form_submit_button("Score application")

if submitted:
    applicant = {
        "applicant_id": applicant_id,
        "income_band": income_band,
        "debt_to_income_ratio": debt_to_income_ratio,
        "credit_history_years": credit_history_years,
        "num_delinquencies_last_2y": num_delinquencies_last_2y,
        "loan_amount_requested": loan_amount_requested,
        "employment_status": employment_status,
        "age_band": age_band,
        "region": region,
    }
    try:
        with httpx.Client(base_url=API_BASE_URL, timeout=30) as client:
            score_response = client.post("/score", json=applicant)
            score_response.raise_for_status()
            score = score_response.json()

            st.subheader("Automated decision")
            decision_color = {"approve": "green", "deny": "red", "refer_to_human": "orange"}
            color = decision_color.get(score["decision"], "blue")
            st.markdown(
                f"**Decision:** :{color}[{score['decision'].upper()}]  \n"
                f"**Probability of default:** {score['probability_of_default']:.1%}  \n"
                f"**Model:** {score['model_name']} v{score['model_version']} "
                f"({score['score_source']} on `{score['endpoint_name']}`)"
            )

            explain_response = client.post("/explain", json=applicant)
            explain_response.raise_for_status()
            explanation = explain_response.json()

            st.subheader("Explanation")
            st.write(explanation["narrative"])
            if explanation["human_review_required"]:
                st.warning("This decision requires human oversight before it is finalised.")

            clauses = ", ".join(explanation["cited_policy_clauses"]) or "none"
            st.caption(
                f"Cited policy clauses: {clauses} · "
                f"Prompt `{explanation['prompt_name']}` v{explanation['prompt_version']} · "
                f"{explanation['input_tokens']}+{explanation['output_tokens']} tokens"
            )

            with st.expander("Top SHAP factors"):
                for factor in explanation["top_factors"]:
                    direction = factor["direction"].replace("_", " ")
                    st.write(f"- **{factor['feature']}** {direction} ({factor['shap_value']:+.3f})")

    except httpx.ConnectError:
        st.error(f"Could not reach the API at {API_BASE_URL}. Run `make serve` first.")

st.divider()
if st.button("Show governance audit log"):
    try:
        with httpx.Client(base_url=API_BASE_URL, timeout=30) as client:
            audit = client.get("/governance/audit").json()
            classification = client.get("/governance/classification").json()
            usage = client.get("/governance/ai_gateway_usage").json()
        st.write("**EU AI Act classification:**", classification)
        st.write("**AI Gateway usage:**", usage)
        st.write("**Audit events:**")
        st.json(audit)
    except httpx.ConnectError:
        st.error(f"Could not reach the API at {API_BASE_URL}.")
