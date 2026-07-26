"""Vector search adapter.

`InMemoryVectorSearch` stands in for Databricks Mosaic AI Vector Search (AI
Search): it indexes a small internal lending-policy corpus with TF-IDF vectors
(swap for a managed embedding model + a real Vector Search index against a Delta
table in production) and retrieves the clauses most relevant to a query built
from an applicant's top risk factors. The explainability agent uses this to
*ground* its narrative in actual policy text instead of letting the LLM invent a
justification — the retrieval step is what makes the explanation auditable.
"""

from dataclasses import dataclass
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LENDING_POLICY_CORPUS: list[tuple[str, str]] = [
    (
        "POLICY-DTI-01",
        "Applicants with a debt-to-income ratio above 0.45 are classified as high risk "
        "and require additional affordability checks before approval.",
    ),
    (
        "POLICY-DELINQ-02",
        "Two or more delinquencies within the last 24 months trigger mandatory manual "
        "underwriting review regardless of the automated risk score.",
    ),
    (
        "POLICY-EMP-03",
        "Unemployed applicants without a co-signer or guarantor are not eligible for "
        "unsecured credit products above 5,000 GBP.",
    ),
    (
        "POLICY-HIST-04",
        "Credit history shorter than 2 years results in a reduced maximum loan amount, "
        "capped at 3x verified annual income.",
    ),
    (
        "POLICY-INCOME-05",
        "Loan amounts exceeding 40% of stated annual income require a secondary income "
        "verification step before disbursement.",
    ),
    (
        "POLICY-APPEAL-06",
        "Applicants have the right to request a human review of any automated credit "
        "decision within 30 days, per Article 22 GDPR and internal appeals policy.",
    ),
]


@dataclass
class PolicyMatch:
    clause_id: str
    text: str
    score: float


class VectorSearchClient(Protocol):
    def search(self, query: str, k: int = 3) -> list[PolicyMatch]: ...


class InMemoryVectorSearch:
    def __init__(self, corpus: list[tuple[str, str]] = LENDING_POLICY_CORPUS) -> None:
        self._ids = [clause_id for clause_id, _ in corpus]
        self._texts = [text for _, text in corpus]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self._texts)

    def search(self, query: str, k: int = 3) -> list[PolicyMatch]:
        query_vector = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self._matrix)[0]
        ranked = sorted(
            zip(self._ids, self._texts, similarities, strict=True),
            key=lambda triple: triple[2],
            reverse=True,
        )
        return [
            PolicyMatch(clause_id=clause_id, text=text, score=float(score))
            for clause_id, text, score in ranked[:k]
            if score > 0
        ]
