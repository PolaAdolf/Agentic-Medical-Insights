"""
module3_search.py
=================
PubMed Search Agent

Responsibilities
----------------
1. Receive extracted context (report_type, patient_info, lab_values).
2. Generate targeted PubMed search queries via DSPy.
3. Execute queries against the PubMed API (via pubmed_api.py utility).
4. Return a list of relevant abstract dicts.

DSPy classes
------------
- QueryGenerationSignature
- QueryGenerationModule
- search_node  (LangGraph node)
"""

import json
import dspy


# 1.  DSPy Signature & Module

class QueryGenerationSignature(dspy.Signature):
    """
    You are a medical research assistant.
    Given a patient's lab context (report type, abnormal values, demographics),
    generate 3–5 specific PubMed search queries that will retrieve relevant,
    high-quality clinical literature to help interpret these results.

    Rules:
    - Use MeSH terms and standard medical terminology where possible.
    - Focus on ABNORMAL values (flagged HIGH or LOW).
    - Include patient demographics (age, gender) when clinically relevant.
    - Each query should target a different clinical angle
      (e.g., differential diagnosis, treatment, prognosis, guidelines).
    - Output a JSON array of query strings. ONLY output the JSON array.

    Example output:
    ["anemia iron deficiency CBC diagnosis adults", "low hemoglobin causes treatment guidelines"]
    """
    context_summary: str = dspy.InputField(
        desc="JSON summary of patient demographics and abnormal lab values"
    )
    queries_json: str = dspy.OutputField(
        desc="JSON array of PubMed search query strings"
    )


class QueryGenerationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(QueryGenerationSignature)

    def forward(self, context_summary: str) -> dspy.Prediction:
        return self.generate(context_summary=context_summary)


# 2.  Helpers

def _build_context_summary(state: dict) -> str:
    """Serialize relevant state fields into a compact JSON string."""
    patient = state.get("patient_info", {})
    abnormal = [
        v for v in state.get("lab_values", [])
        if v.get("flag") in ("HIGH", "LOW")
    ]
    summary = {
        "report_type": state.get("report_type", "UNKNOWN"),
        "patient": {
            "age":    patient.get("age"),
            "gender": patient.get("gender"),
        },
        "abnormal_values": abnormal,
    }
    return json.dumps(summary, indent=2)


def _safe_parse_queries(text: str) -> list[str]:
    """Parse JSON array of queries from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(q) for q in result]
    except json.JSONDecodeError:
        pass
    # Fallback: extract lines that look like queries
    return [line.strip().strip('"').strip("'")
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("[")]


