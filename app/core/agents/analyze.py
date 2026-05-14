"""
module4_analysis.py
===================
Analysis & Personalized Report Generation Agent

Responsibilities
----------------
1. Receive the full context (patient info, lab values) + PubMed abstracts.
2. Use DSPy to synthesize a personalized, evidence-based clinical report.
3. Output a rich Markdown report ready for rendering.

DSPy classes
------------
- ReportGenerationSignature
- ReportGenerationModule
- analysis_node  (LangGraph node)
"""


import json
from datetime import datetime

import dspy


# 1.  DSPy Signature & Module

class ReportGenerationSignature(dspy.Signature):
    """
    You are an expert clinical analyst. Using the patient's lab context and
    the provided PubMed research abstracts, generate a comprehensive,
    personalized medical report in Markdown format.

    The report MUST include:
    1. **Executive Summary** – 2-3 sentence plain-language overview.
    2. **Patient Profile** – demographics and report date.
    3. **Lab Results Analysis** – for each abnormal value:
       - Clinical significance
       - Possible causes (differential)
       - Evidence-based interpretation (cite the abstracts by PMID)
    4. **Normal Values** – brief confirmation of values within range.
    5. **Clinical Recommendations** – actionable next steps (lifestyle,
       follow-up tests, specialist referral) grounded in the literature.
    6. **Evidence Summary** – table: PMID | Title | Relevance.
    7. **Disclaimer** – AI-generated; not a substitute for physician advice.

    IMPORTANT:
    - Be personalized: use the patient's age/gender when discussing risk.
    - Cite PubMed abstracts inline using (PMID: XXXXXXXX).
    - Use clear, empathetic language understandable to a non-specialist.
    - Do NOT invent clinical facts not supported by the context or abstracts.
    """
    patient_context_json: str = dspy.InputField(
        desc="JSON with patient_info, report_type, and lab_values"
    )
    pubmed_abstracts_json: str = dspy.InputField(
        desc="JSON array of {pmid, title, abstract, authors, year}"
    )
    report_markdown: str = dspy.OutputField(
        desc="Full personalized Markdown clinical report"
    )


class ReportGenerationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought for richer, step-by-step reasoning
        self.generate = dspy.ChainOfThought(ReportGenerationSignature)

    def forward(
        self,
        patient_context_json: str,
        pubmed_abstracts_json: str,
    ) -> dspy.Prediction:
        return self.generate(
            patient_context_json=patient_context_json,
            pubmed_abstracts_json=pubmed_abstracts_json,
        )


# 2.  Helpers


def _build_patient_context_json(state: dict) -> str:
    return json.dumps(
        {
            "report_type":  state.get("report_type", "UNKNOWN"),
            "patient_info": state.get("patient_info", {}),
            "lab_values":   state.get("lab_values", []),
        },
        indent=2,
    )


def _build_abstracts_json(state: dict) -> str:
    abstracts = state.get("pubmed_abstracts", [])
    # Truncate abstracts to avoid hitting token limits
    trimmed = []
    for a in abstracts:
        trimmed.append(
            {
                "pmid":     a.get("pmid", ""),
                "title":    a.get("title", ""),
                "abstract": (a.get("abstract", "") or "")[:600],  # first 600 chars
                "authors":  a.get("authors", [])[:3],
                "year":     a.get("year", ""),
            }
        )
    return json.dumps(trimmed, indent=2)


def _add_report_header(report_md: str, state: dict) -> str:
    """Prepend a metadata banner to the generated report."""
    pi = state.get("patient_info", {})
    name = pi.get("patient_name", "Unknown Patient")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    header = (
        f"---\n"
        f"**Agentic Medical Insights** | Patient: {name} | "
        f"Generated: {generated_at}\n\n"
        f"---\n\n"
    )
    return header + report_md
