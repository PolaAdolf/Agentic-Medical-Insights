"""
module5_evaluation.py
=====================
Self-Evaluation Agent

Responsibilities
----------------
1. Receive the final report Markdown.
2. Score it on multiple criteria using a DSPy module.
3. Optionally trigger a re-generation if quality is below threshold.
4. Return the (possibly improved) report + quality scores.

DSPy classes
------------
- EvaluationSignature
- EvaluationModule
- evaluation_node  (LangGraph node)
"""

import json

import dspy


# 1.  DSPy Signature & Module

class EvaluationSignature(dspy.Signature):
    """
    You are a senior medical AI quality reviewer.
    Evaluate the given medical report on five criteria, each scored 1–10:

    1. accuracy        – Are all stated values consistent with the lab context?
    2. completeness    – Are all abnormal values addressed?
    3. evidence_use    – Are PubMed abstracts cited and relevant?
    4. clarity         – Is the language clear for a non-specialist patient?
    5. safety          – Does the report include appropriate disclaimers?
                         Does it avoid dangerous unsupported claims?

    Return ONLY a JSON object:
    {
      "accuracy": <int 1-10>,
      "completeness": <int 1-10>,
      "evidence_use": <int 1-10>,
      "clarity": <int 1-10>,
      "safety": <int 1-10>,
      "overall": <float, average>,
      "feedback": "<one paragraph of specific improvement suggestions>"
    }
    """
    report_markdown: str = dspy.InputField(desc="The full Markdown medical report to evaluate")
    lab_context_json: str = dspy.InputField(desc="Original lab values JSON for accuracy checking")
    evaluation_json: str = dspy.OutputField(desc="JSON evaluation scores and feedback")


class EvaluationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.Predict(EvaluationSignature)

    def forward(self, report_markdown: str, lab_context_json: str) -> dspy.Prediction:
        return self.evaluate(
            report_markdown=report_markdown,
            lab_context_json=lab_context_json,
        )


# 2.  Helpers

QUALITY_THRESHOLD = 7.0  # Re-generate if overall score < this


def _parse_eval(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "accuracy": 5, "completeness": 5, "evidence_use": 5,
            "clarity": 5, "safety": 5, "overall": 5.0,
            "feedback": "Could not parse evaluation output.",
        }


def _scores_to_markdown(scores: dict) -> str:
    criteria = ["accuracy", "completeness", "evidence_use", "clarity", "safety"]
    lines = [
        "\n\n---\n## Quality Evaluation\n",
        "| Criterion | Score |",
        "|-----------|-------|",
    ]
    for c in criteria:
        bar = "█" * int(scores.get(c, 0)) + "░" * (10 - int(scores.get(c, 0)))
        lines.append(f"| {c.replace('_', ' ').title()} | {bar} {scores.get(c,0)}/10 |")
    lines.append(f"\n**Overall Score:** {scores.get('overall', 0):.1f} / 10")
    lines.append(f"\n**Reviewer Feedback:** {scores.get('feedback', '')}")
    return "\n".join(lines)
