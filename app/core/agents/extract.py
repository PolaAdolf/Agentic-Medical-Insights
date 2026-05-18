import dspy
from pydantic import BaseModel
from typing import List, Any
import json
from sympy import re

# =========================
# SCHEMA (ONLY STRUCTURE)
# =========================

class PersonalInfo(BaseModel):
    patient_name: str = "Not found"
    age: str = "Not found"
    gender: str = "Not found"
    report_date: str = "Not found"
    lab_name: str = "Not found"
    physician: str = "Not found"


class MedicalMetric(BaseModel):
    test_name: str
    value: str = "Not found"
    unit: str = "Not found"
    reference_range: str = "Not found"
    flag: str = "Unknown"  # HIGH, LOW, NORMAL, UNKNOWN
    category: str = ""  # e.g. CBC, Blood Test, Vitamin, etc.


class ExtractedContext(BaseModel):
    report_category: str = "Unknown"
    personal_info: PersonalInfo
    metrics: List[MedicalMetric]
    clinical_notes: list[str] 
    confidence: float = 0.0


# =========================
# DSPy SIGNATURE (EXTRACTION ONLY)
# =========================

class ContextExtractionSignature(dspy.Signature):
    """Extract structured data from a markdown lab report.

STRICT RULES:
- Return ONLY valid JSON
- No explanations
- No missing keys
- Extract ALL table rows
"""

    markdown_text: str = dspy.InputField(
         desc="Structured Markdown lab report from OCR pipeline"
    )

    report_category: str = dspy.OutputField(
        desc="One string like 'Blood Test Report', 'CBC', etc."
    )

    personal_info: dict = dspy.OutputField(
        desc="""Return EXACT JSON:
{
  "patient_name": "...",
  "age": "...",
  "gender": "...",
  "report_date": "...",
  "lab_name": "...",
  "physician": "..."
}"""
    )

    metrics: list = dspy.OutputField(
        desc="""Return LIST of JSON objects EXACTLY:
[
  {
    "test_name": "Section - Test Name",
    "value": "...",
    "unit": "...",
    "reference_range": "...",
    "flag": "...",
    "category": "..."
  }
]

Rules:
- Include section name prefix (e.g. 'CBC - Hemoglobin')
- Extract ALL rows from ALL tables
"""
    )
    clinical_notes: list = dspy.OutputField(
        desc='JSON array of strings — physician remarks and interpretations. [] if none.'
    )


# =========================
# SAFE HELPERS (NO MEDICAL LOGIC)
# =========================

def safe_str(x):
    if x is None:
        return "Not found"
    return str(x).strip() or "Not found"

def safe_json(text: any, fallback: Any = {}) -> Any:
    """Parse text as JSON, return empty dict on failure."""

    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return fallback
    
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Last resort: find first {...} or [...] block
    m = re.search(r'(\{.*\}|\[.*\])', cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
 
    return fallback

def clean_metric(m: dict) -> dict:
    return {
        "test_name": safe_str(m.get("test_name") or m.get("metric") or "Unknown"),
        "value": safe_str(m.get("value")),
        "unit": safe_str(m.get("unit")),
        "reference_range": safe_str(m.get("reference_range")),
        "flag": safe_str(m.get("flag")),
        "category": safe_str(m.get("category"))
    }


# =========================
# CONFIDENCE (PURE DATA QUALITY ONLY)
# =========================

def compute_confidence(personal: dict, metrics: list) -> float:
    total = 0
    filled = 0

    for v in personal.values():
        total += 1
        if v != "Not found":
            filled += 1

    for m in metrics:
        for v in m.values():
            total += 1
            if v != "Not found":
                filled += 1

    return round(filled / total, 2) if total else 0.0

# =========================
# Output formatting 
# =========================
def compress(full: dict) -> dict:
    """
    Token-optimised representation for passing to Modules 3 & 4.
    Keeps ALL fields needed downstream (reference_range was missing before).
    """
    pi = full["personal_info"]
    return {
        "rc": full["report_category"],
        "pi": {
            "name":   pi["patient_name"],
            "age":    pi["age"],
            "gender": pi["gender"],
        },
        "data": [
            {
                "t":  m["test_name"],
                "v":  m["value"],
                "u":  m["unit"],
                "rr": m["reference_range"],   
                "f":  m["flag"],
            }
            for m in full["metrics"]
        ],
        "notes": full["clinical_notes"],
        "c":    full["confidence"],
    }


def context_to_markdown(full: dict) -> str:
    """Render extracted context as a clean Markdown summary."""
    pi   = full.get("personal_info", {})
    lv   = full.get("metrics", [])
    rt   = full.get("report_category", "Unknown")
    cn   = full.get("clinical_notes", [])
    conf = full.get("confidence", 0.0)
 
    lines = [f"# Extracted Context: {rt}\n"]
 
    # Patient table
    lines += ["## Patient Information\n", "| Field | Value |", "|-------|-------|"]
    for key, val in pi.items():
        lines.append(f"| {key.replace('_',' ').title()} | {val or 'N/A'} |")
 
    # Lab results table
    lines += [
        "\n## Lab Results\n",
        "| Test | Value | Unit | Reference Range | Flag |",
        "|------|-------|------|-----------------|------|",
    ]
    FLAG_ICON = {"HIGH": "HIGH", "LOW": "LOW", "NORMAL": "NORMAL"}
    for m in lv:
        flag = FLAG_ICON.get(m.get("flag", ""), f"{m.get('flag','?')}")
        lines.append(
            f"| {m.get('test_name','')} | {m.get('value','')} | "
            f"{m.get('unit','')} | {m.get('reference_range','')} | {flag} |"
        )
 
    # Clinical notes
    if cn:
        lines += ["\n## Clinical Notes\n"]
        for note in cn:
            lines.append(f"- {note}")
 
    lines.append(f"\n_Extraction confidence: {conf:.0%}_")
    return "\n".join(lines)

# =========================
# MODULE 2 CORE (EXTRACTION ONLY)
# =========================

class ContextExtractionModule(dspy.Module):

    def __init__(self):
        super().__init__()
        self.model = dspy.Predict(ContextExtractionSignature)

    def forward(self, markdown_text: str):

        if not markdown_text or not markdown_text.strip():
            return self.empty()

        raw = self.model(markdown_text=markdown_text)

        # -------------------------
        # PERSONAL INFO SAFE PARSE
        # -------------------------
        pi = raw.personal_info if isinstance(raw.personal_info, dict) else {}

        personal = PersonalInfo(
            patient_name=safe_str(pi.get("patient_name") or pi.get("name")),
            age=safe_str(pi.get("age")),
            gender=safe_str(pi.get("gender")),
            report_date=safe_str(pi.get("report_date")),
            lab_name=safe_str(pi.get("lab_name")),
            physician=safe_str(pi.get("physician"))
        )
        
        # -------------------------
        # METRICS CLEANING ONLY
        # -------------------------
        raw_metrics = safe_json(raw.metrics, fallback=[])
        metrics: list[dict] = []
        if isinstance(raw.metrics, list):
            for m in raw.metrics:
                if isinstance(m, dict):
                    metrics.append(clean_metric(m))

        # -------------------------
        # Clinical Notes 
        # -------------------------
        raw_notes = safe_json(raw.clinical_notes, [])
        notes: list[str] = []
        if isinstance(raw_notes, list):
            notes = [safe_str(n) for n in raw_notes if n]
        elif isinstance(raw_notes, str) and raw_notes.strip():
            notes = [raw_notes.strip()]
        # -------------------------
        # WRAP OUTPUT
        # -------------------------
        ctx = ExtractedContext(
            report_category=safe_str(raw.report_category),
            personal_info=personal,
            metrics=[MedicalMetric(**m) for m in metrics],
            clinical_notes=notes,
            confidence=compute_confidence(personal.model_dump(), metrics)
        )

        full = ctx.model_dump()
        return {
            "full": full, 
            "compressed": compress(full)
        }

    def empty(self):
        empty_ctx = ExtractedContext(
            personal_info=PersonalInfo()
        )
        full = empty_ctx.model_dump()
        return {
            "full": full, 
            "compressed": compress(full)
        }