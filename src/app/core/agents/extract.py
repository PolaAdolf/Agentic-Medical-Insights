import os
import dspy
from pydantic import BaseModel
from typing import List, Dict, Any

# =========================
# SCHEMA (ONLY STRUCTURE)
# =========================

class PersonalInfo(BaseModel):
    patient_name: str = "Not found"
    age: str = "Not found"
    gender: str = "Not found"
    report_date: str = "Not found"


class MedicalMetric(BaseModel):
    test_name: str
    value: str = "Not found"
    unit: str = "Not found"
    reference_range: str = "Not found"


class ExtractedContext(BaseModel):
    report_category: str = "Unknown"
    personal_info: PersonalInfo
    metrics: List[MedicalMetric]
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

    markdown_text: str = dspy.InputField()

    report_category: str = dspy.OutputField(
        desc="One string like 'Blood Test Report', 'CBC', etc."
    )

    personal_info: dict = dspy.OutputField(
        desc="""Return EXACT JSON:
{
  "patient_name": "...",
  "age": "...",
  "gender": "...",
  "report_date": "..."
}"""
    )

    metrics: list = dspy.OutputField(
        desc="""Return LIST of JSON objects EXACTLY:
[
  {
    "test_name": "Section - Test Name",
    "value": "...",
    "unit": "...",
    "reference_range": "..."
  }
]

Rules:
- Include section name prefix (e.g. 'CBC - Hemoglobin')
- Extract ALL rows from ALL tables
"""
    )


# =========================
# SAFE HELPERS (NO MEDICAL LOGIC)
# =========================

def safe_str(x):
    if x is None:
        return "Not found"
    return str(x)


def clean_metric(m: dict) -> dict:
    return {
        "test_name": safe_str(m.get("test_name") or m.get("metric") or "Unknown"),
        "value": safe_str(m.get("value")),
        "unit": safe_str(m.get("unit")),
        "reference_range": safe_str(m.get("reference_range")),
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
# MODULE 2 CORE (EXTRACTION ONLY)
# =========================

class ContextExtractionModule(dspy.Module):

    def __init__(self):
        super().__init__()
        self.model = dspy.Predict(ContextExtractionSignature)

    def forward(self, markdown_text: str):

        if not markdown_text:
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
        )
        
        # -------------------------
        # METRICS CLEANING ONLY
        # -------------------------
        metrics = []
        if isinstance(raw.metrics, list):
            for m in raw.metrics:
                if isinstance(m, dict):
                    metrics.append(clean_metric(m))

        # -------------------------
        # WRAP OUTPUT
        # -------------------------
        ctx = ExtractedContext(
            report_category=safe_str(raw.report_category),
            personal_info=personal,
            metrics=[MedicalMetric(**m) for m in metrics],
            confidence=compute_confidence(personal.model_dump(), metrics)
        )

        full = ctx.model_dump()

        # -------------------------
        # TOKEN-OPTIMIZED OUTPUT
        # -------------------------
        compressed = {
            "rc": full["report_category"],
            "data": [
                {
                    "t": m["test_name"],
                    "v": m["value"]
                }
                for m in full["metrics"]
            ],
            "meta": {
                "age": full["personal_info"]["age"],
                "gender": full["personal_info"]["gender"]
            },
            "c": full["confidence"]
        }

        return {
            "full": full,
            "compressed": compressed
        }

    def empty(self):
        return {
            "full": ExtractedContext(
                personal_info=PersonalInfo(),
                metrics=[],
                confidence=0.0
            ).model_dump(),
            "compressed": {
                "rc": "Unknown",
                "data": [],
                "meta": {"age": "Not found", "gender": "Not found"},
                "c": 0.0
            }
        }


# =========================
# LANGGRAPH NODE
# =========================

def extraction_node(state: Dict[str, Any]):

    if not dspy.settings.lm:
        dspy.settings.configure(
            lm=dspy.OpenAI(
                model="openrouter/google/gemma-4-31b-it",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                api_base="https://openrouter.ai/api/v1",
                temperature=0
            )
        )

    module = ContextExtractionModule()

    return {
        "module_2_output": module(state.get("md_file_content", ""))
    }


# =========================
# SIMPLE TEST
# =========================


if __name__ == "__main__":

    test = {
        "md_file_content": """
        ### *Patient Information*
*   *Name:* Yashvi M. Patel
*   *Age:* 21 Years
*   *Sex:* Female
*   *UHID:* 556
*   *Registered on:* 02:31 PM 02 Dec, 2X
*   *Collected on:* 03:11 PM 02 Dec, 2X
*   *Reported on:* 04:32 PM 02 Dec, 2X
*   *Sample Collected At:* 123, Shiram Complex, Ahmedabad, Mumbai
*   *Sample Collected By:* Mr. Suresh
*   *Ref. By:* Dr. Hiren Shah

---

### *Complete Blood Count (CBC)*
| Investigation | Result | Status | Reference Value | Unit |
| :--- | :--- | :--- | :--- | :--- |
| *Hemoglobin (Hb)* | 13.00 | Normal | 12.00 - 15.00 | g/dL |
| *Total RBC count* | 4.80 | Normal | 3.80 - 4.80 | mill/cumm |
| *BLOOD INDICES* | | | | |
| Packed Cell Volume (PCV) | 40 | Normal | 36 - 46 | % |
| Mean Corpuscular Volume (MCV) | 88 | Normal | 83 - 101 | fL |
| MCH | 28 | Normal | 27 - 32 | pg |
| MCHC | 32.50 | Normal | 31.50 - 34.50 | g/dL |
| RDW | 13.50 | Normal | 11.60 - 14.00 | % |
| *Total WBC count* | 6000 | Normal | 4000 - 11000 | cumm |
| *DIFFERENTIAL WBC COUNT* | | | | |
| Neutrophils | 60 | Normal | 50 - 70 | % |
| Lymphocytes | 30 | Normal | 20 - 40 | % |
| Eosinophils | 02 | Normal | 00 - 06 | % |
| Monocytes | 05 | Normal | 02 - 10 | % |
| Basophils | 01 | Normal | 00 - 02 | % |
| *Platelet Count* | 250000 | Normal | 150000 - 410000 | cumm |

---

*   *Instruments:* Fully automated cell counter - Mindray 300
*   *Interpretation:* Further perform for Anemia.
*   *Generated on:* 02 Dec, 202X 05:22 PM

*Signatories:*
*   Medical Lab Technician (DMLT, BMLT)
*   Dr. Payal Shah (MD, Pathologist)
*   Dr. Vimal Shah (MD, Pathologist)
        """
    }

    import json
    print(json.dumps(extraction_node(test), indent=2))