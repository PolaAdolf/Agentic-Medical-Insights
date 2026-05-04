"""
module1_ocr.py

Entry Layer — OCR

Pipeline role

  Input  : PDF or image bytes
  Output : structured Markdown string  →  passed to Module 2

DSPy signatures ():
  CleanOCRText     # noise removal, spacing, OCR-error correction
  DetectReportType # CBC / Blood Test / Vitamin / Unknown
  StructureReport  # final Markdown with tables, exact values
"""

import dspy

class CleanOCRText(dspy.Signature):
    """
    Fix OCR errors in raw medical lab text.
    Rules:
    - Preserve ALL numeric values, units, and reference ranges exactly.
    - Fix common OCR mistakes: broken lines, garbled symbols.
    - Do NOT add, remove, or change any medical value.
    - Output plain text only.
    """
    raw_text: str = dspy.InputField(
        desc="Raw OCR output from a medical lab report, may contain noise"
    )
    cleaned_text: str = dspy.OutputField(
        desc="Clean OCR text, remove noise, fix spacing, keep medical values, correct structure"
    )


class DetectReportType(dspy.Signature):
    """
    Identify the type of medical lab report from its text.
    Output exactly one label: CBC, Blood Test, Vitamin, Lipid Panel,
    Thyroid, Urinalysis, Metabolic Panel, or Unknown.
    Output ONLY the label — no explanation, no punctuation.
    """
    text: str = dspy.InputField(
        desc="Cleaned text of a medical lab report"
    )
    report_type: str = dspy.OutputField(
        desc="Detect report type (e.g., CBC, Blood Test, Vitamin, Unknown)"
    )


class StructureReport(dspy.Signature):
    """
    Convert cleaned medical lab text into a well-structured Markdown document.

    Requirements:
    - Top-level heading: report type + lab name if present.
    - ## Patient Information  — Markdown table (Name | Age | Gender | Date | Physician).
    - ## Lab Results          — Markdown table:
      | Analyte | Value | Unit | Reference Range | Flag |
      Flag must be one of: HIGH, LOW, NORMAL, UNKNOWN.
    - Preserve ALL numeric values and units exactly — never round or alter.
    - Do NOT invent values not present in the source text.
    """
    cleaned_text: str = dspy.InputField(
        desc="Cleaned OCR text of the lab report"
    )
    report_type: str = dspy.InputField(
        desc="Detected report type label (e.g. CBC, Blood Test, Vitamin)"
    )
    structured_md: str = dspy.OutputField(
        desc=(
            "Structured Markdown report with patient info table, "
            "lab values table (Analyte | Value | Unit | Reference Range | Flag), "
            "preserve all numeric values exactly"
        )
    )



class OCRPipeline(dspy.Module):
    """
    Three-stage module:
      Predict(CleanOCRText)        
      Predict(DetectReportType) 
      ChainOfThought(StructureReport) 
    """

    def __init__(self):
        super().__init__()
        self.clean     = dspy.Predict(CleanOCRText)
        self.detect    = dspy.Predict(DetectReportType)
        self.structure = dspy.ChainOfThought(StructureReport)

    def forward(self, raw_text):
        cleaned     = self.clean(raw_text=raw_text).cleaned_text
        report_type = self.detect(text=cleaned).report_type.strip()
        structured  = self.structure(
            cleaned_text=cleaned,
            report_type=report_type,
        )
        return dspy.Prediction(
            cleaned_text=cleaned,
            report_type=report_type,
            structured_md=structured.structured_md,
        )
