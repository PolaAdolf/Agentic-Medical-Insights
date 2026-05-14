"""
routes.py
=========
API Endpoint Definitions
"""

import json
import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from app.core.pipeline import visualize_graph , MedicalInsightsPipeline, AgentState

router = APIRouter()

OUTPUT_DIR = Path(os.getenv("REPORT_OUTPUT_DIR", "./outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}

# POST /analyze  – main endpoint
@router.post("/analyze")
async def analyze_report(
    file: UploadFile = File(...)
):

    ext = Path(file.filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail="Unsupported file type",
        )

    file_bytes = await file.read()

    # RUN PIPELINE
    pipeline = MedicalInsightsPipeline()

    initial_state = {
        "file_bytes": file_bytes,
        "filename": file.filename,
        "steps": [],
    }

    final_state = await pipeline.graph.ainvoke(
        initial_state
    )

    # REPORT
    report_md = final_state.get(
        "final_report_markdown",
        ""
    )

    report_path = (
        OUTPUT_DIR /
        f"{Path(file.filename).stem}.md"
    )

    report_path.write_text(
        report_md,
        encoding="utf-8",
    )

    return JSONResponse(
        content={
            "report_type":
                final_state.get("report_type"),

            "patient_info":
                final_state.get("patient_info"),

            "lab_values":
                final_state.get("lab_values"),

            "search_queries":
                final_state.get("search_queries"),

            "pubmed_abstracts":
                final_state.get("pubmed_abstracts"),

            "final_report_markdown":
                report_md,

            "quality_scores":
                final_state.get("quality_scores"),

            "evaluation_passed":
                final_state.get(
                    "evaluation_passed"
                ),

            "ocr_markdown":
                final_state.get(
                    "ocr_markdown"
                ),
        }
    )

# GET /report/{filename}  – retrieve a saved report
@router.get("/report/{filename}", summary="Retrieve a saved Markdown report")
async def get_report(filename: str):
    report_path = OUTPUT_DIR / filename
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return PlainTextResponse(content=report_path.read_text(encoding="utf-8"))
