"""
streamlit_app.py
================
Agentic Medical Insights # Streamlit Frontend

Run with:
    streamlit run app/ui/streamlit_app.py
"""

import io
import json
import os
import time
from pathlib import Path

import requests
import streamlit as st

# Page config 
st.set_page_config(
    page_title="Agentic Medical Insights",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS 
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Space Mono', monospace;
    }
    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00c6a7 0%, #0066ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .stage-badge {
        display: inline-block;
        background: #e0f7f3;
        color: #00796b;
        border-radius: 12px;
        padding: 2px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .flag-high { color: #dc2626; font-weight: 700; }
    .flag-low  { color: #2563eb; font-weight: 700; }
    .flag-norm { color: #16a34a; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

#  Sidebar
with st.sidebar:
    st.markdown("## Settings")
    api_url = st.text_input("API Base URL", value=API_BASE)
    st.markdown("---")
    st.markdown("### Pipeline Stages")
    stages = [
        "1️⃣  OCR & Text Extraction",
        "2️⃣  Context Extraction",
        "3️⃣  PubMed Search",
        "4️⃣  Report Generation",
        "5️⃣  Quality Evaluation",
    ]
    for s in stages:
        st.markdown(f'<span class="stage-badge">{s}</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Powered by**")
    st.markdown("• Gemma 4 via OpenRouter\n• DSPy\n• LangGraph\n• PubMed NCBI\n• FastAPI + Streamlit")

# Hero 
st.markdown('<div class="hero-title">🧬 Agentic Medical Insights</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Upload a CBC lab report (PDF or image) → '
    'get a personalized, evidence-based clinical analysis in seconds.</div>',
    unsafe_allow_html=True,
)

# Upload
col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Drop your lab report here",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
        help="Supports PDF and common image formats.",
    )

with col_info:
    st.info(
        "**Supported reports:**\n"
        "- CBC (Complete Blood Count) ✅\n"
        "- Other panels (experimental)\n\n"
        "Your data is processed locally and not stored externally."
    )

# Analyze Button
if uploaded:
    st.success(f"File loaded: **{uploaded.name}** ({uploaded.size / 1024:.1f} KB)")

    if st.button(" Analyze Report", type="primary", use_container_width=True):

        progress = st.progress(0, text="Initialising pipeline…")
        status   = st.empty()

        stage_messages = [
            (0.15, "🔍 Running OCR and extracting text…"),
            (0.35, "🧩 Extracting structured context…"),
            (0.55, "📚 Searching PubMed for evidence…"),
            (0.75, "📝 Generating personalized report…"),
            (0.90, "✅ Evaluating report quality…"),
        ]

        # Simulate progress while waiting for API
        with st.spinner(""):
            for pct, msg in stage_messages:
                progress.progress(pct, text=msg)
                time.sleep(0.4)

            try:
                resp = requests.post(
                    f"{api_url}/analyze",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=1800,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    " Cannot connect to the FastAPI backend. "
                    "Make sure it's running: `uvicorn app.main:app --reload`"
                )
                st.stop()
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e.response.text}")
                st.stop()

        progress.progress(1.0, text="✨ Analysis complete!")
        time.sleep(0.5)
        progress.empty()

        # Results
        st.markdown("---")
        st.markdown("## 📊 Results")

        # Quality badge
        scores  = data.get("quality_scores", {})
        overall = scores.get("overall", 0)
        passed  = data.get("evaluation_passed", False)
        badge   = "🟢 Passed" if passed else "🟡 Review Recommended"
        st.markdown(f"**Quality Score:** `{overall:.1f}/10`  {badge}")

        # Tabs
        tab_report, tab_labs, tab_pubmed, tab_raw = st.tabs(
            ["📄 Full Report", "🔬 Lab Values", "📚 PubMed Evidence", "🛠 Raw JSON"]
        )

        # Tab 1 – Full Report
        with tab_report:
            report_md = data.get("final_report_markdown", "_No report generated._")
            st.markdown(report_md)
            st.download_button(
                "⬇️ Download Report (.md)",
                data=report_md,
                file_name="medical_insights_report.md",
                mime="text/markdown",
            )

        # Tab 2 – Lab Values
        with tab_labs:
            lab_values = data.get("lab_values", [])
            if lab_values:
                st.markdown(f"**Report Type:** `{data.get('report_type', 'N/A')}`")
                pi = data.get("patient_info", {})
                if pi:
                    st.markdown("**Patient:** " +
                                ", ".join(f"{k.replace('_',' ').title()}: {v}"
                                          for k, v in pi.items() if v))
                st.markdown("### CBC Panel")
                for item in lab_values:
                    flag = item.get("flag", "UNKNOWN")
                    css  = {"HIGH": "flag-high", "LOW": "flag-low", "NORMAL": "flag-norm"}.get(flag, "")
                    flag_html = f'<span class="{css}">{flag}</span>'
                    st.markdown(
                        f"**{item.get('name','')}**: {item.get('value','')} {item.get('unit','')} "
                        f"(Ref: {item.get('reference_range','')}) — {flag_html}",
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("No lab values extracted.")

        # Tab 3 – PubMed Evidence
        with tab_pubmed:
            queries   = data.get("search_queries", [])
            abs_count = data.get("pubmed_abstracts_count", 0)
            st.markdown(f"**{abs_count} abstracts** retrieved using **{len(queries)} queries**:")
            for i, q in enumerate(queries, 1):
                st.markdown(f"`{i}.` {q}")

        # Tab 4 – Raw JSON
        with tab_raw:
            display_data = {k: v for k, v in data.items() if k != "final_report_markdown"}
            st.json(display_data)

# Footer 
st.markdown("---")
st.markdown(
    "<small>This tool is for informational purposes only and does not constitute "
    "medical advice. Always consult a qualified healthcare professional.</small>",
    unsafe_allow_html=True,
)
