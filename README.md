# Agentic Medical Insights

> **An agentic AI pipeline that transforms raw medical lab reports into personalized, evidence-based clinical insights — powered by Gemma 4, DSPy, LangGraph, and PubMed.**

---

## Problem Statement

Every year, millions of patients receive lab reports full of cryptic numbers and flags — with no context, no explanation, and no guidance. A CBC report might show a dangerously low hemoglobin level, but the patient sees only `8.2 g/dL ↓` and has no idea what it means, what caused it, or what to do next.

**Agentic Medical Insights solves this** by acting as an intelligent medical assistant: it reads your report, understands your context, finds the latest clinical evidence from PubMed, and delivers a personalized, plain-language report grounded in peer-reviewed science.

---

## Architecture

```
PDF / Image
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│                   LangGraph AgentGraph                     │
│                                                            │
│  ┌──────────┐   ┌────────────┐   ┌──────────┐             │
│  │ Module 1 │──▶│  Module 2  │──▶│ Module 3 │             │
│  │ OCR +    │   │ Context    │   │ PubMed   │             │
│  │ Clean    │   │ Extraction │   │ Search   │             │
│  └──────────┘   └────────────┘   └──────────┘             │
│       DSPy           DSPy             DSPy                 │
│  ChainOfThought   ChainOfThought  ChainOfThought           │
│                                        │                   │
│                              ┌─────────▼──────────┐        │
│                              │     Module 4        │        │
│                              │  Report Generation  │        │
│                              └─────────┬──────────┘        │
│                                        │                   │
│                              ┌─────────▼──────────┐        │
│                              │     Module 5        │        │
│                              │  Quality Evaluation │        │
│                              └─────────────────────┘        │
└────────────────────────────────────────────────────────────┘
                                        │
                              ┌─────────▼──────────┐
                              │  Personalized MD    │
                              │  Clinical Report    │
                              └─────────────────────┘
```

---

## Project Structure

```
Agentic-Medical-Insights/
├── app/
│   ├── main.py                    # FastAPI entry point
│   ├── api/
│   │   └── routes.py              # API endpoint definitions
│   ├── core/
│   │   ├── agents/
│   │   │   ├── module1_ocr.py         # OCR + DSPy clean
│   │   │   ├── module2_extraction.py  # Context extraction
│   │   │   ├── module3_search.py      # PubMed query generation
│   │   │   ├── module4_analysis.py    # Report generation
│   │   │   └── module5_evaluation.py  # Self-evaluation
│   │   └── pipeline.py            # LangGraph orchestration
│   ├── utils/
│   │   └── pubMed_api.py          # PubMed API wrapper
│   └── ui/
│       └── streamlit_app.py       # Streamlit frontend
├── notebooks/
│   └── evaluation.ipynb           # Experiments & metrics
├── .env                           # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/Agentic-Medical-Insights.git
cd Agentic-Medical-Insights
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env .env.local
# Edit .env.local and add your keys:
#   OPENROUTER_API_KEY  – get free key at https://openrouter.ai
#   PUBMED_EMAIL        – any valid email for NCBI
#   PUBMED_API_KEY     
```

### 3. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Run the Frontend

```bash
streamlit run app/ui/streamlit_app.py
```

Open http://localhost:8501 in your browser. Upload a lab report (PDF or image) and watch the magic happen.

---

## LLM: Gemma 4 via OpenRouter

This project uses **Gemma 4** (Google's open-weight model) via [OpenRouter](https://openrouter.ai), which offers **free access**.

1. Sign up at https://openrouter.ai
2. Generate an API key
3. Add it to `.env` as `OPENROUTER_API_KEY`
4. The model is set to `google/gemma-4-26b-a4b-it:free` 
   
Alternatively, use **Google AI Studio**:
1. Get a key at https://aistudio.google.com
2. Set `GOOGLE_API_KEY` in `.env`

---

## Pipeline Modules

| # | Module | Framework | Input | Output |
|---|--------|-----------|-------|--------|
| 1 | OCR & Clean | DSPy `ChainOfThought` | PDF/Image bytes | Clean Markdown |
| 2 | Context Extraction | DSPy `Predict` | Markdown | Patient info + lab values (JSON) |
| 3 | PubMed Search | DSPy + NCBI Entrez | Extracted context | PubMed abstracts |
| 4 | Report Generation | DSPy `ChainOfThought` | Context + abstracts | Personalized Markdown report |
| 5 | Quality Evaluation | DSPy `Predict` | Report | Scores (1–10) + feedback |

---

## Why DSPy?

DSPy replaces brittle prompt strings with **declarative signatures** and **learnable modules**. Each module can be independently optimized (compiled) using DSPy's teleprompters. This means:
- Prompts are **version-controlled** as Python classes
- Modules can be **fine-tuned** with few-shot examples without changing code
- The pipeline is **composable** and testable

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/analyze` | Upload & analyze a lab report |
| `GET`  | `/api/v1/report/{filename}` | Retrieve a saved report |
| `GET`  | `/` | Health check |

### Example cURL

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@my_cbc_report.pdf" | jq .
```

---

## Disclaimer

This tool is for **informational and research purposes only**. It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical decisions.

---

