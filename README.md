# Agentic-Medical-Insights

### Project Structure

```text
Agentic-Medical-Insights/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── api/
│   │   └── routes.py           # API endpoint definitions
│   ├── core/
│   │   ├── agents/             # Logic for autonomous agents
│   │   │   ├── module1_ocr.py
│   │   │   ├── module2_extraction.py
│   │   │   ├── module3_search.py   # PubMed query generation
│   │   │   ├── module4_analysis.py
│   │   │   └── module5_evaluation.py
│   │   └── pipeline.py         # Orchestration logic & workflow
│   ├── utils/
│   │   ├── file_handler.py     # 
│   │   └── pubmed_api.py       # PubMed API wrapper & utilities
│   └── ui/                     # Integrated frontend
│       └── streamlit_app.py
├── notebooks/
│   └── evaluation.ipynb        # Model experiments & metric tracking
├── .env                        # Environment variables (API keys)
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```
