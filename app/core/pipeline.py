from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict
import operator
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
import json
import os
import requests
from dotenv import load_dotenv
import dspy
from app.core.agents.ocr import OCRPipeline
from app.core.agents.validate import EvaluationModule
from app.utils.file_handler import extract_raw_text
from app.core.agents.extract import ContextExtractionModule, context_to_markdown
from app.core.agents.search import QueryGenerationModule, _build_context_summary, _safe_parse_queries
from app.utils.pubMed_api import search_pubmed, fetch_abstracts
from app.core.agents.analyze import ReportGenerationModule, _build_patient_context_json, _build_abstracts_json, _add_report_header
from app.core.agents.validate import EvaluationModule, _parse_eval, _scores_to_markdown, QUALITY_THRESHOLD
load_dotenv()


def get_gemma_model(variant='26b', temp=0):        
    if variant == '31b':
        gemma_model = "google/gemma-4-31b-it:free"
    else:
        gemma_model = "google/gemma-4-26b-a4b-it:free"

    _ = load_dotenv()

    return ChatOpenRouter(
    model=gemma_model,
    temperature=temp
    )

class OpenRouterLM(dspy.BaseLM):
    def __init__(self, model="openai/gpt-oss-20b:free", temperature=0):
        super().__init__(model=model)
        
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.temperature = temperature

        if not self.api_key:
            raise ValueError("API_KEY is missing in .env")

    def __call__(self, messages=None, prompt=None, **kwargs):
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature
            }
        )

        data = response.json()

        if "choices" not in data:
            raise Exception(f"OpenRouter Error: {data}")

        return [data["choices"][0]["message"]["content"]]

GLOBAL_LM = OpenRouterLM()

dspy.settings.configure(
    lm=GLOBAL_LM
)

class AgentState(TypedDict):

    # Input
    file_bytes:   bytes
    filename:     str
    # Module 1 outputs
    raw_ocr_text: str
    cleaned_ocr_text: str
    report_type: str
    ocr_markdown: str

    # Module 2 outputs
    patient_info: dict
    lab_values: list[dict]
    clinical_notes: dict
    context_markdown: str
    module_2_output: dict

    # Module 3 outputs
    search_queries: list[dict]
    pubmed_abstracts: list[dict]

    # Module 4 outputs
    final_report_markdown: str

    # Module 5 outputs
    quality_scores: dict
    evaluation_passed: bool

    # Tracking
    current_stage: str
    steps: Annotated[List[str], operator.add]


class MedicalInsightsPipeline:
    def __init__(self, checkpointer=MemorySaver()):
        # Initialize the Graph
        workflow = StateGraph(AgentState)
        # Add Nodes (The Modules)
        workflow.add_node("ocr", self.ocr_node)
        workflow.add_node("context_extraction", self.extraction_node)
        workflow.add_node("pubmed_search", self.search_node)
        workflow.add_node("results_analysis", self.analysis_node)
        workflow.add_node("evaluation", self.evaluation_node)

        # Define Edges (The Flow)
        workflow.set_entry_point("ocr")
        workflow.add_edge("ocr", "context_extraction")
        workflow.add_edge("context_extraction", "pubmed_search")
        workflow.add_edge("pubmed_search", "results_analysis")
        workflow.add_edge("results_analysis", "evaluation")
        workflow.add_edge("evaluation", END)

        self.graph = workflow.compile()


    async def ocr_node(self, state: AgentState):
        """
        OCR Node
        
         Reads  : file_bytes, filename
         Writes : raw_ocr_text, cleaned_ocr_text, report_type, ocr_markdown
        """
        print("Performing OCR on the document...")
        ocr_pipeline = OCRPipeline()  
        raw_text = extract_raw_text(state["file_bytes"], state["filename"])
        pred     = ocr_pipeline(raw_text=raw_text)
        print("OCR complete.")
        return {
            **state,
            "raw_ocr_text":     raw_text,
            "cleaned_ocr_text": pred.cleaned_text,
            "report_type":      pred.report_type,
            "ocr_markdown":     pred.structured_md,
            "current_stage":    "ocr_complete",
        }
    
    

    

    async def extraction_node(self, state: AgentState):
        """
        Context Extraction Node
        
         Reads  : ocr_markdown
         Writes : report_type, patient_info, lab_values, clinical_notes, context_markdown
        """
        print("Extracting structured context from OCR output...")
        module = ContextExtractionModule()

        markdown = state.get("ocr_markdown")
        result = module(markdown_text=markdown)
        full= result["full"]
        print("Context extraction complete.")
        return {
            **state,
            "report_type":      full["report_category"],
            "patient_info":     full["personal_info"],
            "lab_values":       full["metrics"],
            "clinical_notes":   {"notes": full["clinical_notes"]},
            "context_markdown": context_to_markdown(full),
            "module_2_output":  result,
            "current_stage":    "extraction_complete",
            "steps":            state.get("steps", []) + ["Context extraction complete"],
        }

    async def search_node(self, state: AgentState):

        """
        LangGraph node: query generation + PubMed search.

        Reads from state
        ----------------
        - ``report_type``  : str
        - ``patient_info`` : dict
        - ``lab_values``   : list[dict]

        Writes to state
        ---------------
        - ``search_queries``  : list[str]
        - ``pubmed_abstracts``: list[dict]  – [{pmid, title, abstract, authors, year}]
        - ``current_stage``   : str
        """
        _query_module = QueryGenerationModule()
        context_summary = _build_context_summary(state)

        # Generate queries
        pred = _query_module(context_summary=context_summary)
        queries = _safe_parse_queries(pred.queries_json)

        # Execute PubMed searches
        all_abstracts: list[dict] = []
        seen_pmids: set[str] = set()

        for query in queries:
            pmids = search_pubmed(query, max_results=5)
            new_pmids = [p for p in pmids if p not in seen_pmids]
            seen_pmids.update(new_pmids)
            abstracts = fetch_abstracts(new_pmids)
            all_abstracts.extend(abstracts)

        return {
            **state,
            "search_queries":   queries,
            "pubmed_abstracts": all_abstracts,
            "current_stage":    "search_complete",
        }



    async def analysis_node(self, state: AgentState):
          
        """
        LangGraph node: synthesis & report generation.

        Reads from state
        ----------------
        - ``patient_info``     : dict
        - ``report_type``      : str
        - ``lab_values``       : list[dict]
        - ``pubmed_abstracts`` : list[dict]

        Writes to state
        ---------------
        - ``final_report_markdown`` : str   – complete Markdown report
        - ``current_stage``         : str
        """
        _report_module = ReportGenerationModule()
        patient_ctx  = _build_patient_context_json(state)
        abstracts_js = _build_abstracts_json(state)

        pred = _report_module(
            patient_context_json=patient_ctx,
            pubmed_abstracts_json=abstracts_js,
        )

        report_md = _add_report_header(pred.report_markdown, state)

        return {
            **state,
            "final_report_markdown": report_md,
            "current_stage":         "report_complete",
        }

    async def evaluation_node(self, state: AgentState):

        """
        LangGraph node: report quality evaluation.

        Reads from state
        ----------------
        - ``final_report_markdown`` : str
        - ``lab_values``            : list[dict]

        Writes to state
        ---------------
        - ``quality_scores``           : dict
        - ``final_report_markdown``    : str  (appended with scores section)
        - ``evaluation_passed``        : bool
        - ``current_stage``            : str
        """
        _eval_module = EvaluationModule()
        report_md = state.get("final_report_markdown", "")
        lab_context = json.dumps(state.get("lab_values", []), indent=2)

        pred = _eval_module(
            report_markdown=report_md,
            lab_context_json=lab_context,
        )
        scores = _parse_eval(pred.evaluation_json)

        # Append quality scorecard to the report
        report_with_scores = report_md + _scores_to_markdown(scores)

        evaluation_passed = scores.get("overall", 0) >= QUALITY_THRESHOLD

        return {
            **state,
            "quality_scores":        scores,
            "final_report_markdown": report_with_scores,
            "evaluation_passed":     evaluation_passed,
            "current_stage":         "evaluation_complete",
        }

def visualize_graph(file_bytes: bytes, filename: str):
    initial_state: AgentState = {
        "file_bytes":    file_bytes,
        "filename":      filename,
        "current_stage": "initialised",
    }
    pipeline = MedicalInsightsPipeline()

    return pipeline.graph.get_graph().draw_mermaid_png()