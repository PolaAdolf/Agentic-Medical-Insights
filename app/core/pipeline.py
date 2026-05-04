from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict
import operator
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

import os
import requests
from dotenv import load_dotenv
import dspy
from app.core.agents.ocr import OCRPipeline
from app.utils.file_handler import extract_raw_text
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
    def __init__(self, model="google/gemma-4-26b-a4b-it:free", temperature=0):
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
    patient_data: Dict
    lab_test_data: Dict
    pubmed_results: List[str]
    final_analysis: str
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

        # Define Edges (The Flow)
        workflow.set_entry_point("ocr")
        workflow.add_edge("ocr", "context_extraction")
        workflow.add_edge("context_extraction", "pubmed_search")
        # workflow.add_conditional_edges("context_extraction", self.is_valid_context, {True: "pubmed_search", False: "context_extraction"})
        workflow.add_edge("pubmed_search", "results_analysis")
        workflow.add_edge("results_analysis", END)

        self.graph = workflow.compile(checkpointer=checkpointer)


    async def ocr_node(self, state: AgentState):
        """
        OCR Node
        
         Reads  : file_bytes, filename
         Writes : raw_ocr_text, cleaned_ocr_text, report_type, ocr_markdown
        """
        dspy.settings.configure(lm=OpenRouterLM())
        ocr_pipeline = OCRPipeline()  
        raw_text = extract_raw_text(state["file_bytes"], state["filename"])
        pred     = ocr_pipeline(raw_text=raw_text)

        return {
            **state,
            "raw_ocr_text":     raw_text,
            "cleaned_ocr_text": pred.cleaned_text,
            "report_type":      pred.report_type,
            "ocr_markdown":     pred.structured_md,
            "current_stage":    "ocr_complete",
        }
    

    

    async def extraction_node(self, state: AgentState):
        pass

    async def search_node(self, state: AgentState):
        pass

    async def analysis_node(self, state: AgentState):
        pass



def visualize_graph():
    pipeline = MedicalInsightsPipeline()

    return pipeline.graph.get_graph().draw_mermaid_png()