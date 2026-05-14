from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict
import operator
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

import requests
from dotenv import load_dotenv
import dspy
import os
import sys

from src.app.core.agents.ocr import OCRPipeline
from src.app.core.agents.extract import extraction_node
from src.app.utils.file_handler import extract_raw_text
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

    # Module 1 Output
    raw_ocr_text: str
    cleaned_ocr_text: str
    report_type: str
    ocr_markdown: str

    # Module 2 Output
    # patient_data: Dict
    # lab_test_data: Dict
    module_2_output: Dict

    # Module 3 Output
    pubmed_results: List[str]

    # Module 4 Output
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
        workflow.add_node("failure", self.return_failure)

        # Define Edges (The Flow)
        workflow.set_entry_point("ocr")
        workflow.add_edge("ocr", "context_extraction")
        workflow.add_edge("context_extraction", END)
        # workflow.add_conditional_edges("ocr", self.is_lab_test_supported, {True: "context_extraction", False: "failure"})
        # workflow.add_edge("context_extraction", "pubmed_search")
        # # workflow.add_conditional_edges("context_extraction", self.is_valid_context, {True: "pubmed_search", False: "context_extraction"})
        # workflow.add_edge("pubmed_search", "results_analysis")
        # workflow.add_edge("results_analysis", END)

        self.graph = workflow.compile(checkpointer=checkpointer)


    async def run_pipeline(self, file_bytes: bytes, filename: str, thread_id: str = "1"):
            """
            Triggers the workflow with the initial state.
            """
            # Initial state following your AgentState TypedDict
            initial_state = {
                "file_bytes": file_bytes,
                "filename": filename,
                "steps": []
            }
            
            # Config for persistence/checkpointer
            config = {"configurable": {"thread_id": thread_id}}
            
            # Execute the graph
            try:
                # We use aysnc invoke since your nodes are async
                result = await self.graph.ainvoke(initial_state, config=config)
                return result
            except Exception as e:
                return {"error": str(e), "steps": ["pipeline_failed"]}

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
        print(state)
        print("Module 1 completed!")
        print("============================================")
        return {
            **state,
            "raw_ocr_text":     raw_text,
            "cleaned_ocr_text": pred.cleaned_text,
            "report_type":      pred.report_type,
            "ocr_markdown":     pred.structured_md,
            "steps":            ["ocr_completed"]
        }
    

    

    async def extraction_node(self, state: AgentState):
        extract =  extraction_node(state)
        # print(extract)
        # print(extract["module_2_output"])
        print(state)
        print("Module 2 completed!")
        print("============================================")
        return  {
            **state,
            # "patient_data" : extract["full"]["personal_info"],
            # "lab_test_data": extract["full"]["metrics"],
            "module_2_output": extract["module_2_output"],
            "steps":        ["Context_extraction_completed"]
        }


    async def search_node(self, state: AgentState):
        pass

    async def analysis_node(self, state: AgentState):
        pass

    def is_lab_test_supported(self, state: AgentState):
        print("My repirt type is", state["report_type"])
        if(state["report_type"] not in ['CBC']):
            return False
        else:
            return True

    def return_failure(self, state: AgentState):
        if(state["steps"][-1] == "ocr_complete"):
            return "Report type not supported"
        else:
            return -1


def visualize_graph():
    pipeline = MedicalInsightsPipeline()

    return pipeline.graph.get_graph().draw_mermaid_png()