from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict
import operator
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

# from agents.ocr import ocr_node
# from agents.extract import extraction_node
# from agents.search import search_node
# from agents.analyze import analysis_node


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



class AgentState(TypedDict):
    input_file: str
    patient_data: Dict
    lab_test_data: Dict
    pubmed_results: List[str]
    final_analysis: str
    steps: Annotated[List[str], operator.add]


class MedicalInsightsPipeline:
    def __init__(self, checkpointer):
            # Initialize the Graph
            workflow = StateGraph(AgentState)

            # Add Nodes
            # workflow.add_node("ocr", ocr_node)
            # workflow.add_node("context_extraction", extraction_node)
            # workflow.add_node("pubmed_search", search_node)
            # workflow.add_node("results_analysis", analysis_node)
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

            workflow.compile(checkpointer=checkpointer)


    async def ocr_node(self, state: AgentState):
        pass

    async def extraction_node(self, state: AgentState):
        pass

    async def search_node(self, state: AgentState):
        pass

    async def analysis_node(self, state: AgentState):
        pass
