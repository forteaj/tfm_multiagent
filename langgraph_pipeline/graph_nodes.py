import os

from typing import Annotated, Literal
from typing import Annotated, Any
from typing_extensions import TypedDict
from llm import get_llm
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph_pipeline.graph_state import RepairState

from pathlib import Path

from agents.test_generator import generate_test_cases
from tools.file_tools import write_text
from tools.json_tools import extract_json

from langgraph_pipeline.graph_state import RepairState

########################
#tools
########################

########################
llm = get_llm()

#llm_with_tools = llm.bind_tools(tools)
########################

########################
#nodes
########################

def generate_tests_node(state: RepairState) -> dict:
    task_dir = Path(state["task_dir"])

    try:
        model_response = generate_test_cases(
            issue=state["issue"],
            bug_code=state["bug_code"],
            module_name="bug1",
        )

        parsed_response = extract_json(model_response)
        generated_test_code = parsed_response["test_code"]

        generated_test_file = task_dir / "test_generated.py"
        write_text(generated_test_file, generated_test_code)

        return {
            "generated_test_code": generated_test_code,
            "generated_tests_valid": True,
            "test_generation_error": None,
        }

    except Exception as exc:
        return {
            "generated_test_code": "",
            "generated_tests_valid": False,
            "test_generation_error": str(exc),
        }

