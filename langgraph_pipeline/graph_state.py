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

########################
#state graph
########################

#ejemplo

#generate testcases
class RepairState(TypedDict, total=False):
    task_dir: str
    task_name: str
    task_path: str

    issue: str
    bug_code: str
    manual_test_code: str
    generated_test_code: str

    generated_tests_valid: bool

    failure_stage: str | None
    error_detail: str | None

