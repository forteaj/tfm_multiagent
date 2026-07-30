import os

from typing import Annotated, Literal
from typing import Annotated
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

class State(TypedDict):
    # messages have the type "list".
    # The add_messages function appends messages to the list, rather than overwriting them
    messages: Annotated[list, add_messages]
graph= StateGraph(State)

########################
#tools
########################

@tool
def  get_weather(location: str):
    """Call to get the current weather"""
    if location.lower() in ["Alicante"]:
        return "it's cold"
    else:
        return "it's hot"

tools = [get_weather]

########################
llm = get_llm()

llm_with_tools = llm.bind_tools(tools)
########################




########################
#nodes
########################
tool_node = ToolNode(tools)

graph.add_node("tool_node", tool_node)


def prompt_node(state: State) -> State:
    new_message = llm_with_tools.invoke(state["messages"])
    return {"messages": [new_message]}

graph.add_node("prompt_node", prompt_node)


def conditional_edge(state: State) -> Literal['tool_node', '__end__']:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    else:
        return "__end__"

graph.add_conditional_edges('prompt_node', conditional_edge)
graph.add_edge("tool_node", "prompt_node")
graph.set_entry_point("prompt_node")


################################
#  execute
################################


APP = graph.compile()

new_state = APP.invoke({"messages": ["What is the weather in Alicante?"]})

print(new_state["messages"][-1].content)