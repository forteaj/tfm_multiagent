from openai import OpenAI
from typing import TypedDict, List, Dict, Literal
from langgraph.graph import StateGraph, START, END
import os
from benchmarks.ult.agents.reviewer import (evaluator_node,reviewer_node,)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "ult"
    / "multiagent"
    / "smoke"
    / "is_even"
)


client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

model_id = client.models.list().data[0].id


class AgentState(TypedDict):
    model_id: str

    function_name: str
    solution_code: str
    output_dir: str
    timeout_seconds: int

    counter: int
    max_rounds: int

    generated_test: str
    reviewed_test: str
    raw_review_response: str
    review_feedback: str
    review_finish_reason: str | None
    review_usage: Dict | None

    accepted: bool
    accepted_tests: List[str]

    syntax_valid: bool
    format_valid: bool
    format_errors: List[str]
    duplicate_name: bool
    test_function_name: str | None

    pytest_exit_code: int | None
    test_passed: bool
    timed_out: bool
    execution_time_seconds: float | None
    execution_feedback: str
    candidate_stdout: str
    candidate_stderr: str

    suite_passed: bool
    line_coverage: float | None
    branch_coverage: float | None
    line_coverage_gain: float
    branch_coverage_gain: float
    improved_coverage: bool

    rounds: List[Dict]

def generator_node(state: AgentState) -> dict:
    """Generate one test candidate."""

    print(f"Generating test for round {state['counter']}")

    prompt = f"""
Generate one pytest test for the following Python function.

Function name:
{state["function_name"]}

Implementation:
{state["solution_code"]}

Tests already accepted:
{state["accepted_tests"]}

Requirements:
- Return only Python code.
- Use exactly one import:
  from solution import {state["function_name"]}
- Generate exactly one test function named test_*.
- Do not use Markdown fences.
"""

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": "You generate concise pytest unit tests.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    generated_test = response.choices[0].message.content or ""

    return {
        "generated_test": generated_test.strip(),
        "reviewed_test": "",
        "review_feedback": "",
        "accepted": False,
    }

def suite_node(state: AgentState) -> dict:
    """Add an accepted test to the accumulated suite."""

    updated_suite = [
        *state["accepted_tests"],
        state["reviewed_test"],
    ]

    return {
        "accepted_tests": updated_suite,
    }
def close_round_node(state: AgentState) -> dict:
    """Close the current round and prepare the next one."""

    return {
        "counter": state["counter"] + 1,
        "accepted": False,
        "generated_test": "",
        "reviewed_test": "",
        "review_feedback": "",
    }


def route_after_evaluation(
    state: AgentState,
) -> Literal["add_to_suite", "close_round"]:
    """Decide whether the candidate should be added to the suite."""

    if state["accepted"]:
        return "add_to_suite"

    return "close_round"


def route_after_round(
    state: AgentState,
) -> Literal["continue", "finish"]:
    """Decide whether another generation round is required."""

    if state["counter"] <= state["max_rounds"]:
        print(f"Starting round {state['counter']}")
        return "continue"

    print("All rounds completed")
    return "finish"

graph = StateGraph(AgentState)

#graph.add_node("router", lambda state:state) #input state will be the putput state
graph.add_node("generator", generator_node)
graph.add_node("reviewer", reviewer_node)
graph.add_node("evaluator", evaluator_node)
graph.add_node("suite_adder", suite_node)
graph.add_node("close_round", close_round_node)


graph.add_edge(START, "generator")
graph.add_edge( "generator","reviewer")
graph.add_edge( "reviewer","evaluator")
graph.add_conditional_edges(
    "evaluator",
    route_after_evaluation,
    {
        "add_to_suite": "suite_adder",
        "close_round": "close_round",
    }
)
graph.add_edge( "suite_adder","close_round")
graph.add_conditional_edges(
    "close_round",
    route_after_round,
    {
        "continue": "generator",
        "finish": END,
    }
)

app = graph.compile()


initial_state: AgentState = {
    "model_id": model_id,
    "function_name": "is_even",
    "solution_code": """
def is_even(number: int) -> bool:
    return number % 2 == 0
""",
    "output_dir": str(OUTPUT_DIR),
    "timeout_seconds": 30,

    "counter": 1,
    "max_rounds": 5,

    "generated_test": "",
    "reviewed_test": "",
    "raw_review_response": "",
    "review_feedback": "",
    "review_finish_reason": None,
    "review_usage": None,

    "accepted": False,
    "accepted_tests": [],

    "syntax_valid": False,
    "format_valid": False,
    "format_errors": [],
    "duplicate_name": False,
    "test_function_name": None,

    "pytest_exit_code": None,
    "test_passed": False,
    "timed_out": False,
    "execution_time_seconds": None,
    "execution_feedback": "",
    "candidate_stdout": "",
    "candidate_stderr": "",

    "suite_passed": False,
    "line_coverage": None,
    "branch_coverage": None,
    "line_coverage_gain": 0.0,
    "branch_coverage_gain": 0.0,
    "improved_coverage": False,

    "rounds": [],
}

final_state = app.invoke(initial_state)

print("\nAccepted tests:")
for test in final_state["accepted_tests"]:
    print(test)
    print("-" * 50)