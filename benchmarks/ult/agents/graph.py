from openai import OpenAI
from typing import TypedDict, List, Dict, Literal
from langgraph.graph import StateGraph, START, END
import json
import time
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
    task_id: str

    function_name: str
    solution_code: str
    output_dir: str
    timeout_seconds: int

    counter: int
    max_rounds: int

    generated_test: str
    generation_time_seconds: float
    generator_finish_reason: str | None
    generator_usage: Dict | None
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
    execution_feedback: str
    candidate_stdout: str
    candidate_stderr: str

    suite_passed: bool
    line_coverage: float | None
    branch_coverage: float | None
    line_coverage_gain: float
    branch_coverage_gain: float
    improved_coverage: bool
    #time
    review_time_seconds: float

    candidate_execution_time_seconds: float
    fallback_suite_execution_time_seconds: float
    execution_time_seconds: float

    rounds: List[Dict]

def generator_node(state: AgentState) -> dict:
    """Generate one test candidate and record its metadata."""

    print(
        f"Generating test for round {state['counter']}"
    )

    previous_tests = (
        "\n\n".join(state["accepted_tests"])
        or "No tests have been accepted yet."
    )

    prompt = f"""
Generate one pytest test for the following Python function.

FUNCTION NAME:
{state["function_name"]}

SOURCE CODE:
{state["solution_code"]}

PREVIOUSLY ACCEPTED TESTS:
{previous_tests}

OUTPUT FORMAT:
- Return only valid Python code.
- Include exactly one import statement.
- Include exactly one test function whose name starts with test_.
- Include exactly one assert statement.
- Test exactly one concrete input case.
- Use a test function name different from all previous tests.
- Do not include comments, explanations, helpers, classes
  or parametrization.
- Do not return the complete accumulated suite.
- Do not use Markdown fences.

The import must be exactly:
from solution import {state["function_name"]}
""".strip()

    generation_start = time.perf_counter()

    response = client.chat.completions.create(
        model=state["model_id"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Python unit-test generator."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_tokens=4096,
    )

    generation_time = round(
        time.perf_counter() - generation_start,
        4,
    )

    choice = response.choices[0]
    raw_response = choice.message.content

    if raw_response is None:
        raise RuntimeError(
            "The generator returned an empty response."
        )

    usage = (
        response.usage.model_dump()
        if response.usage is not None
        else None
    )

    return {
        "generated_test": raw_response.strip(),
        "generation_time_seconds": generation_time,
        "generator_finish_reason": choice.finish_reason,
        "generator_usage": usage,

        "reviewed_test": "",
        "raw_review_response": "",
        "review_feedback": "",
        "review_time_seconds": 0.0,
        "review_finish_reason": None,
        "review_usage": None,

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
        "review_time_seconds": 0.0,
        "candidate_execution_time_seconds": 0.0,
        "fallback_suite_execution_time_seconds": 0.0,
        "execution_time_seconds": 0.0,
        "generation_time_seconds": 0.0,
        "generator_finish_reason": None,
        "generator_usage": None,
    }

def record_round_node(state: AgentState) -> dict:
    """Store the complete result of the current generation round."""

    round_result = {
        "k": state["counter"],

        # Output produced by the generator agent.
        "generator_test": state["generated_test"],

        "generation_time_seconds": state["generation_time_seconds"],
        "generator_finish_reason": state["generator_finish_reason"],
        "generator_usage": state["generator_usage"],

        # Final candidate returned by the reviewer.
        #
        # `generated_test` intentionally contains the reviewed version
        # because the mutation evaluator expects this field to contain
        # the candidate that was actually executed.
        "generated_test": state["reviewed_test"],
        "reviewed_test": state["reviewed_test"],

        # Reviewer information.
        "raw_review_response": state["raw_review_response"],
        "review_feedback": state["review_feedback"],
        "review_finish_reason": state[
            "review_finish_reason"
        ],
        "review_usage": state["review_usage"],

        # Structural validation.
        "test_function_name": state["test_function_name"],
        "syntax_valid": state["syntax_valid"],
        "format_valid": state["format_valid"],
        "format_errors": state["format_errors"],
        "duplicate_name": state["duplicate_name"],

        # Candidate execution.
        "candidate_passed": state["test_passed"],
        "candidate_accepted": state["accepted"],
        "candidate_pytest_exit_code": state[
            "pytest_exit_code"
        ],
        "candidate_timed_out": state["timed_out"],
        #"candidate_execution_time_seconds": state["execution_time_seconds"],
        "candidate_stdout": state["candidate_stdout"],
        "candidate_stderr": state["candidate_stderr"],

        # State of the accumulated suite after this round.
        "accepted_tests": len(state["accepted_tests"]),
        "suite_passed": state["suite_passed"],
        "line_coverage": state["line_coverage"],
        "branch_coverage": state["branch_coverage"],
        "line_coverage_gain": state[
            "line_coverage_gain"
        ],
        "branch_coverage_gain": state[
            "branch_coverage_gain"
        ],
        "improved_coverage": state[
            "improved_coverage"
        ],
        "review_time_seconds": state[
            "review_time_seconds"
        ],

        "candidate_execution_time_seconds": state[
            "candidate_execution_time_seconds"
        ],
        "fallback_suite_execution_time_seconds": state[
            "fallback_suite_execution_time_seconds"
        ],
        "execution_time_seconds": state[
            "execution_time_seconds"
        ],
    }

    updated_rounds = [
        *state["rounds"],
        round_result,
    ]

    print(
        f"Round {state['counter']} recorded | "
        f"Accepted tests: {len(state['accepted_tests'])}"
    )

    return {
        "rounds": updated_rounds,
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

def save_results_node(state: AgentState) -> dict:
    """Build and save the final multi-agent experiment result."""

    output_dir = Path(state["output_dir"])
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = output_dir / "result.json"
    temporary_path = output_dir / "result.json.tmp"

    rounds = state["rounds"]
    max_k = state["max_rounds"]

    valid_generations = sum(
        1
        for round_result in rounds
        if (
            round_result["syntax_valid"]
            and round_result["format_valid"]
        )
    )

    passed_candidates = sum(
        1
        for round_result in rounds
        if round_result["candidate_passed"]
    )

    accepted_candidates = sum(
        1
        for round_result in rounds
        if round_result["candidate_accepted"]
    )

    coverage_improving_tests = sum(
        1
        for round_result in rounds
        if round_result["improved_coverage"]
    )

    generator_prompt_tokens = sum_usage_tokens(
        rounds,
        "generator_usage",
        "prompt_tokens",
    )
    generator_completion_tokens = sum_usage_tokens(
        rounds,
        "generator_usage",
        "completion_tokens",
    )
    generator_total_tokens = sum_usage_tokens(
        rounds,
        "generator_usage",
        "total_tokens",
    )

    reviewer_prompt_tokens = sum_usage_tokens(
        rounds,
        "review_usage",
        "prompt_tokens",
    )
    reviewer_completion_tokens = sum_usage_tokens(
        rounds,
        "review_usage",
        "completion_tokens",
    )
    reviewer_total_tokens = sum_usage_tokens(
        rounds,
        "review_usage",
        "total_tokens",
    )

    total_generation_time = sum_round_value(
        rounds,
        "generation_time_seconds",
    )
    total_review_time = sum_round_value(
        rounds,
        "review_time_seconds",
    )
    total_execution_time = sum_round_value(rounds,"execution_time_seconds", )

    experiment_result = {
        "task_id": state["task_id"],
        "function_name": state["function_name"],
        "model_id": state["model_id"],
        "system": "multiagent_generator_reviewer",
        "max_k": max_k,
        "evaluation_status": "completed",

        # Time.
        "total_generation_time_seconds": (
            total_generation_time
        ),
        "total_review_time_seconds": total_review_time,
        "total_execution_time_seconds": (
            total_execution_time
        ),
        "total_pipeline_time_seconds": round(
            total_generation_time
            + total_review_time
            + total_execution_time,
            4,
        ),

        # Generator tokens.
        "generator_prompt_tokens": (
            generator_prompt_tokens
        ),
        "generator_completion_tokens": (
            generator_completion_tokens
        ),
        "generator_total_tokens": (
            generator_total_tokens
        ),

        # Reviewer tokens.
        "reviewer_prompt_tokens": (
            reviewer_prompt_tokens
        ),
        "reviewer_completion_tokens": (
            reviewer_completion_tokens
        ),
        "reviewer_total_tokens": (
            reviewer_total_tokens
        ),

        # Combined token usage.
        "total_prompt_tokens": (
            generator_prompt_tokens
            + reviewer_prompt_tokens
        ),
        "total_completion_tokens": (
            generator_completion_tokens
            + reviewer_completion_tokens
        ),
        "total_tokens": (
            generator_total_tokens
            + reviewer_total_tokens
        ),

        # Generation and acceptance.
        "generated_tests": len(rounds),
        "reviewed_tests": len(rounds),
        "accepted_tests": accepted_candidates,
        "valid_generation_rate": round(
            valid_generations / max_k,
            4,
        ) if max_k else 0.0,
        "candidate_pass_rate": round(
            passed_candidates / max_k,
            4,
        ) if max_k else 0.0,

        # Coverage.
        "coverage_improving_tests": (
            coverage_improving_tests
        ),
        "coverage_improvement_rate": round(
            coverage_improving_tests / max_k,
            4,
        ) if max_k else 0.0,
        "final_line_coverage": (
            rounds[-1]["line_coverage"]
            if rounds
            else None
        ),
        "final_branch_coverage": (
            rounds[-1]["branch_coverage"]
            if rounds
            else None
        ),
        "suite_pass_at_k": [
            round_result["suite_passed"]
            for round_result in rounds
        ],
        "lcov_at_k": [
            round_result["line_coverage"]
            for round_result in rounds
        ],
        "bcov_at_k": [
            round_result["branch_coverage"]
            for round_result in rounds
        ],

        # Full trace.
        "rounds": rounds,
    }

    temporary_path.write_text(
        json.dumps(
            experiment_result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # Replace result.json only after complete serialization.
    temporary_path.replace(result_path)

    print(
        f"Results saved to: {result_path.resolve()}"
    )

    return {}


#--------------
#aux functions
#--------------
def sum_round_value( rounds: List[Dict], field: str,) -> float:
    """Sum a numeric field from all recorded rounds."""

    return round(
        sum(
            float(round_result.get(field) or 0.0)
            for round_result in rounds
        ),
        4,
    )

def sum_usage_tokens(rounds: List[Dict],usage_field: str,token_field: str,) -> int:
    """Sum token usage stored in generation or review metadata."""

    total = 0

    for round_result in rounds:
        usage = round_result.get(usage_field)

        if isinstance(usage, dict):
            total += int(
                usage.get(token_field, 0) or 0
            )

    return total
#--------------

graph = StateGraph(AgentState)

#graph.add_node("router", lambda state:state) #input state will be the putput state
graph.add_node("generator", generator_node)
graph.add_node("reviewer", reviewer_node)
graph.add_node("evaluator", evaluator_node)
graph.add_node("suite_adder", suite_node)
graph.add_node("close_round", close_round_node)
graph.add_node("record_round",record_round_node)
graph.add_node("save_results",save_results_node)


graph.add_edge(START, "generator")
graph.add_edge( "generator","reviewer")
graph.add_edge( "reviewer","evaluator")
graph.add_conditional_edges(
    "evaluator",
    route_after_evaluation,
    {
        "add_to_suite": "suite_adder",
        "close_round": "record_round",
    }
)
graph.add_edge( "suite_adder","record_round")
graph.add_edge( "record_round","close_round")
graph.add_conditional_edges(
    "close_round",
    route_after_round,
    {
        "continue": "generator",
        "finish": "save_results",
    },
)

graph.add_edge(
    "save_results",
    END,
)

app = graph.compile()


initial_state: AgentState = {
    "model_id": model_id,
    "task_id": "smoke_is_even",
    "review_time_seconds": 0.0,
    "candidate_execution_time_seconds": 0.0,
    "fallback_suite_execution_time_seconds": 0.0,
    "execution_time_seconds": 0.0,
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
    "generation_time_seconds": 0.0,
    "generator_finish_reason": None,
    "generator_usage": None,
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