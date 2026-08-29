from openai import OpenAI
from typing import Any, TypedDict, List, Dict, Literal
from langgraph.graph import StateGraph, START, END
import json
import time
from benchmarks.ult.agents.reviewer import (
    evaluator_node,
    reviewer_node,
)
from benchmarks.ult.evaluator import remove_markdown_fences
from pathlib import Path
import os
from benchmarks.ult.loader import ULTTask, load_ult

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    PROJECT_ROOT
    / "external"
    / "UnLeakedTestBench"
    / "datasets"
    / "ULT_Lite.jsonl"
)

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

class AgentState(TypedDict):
    model_id: str
    task_id: str

    function_name: str
    solution_code: str
    description: str
    cyclomatic_complexity: int | None
    dataset_path: str


    output_dir: str
    timeout_seconds: int

    counter: int
    max_rounds: int

    generated_test: str
    candidate_test: str
    generation_time_seconds: float
    generator_finish_reason: str | None
    generator_usage: Dict | None
    reviewed_test: str
    raw_review_response: str
    review_feedback: str
    review_finish_reason: str | None
    review_usage: Dict | None
    reviewer_called: bool
    repair_changed: bool
    repair_duplicate: bool
    generator_evaluation: Dict | None
    repair_evaluation: Dict | None

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


def format_previous_attempts(
    rounds: List[Dict],
) -> str:
    """Reproduce the baseline history using only generator proposals."""

    if not rounds:
        return "No tests have been generated yet."

    attempted_tests = [
        remove_markdown_fences(
            str(round_result.get("generator_test") or "")
        )
        for round_result in rounds
    ]

    return "\n\n".join(
        test for test in attempted_tests if test
    )


def generator_node(state: AgentState) -> dict:
    """Generate one test candidate and record its metadata."""

    print(
        f"Generating test for round {state['counter']}"
    )

    previous_tests = format_previous_attempts(
        state["rounds"]
    )

    prompt = f"""
You are generating test case number {state["counter"]} for a Python function.

OUTPUT FORMAT:
- Return only valid Python code.
- Include exactly one import statement.
- Include exactly one test function whose name starts with test_.
- Include exactly one assert statement.
- Test exactly one input case.
- Use a test function name different from all previous tests.
- Do not include comments, explanations, helpers, classes or parametrization.

TEST REQUIREMENTS:
- Import the function exactly as follows:
  from solution import {state["function_name"]}
- Generate a case that is meaningfully different from previous tests.
- Prefer an uncovered boundary, branch, exceptional case or input category.
- Use a concrete expected value.
- Do not reproduce or modify the implementation.
- Do not access external resources.
- The test must be deterministic and executable with pytest.

FUNCTION NAME:
{state["function_name"]}

DESCRIPTION:
{state["description"]}

SOURCE CODE:
{state["solution_code"]}

PREVIOUSLY GENERATED TESTS:
{previous_tests}
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
        temperature=0,
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
        "candidate_test": raw_response.strip(),
        "generation_time_seconds": generation_time,
        "generator_finish_reason": choice.finish_reason,
        "generator_usage": usage,

        "reviewed_test": "",
        "raw_review_response": "",
        "review_feedback": "",
        "review_time_seconds": 0.0,
        "review_finish_reason": None,
        "review_usage": None,
        "reviewer_called": False,
        "repair_changed": False,
        "repair_duplicate": False,
        "generator_evaluation": None,
        "repair_evaluation": None,

        "accepted": False,
    }

def suite_node(state: AgentState) -> dict:
    """Add an accepted test to the accumulated suite."""

    updated_suite = [
        *state["accepted_tests"],
        state["candidate_test"],
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
        "candidate_test": "",
        "reviewed_test": "",
        "raw_review_response": "",
        "review_feedback": "",
        "review_time_seconds": 0.0,
        "candidate_execution_time_seconds": 0.0,
        "fallback_suite_execution_time_seconds": 0.0,
        "execution_time_seconds": 0.0,
        "generation_time_seconds": 0.0,
        "generator_finish_reason": None,
        "generator_usage": None,
        "review_finish_reason": None,
        "review_usage": None,
        "reviewer_called": False,
        "repair_changed": False,
        "repair_duplicate": False,
        "generator_evaluation": None,
        "repair_evaluation": None,
    }

def record_round_node(state: AgentState) -> dict:
    """Store the complete result of the current generation round."""

    generator_evaluation = state.get("generator_evaluation") or {}
    repair_evaluation = state.get("repair_evaluation")

    generator_execution_time = float(
        generator_evaluation.get("execution_time_seconds") or 0.0
    )
    repair_execution_time = float(
        (repair_evaluation or {}).get("execution_time_seconds") or 0.0
    )
    total_round_execution_time = round(
        generator_execution_time + repair_execution_time,
        4,
    )

    round_result = {
        "k": state["counter"],

        # Output produced by the generator agent.
        "generator_test": state["generated_test"],

        "generation_time_seconds": state["generation_time_seconds"],
        "generator_finish_reason": state["generator_finish_reason"],
        "generator_usage": state["generator_usage"],

        # Final candidate executed by pytest. `generated_test` keeps this
        # value because the mutation evaluator consumes that field.
        "generated_test": state["candidate_test"],
        "final_test": state["candidate_test"],
        "reviewed_test": (
            state["reviewed_test"]
            if state["reviewer_called"]
            else None
        ),

        # Reviewer information.
        "reviewer_called": state["reviewer_called"],
        "repair_changed": state["repair_changed"],
        "repair_duplicate": state["repair_duplicate"],
        "repair_evaluated": repair_evaluation is not None,
        "raw_review_response": state["raw_review_response"],
        "review_feedback": state["review_feedback"],
        "review_finish_reason": state[
            "review_finish_reason"
        ],
        "review_usage": state["review_usage"],

        # Separate traces for the generator candidate and optional repair.
        "generator_evaluation": generator_evaluation,
        "repair_evaluation": repair_evaluation,
        "generator_candidate_passed": bool(
            generator_evaluation.get("test_passed", False)
        ),
        "generator_syntax_valid": bool(
            generator_evaluation.get("syntax_valid", False)
        ),
        "generator_format_valid": bool(
            generator_evaluation.get("format_valid", False)
        ),

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
        "execution_feedback": state["execution_feedback"],

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
        "execution_time_seconds": total_round_execution_time,
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
) -> Literal["add_to_suite", "review", "close_round"]:
    """Accept, request one repair, or close the current round."""

    if state["accepted"]:
        return "add_to_suite"

    # A timeout is not sent to the LLM: it does not provide a reliable
    # correctness signal and retrying can double an already expensive round.
    if state["timed_out"]:
        return "close_round"

    if not state["reviewer_called"]:
        return "review"

    return "close_round"


def route_after_review(
    state: AgentState,
) -> Literal["evaluate_repair", "close_round"]:
    """Evaluate only a genuinely changed repair candidate."""

    if state["repair_changed"]:
        return "evaluate_repair"

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
            round_result["generator_syntax_valid"]
            and round_result["generator_format_valid"]
        )
    )

    direct_generator_passes = sum(
        1
        for round_result in rounds
        if round_result["generator_candidate_passed"]
    )

    reviewer_calls = sum(
        1
        for round_result in rounds
        if round_result["reviewer_called"]
    )

    evaluated_repairs = sum(
        1
        for round_result in rounds
        if round_result["repair_evaluated"]
    )

    successful_repairs = sum(
        1
        for round_result in rounds
        if (
            isinstance(round_result.get("repair_evaluation"), dict)
            and round_result["repair_evaluation"].get(
                "test_passed",
                False,
            )
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
        "system": "multiagent_generator_conditional_reviewer",
        "dataset_path": state["dataset_path"],
        "cyclomatic_complexity": state[
            "cyclomatic_complexity"
        ],
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
        "reviewed_tests": reviewer_calls,
        "reviewer_calls": reviewer_calls,
        "evaluated_repairs": evaluated_repairs,
        "successful_repairs": successful_repairs,
        "accepted_tests": accepted_candidates,
        "valid_generation_rate": round(
            valid_generations / max_k,
            4,
        ) if max_k else 0.0,
        "candidate_pass_rate": round(
            passed_candidates / max_k,
            4,
        ) if max_k else 0.0,
        "direct_generator_pass_rate": round(
            direct_generator_passes / max_k,
            4,
        ) if max_k else 0.0,
        "repair_success_rate": round(
            successful_repairs / reviewer_calls,
            4,
        ) if reviewer_calls else 0.0,
        "evaluated_repair_success_rate": round(
            successful_repairs / evaluated_repairs,
            4,
        ) if evaluated_repairs else 0.0,

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

def load_task_by_id(dataset_path: Path,task_id: str,) -> ULTTask:
    """Load one ULT task by its string identifier."""

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    tasks = load_ult(dataset_path)

    for task in tasks:
        if str(task.task_id) == str(task_id):
            return task

    raise ValueError(
        f"Task {task_id} was not found in {dataset_path}."
    )
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
graph.add_edge("generator", "evaluator")
graph.add_conditional_edges(
    "evaluator",
    route_after_evaluation,
    {
        "add_to_suite": "suite_adder",
        "review": "reviewer",
        "close_round": "record_round",
    }
)
graph.add_conditional_edges(
    "reviewer",
    route_after_review,
    {
        "evaluate_repair": "evaluator",
        "close_round": "record_round",
    },
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


def build_initial_state(
    task: ULTTask,
    model_id: str,
    dataset_path: Path,
    output_dir: Path,
    max_k: int = 5,
    timeout_seconds: int = 30,
) -> AgentState:
    """Create an independent graph state for one ULT task."""

    return {
        # Task and model.
        "model_id": model_id,
        "task_id": str(task.task_id),
        "function_name": task.func_name,
        "solution_code": task.code,
        "description": task.description,
        "cyclomatic_complexity": task.cyclomatic_complexity,
        "dataset_path": str(dataset_path.resolve()),

        # Execution.
        "output_dir": str(output_dir.resolve()),
        "timeout_seconds": timeout_seconds,

        # Graph control.
        "counter": 1,
        "max_rounds": max_k,

        # Generator.
        "generated_test": "",
        "candidate_test": "",
        "generation_time_seconds": 0.0,
        "generator_finish_reason": None,
        "generator_usage": None,

        # Reviewer.
        "reviewed_test": "",
        "raw_review_response": "",
        "review_feedback": "",
        "review_time_seconds": 0.0,
        "review_finish_reason": None,
        "review_usage": None,
        "reviewer_called": False,
        "repair_changed": False,
        "repair_duplicate": False,

        # Stage-specific deterministic evaluations.
        "generator_evaluation": None,
        "repair_evaluation": None,

        # Acceptance.
        "accepted": False,
        "accepted_tests": [],

        # Validation.
        "syntax_valid": False,
        "format_valid": False,
        "format_errors": [],
        "duplicate_name": False,
        "test_function_name": None,

        # Execution result.
        "pytest_exit_code": None,
        "test_passed": False,
        "timed_out": False,
        "candidate_execution_time_seconds": 0.0,
        "fallback_suite_execution_time_seconds": 0.0,
        "execution_time_seconds": 0.0,
        "execution_feedback": "",
        "candidate_stdout": "",
        "candidate_stderr": "",

        # Accumulated suite.
        "suite_passed": False,
        "line_coverage": None,
        "branch_coverage": None,
        "line_coverage_gain": 0.0,
        "branch_coverage_gain": 0.0,
        "improved_coverage": False,

        # Full trace.
        "rounds": [],
    }


def evaluate_multiagent_task(
    task: ULTTask,
    model_id: str,
    dataset_path: Path,
    output_dir: Path,
    max_k: int = 5,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Run the compiled multi-agent graph for one ULT task."""

    initial_state = build_initial_state(
        task=task,
        model_id=model_id,
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_k=max_k,
        timeout_seconds=timeout_seconds,
    )

    app.invoke(initial_state)

    result_path = output_dir / "result.json"

    if not result_path.is_file():
        raise RuntimeError(
            f"The graph did not create its result file: {result_path}"
        )

    result = json.loads(result_path.read_text(encoding="utf-8"))

    if not isinstance(result, dict):
        raise RuntimeError(
            f"Invalid graph result in: {result_path}"
        )

    return result


def main() -> None:
    """Preserve the original single-task smoke execution."""

    smoke_task_id = "10"
    smoke_output_dir = (
        PROJECT_ROOT
        / "runs"
        / "ult"
        / "multiagent"
        / "smoke"
        / f"task_{smoke_task_id}"
    )

    available_models = client.models.list().data

    if not available_models:
        raise RuntimeError("LM Studio is running, but no model is loaded.")

    model_id = os.getenv(
        "LM_STUDIO_MODEL",
        available_models[0].id,
    )

    task = load_task_by_id(
        dataset_path=DATASET_PATH,
        task_id=smoke_task_id,
    )

    print(f"Model: {model_id}")
    print(f"Task: {task.task_id} - {task.func_name}")
    print(
        "Cyclomatic complexity: "
        f"{task.cyclomatic_complexity}"
    )
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {smoke_output_dir}")

    final_state = app.invoke(
        build_initial_state(
            task=task,
            model_id=model_id,
            dataset_path=DATASET_PATH,
            output_dir=smoke_output_dir,
        )
    )

    print("\nAccepted tests:")
    for test in final_state["accepted_tests"]:
        print(test)
        print("-" * 50)


if __name__ == "__main__":
    main()