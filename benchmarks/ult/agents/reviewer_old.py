"""Reviewer agent and deterministic candidate evaluation node."""
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from benchmarks.ult.evaluator import (
    build_test_suite,
    coverage_gain,
    empty_execution_result,
    get_test_function_name,
    remove_markdown_fences,
    run_generated_test,
    validate_generated_test,
)


client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)


def summarize_feedback_for_prompt(
    feedback: str,
    limit: int = 800,
) -> str:
    """Keep useful pytest details without inflating the LLM prompt."""

    if not feedback.strip():
        return "No additional feedback."

    relevant_lines = [
        line.strip()
        for line in feedback.splitlines()
        if line.strip()
        and (
            line.lstrip().startswith("E ")
            or "AssertionError" in line
            or "SyntaxError" in line
            or "NameError" in line
            or "TypeError" in line
            or "ImportError" in line
            or "AttributeError" in line
            or "Duplicate test function name" in line
            or "timed out" in line.lower()
            or "failed" in line.lower()
        )
    ]

    summary = "\n".join(relevant_lines) or feedback.strip()

    if len(summary) > limit:
        return summary[:limit] + "\n... <feedback truncated>"

    return summary


def format_previous_attempts(
    rounds: list[dict[str, Any]],
) -> str:
    """Format original proposals, reviewed tests and pytest results."""

    if not rounds:
        return "No previous attempts."

    attempts: list[str] = []

    for round_result in rounds:
        status = (
            "accepted"
            if round_result.get("candidate_accepted", False)
            else "rejected"
        )

        feedback = summarize_feedback_for_prompt(
            str(round_result.get("execution_feedback") or "")
        )

        attempts.append(
            f"""
ROUND {round_result.get("k", "unknown")} - {status.upper()}

ORIGINAL GENERATOR OUTPUT:
{round_result.get("generator_test", "Unavailable")}

REVIEWED TEST EXECUTED BY PYTEST:
{round_result.get("generated_test", "Unavailable")}

EVALUATION FEEDBACK:
{feedback}
""".strip()
        )

    return "\n\n".join(attempts)


def reviewer_node(state: dict[str, Any]) -> dict[str, Any]:
    """Repair one candidate after its first deterministic evaluation failed."""

    previous_attempts = format_previous_attempts(
        state.get("rounds", [])
    )

    failed_candidate = remove_markdown_fences(
        state["candidate_test"]
    ).strip()

    pytest_feedback = summarize_feedback_for_prompt(
        state.get("execution_feedback", "")
    )

    prompt = f"""
You are repairing a pytest test that has already failed deterministic validation
or a real pytest execution.

FUNCTION NAME:
{state["function_name"]}

DESCRIPTION:
{state["description"]}

SOURCE CODE:
{state["solution_code"]}

FAILED TEST:
{failed_candidate}

CURRENT VALIDATION/PYTEST FEEDBACK:
{pytest_feedback}

PREVIOUS ATTEMPTS AND RESULTS:
{previous_attempts}

Repair the failed test using the source code and the concrete feedback above.

OUTPUT FORMAT:
- Return only valid Python code.
- Do not use Markdown fences.
- Include exactly one import statement.
- The import must be exactly:
  from solution import {state["function_name"]}
- Include exactly one test function whose name starts with test_.
- Include exactly one assert statement.
- Test exactly one concrete input case.
- Do not include comments, explanations, helper functions, classes
  or parametrization.

REVIEW REQUIREMENTS:
- Repair the current failed test; do not invent an unrelated replacement.
- Keep the same concrete input when possible and correct the expected result.
- If the failure is structural or raises an exception, make only the smallest
  change needed to produce a valid executable test.
- The expected value must agree with the source code.
- The test function name must be different from all accepted tests.
- The test must not duplicate a previously accepted input case.
- Never return a test identical to a previously rejected reviewed test.
- Never return the failed test unchanged.
- Use the current pytest feedback before considering previous attempts.
- Compare against both original generator outputs and reviewed tests.
- Do not compute the expected result by calling the function under test twice
  or by comparing the function with itself.
- Do not return the complete accumulated suite.
- Return only the corrected atomic test.
""".strip()

    review_start = time.perf_counter()

    response = client.chat.completions.create(
        model=state["model_id"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert reviewer of Python pytest tests. "
                    "Return only the corrected atomic test."
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

    review_time = round(
        time.perf_counter() - review_start,
        4,
    )

    choice = response.choices[0]
    raw_response = choice.message.content

    if raw_response is None:
        raise RuntimeError(
            "The reviewer returned an empty response."
        )

    reviewed_test = remove_markdown_fences(raw_response).strip()

    previous_rejected_tests = {
        remove_markdown_fences(
            str(
                round_result.get("generated_test")
                or round_result.get("reviewed_test")
                or ""
            )
        ).strip()
        for round_result in state.get("rounds", [])
        if not round_result.get("candidate_accepted", False)
    }

    repair_duplicate = (
        not reviewed_test
        or reviewed_test == failed_candidate
        or reviewed_test in previous_rejected_tests
    )
    repair_changed = not repair_duplicate

    usage = (
        response.usage.model_dump()
        if response.usage is not None
        else None
    )

    return {
        "raw_review_response": raw_response,
        "reviewed_test": reviewed_test,
        "candidate_test": (
            reviewed_test if repair_changed else failed_candidate
        ),
        "reviewer_called": True,
        "repair_changed": repair_changed,
        "repair_duplicate": repair_duplicate,
        "review_feedback": (
            "Repair candidate produced by the reviewer."
            if repair_changed
            else "Reviewer returned an empty, unchanged or repeated repair; "
            "the repair was not evaluated."
        ),
        "review_time_seconds": review_time,
        "review_finish_reason": choice.finish_reason,
        "review_usage": usage,
    }

def get_accepted_test_names(
    accepted_tests: list[str],
) -> set[str]:
    """Extract the function names from all accepted tests."""

    names: set[str] = set()

    for accepted_test in accepted_tests:
        name = get_test_function_name(accepted_test)

        if name is not None:
            names.add(name)

    return names


def build_evaluation_feedback(
    validation: dict,
    execution: dict,
) -> str:
    """Build concise feedback for the generator and result record."""

    format_errors = validation.get("format_errors", [])

    if format_errors:
        return "; ".join(format_errors)

    if execution["timed_out"]:
        return "Pytest execution timed out."

    if not execution["test_passed"]:
        output = (
            execution["stderr"]
            or execution["stdout"]
            or "The candidate did not pass."
        )

        # Avoid inserting enormous pytest outputs into the graph state.
        return output[:4000]

    return "Candidate accepted."

def evaluator_node(state: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the current generator or repair candidate."""

    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    solution_path = output_dir / "solution.py"
    test_path = output_dir / "test_solution.py"
    evaluation_stage = (
        "repair" if state.get("reviewer_called", False) else "generator"
    )
    candidate_path = output_dir / (
        f"candidate_{state['counter']}_{evaluation_stage}.py"
    )

    candidate = remove_markdown_fences(
        state["candidate_test"]
    )

    solution_path.write_text(
        state["solution_code"],
        encoding="utf-8",
    )

    candidate_path.write_text(
        candidate,
        encoding="utf-8",
    )

    validation = validate_generated_test(
        code=candidate,
        expected_function_name=state["function_name"],
    )

    test_name = get_test_function_name(candidate)

    accepted_names = get_accepted_test_names(
        state["accepted_tests"]
    )

    duplicate_name = (
        test_name is not None
        and test_name in accepted_names
    )

    if duplicate_name:
        validation["format_valid"] = False
        validation["format_errors"].append(
            f"Duplicate test function name: {test_name}"
        )

    candidate_execution = empty_execution_result()
    candidate_accepted = False

    if (
        validation["syntax_valid"]
        and validation["format_valid"]
    ):
        candidate_suite = build_test_suite(
            function_name=state["function_name"],
            accepted_tests=state["accepted_tests"],
            candidate_test=candidate,
        )

        test_path.write_text(
            candidate_suite,
            encoding="utf-8",
        )

        candidate_execution = run_generated_test(
            output_dir=output_dir,
            timeout_seconds=state.get(
                "timeout_seconds",
                30,
            ),
        )

        candidate_accepted = candidate_execution[
            "test_passed"
        ]

    candidate_execution_time = (
        candidate_execution["execution_time_seconds"]
        or 0.0
    )

    # Additional execution performed only when a candidate is rejected
    # and a previously accepted suite exists.
    fallback_suite_execution_time = 0.0

    if candidate_accepted:
        cumulative_execution = candidate_execution

    elif state["accepted_tests"]:
        accepted_suite = build_test_suite(
            function_name=state["function_name"],
            accepted_tests=state["accepted_tests"],
        )

        test_path.write_text(
            accepted_suite,
            encoding="utf-8",
        )

        cumulative_execution = run_generated_test(
            output_dir=output_dir,
            timeout_seconds=state.get(
                "timeout_seconds",
                30,
            ),
        )

        fallback_suite_execution_time = (
            cumulative_execution["execution_time_seconds"]
            or 0.0
        )

    else:
        # Keep a syntactically valid file even if no test was accepted.
        empty_suite = build_test_suite(
            function_name=state["function_name"],
            accepted_tests=[],
        )

        test_path.write_text(
            empty_suite,
            encoding="utf-8",
        )

        cumulative_execution = empty_execution_result()

    round_execution_time = round(
        candidate_execution_time
        + fallback_suite_execution_time,
        4,
    )

    line_gain = coverage_gain(
        current=cumulative_execution["line_coverage"],
        previous=state["line_coverage"],
    )

    branch_gain = coverage_gain(
        current=cumulative_execution["branch_coverage"],
        previous=state["branch_coverage"],
    )

    improved_coverage = (
        line_gain > 0.0
        or branch_gain > 0.0
    )

    feedback = build_evaluation_feedback(
        validation=validation,
        execution=candidate_execution,
    )

    print(
        f"Round {state['counter']} ({evaluation_stage}): "
        f"{'accepted' if candidate_accepted else 'rejected'}"
    )

    print(f"Evaluation: {feedback}")

    print(
        "Coverage: "
        f"LCov={cumulative_execution['line_coverage']} | "
        f"BCov={cumulative_execution['branch_coverage']}"
    )

    print(
        "Execution time: "
        f"candidate={candidate_execution_time:.4f}s | "
        f"fallback={fallback_suite_execution_time:.4f}s | "
        f"total={round_execution_time:.4f}s"
    )

    evaluation_snapshot = {
        "syntax_valid": validation["syntax_valid"],
        "format_valid": validation["format_valid"],
        "format_errors": validation["format_errors"],
        "duplicate_name": duplicate_name,
        "test_function_name": test_name,
        "pytest_exit_code": candidate_execution["pytest_exit_code"],
        "test_passed": candidate_execution["test_passed"],
        "timed_out": candidate_execution["timed_out"],
        "candidate_execution_time_seconds": round(
            candidate_execution_time,
            4,
        ),
        "fallback_suite_execution_time_seconds": round(
            fallback_suite_execution_time,
            4,
        ),
        "execution_time_seconds": round_execution_time,
        "execution_feedback": feedback,
        "candidate_stdout": candidate_execution["stdout"],
        "candidate_stderr": candidate_execution["stderr"],
        "suite_passed": cumulative_execution["test_passed"],
        "line_coverage": cumulative_execution["line_coverage"],
        "branch_coverage": cumulative_execution["branch_coverage"],
        "line_coverage_gain": line_gain,
        "branch_coverage_gain": branch_gain,
        "improved_coverage": improved_coverage,
    }

    stage_result_key = (
        "repair_evaluation"
        if state.get("reviewer_called", False)
        else "generator_evaluation"
    )

    return {
        # Current candidate evaluated by pytest.
        "candidate_test": candidate,
        stage_result_key: evaluation_snapshot,

        # Acceptance and validation.
        "accepted": candidate_accepted,
        "syntax_valid": validation["syntax_valid"],
        "format_valid": validation["format_valid"],
        "format_errors": validation["format_errors"],
        "duplicate_name": duplicate_name,
        "test_function_name": test_name,

        # Candidate execution.
        "pytest_exit_code": candidate_execution[
            "pytest_exit_code"
        ],
        "test_passed": candidate_execution["test_passed"],
        "timed_out": candidate_execution["timed_out"],
        "candidate_execution_time_seconds": round(
            candidate_execution_time,
            4,
        ),
        "fallback_suite_execution_time_seconds": round(
            fallback_suite_execution_time,
            4,
        ),

        # Total execution time of this graph round.
        "execution_time_seconds": round_execution_time,

        "execution_feedback": feedback,
        "candidate_stdout": candidate_execution["stdout"],
        "candidate_stderr": candidate_execution["stderr"],

        # Accumulated-suite result.
        "suite_passed": cumulative_execution["test_passed"],
        "line_coverage": cumulative_execution[
            "line_coverage"
        ],
        "branch_coverage": cumulative_execution[
            "branch_coverage"
        ],
        "line_coverage_gain": line_gain,
        "branch_coverage_gain": branch_gain,
        "improved_coverage": improved_coverage,
    }

