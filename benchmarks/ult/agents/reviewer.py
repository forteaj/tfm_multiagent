"""Reviewer agent and deterministic candidate evaluation node."""

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


def reviewer_node(state: dict[str, Any]) -> dict[str, Any]:
    """Review and correct one generated test candidate."""

    previous_tests = (
        "\n\n".join(state["accepted_tests"])
        or "No tests have been accepted yet."
    )

    prompt = f"""
You are reviewing a pytest test generated for a Python function.

FUNCTION NAME:
{state["function_name"]}

SOURCE CODE:
{state["solution_code"]}

GENERATED TEST:
{state["generated_test"]}

PREVIOUSLY ACCEPTED TESTS:
{previous_tests}

Review and correct the generated test.

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
- The expected value must agree with the source code.
- The test function name must be different from all accepted tests.
- The test must not duplicate a previously accepted input case.
- Do not return the complete accumulated suite.
- Return only the corrected atomic test.
""".strip()

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

    choice = response.choices[0]
    raw_response = choice.message.content

    if raw_response is None:
        raise RuntimeError(
            "The reviewer returned an empty response."
        )

    reviewed_test = remove_markdown_fences(raw_response)

    usage = (
        response.usage.model_dump()
        if response.usage is not None
        else None
    )

    return {
        "raw_review_response": raw_response,
        "reviewed_test": reviewed_test,
        "review_feedback": "Candidate reviewed by the reviewer agent.",
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
    """Evaluate the reviewed candidate using the existing ULT evaluator."""

    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    solution_path = output_dir / "solution.py"
    test_path = output_dir / "test_solution.py"
    candidate_path = (
        output_dir / f"candidate_{state['counter']}.py"
    )

    candidate = remove_markdown_fences(
        state["reviewed_test"]
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
        f"Round {state['counter']}: "
        f"{'accepted' if candidate_accepted else 'rejected'}"
    )

    print(f"Evaluation: {feedback}")

    print(
        "Coverage: "
        f"LCov={cumulative_execution['line_coverage']} | "
        f"BCov={cumulative_execution['branch_coverage']}"
    )

    return {
        # Store the fence-free candidate.
        "reviewed_test": candidate,

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
        "execution_time_seconds": candidate_execution[
            "execution_time_seconds"
        ],
        "execution_feedback": feedback,
        "candidate_stdout": candidate_execution["stdout"],
        "candidate_stderr": candidate_execution["stderr"],

        # Accumulated-suite coverage.
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