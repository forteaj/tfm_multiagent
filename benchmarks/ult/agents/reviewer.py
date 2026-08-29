"""Reviewer agent and deterministic candidate evaluation node."""

import ast
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
    limit: int = 1200,
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
            or "ValueError" in line
            or "IndexError" in line
            or "KeyError" in line
            or "ImportError" in line
            or "AttributeError" in line
            or "RuntimeError" in line
            or "Duplicate test function name" in line
            or "timed out" in line.lower()
            or "failed" in line.lower()
        )
    ]

    summary = "\n".join(relevant_lines) or feedback.strip()

    if len(summary) > limit:
        return summary[:limit] + "\n... <feedback truncated>"

    return summary


def normalize_test(test_code: str) -> str:
    """
    Normalize a test before comparing it with previous attempts.

    AST normalization detects equivalent tests even if whitespace,
    quotes or formatting differ.
    """

    cleaned = remove_markdown_fences(
        str(test_code)
    ).strip()

    if not cleaned:
        return ""

    try:
        tree = ast.parse(cleaned)

        return ast.dump(
            tree,
            include_attributes=False,
        )

    except SyntaxError:
        # Invalid Python cannot be normalized through the AST.
        return cleaned


def get_accepted_test_names(
    accepted_tests: list[str],
) -> set[str]:
    """Extract test-function names from the accepted suite."""

    names: set[str] = set()

    for accepted_test in accepted_tests:
        name = get_test_function_name(
            remove_markdown_fences(accepted_test)
        )

        if name is not None:
            names.add(name)

    return names


def get_previous_test_names(
    rounds: list[dict[str, Any]],
) -> set[str]:
    """Collect all test names proposed in previous rounds."""

    names: set[str] = set()

    test_fields = (
        "generator_test",
        "generated_test",
        "reviewed_test",
        "final_test",
    )

    for round_result in rounds:
        for field in test_fields:
            test = round_result.get(field)

            if not test:
                continue

            cleaned_test = remove_markdown_fences(
                str(test)
            ).strip()

            name = get_test_function_name(cleaned_test)

            if name is not None:
                names.add(name)

    return names


def get_previous_rejected_tests(
    rounds: list[dict[str, Any]],
) -> set[str]:
    """
    Collect normalized versions of every rejected proposal.

    All relevant fields are inspected instead of selecting only
    generated_test or reviewed_test.
    """

    rejected_tests: set[str] = set()

    test_fields = (
        "generator_test",
        "generated_test",
        "reviewed_test",
        "final_test",
    )

    for round_result in rounds:
        if round_result.get("candidate_accepted", False):
            continue

        for field in test_fields:
            test = round_result.get(field)

            if not test:
                continue

            normalized = normalize_test(str(test))

            if normalized:
                rejected_tests.add(normalized)

    return rejected_tests


def build_failure_specific_instructions(
    state: dict[str, Any],
) -> str:
    """Return focused instructions for the current failure type."""

    feedback = str(
        state.get("execution_feedback") or ""
    )

    feedback_lower = feedback.lower()
    syntax_valid = state.get("syntax_valid", True)
    format_valid = state.get("format_valid", True)
    format_errors = state.get("format_errors", [])

    if not syntax_valid:
        return """
FAILURE TYPE: SYNTAX ERROR

- Correct only the invalid Python syntax.
- Preserve the current test name, input and intended assertion.
- Do not replace the test with another case.
- Remove Markdown fences and non-Python text.
- Return a minimal syntactically valid atomic test.
""".strip()

    if not format_valid:
        errors = (
            "; ".join(str(error) for error in format_errors)
            or "The candidate violates the required atomic-test format."
        )

        return f"""
FAILURE TYPE: FORMAT VALIDATION ERROR

FORMAT ERRORS:
{errors}

- Rebuild the current test using the required atomic format.
- Preserve its concrete input and expected behavior whenever possible.
- Remove comments, helper functions, classes, parametrization,
  additional imports and additional asserts.
- Do not replace it with a different test case.
""".strip()

    if "assertionerror" in feedback_lower:
        return """
FAILURE TYPE: ASSERTION ERROR

Pytest compares:

    assert ACTUAL_RESULT == CURRENT_EXPECTED_RESULT

Interpret the failure literally:

- The value on the LEFT side of `==` is the actual function result.
- The value on the RIGHT side is the incorrect expected value.
- In `At index N diff: ACTUAL != EXPECTED`, the first value is actual.
- Preserve the current test name and concrete input.
- Replace only the incorrect expected value whenever possible.
- Copy the exact actual value reported by Pytest.
- Preserve its exact Python type and structure.
- Preserve spaces, signs, capitalization, tuple/list structure and complete
  floating-point digits.
- Do not round, simplify or manually recalculate a value that Pytest already
  reports.
- Never copy an ellipsis (`...`) from truncated output.
- Do not modify the input merely to avoid the assertion failure.
""".strip()

    exception_types = (
        "typeerror",
        "valueerror",
        "indexerror",
        "keyerror",
        "attributeerror",
        "nameerror",
        "importerror",
        "runtimeerror",
    )

    if any(
        exception_type in feedback_lower
        for exception_type in exception_types
    ):
        return """
FAILURE TYPE: RUNTIME EXCEPTION

- Read the final exception and the failing source-code line.
- Determine whether the test uses an invalid object, attribute, argument type,
  input size or input shape.
- Preserve the current test name.
- Make the smallest possible correction to the current concrete case.
- If the current input violates the function's domain, change it to the nearest
  clearly valid concrete input supported by the source code.
- Do not invent an expected result while the function still raises.
- Do not add `pytest.raises`; the output contract permits only the exact
  solution import.
- Do not replace the candidate with an unrelated test.
""".strip()

    return """
FAILURE TYPE: OTHER EXECUTION FAILURE

- Use the traceback and source code to identify the smallest correction.
- Preserve the current test name and concrete input whenever possible.
- Do not replace the candidate with an unrelated test.
""".strip()


def reviewer_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Repair the current candidate after deterministic evaluation failed."""

    rounds = state.get("rounds", [])

    failed_candidate = remove_markdown_fences(
        state["candidate_test"]
    ).strip()

    current_test_name = get_test_function_name(
        failed_candidate
    )

    pytest_feedback = summarize_feedback_for_prompt(
        state.get("execution_feedback", "")
    )

    accepted_test_names = get_accepted_test_names(
        state.get("accepted_tests", [])
    )

    accepted_names_text = (
        ", ".join(sorted(accepted_test_names))
        if accepted_test_names
        else "No accepted test names."
    )

    current_name_text = (
        current_test_name
        if current_test_name is not None
        else "The current test name could not be extracted."
    )

    failure_instructions = build_failure_specific_instructions(
        state
    )

    prompt = f"""
Repair exactly one failed pytest candidate.

The source implementation is the reference behavior. The objective is to
produce a characterization test that passes against this implementation and
can later be evaluated through mutation testing.

FUNCTION UNDER TEST:
{state["function_name"]}

FUNCTION DESCRIPTION:
{state["description"]}

SOURCE IMPLEMENTATION:
{state["solution_code"]}

ACCEPTED TEST NAMES:
{accepted_names_text}

CURRENT TEST NAME:
{current_name_text}

CURRENT FAILED TEST:
{failed_candidate}

CURRENT VALIDATION/PYTEST FEEDBACK:
{pytest_feedback}

FAILURE-SPECIFIC INSTRUCTIONS:
{failure_instructions}

REPAIR PRIORITY:
1. Repair only the current candidate.
2. Use the concrete Pytest feedback as the strongest evidence.
3. Make the smallest change that fixes the reported failure.
4. Preserve the test name and input unless the feedback proves that one of
   them is invalid.
5. Never use or reconstruct a candidate from another round.

MANDATORY OUTPUT CONTRACT:
- Return only Python code.
- Do not use Markdown fences.
- Use exactly this import:
  from solution import {state["function_name"]}
- Include exactly one test function named test_*.
- Include exactly one assert statement.
- Test exactly one concrete input case.
- Do not include any other import.
- Do not include comments or explanations.
- Do not include helper functions or classes.
- Do not include parametrization.
- Do not include manual expected-value calculations.
- Do not include the accumulated test suite.
- Do not return the failed candidate unchanged.

Before responding, silently verify that the output follows every mandatory
rule. Then return only the corrected atomic test.
""".strip()

    review_start = time.perf_counter()

    response = client.chat.completions.create(
        model=state["model_id"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You repair exactly the current pytest candidate. "
                    "Treat the source implementation and concrete Pytest "
                    "output as authoritative. Make the smallest "
                    "evidence-based change. Never select another test case. "
                    "Return only Python code."
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

    reviewed_test = remove_markdown_fences(
        raw_response
    ).strip()

    reviewed_test_name = get_test_function_name(
        reviewed_test
    )

    previous_test_names = get_previous_test_names(rounds)
    previous_rejected_tests = get_previous_rejected_tests(
        rounds
    )

    normalized_failed_candidate = normalize_test(
        failed_candidate
    )

    normalized_reviewed_test = normalize_test(
        reviewed_test
    )

    duplicate_name_feedback = (
        "duplicate test function name"
        in pytest_feedback.lower()
    )

    empty_response = not reviewed_test

    unchanged_candidate = (
        normalized_reviewed_test
        == normalized_failed_candidate
    )

    repeated_rejected_test = (
        normalized_reviewed_test
        and normalized_reviewed_test
        in previous_rejected_tests
    )

    accepted_name_duplicate = (
        reviewed_test_name is not None
        and reviewed_test_name in accepted_test_names
    )

    reused_previous_name = (
        reviewed_test_name is not None
        and reviewed_test_name in previous_test_names
        and reviewed_test_name != current_test_name
    )

    unrelated_name_change = (
        current_test_name is not None
        and reviewed_test_name is not None
        and reviewed_test_name != current_test_name
        and not duplicate_name_feedback
    )

    missing_test_name = reviewed_test_name is None

    rejection_reasons: list[str] = []

    if empty_response:
        rejection_reasons.append(
            "the reviewer returned an empty response"
        )

    if unchanged_candidate:
        rejection_reasons.append(
            "the reviewer returned the failed candidate unchanged"
        )

    if repeated_rejected_test:
        rejection_reasons.append(
            "the reviewer repeated a previously rejected test"
        )

    if accepted_name_duplicate:
        rejection_reasons.append(
            "the reviewer reused an accepted test function name"
        )

    if reused_previous_name:
        rejection_reasons.append(
            "the reviewer selected a test from a previous round"
        )

    if unrelated_name_change:
        rejection_reasons.append(
            "the reviewer changed the current test function name "
            "without a duplicate-name failure"
        )

    if missing_test_name:
        rejection_reasons.append(
            "the reviewer output has no valid test function name"
        )

    repair_duplicate = bool(rejection_reasons)
    repair_changed = not repair_duplicate

    usage = (
        response.usage.model_dump()
        if response.usage is not None
        else None
    )

    if repair_changed:
        review_feedback = (
            "Repair candidate produced by the reviewer."
        )
    else:
        review_feedback = (
            "Reviewer repair rejected: "
            + "; ".join(rejection_reasons)
            + "."
        )

    return {
        "raw_review_response": raw_response,
        "reviewed_test": reviewed_test,
        "candidate_test": (
            reviewed_test
            if repair_changed
            else failed_candidate
        ),
        "reviewer_called": True,
        "repair_changed": repair_changed,
        "repair_duplicate": repair_duplicate,
        "review_feedback": review_feedback,
        "review_time_seconds": review_time,
        "review_finish_reason": choice.finish_reason,
        "review_usage": usage,
    }


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
