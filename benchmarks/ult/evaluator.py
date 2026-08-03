import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI


client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)


def remove_markdown_fences(response: str) -> str:
    match = re.search(
        r"```(?:python)?\s*(.*?)```",
        response,
        flags=re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return response.strip()


def generate_test(
    task,
    model_id: str,
    previous_tests: str,
    round_number: int,
) -> tuple[str, str, dict]:
    previous_section = (
        previous_tests
        or "No tests have been generated yet."
    )

    prompt = f"""
You are generating test case number {round_number} for a Python function.

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
  from solution import {task.func_name}
- Generate a case that is meaningfully different from previous tests.
- Prefer an uncovered boundary, branch, exceptional case or input category.
- Use a concrete expected value.
- Do not reproduce or modify the implementation.
- Do not access external resources.
- The test must be deterministic and executable with pytest.

FUNCTION NAME:
{task.func_name}

DESCRIPTION:
{task.description}

SOURCE CODE:
{task.code}

PREVIOUSLY GENERATED TESTS:
{previous_section}
""".strip()

    response = client.chat.completions.create(
        model=model_id,
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

    choice = response.choices[0]
    raw_response = choice.message.content

    if raw_response is None:
        raise RuntimeError(
            "LM Studio returned an empty response."
        )

    generated_test = remove_markdown_fences(raw_response)

    usage = (
        response.usage.model_dump()
        if response.usage is not None
        else None
    )
    generation_metadata = {
        "finish_reason": choice.finish_reason,
        "usage": usage,
    }

    return raw_response, generated_test, generation_metadata


def validate_generated_test(
    code: str,
    expected_function_name: str,
) -> dict:
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return {
            "syntax_valid": False,
            "format_valid": False,
            "format_errors": [
                f"SyntaxError on line {error.lineno}: {error.msg}"
            ],
        }

    imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    test_functions = [
        node
        for node in tree.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )
        and node.name.startswith("test_")
    ]
    all_functions = [
        node
        for node in tree.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )
    ]
    test_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]
    asserts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
    ]

    errors = []

    if len(imports) != 1:
        errors.append(
            f"Expected 1 import statement, found {len(imports)}."
        )
    else:
        imported_node = imports[0]
        correct_import = (
            isinstance(imported_node, ast.ImportFrom)
            and imported_node.module == "solution"
            and imported_node.level == 0
            and len(imported_node.names) == 1
            and imported_node.names[0].name
            == expected_function_name
            and imported_node.names[0].asname is None
        )

        if not correct_import:
            errors.append(
                "The import must be exactly: "
                f"from solution import {expected_function_name}"
            )

    if len(test_functions) != 1:
        errors.append(
            "Expected 1 test function, "
            f"found {len(test_functions)}."
        )

    if len(all_functions) != 1:
        errors.append(
            "Helper functions or additional functions are not allowed."
        )

    if test_classes:
        errors.append("Test classes are not allowed.")

    if len(asserts) != 1:
        errors.append(
            f"Expected 1 assert statement, found {len(asserts)}."
        )

    return {
        "syntax_valid": True,
        "format_valid": not errors,
        "format_errors": errors,
    }


def empty_execution_result() -> dict:
    return {
        "pytest_exit_code": None,
        "test_passed": False,
        "timed_out": False,
        "execution_time_seconds": None,
        "stdout": "",
        "stderr": "",
        "line_coverage": None,
        "branch_coverage": None,
    }


def run_generated_test(
    output_dir: Path,
    timeout_seconds: int = 30,
) -> dict:
    coverage_path = output_dir / "coverage.json"

    if coverage_path.exists():
        coverage_path.unlink()

    start_time = time.perf_counter()

    try:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "test_solution.py",
                "-q",
                "--cov=solution",
                "--cov-branch",
                f"--cov-report=json:{coverage_path.name}",
                "--cov-report=term-missing",
            ],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        result = {
            "pytest_exit_code": process.returncode,
            "test_passed": process.returncode == 0,
            "timed_out": False,
            "execution_time_seconds": round(
                time.perf_counter() - start_time,
                4,
            ),
            "stdout": process.stdout,
            "stderr": process.stderr,
            "line_coverage": None,
            "branch_coverage": None,
        }

        if coverage_path.exists():
            coverage_data = json.loads(
                coverage_path.read_text(encoding="utf-8")
            )
            totals = coverage_data.get("totals", {})
            statement_count = totals.get(
                "num_statements",
                0,
            )
            covered_lines = totals.get("covered_lines", 0)
            branch_count = totals.get("num_branches", 0)
            covered_branches = totals.get(
                "covered_branches",
                0,
            )

            if statement_count > 0:
                result["line_coverage"] = round(
                    100 * covered_lines / statement_count,
                    2,
                )

            if branch_count > 0:
                result["branch_coverage"] = round(
                    100 * covered_branches / branch_count,
                    2,
                )

        return result

    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""

        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")

        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")

        return {
            "pytest_exit_code": None,
            "test_passed": False,
            "timed_out": True,
            "execution_time_seconds": round(
                time.perf_counter() - start_time,
                4,
            ),
            "stdout": stdout,
            "stderr": stderr,
            "line_coverage": None,
            "branch_coverage": None,
        }


def remove_imports(code: str) -> str:
    lines = code.splitlines()

    return "\n".join(
        line
        for line in lines
        if not line.strip().startswith(
            "from solution import "
        )
    ).strip()


def build_test_suite(
    function_name: str,
    accepted_tests: list[str],
    candidate_test: str | None = None,
) -> str:
    parts = [f"from solution import {function_name}"]

    for test in accepted_tests:
        parts.append(remove_imports(test))

    if candidate_test is not None:
        parts.append(remove_imports(candidate_test))

    return "\n\n".join(parts).strip() + "\n"


def get_test_function_name(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in tree.body:
        if (
            isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            )
            and node.name.startswith("test_")
        ):
            return node.name

    return None


def coverage_gain(
    current: float | None,
    previous: float | None,
) -> float:
    if current is None:
        return 0.0

    if previous is None:
        return current

    return round(current - previous, 2)


def evaluate_task(
    task,
    model_id: str,
    output_dir: Path,
    max_k: int = 5,
    timeout_seconds: int = 30,
) -> dict:
    """Evaluate one ULT task with an accumulated suite of up to k tests."""
    output_dir.mkdir(parents=True, exist_ok=True)

    solution_path = output_dir / "solution.py"
    test_path = output_dir / "test_solution.py"
    result_path = output_dir / "result.json"

    solution_path.write_text(task.code, encoding="utf-8")

    accepted_tests: list[str] = []
    attempted_tests: list[str] = []
    accepted_names: set[str] = set()
    round_results: list[dict] = []
    total_generation_time = 0.0
    total_execution_time = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    previous_line_coverage: float | None = None
    previous_branch_coverage: float | None = None

    for round_number in range(1, max_k + 1):
        previous_tests = "\n\n".join(attempted_tests)

        generation_start = time.perf_counter()
        (
            raw_model_response,
            generated_test,
            generation_metadata,
        ) = generate_test(
            task=task,
            model_id=model_id,
            previous_tests=previous_tests,
            round_number=round_number,
        )
        attempted_tests.append(generated_test)
        generation_time = round(
            time.perf_counter() - generation_start,
            4,
        )
        total_generation_time += generation_time

        usage = generation_metadata["usage"]
        if usage is not None:
            total_prompt_tokens += usage.get(
                "prompt_tokens",
                0,
            ) or 0
            total_completion_tokens += usage.get(
                "completion_tokens",
                0,
            ) or 0
            total_tokens += usage.get(
                "total_tokens",
                0,
            ) or 0

        candidate_path = (
            output_dir / f"candidate_{round_number}.py"
        )
        candidate_path.write_text(
            generated_test,
            encoding="utf-8",
        )

        validation = validate_generated_test(
            generated_test,
            task.func_name,
        )
        test_name = get_test_function_name(generated_test)
        duplicate_name = (
            test_name is not None
            and test_name in accepted_names
        )

        if duplicate_name:
            validation["format_valid"] = False
            validation["format_errors"].append(
                f"Duplicate test function name: {test_name}"
            )

        candidate_accepted = False
        candidate_execution = empty_execution_result()

        if (
            validation["syntax_valid"]
            and validation["format_valid"]
        ):
            candidate_suite = build_test_suite(
                function_name=task.func_name,
                accepted_tests=accepted_tests,
                candidate_test=generated_test,
            )
            test_path.write_text(
                candidate_suite,
                encoding="utf-8",
            )
            candidate_execution = run_generated_test(
                output_dir,
                timeout_seconds=timeout_seconds,
            )

            execution_time = candidate_execution[
                "execution_time_seconds"
            ]
            if execution_time is not None:
                total_execution_time += execution_time

            if candidate_execution["test_passed"]:
                accepted_tests.append(generated_test)

                if test_name is not None:
                    accepted_names.add(test_name)

                candidate_accepted = True

        accepted_suite = build_test_suite(
            task.func_name,
            accepted_tests,
        )
        test_path.write_text(
            accepted_suite,
            encoding="utf-8",
        )

        if candidate_accepted:
            cumulative_execution = candidate_execution
        elif accepted_tests:
            cumulative_execution = run_generated_test(
                output_dir,
                timeout_seconds=timeout_seconds,
            )
            execution_time = cumulative_execution[
                "execution_time_seconds"
            ]
            if execution_time is not None:
                total_execution_time += execution_time
        else:
            cumulative_execution = empty_execution_result()

        line_gain = coverage_gain(
            cumulative_execution["line_coverage"],
            previous_line_coverage,
        )
        branch_gain = coverage_gain(
            cumulative_execution["branch_coverage"],
            previous_branch_coverage,
        )
        improved_coverage = (
            line_gain > 0.0 or branch_gain > 0.0
        )

        round_result = {
            "k": round_number,
            "generation_time_seconds": generation_time,
            "finish_reason": generation_metadata[
                "finish_reason"
            ],
            "usage": usage,
            "raw_model_response": raw_model_response,
            "generated_test": generated_test,
            "test_function_name": test_name,
            "syntax_valid": validation["syntax_valid"],
            "format_valid": validation["format_valid"],
            "format_errors": validation["format_errors"],
            "duplicate_name": duplicate_name,
            "candidate_passed": candidate_execution[
                "test_passed"
            ],
            "candidate_accepted": candidate_accepted,
            "candidate_pytest_exit_code": (
                candidate_execution["pytest_exit_code"]
            ),
            "candidate_timed_out": candidate_execution[
                "timed_out"
            ],
            "candidate_stdout": candidate_execution["stdout"],
            "candidate_stderr": candidate_execution["stderr"],
            "accepted_tests": len(accepted_tests),
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
        round_results.append(round_result)

        previous_line_coverage = cumulative_execution[
            "line_coverage"
        ]
        previous_branch_coverage = cumulative_execution[
            "branch_coverage"
        ]

        print(
            f"  Round {round_number}/{max_k} | "
            f"Accepted: {candidate_accepted} | "
            f"LCov: {cumulative_execution['line_coverage']} | "
            f"BCov: {cumulative_execution['branch_coverage']}"
        )

    valid_generations = sum(
        1
        for result in round_results
        if result["syntax_valid"] and result["format_valid"]
    )
    passed_candidates = sum(
        1
        for result in round_results
        if result["candidate_passed"]
    )
    coverage_improving_tests = sum(
        1
        for result in round_results
        if result["improved_coverage"]
    )

    experiment_result = {
        "task_id": task.task_id,
        "function_name": task.func_name,
        "model_id": model_id,
        "max_k": max_k,
        "evaluation_status": "completed",
        "total_generation_time_seconds": round(
            total_generation_time,
            4,
        ),
        "total_execution_time_seconds": round(
            total_execution_time,
            4,
        ),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "generated_tests": max_k,
        "accepted_tests": len(accepted_tests),
        "valid_generation_rate": round(
            valid_generations / max_k,
            4,
        ) if max_k else 0.0,
        "candidate_pass_rate": round(
            passed_candidates / max_k,
            4,
        ) if max_k else 0.0,
        "coverage_improving_tests": coverage_improving_tests,
        "coverage_improvement_rate": round(
            coverage_improving_tests / max_k,
            4,
        ) if max_k else 0.0,
        "final_line_coverage": (
            round_results[-1]["line_coverage"]
            if round_results
            else None
        ),
        "final_branch_coverage": (
            round_results[-1]["branch_coverage"]
            if round_results
            else None
        ),
        "suite_pass_at_k": [
            result["suite_passed"]
            for result in round_results
        ],
        "lcov_at_k": [
            result["line_coverage"]
            for result in round_results
        ],
        "bcov_at_k": [
            result["branch_coverage"]
            for result in round_results
        ],
        "rounds": round_results,
    }

    result_path.write_text(
        json.dumps(
            experiment_result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return experiment_result