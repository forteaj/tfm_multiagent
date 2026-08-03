import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

from benchmarks.ult.loader import load_ult


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "external"
    / "UnLeakedTestBench"
    / "datasets"
    / "ULT_Lite.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "runs" / "ult" / "smoke"
MAX_K = 5

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
) -> str:
    previous_section = previous_tests or "No tests have been generated yet."

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

PREVIOUSLY ACCEPTED TESTS:
{previous_section}
""".strip()

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": "You are an expert Python unit-test generator.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        max_tokens=2048,
    )

    raw_response = response.choices[0].message.content

    if raw_response is None:
        raise RuntimeError("LM Studio returned an empty response.")

    return remove_markdown_fences(raw_response)


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
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    all_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
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
            and imported_node.names[0].name == expected_function_name
            and imported_node.names[0].asname is None
        )

        if not correct_import:
            errors.append(
                "The import must be exactly: "
                f"from solution import {expected_function_name}"
            )

    if len(test_functions) != 1:
        errors.append(
            f"Expected 1 test function, found {len(test_functions)}."
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


def run_generated_test(output_dir: Path) -> dict:
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
            timeout=30,
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
            statement_count = totals.get("num_statements", 0)
            covered_lines = totals.get("covered_lines", 0)
            branch_count = totals.get("num_branches", 0)
            covered_branches = totals.get("covered_branches", 0)

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
        if not line.strip().startswith("from solution import ")
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
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        ):
            return node.name

    return None


def main() -> None:
    available_models = client.models.list().data

    if not available_models:
        raise RuntimeError(
            "LM Studio is running, but no model is loaded."
        )

    default_model = available_models[0].id
    model_id = os.getenv("LM_STUDIO_MODEL", default_model)

    tasks = load_ult(DATASET_PATH, limit=1)

    if not tasks:
        raise RuntimeError(
            f"No tasks were found in the dataset: {DATASET_PATH}"
        )

    task = tasks[0]

    print(f"Generating tests with model: {model_id}")
    print(f"Task: {task.task_id} - {task.func_name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    solution_path = OUTPUT_DIR / "solution.py"
    test_path = OUTPUT_DIR / "test_solution.py"
    result_path = OUTPUT_DIR / "result.json"

    solution_path.write_text(task.code, encoding="utf-8")

    accepted_tests: list[str] = []
    accepted_names: set[str] = set()
    round_results: list[dict] = []
    total_generation_time = 0.0
    total_execution_time = 0.0

    for round_number in range(1, MAX_K + 1):
        print(f"\n--- ROUND {round_number}/{MAX_K} ---")

        previous_tests = "\n\n".join(accepted_tests)
        generation_start = time.perf_counter()
        generated_test = generate_test(
            task=task,
            model_id=model_id,
            previous_tests=previous_tests,
            round_number=round_number,
        )
        generation_time = round(
            time.perf_counter() - generation_start,
            4,
        )
        total_generation_time += generation_time

        candidate_path = OUTPUT_DIR / f"candidate_{round_number}.py"
        candidate_path.write_text(generated_test, encoding="utf-8")

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

        if validation["syntax_valid"] and validation["format_valid"]:
            candidate_suite = build_test_suite(
                function_name=task.func_name,
                accepted_tests=accepted_tests,
                candidate_test=generated_test,
            )
            test_path.write_text(candidate_suite, encoding="utf-8")
            candidate_execution = run_generated_test(OUTPUT_DIR)

            if candidate_execution["execution_time_seconds"] is not None:
                total_execution_time += candidate_execution[
                    "execution_time_seconds"
                ]

            if candidate_execution["test_passed"]:
                accepted_tests.append(generated_test)

                if test_name is not None:
                    accepted_names.add(test_name)

                candidate_accepted = True

        accepted_suite = build_test_suite(
            task.func_name,
            accepted_tests,
        )
        test_path.write_text(accepted_suite, encoding="utf-8")

        if candidate_accepted:
            cumulative_execution = candidate_execution
        elif accepted_tests:
            cumulative_execution = run_generated_test(OUTPUT_DIR)

            if cumulative_execution["execution_time_seconds"] is not None:
                total_execution_time += cumulative_execution[
                    "execution_time_seconds"
                ]
        else:
            cumulative_execution = empty_execution_result()

        round_result = {
            "k": round_number,
            "generation_time_seconds": generation_time,
            "generated_test": generated_test,
            "test_function_name": test_name,
            "syntax_valid": validation["syntax_valid"],
            "format_valid": validation["format_valid"],
            "format_errors": validation["format_errors"],
            "duplicate_name": duplicate_name,
            "candidate_passed": candidate_execution["test_passed"],
            "candidate_accepted": candidate_accepted,
            "candidate_pytest_exit_code": candidate_execution[
                "pytest_exit_code"
            ],
            "candidate_timed_out": candidate_execution["timed_out"],
            "candidate_stdout": candidate_execution["stdout"],
            "candidate_stderr": candidate_execution["stderr"],
            "accepted_tests": len(accepted_tests),
            "suite_passed": cumulative_execution["test_passed"],
            "line_coverage": cumulative_execution["line_coverage"],
            "branch_coverage": cumulative_execution["branch_coverage"],
        }
        round_results.append(round_result)

        print(generated_test)
        print(
            f"Accepted: {candidate_accepted} | "
            f"Suite size: {len(accepted_tests)} | "
            f"LCov@{round_number}: "
            f"{cumulative_execution['line_coverage']} | "
            f"BCov@{round_number}: "
            f"{cumulative_execution['branch_coverage']}"
        )

    experiment_result = {
        "task_id": task.task_id,
        "function_name": task.func_name,
        "model_id": model_id,
        "max_k": MAX_K,
        "total_generation_time_seconds": round(
            total_generation_time,
            4,
        ),
        "total_execution_time_seconds": round(
            total_execution_time,
            4,
        ),
        "generated_tests": MAX_K,
        "accepted_tests": len(accepted_tests),
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

    print("\n--- FINAL RESULTS ---")
    print(
        json.dumps(
            experiment_result,
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\nFinal suite saved to: {test_path}")
    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()