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


def generate_test(task, model_id: str) -> str:
    prompt = f"""
You are generating one atomic pytest test case for a Python function.

OUTPUT FORMAT:
- Return only valid Python code.
- Include exactly one import statement.
- Include exactly one test function whose name starts with test_.
- Include exactly one assert statement.
- Test exactly one input case.
- Do not include comments or explanations.
- Do not include helper functions, classes, parametrization, Markdown,
  special tokens, placeholders, or any text outside the Python code.

TEST REQUIREMENTS:
- Import the function exactly as follows:
  from solution import {task.func_name}
- Choose one simple input with an expected result that can be determined confidently.
- Test observable behaviour using a concrete expected value.
- Do not reproduce, redefine, mock, or modify the implementation.
- Do not access external resources.
- The test must be deterministic and executable with pytest.

FUNCTION NAME:
{task.func_name}

DESCRIPTION:
{task.description}

SOURCE CODE:
{task.code}
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

    print(f"Generating a test with model: {model_id}")
    print(f"Task: {task.task_id} - {task.func_name}")

    generation_start = time.perf_counter()
    generated_test = generate_test(task, model_id)
    generation_time = round(
        time.perf_counter() - generation_start,
        4,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    solution_path = OUTPUT_DIR / "solution.py"
    test_path = OUTPUT_DIR / "test_solution.py"
    result_path = OUTPUT_DIR / "result.json"

    solution_path.write_text(
        task.code,
        encoding="utf-8",
    )

    test_path.write_text(
        generated_test,
        encoding="utf-8",
    )

    validation = validate_generated_test(
        generated_test,
        task.func_name,
    )

    if validation["syntax_valid"]:
        execution = run_generated_test(OUTPUT_DIR)
    else:
        execution = empty_execution_result()

    experiment_result = {
        "task_id": task.task_id,
        "function_name": task.func_name,
        "model_id": model_id,
        "generation_time_seconds": generation_time,
        **validation,
        **execution,
    }

    result_path.write_text(
        json.dumps(
            experiment_result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nImplementation saved to: {solution_path}")
    print(f"Generated test saved to: {test_path}")

    print("\n--- GENERATED TEST ---")
    print(generated_test)

    print("\n--- EVALUATION ---")
    print(
        json.dumps(
            experiment_result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"\nResult saved to: {result_path}")


if __name__ == "__main__":
    main()