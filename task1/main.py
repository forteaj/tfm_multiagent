from pathlib import Path
import json
import subprocess

from llm import get_llm


ROOT_DIR = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT_DIR / "toy_tasks" / "task1"

BUG_FILE = TASK_DIR / "bug1.py"
TEST_FILE = TASK_DIR / "test_bug1.py"
ISSUE_FILE = TASK_DIR / "issue"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def run_tests(task_dir: Path) -> dict:
    result = subprocess.run(
        ["pytest", "-q"],
        cwd=task_dir,
        capture_output=True,
        text=True,
        timeout=30,
    )

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
    }


def extract_json(text: str) -> dict:
    """
    Tries to extract JSON from the model response.
    This is useful because local models sometimes add extra text.
    """
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model response:\n{text}")

    json_text = text[start:end + 1]
    return json.loads(json_text)


def ask_model_for_fix(issue: str, bug_code: str, test_code: str, test_output: str) -> str:
    llm = get_llm()

    prompt = f"""
You are a software repair agent.

Your task is to fix the bug described in the issue.

You will receive:
1. The issue description.
2. The current buggy Python file.
3. The test file.
4. The failing pytest output.

Return ONLY valid JSON with this exact structure:

{{
  "explanation": "short explanation of the bug",
  "fixed_code": "complete corrected content of bug1.py"
}}

Do not include markdown.
Do not include ```json.
Do not include any text outside the JSON.

ISSUE:
{issue}

BUGGY FILE bug1.py:
{bug_code}

TEST FILE testBug1.py:
{test_code}

PYTEST OUTPUT:
{test_output}
"""

    response = llm.invoke(prompt)
    return response.content


def main() -> None:
    print("=== Multi-Agent TFM: First Bug Fix Test ===\n")

    issue = read_text(ISSUE_FILE)
    bug_code = read_text(BUG_FILE)
    test_code = read_text(TEST_FILE)

    print("Running tests before fix...\n")
    before = run_tests(TASK_DIR)

    print("TESTS BEFORE:")
    print(before["stdout"])
    print(before["stderr"])

    if before["passed"]:
        print("Tests already pass. No fix needed.")
        return

    print("\nAsking model for a fix...\n")

    model_response = ask_model_for_fix(
        issue=issue,
        bug_code=bug_code,
        test_code=test_code,
        test_output=before["stdout"] + "\n" + before["stderr"],
    )

    print("MODEL RESPONSE:")
    print(model_response)

    try:
        parsed = extract_json(model_response)
    except Exception as exc:
        print("\nCould not parse model response as JSON.")
        print(exc)
        return

    fixed_code = parsed["fixed_code"]

    print("\nApplying fix to bug1.py...\n")
    write_text(BUG_FILE, fixed_code)

    print("Running tests after fix...\n")
    after = run_tests(TASK_DIR)

    print("TESTS AFTER:")
    print(after["stdout"])
    print(after["stderr"])

    if after["passed"]:
        print("\nSUCCESS: The bug was fixed and all tests passed.")
    else:
        print("\nFAILURE: The generated fix did not pass the tests.")


if __name__ == "__main__":
    main()