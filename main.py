from pathlib import Path

from agents.test_generator import generate_test_cases
from agents.repair_agent import ask_model_for_fix

from tools.file_tools import read_text, write_text
from tools.test_tools import run_tests
from tools.json_tools import extract_json


ROOT_DIR = Path(__file__).resolve().parent
TASK_DIR = ROOT_DIR / "toy_tasks" / "task1"

BUG_FILE = TASK_DIR / "bug1.py"
TEST_FILE = TASK_DIR / "test_bug1.py"
ISSUE_FILE = TASK_DIR / "issue"
GENERATED_TEST_FILE = TASK_DIR / "test_generated.py"


def generate_and_save_tests(issue: str, bug_code: str) -> str:
    print("\nGenerating automatic test cases...\n")

    model_response = generate_test_cases(
        issue=issue,
        bug_code=bug_code,
        module_name="bug1",
    )

    print("TEST GENERATOR RESPONSE:")
    print(model_response)

    parsed = extract_json(model_response)

    test_file_name = parsed.get("test_file_name", "test_generated.py")
    test_code = parsed["test_code"]

    if test_file_name != "test_generated.py":
        print(
            f"\nWarning: model proposed test file name '{test_file_name}'. "
            f"Using 'test_generated.py' instead."
        )

    write_text(GENERATED_TEST_FILE, test_code)

    print(f"\nGenerated tests saved to: {GENERATED_TEST_FILE}\n")

    return test_code


def main() -> None:
    print("=== Multi-Agent TFM: Automatic Test Generation + Bug Fix Test ===\n")

    issue = read_text(ISSUE_FILE)
    bug_code = read_text(BUG_FILE)

    if TEST_FILE.exists():
        test_code = read_text(TEST_FILE)
    else:
        test_code = ""

    try:
        generated_test_code = generate_and_save_tests(
            issue=issue,
            bug_code=bug_code,
        )
    except Exception as exc:
        print("\nCould not generate automatic tests.")
        print(exc)
        return

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
        generated_test_code=generated_test_code,
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