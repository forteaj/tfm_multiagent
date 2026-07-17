from pathlib import Path
import argparse

from agents.test_generator import generate_test_cases
from agents.repair_agent import ask_model_for_fix

from tools.file_tools import read_text, write_text
from tools.test_tools import run_tests
from tools.json_tools import extract_json

from tools.log_tools import append_run_log




ROOT_DIR = Path(__file__).resolve().parent
LOG_FILE = ROOT_DIR / "runs" / "results.jsonl"
#TASK_DIR = ROOT_DIR / "toy_tasks" / "task1"

#BUG_FILE = TASK_DIR / "bug1.py"
#TEST_FILE = TASK_DIR / "test_bug1.py"
#ISSUE_FILE = TASK_DIR / "issue"
#GENERATED_TEST_FILE = TASK_DIR / "test_generated.py"

def parse_args():
    parser = argparse.ArgumentParser(description="Run the multi-agent bug fixing pipeline on a task." )

    parser.add_argument("--task", required=True,   help="Path to the task directory, e.g. toy_tasks/task1", )

    return parser.parse_args()


def resolve_task_dir(task_argument: str) -> Path:
    task_dir = Path(task_argument)

    if not task_dir.is_absolute():
        task_dir = ROOT_DIR / task_dir

    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory does not exist: {task_dir}")

    if not task_dir.is_dir():
        raise NotADirectoryError(f"Task path is not a directory: {task_dir}")

    return task_dir



def generate_and_save_tests(issue: str, bug_code: str, task_dir: Path) -> str:
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

    generated_test_file = task_dir / "test_generated.py"

    if test_file_name != "test_generated.py":
        print(
            f"\nWarning: model proposed test file name '{test_file_name}'. "
            f"Using 'test_generated.py' instead."
        )

    write_text(generated_test_file, test_code)

    print(f"\nGenerated tests saved to: {generated_test_file}\n")

    return test_code



def main() -> None:
    args = parse_args()
    task_dir = resolve_task_dir(args.task)

    run_log = {
        "task_name": task_dir.name,
        "task_path": str(task_dir.relative_to(ROOT_DIR)),
        "generated_tests_valid": False,
        "repair_response_valid": False,
        "tests_before_passed": None,
        "tests_after_passed": None,
        "tests_before_summary": None,
        "tests_after_summary": None,
        "resolved": False,
        "error": None,
    }

    bug_file = task_dir / "bug1.py"
    test_file = task_dir / "test_bug1.py"
    issue_file = task_dir / "issue"

    print("=== Multi-Agent TFM: Automatic Test Generation + Bug Fix Test ===\n")
    print(f"Running task: {task_dir}\n")

    if not bug_file.exists():
        message = f"Missing bug file: {bug_file}"
        print(message)
        run_log["error"] = message
        append_run_log(LOG_FILE, run_log)
        return

    if not issue_file.exists():
        message = f"Missing issue file: {issue_file}"
        print(message)
        run_log["error"] = message
        append_run_log(LOG_FILE, run_log)
        return

    issue = read_text(issue_file)
    bug_code = read_text(bug_file)

    if test_file.exists():
        test_code = read_text(test_file)
    else:
        test_code = ""

    try:
        generated_test_code = generate_and_save_tests(
            issue=issue,
            bug_code=bug_code,
            task_dir=task_dir,
        )
        run_log["generated_tests_valid"] = True

    except Exception as exc:
        print("\nCould not generate automatic tests.")
        print(exc)
        print("\nContinuing with manual tests only.")

        generated_test_code = ""
        run_log["generated_tests_valid"] = False
        run_log["error"] = f"test_generation_failed_but_continued: {exc}"

    print("Running tests before fix...\n")
    before = run_tests(task_dir)

    run_log["tests_before_passed"] = before["passed"]
    run_log["tests_before_summary"] = before["summary"]

    print("TESTS BEFORE:")
    print(before["stdout"])
    print(before["stderr"])

    if before["passed"]:
        print("Tests already pass. No fix needed.")

        run_log["resolved"] = True
        run_log["tests_after_passed"] = True
        run_log["tests_after_summary"] = before["summary"]
        append_run_log(LOG_FILE, run_log)
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
        run_log["repair_response_valid"] = True

    except Exception as exc:
        print("\nCould not parse model response as JSON.")
        print(exc)

        run_log["error"] = f"repair_response_parse_failed: {exc}"
        append_run_log(LOG_FILE, run_log)
        return

    fixed_code = parsed["fixed_code"]

    print("\nApplying fix to bug1.py...\n")
    write_text(bug_file, fixed_code)

    print("Running tests after fix...\n")
    after = run_tests(task_dir)

    run_log["tests_after_passed"] = after["passed"]
    run_log["tests_after_summary"] = after["summary"]
    run_log["resolved"] = after["passed"]

    print("TESTS AFTER:")
    print(after["stdout"])
    print(after["stderr"])

    if after["passed"]:
        print("\nSUCCESS: The bug was fixed and all tests passed.")
    else:
        print("\nFAILURE: The generated fix did not pass the tests.")

    append_run_log(LOG_FILE, run_log)


if __name__ == "__main__":
    main()