from pathlib import Path
import json
from collections import Counter


ROOT_DIR = Path(__file__).resolve().parent
RESULTS_FILE = ROOT_DIR / "runs" / "results.jsonl"


def load_results(path: Path) -> list[dict]:
    results = []

    if not path.exists():
        print(f"Results file does not exist: {path}")
        return results

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:100]}...")

    return results


def percentage(part: int, total: int) -> float:
    if total == 0:
        return 0.0

    return part / total * 100


def get_summary_total(summary: dict | None) -> int:
    if not summary:
        return 0

    return summary.get("total", 0)


def get_summary_passed(summary: dict | None) -> int:
    if not summary:
        return 0

    return summary.get("passed", 0)


def get_summary_failed(summary: dict | None) -> int:
    if not summary:
        return 0

    return summary.get("failed", 0)


def get_summary_errors(summary: dict | None) -> int:
    if not summary:
        return 0

    return summary.get("errors", 0)


def infer_failure_stage(result: dict) -> str:
    """
    Returns a clean failure category.

    New logs should ideally contain:
        failure_stage
        error_detail

    Older logs may only contain:
        error

    This function keeps compatibility with both formats.
    """

    resolved = result.get("resolved") is True
    failure_stage = result.get("failure_stage")

    if resolved:
        return "resolved"

    if failure_stage:
        return failure_stage

    old_error = result.get("error")

    if old_error:
        if old_error.startswith("test_generation_failed_but_continued"):
            return "test_generation_failed_but_continued"

        if old_error.startswith("test_generation_failed"):
            return "test_generation_failed"

        if old_error.startswith("repair_response_parse_failed"):
            return "repair_response_parse_failed"

        if "Expecting ',' delimiter" in old_error:
            return "json_parse_failed"

        return "unknown_error"

    after_summary = result.get("tests_after_summary") or {}
    after_failed = get_summary_failed(after_summary)
    after_errors = get_summary_errors(after_summary)

    if after_failed > 0 or after_errors > 0:
        return "tests_still_failing_after_repair"

    if result.get("repair_response_valid") is False:
        return "repair_response_invalid"

    if result.get("generated_tests_valid") is False:
        return "test_generation_invalid"

    return "unresolved_without_error"


def get_error_detail(result: dict) -> str | None:
    """
    Gets the technical error detail.

    New format:
        error_detail

    Old format:
        error
    """

    if result.get("error_detail"):
        return result["error_detail"]

    if result.get("error"):
        return result["error"]

    return None


def get_final_outcome(result: dict) -> str:
    if result.get("resolved") is True:
        return "resolved"

    return "unresolved"


def analyze_results(results: list[dict]) -> None:
    total = len(results)

    if total == 0:
        print("No results found.")
        return

    resolved = sum(1 for result in results if result.get("resolved") is True)
    failed = total - resolved

    generated_tests_valid = sum(
        1 for result in results
        if result.get("generated_tests_valid") is True
    )

    repair_response_valid = sum(
        1 for result in results
        if result.get("repair_response_valid") is True
    )

    outcomes = Counter(
        get_final_outcome(result)
        for result in results
    )

    failure_stages = Counter(
        infer_failure_stage(result)
        for result in results
    )

    error_details = Counter(
        detail
        for result in results
        if (detail := get_error_detail(result))
    )

    tests_before_total = [
        get_summary_total(result.get("tests_before_summary"))
        for result in results
    ]

    tests_after_total = [
        get_summary_total(result.get("tests_after_summary"))
        for result in results
    ]

    tests_after_passed = [
        get_summary_passed(result.get("tests_after_summary"))
        for result in results
    ]

    tests_after_failed = [
        get_summary_failed(result.get("tests_after_summary"))
        for result in results
    ]

    tests_after_errors = [
        get_summary_errors(result.get("tests_after_summary"))
        for result in results
    ]

    avg_tests_before = sum(tests_before_total) / total
    avg_tests_after = sum(tests_after_total) / total
    avg_tests_after_passed = sum(tests_after_passed) / total
    avg_tests_after_failed = sum(tests_after_failed) / total
    avg_tests_after_errors = sum(tests_after_errors) / total

    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"Total runs:                    {total}")
    print(f"Resolved:                      {resolved}")
    print(f"Unresolved:                    {failed}")
    print(f"Resolution rate:               {percentage(resolved, total):.2f}%")
    print()
    print(f"Generated tests valid:         {generated_tests_valid}/{total} "
          f"({percentage(generated_tests_valid, total):.2f}%)")
    print(f"Repair responses valid:        {repair_response_valid}/{total} "
          f"({percentage(repair_response_valid, total):.2f}%)")
    print()
    print(f"Avg. tests before fix:         {avg_tests_before:.2f}")
    print(f"Avg. tests after fix:          {avg_tests_after:.2f}")
    print(f"Avg. passed tests after fix:   {avg_tests_after_passed:.2f}")
    print(f"Avg. failed tests after fix:   {avg_tests_after_failed:.2f}")
    print(f"Avg. error tests after fix:    {avg_tests_after_errors:.2f}")

    print("\nFinal outcomes:")
    for outcome, count in outcomes.most_common():
        print(f"- {outcome}: {count}")

    print("\nFailure stages:")
    for stage, count in failure_stages.most_common():
        print(f"- {stage}: {count}")

    if error_details:
        print("\nError details:")
        for detail, count in error_details.most_common():
            print(f"- {detail}: {count}")

    print("\nPer-task results:")
    print("-" * 80)

    for result in results:
        task_name = result.get("task_name", "unknown_task")
        resolved_flag = result.get("resolved") is True
        generated_ok = result.get("generated_tests_valid")
        repair_ok = result.get("repair_response_valid")

        status = "RESOLVED" if resolved_flag else "FAILED"
        failure_stage = infer_failure_stage(result)
        error_detail = get_error_detail(result)

        before_summary = result.get("tests_before_summary") or {}
        after_summary = result.get("tests_after_summary") or {}

        passed_before = before_summary.get("passed", 0)
        failed_before = before_summary.get("failed", 0)
        errors_before = before_summary.get("errors", 0)

        passed_after = after_summary.get("passed", 0)
        failed_after = after_summary.get("failed", 0)
        errors_after = after_summary.get("errors", 0)

        print(
            f"{task_name:<25} "
            f"{status:<10} "
            f"gen_tests={str(generated_ok):<5} "
            f"repair_valid={str(repair_ok):<5} "
            f"before: {passed_before} passed, {failed_before} failed, {errors_before} errors | "
            f"after: {passed_after} passed, {failed_after} failed, {errors_after} errors | "
            f"stage={failure_stage}"
        )

        if error_detail:
            print(f"{'':<25} error_detail={error_detail}")

    print("=" * 80)


def main() -> None:
    results = load_results(RESULTS_FILE)
    analyze_results(results)


if __name__ == "__main__":
    main()