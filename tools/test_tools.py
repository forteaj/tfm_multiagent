from pathlib import Path
import subprocess
import re


def run_tests(task_dir: Path) -> dict:
    result = subprocess.run(
        ["pytest", "-q", "--ignore=original"],
        cwd=task_dir,
        capture_output=True,
        text=True,
        timeout=30,
    )

    output = result.stdout + "\n" + result.stderr

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output": output,
        "passed": result.returncode == 0,
        "summary": extract_pytest_summary(output),
    }


def extract_pytest_summary(output: str) -> dict:
    """
    Extracts simple pytest summary information from the output.
    This is intentionally simple and can be improved later.
    """
    summary = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "total": 0,
    }

    patterns = {
        "passed": r"(\d+) passed",
        "failed": r"(\d+) failed",
        "errors": r"(\d+) error",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            summary[key] = int(match.group(1))

    summary["total"] = summary["passed"] + summary["failed"] + summary["errors"]

    return summary