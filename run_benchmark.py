from pathlib import Path
import argparse
import subprocess
import sys
import time
import json

from tools.task_tools import restore_task_from_original

###
#1. Lee argumentos.
#2. Resuelve la carpeta toy_tasks.
#3. Busca tareas válidas.
#4. Para cada tarea:
#   4.1 Lee cuántos logs había antes.
#   4.2 Ejecuta python main.py --task toy_tasks/taskX.
#   4.3 Captura stdout/stderr.
#   4.4 Mide duración.
#   4.5 Lee el nuevo log generado por main.py.
#   4.6 Guarda el resultado.
#5. Al final imprime resumen.
###

ROOT_DIR = Path(__file__).resolve().parent
MAIN_FILE = ROOT_DIR / "main.py"
LOG_FILE = ROOT_DIR / "runs" / "results.jsonl"





#read arguments from terminal
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the bug fixing pipeline over multiple toy tasks."
    )

    parser.add_argument(
        "--tasks",
        required=True,
        help="Path to the directory containing task folders, e.g. toy_tasks",
    )

    parser.add_argument(
        "--pattern",
        default="task*",
        help="Pattern used to select task folders. Default: task*",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Maximum execution time per task in seconds. Default: 300",
    )

    return parser.parse_args()

#turn argument to real path
def resolve_tasks_dir(tasks_argument: str) -> Path:
    tasks_dir = Path(tasks_argument)

    if not tasks_dir.is_absolute():
        tasks_dir = ROOT_DIR / tasks_dir

    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory does not exist: {tasks_dir}")

    if not tasks_dir.is_dir():
        raise NotADirectoryError(f"Tasks path is not a directory: {tasks_dir}")

    return tasks_dir

#check every task and make sure they have an issue and bug1.py file
def discover_tasks(tasks_dir: Path, pattern: str) -> list[Path]:
    task_dirs = [
        path
        for path in sorted(tasks_dir.glob(pattern))
        if path.is_dir()
    ]

    valid_tasks = []

    for task_dir in task_dirs:
        issue_file = task_dir / "original" / "issue"
        bug_file = task_dir / "original" / "bug1.py"

        if issue_file.exists() and bug_file.exists():
            valid_tasks.append(task_dir)
        else:
            print(f"Skipping {task_dir.name}: missing issue or bug1.py")

    return valid_tasks


def read_log_lines() -> list[dict]:
    if not LOG_FILE.exists():
        return []

    entries = []

    with LOG_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return entries


def run_single_task(task_dir: Path, timeout: int) -> dict:
    #external function to restore the tasks
    restore_task_from_original(task_dir)

    relative_task_path = task_dir.relative_to(ROOT_DIR)

    command = [
        sys.executable,
        str(MAIN_FILE),
        "--task",
        str(relative_task_path),
    ]

    print("\n" + "=" * 80)
    print(f"Running task: {relative_task_path}")
    print("=" * 80)

    logs_before = read_log_lines()
    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        duration = round(time.time() - start_time, 2)

        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        logs_after = read_log_lines()
        new_logs = logs_after[len(logs_before):]

        latest_log = new_logs[-1] if new_logs else None

        return {
            "task_name": task_dir.name,
            "task_path": str(relative_task_path),
            "exit_code": result.returncode,
            "duration_seconds": duration,
            "log_found": latest_log is not None,
            "log": latest_log,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        duration = round(time.time() - start_time, 2)

        print(f"TIMEOUT: task exceeded {timeout} seconds.")

        return {
            "task_name": task_dir.name,
            "task_path": str(relative_task_path),
            "exit_code": None,
            "duration_seconds": duration,
            "log_found": False,
            "log": None,
            "stdout": "",
            "stderr": f"Timeout after {timeout} seconds",
        }


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    total = len(results)
    resolved = 0
    failed = 0
    no_log = 0

    for result in results:
        log = result["log"]

        if log is None:
            no_log += 1
            status = "NO LOG"
        elif log.get("resolved") is True:
            resolved += 1
            status = "RESOLVED"
        else:
            failed += 1
            status = "FAILED"

        print(
            f"{result['task_name']:<20} "
            f"{status:<10} "
            f"{result['duration_seconds']}s"
        )

    print("-" * 80)
    print(f"Total tasks: {total}")
    print(f"Resolved:    {resolved}")
    print(f"Failed:      {failed}")
    print(f"No log:      {no_log}")

    if total > 0:
        resolution_rate = resolved / total * 100
        print(f"Resolution rate: {resolution_rate:.2f}%")

    print("=" * 80)


def main() -> None:
    args = parse_args()

    tasks_dir = resolve_tasks_dir(args.tasks)
    task_dirs = discover_tasks(tasks_dir, args.pattern)

    if not task_dirs:
        print("No valid tasks found.")
        return

    print(f"Found {len(task_dirs)} valid tasks.")

    results = []

    for task_dir in task_dirs:
        result = run_single_task(
            task_dir=task_dir,
            timeout=args.timeout,
        )
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()