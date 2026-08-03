import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from benchmarks.ult.evaluator import evaluate_task
from benchmarks.ult.loader import load_ult


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "external"
    / "UnLeakedTestBench"
    / "datasets"
    / "ULT_Lite.jsonl"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "ult"
    / "baseline"
)


client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)


INVALID_TASKS = {
    "186": {
        "error_type": "missing_import",
        "error_message": (
            "The benchmark function uses re.search(), "
            "but its source code does not import re."
        ),
    },
}


TERMINAL_EVALUATION_STATUSES = {
    "completed",
    "invalid_task_environment",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of tasks to evaluate.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of generated test candidates per task.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an existing run, skipping tasks already completed "
            "or classified as invalid."
        ),
    )

    return parser.parse_args()


def append_jsonl(path: Path, result: dict) -> None:
    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                result,
                ensure_ascii=False,
            )
            + "\n"
        )


def load_completed_task_ids(
    results_path: Path,
    model_id: str,
    max_k: int,
) -> set[str]:
    completed_task_ids: set[str] = set()

    if not results_path.exists():
        return completed_task_ids

    with results_path.open(
        "r",
        encoding="utf-8",
    ) as results_file:
        for line_number, line in enumerate(
            results_file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                result = json.loads(line)

            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in {results_path} "
                    f"at line {line_number}: {error}"
                ) from error

            result_model = result.get("model_id")
            result_max_k = result.get("max_k")
            evaluation_status = result.get(
                "evaluation_status"
            )

            if result_model != model_id:
                continue

            if result_max_k != max_k:
                continue

            if (
                evaluation_status
                in TERMINAL_EVALUATION_STATUSES
            ):
                completed_task_ids.add(
                    str(result["task_id"])
                )

    return completed_task_ids


def main() -> None:
    args = parse_args()

    available_models = client.models.list().data

    if not available_models:
        raise RuntimeError(
            "LM Studio is running, but no model is loaded."
        )

    default_model = available_models[0].id

    model_id = os.getenv(
        "LM_STUDIO_MODEL",
        default_model,
    )

    tasks = load_ult(
        DATASET_PATH,
        limit=args.limit,
    )

    if not tasks:
        raise RuntimeError(
            f"No tasks found in: {DATASET_PATH}"
        )

    model_directory_name = model_id.replace("/", "_")

    run_output_dir = (
        OUTPUT_ROOT
        / model_directory_name
    )

    run_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = run_output_dir / "results.jsonl"

    if results_path.exists() and not args.resume:
        raise FileExistsError(
            f"Results already exist at {results_path}. "
            "Use --resume to continue the run or move the "
            "previous results before starting a new run."
        )

    completed_task_ids = (
        load_completed_task_ids(
            results_path=results_path,
            model_id=model_id,
            max_k=args.k,
        )
        if args.resume
        else set()
    )

    print(f"Model: {model_id}")
    print(f"Tasks: {len(tasks)}")
    print(f"Candidates per task: {args.k}")

    if args.resume:
        print(
            "Resume enabled: "
            f"{len(completed_task_ids)} terminal tasks found."
        )

    for task_number, task in enumerate(
        tasks,
        start=1,
    ):
        print(
            f"\n[{task_number}/{len(tasks)}] "
            f"Task {task.task_id}: {task.func_name}"
        )

        task_id = str(task.task_id)

        if task_id in completed_task_ids:
            print(
                "Task skipped: terminal result already exists."
            )
            continue

        task_output_dir = (
            run_output_dir
            / f"task_{task.task_id}"
        )

        invalid_task = INVALID_TASKS.get(task_id)

        if invalid_task is not None:
            result = {
                "task_id": task.task_id,
                "function_name": task.func_name,
                "model_id": model_id,
                "max_k": args.k,
                "evaluation_status": (
                    "invalid_task_environment"
                ),
                "infrastructure_error": True,
                "infrastructure_error_type": (
                    invalid_task["error_type"]
                ),
                "infrastructure_error_message": (
                    invalid_task["error_message"]
                ),
            }

            print(
                "Task skipped: "
                f"{invalid_task['error_type']} — "
                f"{invalid_task['error_message']}"
            )

        else:
            try:
                result = evaluate_task(
                    task=task,
                    model_id=model_id,
                    output_dir=task_output_dir,
                    max_k=args.k,
                )

            except Exception as error:
                result = {
                    "task_id": task.task_id,
                    "function_name": task.func_name,
                    "model_id": model_id,
                    "max_k": args.k,
                    "evaluation_status": "evaluation_error",
                    "infrastructure_error": True,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }

                print(
                    "Evaluation failed: "
                    f"{type(error).__name__}: {error}"
                )

        append_jsonl(
            results_path,
            result,
        )

    print("\nBaseline completed.")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()