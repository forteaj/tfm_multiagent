"""Run the single-agent baseline over an ULT dataset.

With no ``--dataset`` argument, the script preserves the pilot behaviour and
evaluates the first 10 tasks from ULT Lite.  When a selected dataset is passed,
all of its tasks are evaluated unless ``--limit`` is also specified.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any

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

OUTPUT_ROOT = PROJECT_ROOT / "runs" / "ult" / "baseline"


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
    parser = argparse.ArgumentParser(
        description="Run the ULT single-agent baseline with LM Studio."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Dataset to evaluate. If omitted, ULT_Lite.jsonl is used. "
            "A custom selected dataset is evaluated in full by default."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of tasks. The default is 10 for the original "
            "ULT Lite dataset and all tasks for a custom dataset."
        ),
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of generated test candidates per task.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Exact output directory. By default, custom datasets receive "
            "their own directory below the model directory."
        ),
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


def append_jsonl(path: Path, result: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def safe_directory_name(value: str) -> str:
    safe_value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe_value or "dataset"


def default_run_output_dir(
    model_id: str,
    custom_dataset: Path | None,
) -> Path:
    model_directory = OUTPUT_ROOT / safe_directory_name(model_id)

    if custom_dataset is None:
        return model_directory

    if custom_dataset.name == "selected_tasks.jsonl":
        dataset_name = custom_dataset.parent.name
    else:
        dataset_name = custom_dataset.stem

    return model_directory / safe_directory_name(dataset_name)


def load_selection_metadata(
    dataset_path: Path,
) -> dict[str, dict[str, Any]]:
    """Load task difficulty metadata from a sibling selection manifest."""
    manifest_path = dataset_path.with_name("selection_manifest.json")

    if not manifest_path.exists():
        return {}

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read selection manifest {manifest_path}: {error}"
        ) from error

    selected_tasks = manifest.get("selected_tasks", [])
    if not isinstance(selected_tasks, list):
        raise RuntimeError(
            f"Invalid selected_tasks field in {manifest_path}."
        )

    metadata: dict[str, dict[str, Any]] = {}

    for item in selected_tasks:
        if not isinstance(item, dict) or "task_id" not in item:
            continue

        task_id = str(item["task_id"])
        metadata[task_id] = {
            "cyclomatic_complexity": item.get(
                "cyclomatic_complexity"
            ),
            "complexity_stratum": item.get("stratum"),
        }

    return metadata


def load_completed_task_ids(
    results_path: Path,
    model_id: str,
    max_k: int,
    dataset_sha256: str,
) -> set[str]:
    completed_task_ids: set[str] = set()

    if not results_path.exists():
        return completed_task_ids

    with results_path.open("r", encoding="utf-8") as results_file:
        for line_number, line in enumerate(results_file, start=1):
            if not line.strip():
                continue

            try:
                result = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in {results_path} "
                    f"at line {line_number}: {error}"
                ) from error

            if result.get("model_id") != model_id:
                continue

            if result.get("max_k") != max_k:
                continue

            result_dataset_hash = result.get("dataset_sha256")
            if (
                result_dataset_hash is not None
                and result_dataset_hash != dataset_sha256
            ):
                continue

            if (
                result.get("evaluation_status")
                in TERMINAL_EVALUATION_STATUSES
            ):
                completed_task_ids.add(str(result["task_id"]))

    return completed_task_ids


def load_tasks(dataset_path: Path, limit: int | None) -> list[Any]:
    if limit is None:
        return load_ult(dataset_path)

    return load_ult(dataset_path, limit=limit)


def add_run_metadata(
    result: dict[str, Any],
    dataset_path: Path,
    dataset_hash: str,
    task_metadata: dict[str, Any] | None,
) -> None:
    result["dataset_path"] = str(dataset_path)
    result["dataset_sha256"] = dataset_hash

    if task_metadata:
        result.update(
            {
                key: value
                for key, value in task_metadata.items()
                if value is not None
            }
        )


def main() -> None:
    args = parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be a positive integer.")

    if args.k < 1:
        raise ValueError("--k must be a positive integer.")

    custom_dataset = args.dataset
    dataset_path = (custom_dataset or DATASET_PATH).resolve()

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Preserve the old 10-task pilot default, but evaluate a selected custom
    # dataset in full unless the caller explicitly provides --limit.
    effective_limit = args.limit
    if effective_limit is None and custom_dataset is None:
        effective_limit = 10

    available_models = client.models.list().data

    if not available_models:
        raise RuntimeError("LM Studio is running, but no model is loaded.")

    default_model = available_models[0].id
    model_id = os.getenv("LM_STUDIO_MODEL", default_model)

    tasks = load_tasks(dataset_path, effective_limit)

    if not tasks:
        raise RuntimeError(f"No tasks found in: {dataset_path}")

    dataset_hash = file_sha256(dataset_path)
    selection_metadata = load_selection_metadata(dataset_path)

    run_output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_run_output_dir(model_id, custom_dataset)
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_output_dir / "results.jsonl"

    if results_path.exists() and not args.resume:
        raise FileExistsError(
            f"Results already exist at {results_path}. "
            "Use --resume to continue the run or choose --output-dir."
        )

    completed_task_ids = (
        load_completed_task_ids(
            results_path=results_path,
            model_id=model_id,
            max_k=args.k,
            dataset_sha256=dataset_hash,
        )
        if args.resume
        else set()
    )

    print(f"Model: {model_id}")
    print(f"Dataset: {dataset_path}")
    print(f"Tasks: {len(tasks)}")
    print(f"Candidates per task: {args.k}")
    print(f"Output: {run_output_dir}")

    if selection_metadata:
        print(
            "Selection metadata: "
            f"loaded for {len(selection_metadata)} tasks."
        )

    if args.resume:
        print(
            "Resume enabled: "
            f"{len(completed_task_ids)} terminal tasks found."
        )

    for task_number, task in enumerate(tasks, start=1):
        print(
            f"\n[{task_number}/{len(tasks)}] "
            f"Task {task.task_id}: {task.func_name}"
        )

        task_id = str(task.task_id)

        if task_id in completed_task_ids:
            print("Task skipped: terminal result already exists.")
            continue

        task_output_dir = run_output_dir / f"task_{task.task_id}"
        invalid_task = INVALID_TASKS.get(task_id)

        if invalid_task is not None:
            result = {
                "task_id": task.task_id,
                "function_name": task.func_name,
                "model_id": model_id,
                "max_k": args.k,
                "evaluation_status": "invalid_task_environment",
                "infrastructure_error": True,
                "infrastructure_error_type": invalid_task["error_type"],
                "infrastructure_error_message": invalid_task[
                    "error_message"
                ],
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

        add_run_metadata(
            result=result,
            dataset_path=dataset_path,
            dataset_hash=dataset_hash,
            task_metadata=selection_metadata.get(task_id),
        )
        append_jsonl(results_path, result)

    print("\nBaseline completed.")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()