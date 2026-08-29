"""Coordinate mutation evaluation over saved ULT baseline results."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence

from benchmarks.ult.mutation.cosmic_ray_engine import (
    find_cosmic_ray,
    run_suite_mutation,
    truncate_diagnostic,
)
from benchmarks.ult.mutation.suite_builder import reconstruct_suites


TERMINAL_INPUT_STATUSES = {"completed"}


@dataclass(frozen=True)
class DatasetTask:
    """Dataset fields required to evaluate one generated test suite."""

    task_id: str
    function_name: str
    code: str


@dataclass
class TaskMutationEvaluation:
    """Compact task result plus normalized records for the auxiliary JSONL files."""

    result: dict[str, Any]
    mutant_catalog: list[dict[str, Any]]
    mutation_details: list[dict[str, Any]]


def separate_mutant_artifacts(
    task_id: str,
    k: int,
    mutants: Sequence[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Store invariant mutant data once and return compact per-run outcomes."""
    details: list[dict[str, Any]] = []
    for mutant in mutants:
        mutant_id = str(mutant["mutant_id"])
        catalog_record = {
            "task_id": task_id,
            "mutant_id": mutant_id,
            "operator": mutant.get("operator"),
            "occurrence": mutant.get("occurrence"),
            "start_pos": mutant.get("start_pos"),
            "end_pos": mutant.get("end_pos"),
            "definition_name": mutant.get("definition_name"),
            "diff": mutant.get("diff"),
        }
        previous = catalog.get(mutant_id)
        if previous is None:
            catalog[mutant_id] = catalog_record
        elif previous != catalog_record:
            raise ValueError(
                f"Mutant definition changed between rounds for task {task_id}: "
                f"{mutant_id}"
            )

        status = str(mutant.get("status"))
        detail = {
            "task_id": task_id,
            "k": k,
            "mutant_id": mutant_id,
            "status": status,
        }
        # Passing/failing pytest output is routine and extremely repetitive.
        # Retain bounded diagnostics only for non-standard outcomes.
        if status in {"timeout", "incompetent", "error"}:
            detail["worker_outcome"] = mutant.get("worker_outcome")
            diagnostic = truncate_diagnostic(mutant.get("output"))
            if diagnostic:
                detail["output"] = diagnostic
        details.append(detail)
    return details


def read_json_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSON object, JSON array or JSONL file."""
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}: {error}"
                ) from error
            if not isinstance(item, dict):
                raise ValueError(f"Expected an object in {path} at line {line_number}.")
            records.append(item)
        return records

    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ValueError(f"Expected a JSON object, array of objects or JSONL in {path}.")


def first_value(record: dict[str, Any], names: Iterable[str]) -> Any:
    """Return the first non-null value found under one of the candidate names."""
    for name in names:
        value = record.get(name)
        if value is not None:
            return value
    return None


def load_dataset(path: Path) -> dict[str, DatasetTask]:
    """Index the selected ULT tasks by their string task ID."""
    tasks: dict[str, DatasetTask] = {}
    for index, record in enumerate(read_json_records(path)):
        task_id = first_value(record, ("task_id", "id"))
        function_name = first_value(
            record,
            ("func_name", "function_name", "entry_point", "name"),
        )
        code = first_value(
            record,
            ("code", "source_code", "solution", "canonical_solution"),
        )
        if task_id is None or not isinstance(function_name, str) or not function_name:
            raise ValueError(f"Dataset record {index} has incomplete task metadata.")
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"Dataset task {task_id} has no implementation code.")

        key = str(task_id)
        if key in tasks:
            raise ValueError(f"Duplicate dataset task ID: {key}")
        tasks[key] = DatasetTask(key, function_name, code)
    return tasks


def evaluate_result_record(
    record: dict[str, Any],
    task: DatasetTask,
    cosmic_ray: str,
    timeout_seconds: float,
    work_dir: Path,
    *,
    final_only: bool = False,
) -> TaskMutationEvaluation:
    """Evaluate every cumulative suite, or only the final one, for one task."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "solution.py").write_text(task.code, encoding="utf-8")

    result_function = record.get("function_name")
    if result_function != task.function_name:
        raise ValueError(
            f"Function mismatch for task {task.task_id}: results use "
            f"{result_function!r}, dataset uses {task.function_name!r}."
        )

    base = {
        "schema_version": 2,
        "task_id": task.task_id,
        "function_name": task.function_name,
        "model_id": record.get("model_id"),
        "max_k": record.get("max_k", len(record.get("rounds") or [])),
        "evaluation_mode": "final_only" if final_only else "all_k",
    }
    if record.get("evaluation_status") not in TERMINAL_INPUT_STATUSES:
        return TaskMutationEvaluation(
            {**base, "evaluation_status": "excluded_input_record", "by_k": []},
            [],
            [],
        )

    snapshots = reconstruct_suites(record)
    if final_only and snapshots:
        # Each snapshot already contains the accumulated accepted suite at k.
        # Therefore the last snapshot is the complete suite produced by the run.
        snapshots = [snapshots[-1]]

    if not any(item["suite_code"] for item in snapshots):
        return TaskMutationEvaluation(
            {
                **base,
                "evaluation_status": "no_accepted_tests",
                "by_k": [
                    {
                        "k": item["k"],
                        "status": "no_accepted_tests",
                        "accepted_tests": 0,
                        "mutation_score": None,
                        "mutation_score_percent": None,
                    }
                    for item in snapshots
                ],
            },
            [],
            [],
        )

    by_k: list[dict[str, Any]] = []
    previous_executed: dict[str, Any] | None = None
    expected_mutant_ids: set[str] | None = None
    catalog: dict[str, dict[str, Any]] = {}
    mutation_details: list[dict[str, Any]] = []
    for snapshot in snapshots:
        k = snapshot["k"]
        if snapshot["suite_code"] is None:
            by_k.append(
                {
                    "k": k,
                    "status": "no_accepted_tests",
                    "accepted_tests": 0,
                    "mutation_score": None,
                    "mutation_score_percent": None,
                }
            )
            continue

        if not snapshot["suite_changed"] and previous_executed is not None:
            reused = {
                "k": k,
                "status": previous_executed["status"],
                "reused_from_k": previous_executed["k"],
                "accepted_tests": snapshot["accepted_tests"],
                **{
                    name: previous_executed.get(name)
                    for name in (
                        "mutants_total",
                        "mutants_evaluable",
                        "killed",
                        "survived",
                        "timeout",
                        "incompetent",
                        "error",
                        "mutation_score",
                        "mutation_score_percent",
                    )
                },
            }
            by_k.append(reused)
            continue

        result = run_suite_mutation(
            work_dir,
            snapshot["suite_code"],
            cosmic_ray,
            timeout_seconds,
            k,
        )
        result["accepted_tests"] = snapshot["accepted_tests"]
        if result["status"] != "completed":
            by_k.append(result)
            return TaskMutationEvaluation(
                {
                    **base,
                    "evaluation_status": "invalid_baseline_suite",
                    "by_k": by_k,
                },
                list(catalog.values()),
                mutation_details,
            )

        mutants = result.pop("mutants")
        mutant_ids = {item["mutant_id"] for item in mutants}
        if expected_mutant_ids is None:
            expected_mutant_ids = mutant_ids
        elif mutant_ids != expected_mutant_ids:
            result["mutant_set_mismatch"] = True
            result["missing_mutant_ids"] = sorted(expected_mutant_ids - mutant_ids)
            result["new_mutant_ids"] = sorted(mutant_ids - expected_mutant_ids)
        mutation_details.extend(
            separate_mutant_artifacts(task.task_id, k, mutants, catalog)
        )
        by_k.append(result)
        previous_executed = result

    return TaskMutationEvaluation(
        {**base, "evaluation_status": "completed", "by_k": by_k},
        list(catalog.values()),
        mutation_details,
    )


def result_at_k(record: dict[str, Any], k: int) -> dict[str, Any] | None:
    """Find one round inside an evaluated task record."""
    for item in record.get("by_k") or []:
        if item.get("k") == k:
            return item
    return None


def build_mutation_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate strict task scores into macro and pooled Mut@k metrics."""
    modes = {str(item.get("evaluation_mode", "all_k")) for item in records}
    if len(modes) > 1:
        raise ValueError(
            "Cannot summarize mutation records produced in different "
            f"evaluation modes: {sorted(modes)}"
        )
    evaluation_mode = next(iter(modes), "all_k")

    included = [
        item
        for item in records
        if item.get("evaluation_status") in {"completed", "no_accepted_tests"}
    ]
    max_k = max((int(item.get("max_k") or 0) for item in included), default=0)
    evaluated_ks = (
        [max_k]
        if evaluation_mode == "final_only" and max_k
        else range(1, max_k + 1)
    )
    by_k: list[dict[str, Any]] = []
    for k in evaluated_ks:
        rows = [result_at_k(record, k) for record in included]
        measured = [
            row
            for row in rows
            if row is not None
            and isinstance(row.get("mutation_score"), (int, float))
        ]
        scores = [float(row["mutation_score"]) for row in measured]
        killed = sum(int(row.get("killed", 0)) for row in measured)
        evaluable = sum(int(row.get("mutants_evaluable", 0)) for row in measured)
        # ULT-style macro average: every selected function is in N; functions
        # without an accepted executable suite contribute zero.
        official = sum(scores) / len(included) if included else None
        by_k.append(
            {
                "k": k,
                "n_functions": len(included),
                "tasks_with_mutation_score": len(measured),
                "official_mut_at_k": (
                    round(official, 6) if official is not None else None
                ),
                "official_mut_at_k_percent": (
                    round(100 * official, 2) if official is not None else None
                ),
                "mean_measured_task_score": (
                    round(sum(scores) / len(scores), 6) if scores else None
                ),
                "pooled_mutation_score": (
                    round(killed / evaluable, 6) if evaluable else None
                ),
                "killed": killed,
                "mutants_evaluable": evaluable,
            }
        )
    return {
        "records": len(records),
        "n_functions": len(included),
        "evaluation_mode": evaluation_mode,
        "status_counts": dict(
            Counter(str(item.get("evaluation_status")) for item in records)
        ),
        "by_k": by_k,
        "definitions": {
            "task_mutation_score": "killed / (killed + survived)",
            "official_mut_at_k": (
                "sum(task mutation score at k, using 0 when unavailable) / N"
            ),
            "pooled_mutation_score": "sum(killed) / sum(killed + survived)",
            "timeout_policy": (
                "reported separately and excluded from the strict denominator"
            ),
        },
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one complete task result so interrupted runs remain resumable."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl_records(
    path: Path,
    records: Sequence[dict[str, Any]],
    existing_keys: set[tuple[Any, ...]],
    key_fields: Sequence[str],
) -> None:
    """Append normalized records once, including after an interrupted resume."""
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            key = tuple(record.get(field) for field in key_fields)
            if key in existing_keys:
                continue
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing_keys.add(key)


def run_mutation_evaluation(
    dataset_path: Path,
    results_path: Path,
    output_dir: Path,
    *,
    task_ids: Sequence[str] | None = None,
    limit: int | None = None,
    timeout_seconds: float = 30.0,
    cosmic_ray_command: str | None = None,
    keep_workdirs: bool = False,
    resume: bool = False,
    final_only: bool = False,
) -> tuple[Path, Path, Path, Path]:
    """Run mutation testing and return compact, catalog, detail and summary paths."""
    dataset = load_dataset(dataset_path.resolve())
    input_records = read_json_records(results_path.resolve())

    selected_ids = {str(item) for item in task_ids or []}
    if selected_ids:
        input_records = [
            item
            for item in input_records
            if str(item.get("task_id")) in selected_ids
        ]
    if limit is not None:
        input_records = input_records[:limit]

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mutation_results_path = output_dir / "mutation_results.jsonl"
    mutant_catalog_path = output_dir / "mutant_catalog.jsonl"
    mutation_details_path = output_dir / "mutation_details.jsonl"
    output_paths = (
        mutation_results_path,
        mutant_catalog_path,
        mutation_details_path,
    )
    existing = (
        read_json_records(mutation_results_path)
        if resume and mutation_results_path.exists()
        else []
    )
    evaluation_mode = "final_only" if final_only else "all_k"
    incompatible_modes = {
        str(item.get("evaluation_mode", "all_k"))
        for item in existing
        if str(item.get("evaluation_mode", "all_k")) != evaluation_mode
    }
    if incompatible_modes:
        raise ValueError(
            "The existing mutation results use a different evaluation mode "
            f"({sorted(incompatible_modes)}). Use a new output directory."
        )
    completed_ids = {str(item.get("task_id")) for item in existing}
    if resume and any(
        "mutants" in round_result
        for record in existing
        for round_result in record.get("by_k") or []
    ):
        raise ValueError(
            "The existing mutation_results.jsonl uses the legacy expanded "
            "format. Run the compact evaluator in a new output directory."
        )
    if any(path.exists() for path in output_paths) and not resume:
        raise FileExistsError(
            f"Output exists in {output_dir}. "
            "Use --resume or another directory."
        )

    existing_catalog = (
        read_json_records(mutant_catalog_path)
        if resume and mutant_catalog_path.exists()
        else []
    )
    existing_details = (
        read_json_records(mutation_details_path)
        if resume and mutation_details_path.exists()
        else []
    )
    catalog_keys = {
        (item.get("task_id"), item.get("mutant_id")) for item in existing_catalog
    }
    detail_keys = {
        (item.get("task_id"), item.get("k"), item.get("mutant_id"))
        for item in existing_details
    }

    cosmic_ray = find_cosmic_ray(cosmic_ray_command)
    produced = list(existing)
    for number, record in enumerate(input_records, start=1):
        task_id = str(record.get("task_id"))
        if task_id in completed_ids:
            print(f"[{number}/{len(input_records)}] Task {task_id}: already evaluated")
            continue

        task = dataset.get(task_id)
        if task is None:
            result = {
                "task_id": task_id,
                "model_id": record.get("model_id"),
                "evaluation_mode": evaluation_mode,
                "evaluation_status": "missing_dataset_task",
                "by_k": [],
            }
            evaluation = TaskMutationEvaluation(result, [], [])
        else:
            print(f"[{number}/{len(input_records)}] Task {task_id}: mutation testing")
            try:
                if keep_workdirs:
                    evaluation = evaluate_result_record(
                        record,
                        task,
                        cosmic_ray,
                        timeout_seconds,
                        output_dir / "work" / f"task_{task_id}",
                        final_only=final_only,
                    )
                else:
                    with tempfile.TemporaryDirectory(
                        prefix=f"ult_mut_{task_id}_"
                    ) as temporary_directory:
                        evaluation = evaluate_result_record(
                            record,
                            task,
                            cosmic_ray,
                            timeout_seconds,
                            Path(temporary_directory),
                            final_only=final_only,
                        )
            except Exception as error:
                result = {
                    "task_id": task_id,
                    "function_name": task.function_name,
                    "model_id": record.get("model_id"),
                    "max_k": record.get(
                        "max_k",
                        len(record.get("rounds") or []),
                    ),
                    "evaluation_mode": evaluation_mode,
                    "evaluation_status": "mutation_infrastructure_error",
                    "error_type": type(error).__name__,
                    "error_message": truncate_diagnostic(error),
                    "by_k": [],
                }
                evaluation = TaskMutationEvaluation(result, [], [])
                print(f"  Error: {type(error).__name__}: {error}")

        result = evaluation.result
        append_jsonl_records(
            mutant_catalog_path,
            evaluation.mutant_catalog,
            catalog_keys,
            ("task_id", "mutant_id"),
        )
        append_jsonl_records(
            mutation_details_path,
            evaluation.mutation_details,
            detail_keys,
            ("task_id", "k", "mutant_id"),
        )
        # Write the compact task record last. On resume it is the completion marker.
        append_jsonl(mutation_results_path, result)
        produced.append(result)

    summary = build_mutation_summary(produced)
    summary_path = output_dir / "mutation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return (
        mutation_results_path,
        mutant_catalog_path,
        mutation_details_path,
        summary_path,
    )