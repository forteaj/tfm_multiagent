"""Mutation-test previously generated ULT suites with Cosmic Ray.

The baseline JSONL stores every generated candidate and whether it was
accepted. This module reconstructs the cumulative suite after each query
(``k``), validates it against the original implementation, and runs Cosmic
Ray only when the accepted suite changes.

The per-task mutation score is ``killed / (killed + survived)``. Timeouts,
incompetent mutants and infrastructure errors are reported separately and do
not silently enter that denominator.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Sequence


TERMINAL_INPUT_STATUSES = {"completed"}
MUTANT_STATUSES = {"killed", "survived", "timeout", "incompetent", "error"}


@dataclass(frozen=True)
class DatasetTask:
    task_id: str
    function_name: str
    code: str


class CommandError(RuntimeError):
    """Raised when a required subprocess command fails."""

    def __init__(self, command: Sequence[str], process: subprocess.CompletedProcess[str]):
        rendered = " ".join(shlex.quote(part) for part in command)
        output = (process.stderr or process.stdout or "").strip()
        super().__init__(f"Command failed ({process.returncode}): {rendered}\n{output}")
        self.command = list(command)
        self.returncode = process.returncode
        self.stdout = process.stdout
        self.stderr = process.stderr


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
    for name in names:
        value = record.get(name)
        if value is not None:
            return value
    return None


def load_dataset(path: Path) -> dict[str, DatasetTask]:
    tasks: dict[str, DatasetTask] = {}
    for index, record in enumerate(read_json_records(path)):
        task_id = first_value(record, ("task_id", "id"))
        function_name = first_value(
            record, ("func_name", "function_name", "entry_point", "name")
        )
        code = first_value(
            record, ("code", "source_code", "solution", "canonical_solution")
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


def remove_solution_import(code: str) -> str:
    """Remove the protocol import before joining independently generated tests."""
    return "\n".join(
        line
        for line in code.splitlines()
        if not line.strip().startswith("from solution import ")
    ).strip()


def build_suite(function_name: str, accepted_tests: Sequence[str]) -> str:
    parts = [f"from solution import {function_name}"]
    parts.extend(remove_solution_import(test) for test in accepted_tests)
    return "\n\n".join(part for part in parts if part).strip() + "\n"


def reconstruct_suites(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the cumulative accepted suite after each generation round."""
    function_name = record.get("function_name")
    if not isinstance(function_name, str) or not function_name:
        raise ValueError(f"Result task {record.get('task_id')} has no function_name.")

    accepted: list[str] = []
    snapshots: list[dict[str, Any]] = []
    for position, round_result in enumerate(record.get("rounds") or [], start=1):
        k = int(round_result.get("k", position))
        changed = False
        if round_result.get("candidate_accepted") is True:
            generated = round_result.get("generated_test")
            if not isinstance(generated, str) or not generated.strip():
                raise ValueError(
                    f"Task {record.get('task_id')} round {k} is accepted but has no test code."
                )
            accepted.append(generated)
            changed = True
        snapshots.append(
            {
                "k": k,
                "accepted_tests": len(accepted),
                "suite_changed": changed,
                "suite_code": build_suite(function_name, accepted) if accepted else None,
            }
        )
    return snapshots


def run_command(
    command: Sequence[str], cwd: Path, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        list(command), cwd=cwd, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False,
    )
    if check and process.returncode != 0:
        raise CommandError(command, process)
    return process


def find_cosmic_ray(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    discovered = shutil.which("cosmic-ray")
    if discovered:
        return discovered
    sibling = Path(sys.executable).with_name(
        "cosmic-ray.exe" if sys.platform == "win32" else "cosmic-ray"
    )
    if sibling.is_file():
        return str(sibling)
    raise FileNotFoundError(
        "Cosmic Ray was not found. Install it in the active environment with "
        "'python -m pip install cosmic-ray==8.4.6'."
    )


def cosmic_config(python_executable: str, timeout_seconds: float) -> str:
    test_command = f"{shlex.quote(python_executable)} -m pytest -q -x test_solution.py"
    # JSON strings are valid TOML basic strings and safely escape Windows paths.
    return (
        "[cosmic-ray]\n"
        'module-path = "solution.py"\n'
        f"timeout = {timeout_seconds:g}\n"
        "excluded-modules = []\n"
        f"test-command = {json.dumps(test_command)}\n\n"
        "[cosmic-ray.distributor]\n"
        'name = "local"\n'
    )


def mutant_identity(work_item: dict[str, Any]) -> str:
    """Build a stable ID; Cosmic Ray job IDs change between sessions."""
    mutations = work_item.get("mutations") or []
    canonical = json.dumps(mutations, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()[:20]


def classify_mutant(result: dict[str, Any] | None) -> str:
    if result is None:
        return "error"
    output = str(result.get("output") or "").strip().lower()
    if output == "timeout":
        return "timeout"
    test_outcome = result.get("test_outcome")
    if test_outcome in {"killed", "survived", "incompetent"}:
        return str(test_outcome)
    return "error"


def parse_cosmic_dump(text: str) -> list[dict[str, Any]]:
    mutants: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid Cosmic Ray dump line {line_number}: {error}") from error
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"Unexpected Cosmic Ray dump item at line {line_number}.")
        work_item, result = value
        if not isinstance(work_item, dict) or (result is not None and not isinstance(result, dict)):
            raise ValueError(f"Invalid Cosmic Ray work item at line {line_number}.")
        mutations = work_item.get("mutations") or []
        first = mutations[0] if mutations else {}
        mutants.append(
            {
                "mutant_id": mutant_identity(work_item),
                "operator": first.get("operator_name"),
                "occurrence": first.get("occurrence"),
                "start_pos": first.get("start_pos"),
                "end_pos": first.get("end_pos"),
                "definition_name": first.get("definition_name"),
                "status": classify_mutant(result),
                "diff": result.get("diff") if result else None,
                "worker_outcome": result.get("worker_outcome") if result else None,
                "output": result.get("output") if result else None,
            }
        )
    return mutants


def summarize_mutants(mutants: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(item.get("status")) for item in mutants)
    unknown = set(counts) - MUTANT_STATUSES
    if unknown:
        raise ValueError(f"Unknown mutant statuses: {sorted(unknown)}")
    killed = counts["killed"]
    survived = counts["survived"]
    evaluable = killed + survived
    score = killed / evaluable if evaluable else None
    return {
        "mutants_total": len(mutants),
        "mutants_evaluable": evaluable,
        "killed": killed,
        "survived": survived,
        "timeout": counts["timeout"],
        "incompetent": counts["incompetent"],
        "error": counts["error"],
        "mutation_score": round(score, 6) if score is not None else None,
        "mutation_score_percent": round(100 * score, 2) if score is not None else None,
    }


def run_suite_mutation(
    work_dir: Path,
    suite_code: str,
    cosmic_ray: str,
    timeout_seconds: float,
    k: int,
) -> dict[str, Any]:
    (work_dir / "test_solution.py").write_text(suite_code, encoding="utf-8")
    config_path = work_dir / "cosmic-ray.toml"
    config_path.write_text(cosmic_config(sys.executable, timeout_seconds), encoding="utf-8")

    baseline = run_command([cosmic_ray, "baseline", config_path.name], work_dir, check=False)
    if baseline.returncode != 0:
        return {
            "k": k,
            "status": "invalid_baseline_suite",
            "baseline_exit_code": baseline.returncode,
            "baseline_stdout": baseline.stdout,
            "baseline_stderr": baseline.stderr,
        }

    session_path = work_dir / f"session_k{k}.sqlite"
    run_command(
        [cosmic_ray, "init", "--force", config_path.name, session_path.name],
        work_dir,
    )
    run_command([cosmic_ray, "exec", config_path.name, session_path.name], work_dir)
    dump = run_command([cosmic_ray, "dump", session_path.name], work_dir)
    mutants = parse_cosmic_dump(dump.stdout)
    return {
        "k": k,
        "status": "completed",
        "baseline_exit_code": 0,
        **summarize_mutants(mutants),
        "mutants": mutants,
    }


def evaluate_result_record(
    record: dict[str, Any],
    task: DatasetTask,
    cosmic_ray: str,
    timeout_seconds: float,
    work_dir: Path,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "solution.py").write_text(task.code, encoding="utf-8")

    result_function = record.get("function_name")
    if result_function != task.function_name:
        raise ValueError(
            f"Function mismatch for task {task.task_id}: results use "
            f"{result_function!r}, dataset uses {task.function_name!r}."
        )

    base = {
        "task_id": task.task_id,
        "function_name": task.function_name,
        "model_id": record.get("model_id"),
        "max_k": record.get("max_k", len(record.get("rounds") or [])),
    }
    if record.get("evaluation_status") not in TERMINAL_INPUT_STATUSES:
        return {**base, "evaluation_status": "excluded_input_record", "by_k": []}

    snapshots = reconstruct_suites(record)
    if not any(item["suite_code"] for item in snapshots):
        return {
            **base,
            "evaluation_status": "no_accepted_tests",
            "by_k": [
                {
                    "k": item["k"], "status": "no_accepted_tests",
                    "accepted_tests": 0, "mutation_score": None,
                    "mutation_score_percent": None,
                }
                for item in snapshots
            ],
        }

    by_k: list[dict[str, Any]] = []
    previous_executed: dict[str, Any] | None = None
    expected_mutant_ids: set[str] | None = None
    for snapshot in snapshots:
        k = snapshot["k"]
        if snapshot["suite_code"] is None:
            by_k.append(
                {"k": k, "status": "no_accepted_tests", "accepted_tests": 0,
                 "mutation_score": None, "mutation_score_percent": None}
            )
            continue

        if not snapshot["suite_changed"] and previous_executed is not None:
            reused = {**previous_executed, "k": k, "reused_from_k": previous_executed["k"]}
            reused["accepted_tests"] = snapshot["accepted_tests"]
            by_k.append(reused)
            previous_executed = reused
            continue

        result = run_suite_mutation(
            work_dir, snapshot["suite_code"], cosmic_ray, timeout_seconds, k
        )
        result["accepted_tests"] = snapshot["accepted_tests"]
        if result["status"] != "completed":
            by_k.append(result)
            return {**base, "evaluation_status": "invalid_baseline_suite", "by_k": by_k}

        mutant_ids = {item["mutant_id"] for item in result["mutants"]}
        if expected_mutant_ids is None:
            expected_mutant_ids = mutant_ids
        elif mutant_ids != expected_mutant_ids:
            result["mutant_set_mismatch"] = True
            result["missing_mutant_ids"] = sorted(expected_mutant_ids - mutant_ids)
            result["new_mutant_ids"] = sorted(mutant_ids - expected_mutant_ids)
        by_k.append(result)
        previous_executed = result

    return {**base, "evaluation_status": "completed", "by_k": by_k}


def result_at_k(record: dict[str, Any], k: int) -> dict[str, Any] | None:
    for item in record.get("by_k") or []:
        if item.get("k") == k:
            return item
    return None


def build_mutation_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    included = [
        item for item in records
        if item.get("evaluation_status") in {"completed", "no_accepted_tests"}
    ]
    max_k = max((int(item.get("max_k") or 0) for item in included), default=0)
    by_k: list[dict[str, Any]] = []
    for k in range(1, max_k + 1):
        rows = [result_at_k(record, k) for record in included]
        measured = [
            row for row in rows
            if row is not None and isinstance(row.get("mutation_score"), (int, float))
        ]
        scores = [float(row["mutation_score"]) for row in measured]
        killed = sum(int(row.get("killed", 0)) for row in measured)
        evaluable = sum(int(row.get("mutants_evaluable", 0)) for row in measured)
        # ULT-style macro average: every selected function is in N; functions
        # without an accepted executable suite contribute zero.
        official = sum(scores) / len(included) if included else None
        by_k.append(
            {
                "k": k, "n_functions": len(included),
                "tasks_with_mutation_score": len(measured),
                "official_mut_at_k": round(official, 6) if official is not None else None,
                "official_mut_at_k_percent": round(100 * official, 2) if official is not None else None,
                "mean_measured_task_score": round(sum(scores) / len(scores), 6) if scores else None,
                "pooled_mutation_score": round(killed / evaluable, 6) if evaluable else None,
                "killed": killed, "mutants_evaluable": evaluable,
            }
        )
    return {
        "records": len(records), "n_functions": len(included),
        "status_counts": dict(Counter(str(item.get("evaluation_status")) for item in records)),
        "by_k": by_k,
        "definitions": {
            "task_mutation_score": "killed / (killed + survived)",
            "official_mut_at_k": "sum(task mutation score at k, using 0 when unavailable) / N",
            "pooled_mutation_score": "sum(killed) / sum(killed + survived)",
            "timeout_policy": "reported separately and excluded from the strict denominator",
        },
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved ULT test generations with Cosmic Ray."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--cosmic-ray-command", default=None)
    parser.add_argument("--keep-workdirs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive.")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive.")

    dataset = load_dataset(args.dataset.resolve())
    input_records = read_json_records(args.results.resolve())
    selected_ids = {str(item) for item in args.task_id}
    if selected_ids:
        input_records = [item for item in input_records if str(item.get("task_id")) in selected_ids]
    if args.limit is not None:
        input_records = input_records[: args.limit]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "mutation_results.jsonl"
    existing = read_json_records(results_path) if args.resume and results_path.exists() else []
    completed_ids = {str(item.get("task_id")) for item in existing}
    if results_path.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {results_path}. Use --resume or another directory.")

    cosmic_ray = find_cosmic_ray(args.cosmic_ray_command)
    produced = list(existing)
    for number, record in enumerate(input_records, start=1):
        task_id = str(record.get("task_id"))
        if task_id in completed_ids:
            print(f"[{number}/{len(input_records)}] Task {task_id}: already evaluated")
            continue
        task = dataset.get(task_id)
        if task is None:
            result = {
                "task_id": task_id, "model_id": record.get("model_id"),
                "evaluation_status": "missing_dataset_task", "by_k": [],
            }
        else:
            print(f"[{number}/{len(input_records)}] Task {task_id}: mutation testing")
            try:
                if args.keep_workdirs:
                    result = evaluate_result_record(
                        record, task, cosmic_ray, args.timeout,
                        output_dir / "work" / f"task_{task_id}",
                    )
                else:
                    with tempfile.TemporaryDirectory(prefix=f"ult_mut_{task_id}_") as temp:
                        result = evaluate_result_record(
                            record, task, cosmic_ray, args.timeout, Path(temp)
                        )
            except Exception as error:
                result = {
                    "task_id": task_id,
                    "function_name": task.function_name,
                    "model_id": record.get("model_id"),
                    "max_k": record.get("max_k", len(record.get("rounds") or [])),
                    "evaluation_status": "mutation_infrastructure_error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "by_k": [],
                }
                print(f"  Error: {type(error).__name__}: {error}")
        append_jsonl(results_path, result)
        produced.append(result)

    summary = build_mutation_summary(produced)
    summary_path = output_dir / "mutation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Mutation results: {results_path}")
    print(f"Mutation summary: {summary_path}")


if __name__ == "__main__":
    main()
