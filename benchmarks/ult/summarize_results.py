"""Aggregate and display results produced by the ULT baseline evaluator.

Examples:
    python -m benchmarks.ult.summarize_results runs/ult/baseline/results.jsonl
    python -m benchmarks.ult.summarize_results results.jsonl --output summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def load_json_records(path: Path) -> list[dict[str, Any]]:
    """Load one JSON object, a JSON array, JSONL, or all JSON files in a directory."""
    if path.is_dir():
        records: list[dict[str, Any]] = []
        candidates = sorted(path.glob("*.jsonl")) or sorted(path.glob("*.json"))
        if not candidates:
            raise ValueError(f"No .jsonl or .json files found in {path}")
        for candidate in candidates:
            if candidate.name == "summary.json":
                continue
            records.extend(load_json_records(candidate))
        return records

    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {path}"
                )
            records.append(value)
        return records

    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ValueError(f"Expected a JSON object, array of objects, or JSONL in {path}")


def safe_number(value: Any, default: float = 0.0) -> float:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def percentage(numerator: int | float, denominator: int | float) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def rounded_mean(values: Iterable[float | int | None]) -> float | None:
    usable = [float(value) for value in values if isinstance(value, (int, float))]
    return round(mean(usable), 2) if usable else None


def reasoning_tokens(round_result: dict[str, Any]) -> int:
    usage = round_result.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return int(safe_number(details.get("reasoning_tokens")))


def is_evaluable(record: dict[str, Any]) -> bool:
    """A result is evaluable when the evaluator completed and produced rounds."""
    return record.get("evaluation_status") == "completed" and bool(record.get("rounds"))


def value_at_k(record: dict[str, Any], field: str, k: int) -> Any:
    rounds = record.get("rounds") or []
    if k <= len(rounds):
        return rounds[k - 1].get(field)
    return None


def task_row(record: dict[str, Any]) -> dict[str, Any]:
    rounds = record.get("rounds") or []
    valid = sum(
        bool(item.get("syntax_valid") and item.get("format_valid"))
        for item in rounds
    )
    accepted = sum(bool(item.get("candidate_accepted")) for item in rounds)
    improved = sum(bool(item.get("improved_coverage")) for item in rounds)
    return {
        "task_id": str(record.get("task_id", "")),
        "function_name": record.get("function_name"),
        "evaluation_status": record.get("evaluation_status", "unknown"),
        "generations": len(rounds),
        "valid_generations": valid,
        "accepted_candidates": accepted,
        "coverage_improving_candidates": improved,
        "final_line_coverage": record.get("final_line_coverage"),
        "final_branch_coverage": record.get("final_branch_coverage"),
        "generation_time_seconds": round(
            safe_number(record.get("total_generation_time_seconds")), 4
        ),
        "execution_time_seconds": round(
            safe_number(record.get("total_execution_time_seconds")), 4
        ),
        "total_tokens": int(safe_number(record.get("total_tokens"))),
        "reasoning_tokens": sum(reasoning_tokens(item) for item in rounds),
        "empty_responses": sum(
            not (item.get("raw_model_response") or "").strip() for item in rounds
        ),
        "truncated_responses": sum(
            item.get("finish_reason") == "length" for item in rounds
        ),
    }


def build_summary(records: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    evaluable = [record for record in records if is_evaluable(record)]
    excluded = [record for record in records if not is_evaluable(record)]
    rounds = [item for record in evaluable for item in (record.get("rounds") or [])]

    valid_generations = sum(
        bool(item.get("syntax_valid") and item.get("format_valid")) for item in rounds
    )
    accepted_candidates = sum(bool(item.get("candidate_accepted")) for item in rounds)
    improving_candidates = sum(bool(item.get("improved_coverage")) for item in rounds)
    tasks_with_accepted = sum(
        any(item.get("candidate_accepted") for item in record.get("rounds", []))
        for record in evaluable
    )

    max_k = max((len(record.get("rounds") or []) for record in evaluable), default=0)
    official_by_k = []
    diagnostic_by_k = []
    lcov_at_1 = 0.0
    bcov_at_1 = 0.0
    for k in range(1, max_k + 1):
        correct_tests = sum(
            sum(
                bool(item.get("candidate_accepted"))
                for item in (record.get("rounds") or [])[:k]
            )
            for record in evaluable
        )
        # ULT averages coverage over every function. A function with no
        # accepted/executable test contributes zero coverage.
        official_line_values = [
            safe_number(value_at_k(record, "line_coverage", k))
            for record in evaluable
        ]
        official_branch_values = [
            safe_number(value_at_k(record, "branch_coverage", k))
            for record in evaluable
        ]
        lcov = rounded_mean(official_line_values) or 0.0
        bcov = rounded_mean(official_branch_values) or 0.0
        if k == 1:
            lcov_at_1 = lcov
            bcov_at_1 = bcov

        official_by_k.append(
            {
                "k": k,
                "correct_tests": correct_tests,
                "total_test_queries": len(evaluable) * k,
                "pass_at_k": percentage(correct_tests, len(evaluable) * k),
                "lcov_at_k": lcov,
                "bcov_at_k": bcov,
                "delta_lcov_at_k": round(lcov - lcov_at_1, 2),
                "delta_bcov_at_k": round(bcov - bcov_at_1, 2),
                "mut_at_k": None,
            }
        )

        suite_successes = sum(value_at_k(record, "suite_passed", k) is True for record in evaluable)
        line_values = [value_at_k(record, "line_coverage", k) for record in evaluable]
        branch_values = [value_at_k(record, "branch_coverage", k) for record in evaluable]
        measured_line = [value for value in line_values if isinstance(value, (int, float))]
        measured_branch = [value for value in branch_values if isinstance(value, (int, float))]
        diagnostic_by_k.append(
            {
                "k": k,
                "tasks_with_passing_suite": suite_successes,
                "task_success_rate": percentage(suite_successes, len(evaluable)),
                "mean_line_coverage": rounded_mean(measured_line),
                "tasks_with_line_coverage": len(measured_line),
                "mean_branch_coverage": rounded_mean(measured_branch),
                "tasks_with_branch_coverage": len(measured_branch),
            }
        )

    final_line = [record.get("final_line_coverage") for record in evaluable]
    final_branch = [record.get("final_branch_coverage") for record in evaluable]
    measured_final_line = [value for value in final_line if isinstance(value, (int, float))]
    measured_final_branch = [value for value in final_branch if isinstance(value, (int, float))]

    task_ids = [str(record.get("task_id")) for record in records]
    duplicate_ids = sorted(task_id for task_id, count in Counter(task_ids).items() if count > 1)
    finish_reasons = Counter(str(item.get("finish_reason", "unknown")) for item in rounds)

    total_generation_time = sum(
        safe_number(record.get("total_generation_time_seconds")) for record in evaluable
    )
    total_execution_time = sum(
        safe_number(record.get("total_execution_time_seconds")) for record in evaluable
    )
    total_prompt_tokens = sum(int(safe_number(record.get("total_prompt_tokens"))) for record in evaluable)
    total_completion_tokens = sum(
        int(safe_number(record.get("total_completion_tokens"))) for record in evaluable
    )
    total_tokens = sum(int(safe_number(record.get("total_tokens"))) for record in evaluable)
    total_reasoning_tokens = sum(reasoning_tokens(item) for item in rounds)

    return {
        "source": str(source),
        "models": sorted({str(record.get("model_id")) for record in records if record.get("model_id")}),
        "tasks": {
            "records_loaded": len(records),
            "evaluable": len(evaluable),
            "excluded": len(excluded),
            "with_accepted_test": tasks_with_accepted,
            "success_rate": percentage(tasks_with_accepted, len(evaluable)),
            "duplicate_task_ids": duplicate_ids,
            "excluded_details": [
                {
                    "task_id": str(record.get("task_id", "")),
                    "function_name": record.get("function_name"),
                    "evaluation_status": record.get("evaluation_status", "unknown"),
                    "error": record.get("error"),
                }
                for record in excluded
            ],
        },
        "candidates": {
            "generated": len(rounds),
            "valid": valid_generations,
            "valid_rate": percentage(valid_generations, len(rounds)),
            "accepted": accepted_candidates,
            "acceptance_rate": percentage(accepted_candidates, len(rounds)),
            "coverage_improving": improving_candidates,
            "coverage_improvement_rate": percentage(improving_candidates, len(rounds)),
            "empty_responses": sum(not (item.get("raw_model_response") or "").strip() for item in rounds),
            "truncated_responses": sum(item.get("finish_reason") == "length" for item in rounds),
            "timed_out": sum(bool(item.get("candidate_timed_out")) for item in rounds),
            "finish_reasons": dict(sorted(finish_reasons.items())),
        },
        "coverage": {
            "mean_final_line": rounded_mean(measured_final_line),
            "tasks_with_final_line_measurement": len(measured_final_line),
            "mean_final_branch": rounded_mean(measured_final_branch),
            "tasks_with_final_branch_measurement": len(measured_final_branch),
            "by_k": diagnostic_by_k,
        },
        "official_ult_metrics": {
            "definition": {
                "pass_at_k": "correct tests among the first k queries divided by N * k",
                "lcov_at_k": "mean cumulative line coverage after k queries across all N functions",
                "bcov_at_k": "mean cumulative branch coverage after k queries across all N functions",
                "delta_coverage": "coverage at k minus coverage at the first query",
                "null_coverage_policy": "0.0: tasks without an executable accepted suite contribute zero",
            },
            "n_functions": len(evaluable),
            "by_k": official_by_k,
            "mutation_testing_available": False,
            "mutation_note": (
                "Mut@k cannot be calculated from the current result schema; "
                "mutation execution and killed/total mutant counts are required."
            ),
        },
        "performance": {
            "total_generation_time_seconds": round(total_generation_time, 2),
            "mean_generation_time_per_candidate_seconds": (
                round(total_generation_time / len(rounds), 2) if rounds else 0.0
            ),
            "total_execution_time_seconds": round(total_execution_time, 2),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "total_reasoning_tokens": total_reasoning_tokens,
            "reasoning_share_of_completion_tokens": (
                percentage(total_reasoning_tokens, total_completion_tokens)
            ),
            "mean_tokens_per_candidate": (
                round(total_tokens / len(rounds), 2) if rounds else 0.0
            ),
        },
        "per_task": [task_row(record) for record in records],
    }


def format_value(value: Any, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value}{suffix}"


def print_summary(summary: dict[str, Any]) -> None:
    tasks = summary["tasks"]
    candidates = summary["candidates"]
    coverage = summary["coverage"]
    performance = summary["performance"]
    official = summary["official_ult_metrics"]

    print("\nULT-COMPATIBLE METRICS")
    print("=" * 70)
    print(f"Models:                         {', '.join(summary['models']) or 'unknown'}")
    print(f"Functions evaluated (N):        {official['n_functions']}")
    if official["by_k"]:
        print("\n" + f"{'k':>3} {'Pass@k':>10} {'LCov@k':>10} {'BCov@k':>10} {'Delta LCov':>12} {'Delta BCov':>12}")
        print("-" * 70)
        for item in official["by_k"]:
            print(
                f"{item['k']:>3} "
                f"{item['pass_at_k']:>9.2f}% "
                f"{item['lcov_at_k']:>9.2f}% "
                f"{item['bcov_at_k']:>9.2f}% "
                f"{item['delta_lcov_at_k']:>11.2f} "
                f"{item['delta_bcov_at_k']:>11.2f}"
            )
    print("\nMut@k:                         unavailable (mutation testing not run)")

    print("\nADDITIONAL DIAGNOSTIC METRICS")
    print("=" * 70)
    print(f"Records loaded:                 {tasks['records_loaded']}")
    print(f"Evaluable tasks:                {tasks['evaluable']}")
    print(f"Excluded tasks:                 {tasks['excluded']}")
    print(
        "Tasks with accepted tests:     "
        f"{tasks['with_accepted_test']}/{tasks['evaluable']} "
        f"({tasks['success_rate']:.2f}%)"
    )
    print(
        "Valid generations:             "
        f"{candidates['valid']}/{candidates['generated']} "
        f"({candidates['valid_rate']:.2f}%)"
    )
    print(
        "Accepted candidates:           "
        f"{candidates['accepted']}/{candidates['generated']} "
        f"({candidates['acceptance_rate']:.2f}%)"
    )
    print(
        "Coverage-improving candidates: "
        f"{candidates['coverage_improving']}/{candidates['generated']} "
        f"({candidates['coverage_improvement_rate']:.2f}%)"
    )
    print(
        "Mean final line coverage:       "
        f"{format_value(coverage['mean_final_line'], '%')} "
        f"(n={coverage['tasks_with_final_line_measurement']})"
    )
    print(
        "Mean final branch coverage:     "
        f"{format_value(coverage['mean_final_branch'], '%')} "
        f"(n={coverage['tasks_with_final_branch_measurement']})"
    )
    print(f"Total generation time:          {performance['total_generation_time_seconds']:.2f} s")
    print(
        "Mean generation time:           "
        f"{performance['mean_generation_time_per_candidate_seconds']:.2f} s/candidate"
    )
    print(f"Total execution time:           {performance['total_execution_time_seconds']:.2f} s")
    print(f"Total tokens:                   {performance['total_tokens']}")
    print(f"Reasoning tokens:               {performance['total_reasoning_tokens']}")
    print(f"Empty responses:                {candidates['empty_responses']}")
    print(f"Truncated responses:            {candidates['truncated_responses']}")

    if coverage["by_k"]:
        print("\nTASK-LEVEL SUCCESS AND MEASURED COVERAGE BY K")
        print("-" * 70)
        print(f"{'k':>3} {'passing tasks':>16} {'success':>10} {'mean LCov':>12} {'mean BCov':>12}")
        for item in coverage["by_k"]:
            passing = f"{item['tasks_with_passing_suite']}/{tasks['evaluable']}"
            print(
                f"{item['k']:>3} {passing:>16} "
                f"{item['task_success_rate']:>9.2f}% "
                f"{format_value(item['mean_line_coverage'], '%'):>12} "
                f"{format_value(item['mean_branch_coverage'], '%'):>12}"
            )

    if tasks["excluded_details"]:
        print("\nEXCLUDED TASKS")
        print("-" * 70)
        for item in tasks["excluded_details"]:
            detail = item.get("error") or item["evaluation_status"]
            print(f"- {item['task_id']} ({item['function_name']}): {detail}")

    if tasks["duplicate_task_ids"]:
        print(f"\nWarning: duplicate task IDs: {', '.join(tasks['duplicate_task_ids'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ULT baseline JSON/JSONL results.")
    parser.add_argument("results", type=Path, help="Results JSON, JSONL, or directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Summary JSON path (default: summary.json beside the input)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_json_records(args.results)
    if not records:
        raise SystemExit(f"No result records found in {args.results}")

    summary = build_summary(records, args.results)
    print_summary(summary)

    output_path = args.output or (
        args.results / "summary.json" if args.results.is_dir() else args.results.with_name("summary.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSummary written to: {output_path}")


if __name__ == "__main__":
    main()