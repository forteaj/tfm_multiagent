"""Command-line entry point for ULT mutation testing."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.ult.mutation.runner import run_mutation_evaluation


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

    results_path, catalog_path, details_path, summary_path = run_mutation_evaluation(
        dataset_path=args.dataset,
        results_path=args.results,
        output_dir=args.output_dir,
        task_ids=args.task_id,
        limit=args.limit,
        timeout_seconds=args.timeout,
        cosmic_ray_command=args.cosmic_ray_command,
        keep_workdirs=args.keep_workdirs,
        resume=args.resume,
    )
    print(f"Mutation results: {results_path}")
    print(f"Mutant catalog: {catalog_path}")
    print(f"Mutation details: {details_path}")
    print(f"Mutation summary: {summary_path}")


if __name__ == "__main__":
    main()