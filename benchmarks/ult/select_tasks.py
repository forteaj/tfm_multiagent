"""Validate and select a reproducible stratified sample from ULT Lite.

The script performs model-independent checks, calculates cyclomatic
complexity with Radon, prints the available difficulty distribution and writes
both the selected dataset (JSONL) and an audit manifest (JSON).

Example:
    python -m benchmarks.ult.select_tasks \
        external/UnLeakedTestBench/datasets/ULT_Lite.jsonl \
        --sample-size 100 --seed 42
"""

from __future__ import annotations

import argparse
import ast
import builtins
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
import symtable
import sys
import tempfile
from typing import Any, Iterable, Sequence

try:
    from radon.complexity import cc_visit
except ImportError as error:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "Missing dependency 'radon'. Install it with: pip install radon"
    ) from error


DEFAULT_DATASET = Path(
    "external/UnLeakedTestBench/datasets/ULT_Lite.jsonl"
)
SPECIAL_GLOBALS = {
    "__builtins__",
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
}


@dataclass(frozen=True)
class Candidate:
    index: int
    task_id: str
    function_name: str
    complexity: int
    stratum: str
    record: dict[str, Any]


@dataclass(frozen=True)
class Exclusion:
    index: int
    task_id: str
    function_name: str | None
    reason: str
    detail: str


def read_dataset(path: Path) -> list[dict[str, Any]]:
    """Read either a JSON array or newline-delimited JSON objects."""
    raw_text = path.read_text(encoding="utf-8-sig")
    if not raw_text.strip():
        return []

    if raw_text.lstrip().startswith("["):
        try:
            value = json.loads(raw_text)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON dataset: {error}") from error
        if not isinstance(value, list):
            raise ValueError("The top-level JSON value must be a list.")
        if not all(isinstance(record, dict) for record in value):
            raise ValueError("Every dataset item must be a JSON object.")
        return value

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON on line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"Line {line_number} is not a JSON object.")
        records.append(record)
    return records


def first_value(record: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = record.get(name)
        if value is not None:
            return value
    return None


def task_fields(record: dict[str, Any], index: int) -> tuple[str, str, str]:
    task_id = first_value(record, ("task_id", "id"))
    function_name = first_value(
        record,
        ("func_name", "function_name", "entry_point", "name"),
    )
    code = first_value(
        record,
        ("code", "source_code", "solution", "canonical_solution"),
    )

    if task_id is None:
        raise ValueError(f"Record {index} has no task ID.")
    if not isinstance(function_name, str) or not function_name:
        raise ValueError(f"Record {index} has no function name.")
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"Record {index} has no source code.")
    return str(task_id), function_name, code


def find_function(tree: ast.Module, function_name: str) -> ast.AST | None:
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    return matches[0] if len(matches) == 1 else None


def module_definitions(table: symtable.SymbolTable) -> set[str]:
    return {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned()
        or symbol.is_imported()
        or symbol.is_parameter()
        or symbol.is_namespace()
    }


def global_references(table: symtable.SymbolTable) -> set[str]:
    references = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_referenced() and symbol.is_global()
    }
    for child in table.get_children():
        references.update(global_references(child))
    return references


def undefined_global_names(code: str, function_name: str) -> list[str]:
    """Return obvious missing globals without executing the benchmark code."""
    table = symtable.symtable(code, "solution.py", "exec")
    available = module_definitions(table)
    available.update(dir(builtins))
    available.update(SPECIAL_GLOBALS)

    target_tables = [
        child
        for child in table.get_children()
        if child.get_name() == function_name and child.get_type() == "function"
    ]
    if len(target_tables) != 1:
        return []

    referenced = global_references(target_tables[0])

    # Annotations, decorators and default values are evaluated in module scope.
    module_missing = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_referenced()
        and not symbol.is_assigned()
        and not symbol.is_imported()
        and symbol.get_name() not in available
    }
    return sorted((referenced | module_missing) - available)


def function_complexity(code: str, function_name: str, line: int) -> int:
    matching = [
        block
        for block in cc_visit(code)
        if block.name == function_name and block.lineno == line
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Radon found {len(matching)} matching complexity blocks."
        )
    return int(matching[0].complexity)


def import_check(code: str, function_name: str, timeout: float) -> str | None:
    """Import solution.py in an isolated subprocess and return an error detail."""
    with tempfile.TemporaryDirectory(prefix="ult_validate_") as temp_dir:
        solution_path = Path(temp_dir) / "solution.py"
        solution_path.write_text(code, encoding="utf-8")
        check = (
            "import importlib.util, pathlib; "
            "p=pathlib.Path('solution.py').resolve(); "
            "s=importlib.util.spec_from_file_location('solution', p); "
            "m=importlib.util.module_from_spec(s); "
            "s.loader.exec_module(m); "
            f"assert callable(getattr(m, {function_name!r}, None))"
        )
        try:
            process = subprocess.run(
                [sys.executable, "-I", "-c", check],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"import timed out after {timeout:g} seconds"

    if process.returncode == 0:
        return None
    output = (process.stderr or process.stdout).strip().splitlines()
    return output[-1] if output else f"exit code {process.returncode}"


def stratum_for(complexity: int, boundaries: Sequence[int]) -> str:
    if complexity < boundaries[0]:
        return f"<{boundaries[0]}"
    for lower, upper in zip(boundaries, boundaries[1:]):
        if lower <= complexity < upper:
            return f"[{lower},{upper})"
    return f"[{boundaries[-1]},+inf)"


def validate_records(
    records: Sequence[dict[str, Any]],
    boundaries: Sequence[int],
    import_timeout: float,
    run_import_check: bool,
) -> tuple[list[Candidate], list[Exclusion]]:
    candidates: list[Candidate] = []
    exclusions: list[Exclusion] = []
    seen_ids: set[str] = set()

    for index, record in enumerate(records):
        task_id = str(first_value(record, ("task_id", "id")) or index)
        function_name = first_value(
            record,
            ("func_name", "function_name", "entry_point", "name"),
        )
        try:
            task_id, function_name, code = task_fields(record, index)
            if task_id in seen_ids:
                raise ValueError(f"duplicate task ID {task_id}")
            seen_ids.add(task_id)

            try:
                tree = ast.parse(code, filename=f"task_{task_id}.py")
            except SyntaxError as error:
                exclusions.append(
                    Exclusion(
                        index,
                        task_id,
                        function_name,
                        "syntax_error",
                        f"line {error.lineno}: {error.msg}",
                    )
                )
                continue

            function_node = find_function(tree, function_name)
            if function_node is None:
                exclusions.append(
                    Exclusion(
                        index,
                        task_id,
                        function_name,
                        "target_function_missing_or_ambiguous",
                        "expected exactly one top-level target function",
                    )
                )
                continue

            missing_names = undefined_global_names(code, function_name)
            if missing_names:
                exclusions.append(
                    Exclusion(
                        index,
                        task_id,
                        function_name,
                        "undefined_global_names",
                        ", ".join(missing_names),
                    )
                )
                continue

            if run_import_check:
                import_error = import_check(
                    code,
                    function_name,
                    import_timeout,
                )
                if import_error:
                    exclusions.append(
                        Exclusion(
                            index,
                            task_id,
                            function_name,
                            "import_failed",
                            import_error,
                        )
                    )
                    continue

            complexity = function_complexity(
                code,
                function_name,
                function_node.lineno,
            )
            candidates.append(
                Candidate(
                    index=index,
                    task_id=task_id,
                    function_name=function_name,
                    complexity=complexity,
                    stratum=stratum_for(complexity, boundaries),
                    record=record,
                )
            )
        except Exception as error:  # keep an audit trail for malformed tasks
            exclusions.append(
                Exclusion(
                    index=index,
                    task_id=task_id,
                    function_name=(
                        function_name if isinstance(function_name, str) else None
                    ),
                    reason="validation_error",
                    detail=str(error),
                )
            )
    return candidates, exclusions


def proportional_quotas(
    counts: dict[str, int],
    sample_size: int,
    minimum_per_stratum: int,
) -> dict[str, int]:
    """Allocate exact proportional quotas subject to stratum minimums."""
    if sample_size > sum(counts.values()):
        raise ValueError(
            f"Requested {sample_size} tasks, but only {sum(counts.values())} "
            "valid tasks are available."
        )
    strata = sorted(counts)
    quotas: dict[str, int] = {
        name: min(minimum_per_stratum, counts[name]) for name in strata
    }
    if sum(quotas.values()) > sample_size:
        quotas = {name: 0 for name in strata}

    total = sum(counts.values())
    targets = {
        name: sample_size * counts[name] / total for name in strata
    }
    while sum(quotas.values()) < sample_size:
        available = [name for name in strata if quotas[name] < counts[name]]
        if not available:
            raise RuntimeError("Could not allocate the requested sample size.")
        # Allocate next to the most underrepresented stratum relative to its
        # ideal proportional quota. Alphabetical order breaks exact ties.
        chosen = min(
            available,
            key=lambda name: (-(targets[name] - quotas[name]), name),
        )
        quotas[chosen] += 1
    return quotas


def select_sample(
    candidates: Sequence[Candidate],
    sample_size: int,
    seed: int,
    minimum_per_stratum: int,
) -> tuple[list[Candidate], dict[str, int]]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.stratum].append(candidate)

    counts = {name: len(items) for name, items in grouped.items()}
    quotas = proportional_quotas(counts, sample_size, minimum_per_stratum)
    rng = random.Random(seed)
    selected: list[Candidate] = []
    for name in sorted(grouped):
        items = sorted(grouped[name], key=lambda item: item.index)
        selected.extend(rng.sample(items, quotas[name]))
    return sorted(selected, key=lambda item: item.index), quotas


def percent(count: int, total: int) -> str:
    return f"{100 * count / total:.2f}%" if total else "0.00%"


def print_distribution(
    title: str,
    counts: dict[str, int],
    total: int,
    quotas: dict[str, int] | None = None,
) -> None:
    print(f"\n{title}")
    print("=" * 70)
    if quotas is None:
        print(f"{'stratum':<16}{'available':>12}{'share':>12}")
    else:
        print(
            f"{'stratum':<16}{'available':>12}{'share':>12}{'selected':>12}"
        )
    print("-" * 70)
    for name in sorted(counts):
        row = f"{name:<16}{counts[name]:>12}{percent(counts[name], total):>12}"
        if quotas is not None:
            row += f"{quotas.get(name, 0):>12}"
        print(row)


def write_outputs(
    output_dir: Path,
    dataset_path: Path,
    records: Sequence[dict[str, Any]],
    candidates: Sequence[Candidate],
    selected: Sequence[Candidate],
    exclusions: Sequence[Exclusion],
    boundaries: Sequence[int],
    quotas: dict[str, int],
    sample_size: int,
    seed: int,
    minimum_per_stratum: int,
    run_import_check: bool,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_tasks.jsonl"
    manifest_path = output_dir / "selection_manifest.json"

    with selected_path.open("w", encoding="utf-8", newline="\n") as stream:
        for candidate in selected:
            stream.write(json.dumps(candidate.record, ensure_ascii=False) + "\n")

    valid_counts = Counter(item.stratum for item in candidates)
    selected_counts = Counter(item.stratum for item in selected)
    manifest = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256(dataset_path.read_bytes()).hexdigest(),
        "records_total": len(records),
        "valid_tasks": len(candidates),
        "excluded_tasks": len(exclusions),
        "sample_size": sample_size,
        "seed": seed,
        "selection_method": (
            "stratified_proportional_largest_deficit_with_minimum"
        ),
        "minimum_per_stratum": minimum_per_stratum,
        "complexity_metric": "Radon cyclomatic complexity",
        "stratum_boundaries": list(boundaries),
        "import_check_enabled": run_import_check,
        "valid_distribution": dict(sorted(valid_counts.items())),
        "selected_distribution": dict(sorted(selected_counts.items())),
        "quotas": dict(sorted(quotas.items())),
        "selected_tasks": [
            {
                "dataset_index": item.index,
                "task_id": item.task_id,
                "function_name": item.function_name,
                "cyclomatic_complexity": item.complexity,
                "stratum": item.stratum,
            }
            for item in selected
        ],
        "exclusions": [asdict(item) for item in exclusions],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return selected_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate ULT tasks and create a reproducible stratified sample."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"ULT JSONL dataset (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bins",
        type=int,
        nargs="+",
        default=[10, 20, 30, 40],
        metavar="N",
        help="Lower bounds for complexity strata.",
    )
    parser.add_argument(
        "--min-per-stratum",
        type=int,
        default=1,
        help="Minimum selected from each non-empty stratum (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination directory. By default it includes sample size and "
            "seed under benchmarks/ult/selections/."
        ),
    )
    parser.add_argument(
        "--import-timeout",
        type=float,
        default=5.0,
        help="Seconds allowed for importing each solution.",
    )
    parser.add_argument(
        "--skip-import-check",
        action="store_true",
        help="Skip isolated import checks (static checks still run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    boundaries = sorted(set(args.bins))
    if not boundaries or boundaries[0] < 1:
        raise SystemExit("--bins must contain positive integers.")
    if args.sample_size < 1:
        raise SystemExit("--sample-size must be positive.")
    if args.min_per_stratum < 0:
        raise SystemExit("--min-per-stratum cannot be negative.")
    if args.import_timeout <= 0:
        raise SystemExit("--import-timeout must be positive.")
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    output_dir = args.output_dir or (
        Path("benchmarks/ult/selections")
        / f"{args.dataset.stem}_{args.sample_size}_seed{args.seed}"
    )
    records = read_dataset(args.dataset)
    candidates, exclusions = validate_records(
        records,
        boundaries,
        args.import_timeout,
        not args.skip_import_check,
    )
    valid_counts = dict(Counter(item.stratum for item in candidates))
    selected, quotas = select_sample(
        candidates,
        args.sample_size,
        args.seed,
        args.min_per_stratum,
    )

    print("ULT STRATIFIED TASK SELECTION")
    print("=" * 70)
    print(f"Dataset records:                 {len(records)}")
    print(f"Valid self-contained tasks:      {len(candidates)}")
    print(f"Excluded tasks:                  {len(exclusions)}")
    print(f"Selected tasks:                  {len(selected)}")
    print(f"Random seed:                     {args.seed}")
    print_distribution(
        "VALID DATASET DISTRIBUTION",
        valid_counts,
        len(candidates),
        quotas,
    )
    if exclusions:
        reason_counts = Counter(item.reason for item in exclusions)
        print("\nEXCLUSIONS BY REASON")
        print("=" * 70)
        for reason, count in sorted(reason_counts.items()):
            print(f"{reason:<45}{count:>8}")

    selected_path, manifest_path = write_outputs(
        output_dir,
        args.dataset,
        records,
        candidates,
        selected,
        exclusions,
        boundaries,
        quotas,
        args.sample_size,
        args.seed,
        args.min_per_stratum,
        not args.skip_import_check,
    )
    print(f"\nSelected dataset written to: {selected_path}")
    print(f"Audit manifest written to:    {manifest_path}")


if __name__ == "__main__":
    main()