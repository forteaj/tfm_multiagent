"""Cosmic Ray adapter used by the ULT mutation evaluator."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Sequence


MUTANT_STATUSES = {"killed", "survived", "timeout", "incompetent", "error"}
MAX_DIAGNOSTIC_CHARS = 4_000


class CommandError(RuntimeError):
    """Raised when a required subprocess command fails."""

    def __init__(
        self,
        command: Sequence[str],
        process: subprocess.CompletedProcess[str],
    ) -> None:
        rendered = " ".join(shlex.quote(part) for part in command)
        output = (process.stderr or process.stdout or "").strip()
        super().__init__(f"Command failed ({process.returncode}): {rendered}\n{output}")
        self.command = list(command)
        self.returncode = process.returncode
        self.stdout = process.stdout
        self.stderr = process.stderr


def run_command(
    command: Sequence[str],
    cwd: Path,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command in an isolated task directory."""
    process = subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and process.returncode != 0:
        raise CommandError(command, process)
    return process


def find_cosmic_ray(explicit: str | None = None) -> str:
    """Locate Cosmic Ray in PATH or beside the active Python executable."""
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
    """Build the TOML configuration for a single isolated ULT task."""
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
    """Build a stable ID because Cosmic Ray job IDs change between sessions."""
    mutations = work_item.get("mutations") or []
    canonical = json.dumps(mutations, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()[:20]


def classify_mutant(result: dict[str, Any] | None) -> str:
    """Map a Cosmic Ray result to the experiment's explicit status set."""
    if result is None:
        return "error"
    output = str(result.get("output") or "").strip().lower()
    if output == "timeout":
        return "timeout"
    test_outcome = result.get("test_outcome")
    if test_outcome in {"killed", "survived", "incompetent"}:
        return str(test_outcome)
    return "error"


def truncate_diagnostic(value: Any, limit: int = MAX_DIAGNOSTIC_CHARS) -> str | None:
    """Bound exceptional command output so one traceback cannot dominate JSONL."""
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n... <truncated {omitted} characters>"


def parse_cosmic_dump(text: str) -> list[dict[str, Any]]:
    """Parse Cosmic Ray's JSONL dump into stable, serializable mutant records."""
    mutants: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid Cosmic Ray dump line {line_number}: {error}"
            ) from error
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"Unexpected Cosmic Ray dump item at line {line_number}.")

        work_item, result = value
        if not isinstance(work_item, dict) or (
            result is not None and not isinstance(result, dict)
        ):
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
    """Calculate the strict score while retaining non-evaluable outcomes."""
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
    """Validate one suite and execute every generated mutant against it."""
    (work_dir / "test_solution.py").write_text(suite_code, encoding="utf-8")
    config_path = work_dir / "cosmic-ray.toml"
    config_path.write_text(
        cosmic_config(sys.executable, timeout_seconds),
        encoding="utf-8",
    )

    baseline = run_command(
        [cosmic_ray, "baseline", config_path.name],
        work_dir,
        check=False,
    )
    if baseline.returncode != 0:
        return {
            "k": k,
            "status": "invalid_baseline_suite",
            "baseline_exit_code": baseline.returncode,
            "baseline_stdout": truncate_diagnostic(baseline.stdout),
            "baseline_stderr": truncate_diagnostic(baseline.stderr),
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