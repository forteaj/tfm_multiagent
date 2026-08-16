"""Reconstruct cumulative pytest suites from saved baseline rounds."""

from __future__ import annotations

from typing import Any, Sequence


def remove_solution_import(code: str) -> str:
    """Remove the protocol import before joining generated test candidates."""
    return "\n".join(
        line
        for line in code.splitlines()
        if not line.strip().startswith("from solution import ")
    ).strip()


def build_suite(function_name: str, accepted_tests: Sequence[str]) -> str:
    """Combine accepted candidates into one suite with one solution import."""
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
                    f"Task {record.get('task_id')} round {k} is accepted "
                    "but has no test code."
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