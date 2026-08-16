from __future__ import annotations

import json

import pytest

from benchmarks.ult.mutation_runner import (
    build_mutation_summary,
    build_suite,
    mutant_identity,
    parse_cosmic_dump,
    reconstruct_suites,
    summarize_mutants,
)


def test_reconstructs_only_accepted_tests_at_each_k():
    record = {
        "task_id": "10",
        "function_name": "f",
        "rounds": [
            {
                "k": 1,
                "candidate_accepted": True,
                "generated_test": "from solution import f\n\ndef test_a():\n    assert f(1) == 1",
            },
            {
                "k": 2,
                "candidate_accepted": False,
                "generated_test": "from solution import f\n\ndef test_bad():\n    assert False",
            },
            {
                "k": 3,
                "candidate_accepted": True,
                "generated_test": "from solution import f\n\ndef test_b():\n    assert f(0) == 0",
            },
        ],
    }

    suites = reconstruct_suites(record)

    assert [item["accepted_tests"] for item in suites] == [1, 1, 2]
    assert [item["suite_changed"] for item in suites] == [True, False, True]
    assert "test_bad" not in suites[1]["suite_code"]
    assert suites[2]["suite_code"].count("from solution import f") == 1
    assert "test_a" in suites[2]["suite_code"]
    assert "test_b" in suites[2]["suite_code"]


def test_build_suite_has_one_import():
    suite = build_suite(
        "f",
        [
            "from solution import f\n\ndef test_a():\n    assert f(1) == 1",
            "from solution import f\n\ndef test_b():\n    assert f(2) == 2",
        ],
    )
    assert suite.count("from solution import f") == 1


def test_cosmic_dump_classifies_timeout_and_builds_stable_id():
    work = {
        "job_id": "random-a",
        "mutations": [
            {
                "module_path": "solution.py",
                "operator_name": "core/NumberReplacer",
                "occurrence": 0,
                "start_pos": [2, 10],
                "end_pos": [2, 11],
                "operator_args": {},
                "definition_name": "f",
            }
        ],
    }
    timeout_line = json.dumps(
        [work, {"worker_outcome": "normal", "test_outcome": "killed", "output": "timeout"}]
    )
    killed_line = json.dumps(
        [work, {"worker_outcome": "normal", "test_outcome": "killed", "output": "failed", "diff": "x"}]
    )

    timeout_mutant = parse_cosmic_dump(timeout_line)[0]
    killed_mutant = parse_cosmic_dump(killed_line)[0]

    assert timeout_mutant["status"] == "timeout"
    assert killed_mutant["status"] == "killed"
    assert mutant_identity(work) == mutant_identity({**work, "job_id": "random-b"})


def test_strict_score_excludes_timeout_and_incompetent():
    summary = summarize_mutants(
        [
            {"status": "killed"},
            {"status": "survived"},
            {"status": "timeout"},
            {"status": "incompetent"},
        ]
    )
    assert summary["mutants_total"] == 4
    assert summary["mutants_evaluable"] == 2
    assert summary["mutation_score"] == 0.5


def test_official_summary_uses_zero_for_task_without_suite():
    records = [
        {
            "task_id": "a",
            "max_k": 1,
            "evaluation_status": "completed",
            "by_k": [
                {"k": 1, "mutation_score": 0.5, "killed": 1, "mutants_evaluable": 2}
            ],
        },
        {
            "task_id": "b",
            "max_k": 1,
            "evaluation_status": "no_accepted_tests",
            "by_k": [{"k": 1, "mutation_score": None}],
        },
    ]

    summary = build_mutation_summary(records)

    assert summary["by_k"][0]["official_mut_at_k"] == 0.25
    assert summary["by_k"][0]["mean_measured_task_score"] == 0.5
    assert summary["by_k"][0]["pooled_mutation_score"] == 0.5


def test_rejects_accepted_round_without_code():
    with pytest.raises(ValueError, match="accepted but has no test code"):
        reconstruct_suites(
            {
                "task_id": "x",
                "function_name": "f",
                "rounds": [{"k": 1, "candidate_accepted": True}],
            }
        )
