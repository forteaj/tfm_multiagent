# ULT mutation testing

`mutation_runner.py` reuses the accepted candidates already stored in a
baseline `results.jsonl`; it never calls the language model again.

## Install

```bash
python -m pip install -r requirements.txt
```

The experiment pins Cosmic Ray 8.4.6 so all models use the same mutation
operators and result format.

## Five-task pilot

```bash
python -m benchmarks.ult.run_mutation \
  --dataset runs/ult/selections/ULT_Lite_100_seed42/selected_tasks.jsonl \
  --results runs/ult/baseline/MODEL/ULT_Lite_100_seed42/results.jsonl \
  --output-dir runs/ult/mutation/MODEL/pilot \
  --limit 5 \
  --timeout 30 \
  --keep-workdirs
```

Use `--task-id 10 --task-id 123` instead of `--limit` to choose exact tasks.
`--keep-workdirs` preserves Cosmic Ray SQLite sessions and diffs for auditing;
omit it for the full experiment to save disk space. If a run is interrupted,
repeat the command with `--resume`.

## Full run

```bash
python -m benchmarks.ult.run_mutation \
  --dataset runs/ult/selections/ULT_Lite_100_seed42/selected_tasks.jsonl \
  --results runs/ult/baseline/MODEL/ULT_Lite_100_seed42/results.jsonl \
  --output-dir runs/ult/mutation/MODEL/ULT_Lite_100_seed42 \
  --timeout 30 \
  --resume
```

The output directory contains:

- `mutation_results.jsonl`: detailed task, round and mutant results;
- `mutation_summary.json`: `Mut@1...Mut@k` and diagnostic aggregates.

For each task and round, the strict score is:

```text
killed / (killed + survived)
```

Timeouts, incompetent mutants and infrastructure errors are counted
separately. The summary provides three views:

- `official_mut_at_k`: macro average over all selected/evaluable ULT
  functions, assigning zero to a function without an accepted suite;
- `mean_measured_task_score`: macro average only over tasks with a score;
- `pooled_mutation_score`: all killed mutants divided by all evaluable
  mutants, which gives functions with more mutation points more weight.

Cosmic Ray's random `job_id` is not used to compare rounds. The runner derives
a stable mutant ID from the operator, source location, occurrence and operator
arguments, and reports if the generated mutant set changes between rounds.
