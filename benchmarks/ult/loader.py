from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class ULTTask:
    task_id: str
    func_name: str
    code: str
    description: str
    cyclomatic_complexity: int | None

def read_records(dataset_path: Path) -> list[dict]:
    content = dataset_path.read_text(encoding="utf-8").strip()

    if not content:
        return []

    # Primero intenta leer una lista JSON completa.
    try:
        parsed = json.loads(content)

        if isinstance(parsed, list):
            return parsed

        if isinstance(parsed, dict):
            return [parsed]

        raise ValueError(
            f"Unsupported JSON root type: {type(parsed).__name__}"
        )

    except json.JSONDecodeError:
        pass

    # Si no es JSON completo, lo interpreta como JSONL.
    records = []

    for line_number, line in enumerate(
        content.splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSONL in {dataset_path}, "
                f"line {line_number}: {error}"
            ) from error

        if not isinstance(record, dict):
            raise ValueError(
                f"Expected a JSON object in {dataset_path}, "
                f"line {line_number}."
            )

        records.append(record)

    return records




def load_ult(path: Path, limit: int | None = None) -> list[ULTTask]:
    records = read_records(Path(path))

    tasks = [
        ULTTask(
            task_id=str(record["task_id"]),
            func_name=record["func_name"],
            code=record["code"],
            description=record["prompt"],
            cyclomatic_complexity=record.get("cyclomatic_complexity"),
        )
        for record in records
    ]

    return tasks[:limit] if limit is not None else tasks