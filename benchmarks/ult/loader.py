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


def load_ult(path: Path, limit: int | None = None) -> list[ULTTask]:
    with path.open(encoding="utf-8") as file:
        records = json.load(file)

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