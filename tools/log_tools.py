from pathlib import Path
import json
from datetime import datetime


def append_run_log(log_path: Path, data: dict) -> None:
    """
    Appends one execution result to a JSONL log file.
    Each line is an independent JSON object.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    data["timestamp"] = datetime.now().isoformat(timespec="seconds")

    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")