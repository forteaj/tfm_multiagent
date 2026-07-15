from pathlib import Path
import shutil

#copy task from the original folder to the task folder, so it is not needed to restore the testcases everytime
def restore_task_from_original(task_dir: Path) -> None:
    original_dir = task_dir / "original"

    if not original_dir.exists():
        raise FileNotFoundError(f"Missing original directory: {original_dir}")

    if not original_dir.is_dir():
        raise NotADirectoryError(f"Original path is not a directory: {original_dir}")

    for child in original_dir.iterdir():
        if child.is_file():
            target_file = task_dir / child.name
            shutil.copy2(child, target_file)

    #remove old generated testcases if they exist
    generated_test_file = task_dir / "test_generated.py"

    if generated_test_file.exists():
        generated_test_file.unlink()