from pathlib import Path
import shutil


original_dir = Path("C:/Users/Jose/masterIA/TFM/tfm_multiagent/toy_tasks")

def copy_original_to_work_dir(original_dir: Path, work_dir: Path):

    for child in original_dir.iterdir():
        shutil.copy2(child, work_dir)


#shutil.copy2('/src/dir/file.ext', '/dst/dir/newname.ext')