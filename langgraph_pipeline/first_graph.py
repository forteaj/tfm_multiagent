from pathlib import Path

from langgraph_pipeline.graph_builder import build_graph
from tools.file_tools import read_text


ROOT_DIR = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT_DIR / "toy_tasks" / "task7"


def main() -> None:
    graph = build_graph()

    initial_state = {
        "task_dir": str(TASK_DIR),
        "task_name": TASK_DIR.name,
        "task_path": str(TASK_DIR.relative_to(ROOT_DIR)),
        "issue": read_text(TASK_DIR / "issue"),
        "bug_code": read_text(TASK_DIR / "bug1.py"),
        "manual_test_code": read_text(TASK_DIR / "test_bug1.py"),
    }

    final_state = graph.invoke(initial_state)

    print("\nFINAL STATE:")
    print(f"Task: {final_state['task_name']}")
    print(f"Generated tests valid: {final_state['generated_tests_valid']}")

    if final_state.get("test_generation_error"):
        print("\nTest generation error:")
        print(final_state["test_generation_error"])

    if final_state.get("generated_test_code"):
        print("\nGenerated test code:")
        print(final_state["generated_test_code"])


if __name__ == "__main__":
    main()