from langgraph.graph import StateGraph, START, END

from langgraph_pipeline.graph_state import RepairState
from langgraph_pipeline.graph_nodes import generate_tests_node


def build_graph():
    graph = StateGraph(RepairState)

    graph.add_node("generate_tests", generate_tests_node)
    graph.add_edge(START, "generate_tests")
    graph.add_edge("generate_tests", END)


    return graph.compile()


"""

def build_graph():
    graph = StateGraph(RepairState)

    graph.add_node("load_task", load_task_node)
    graph.add_node("generate_tests", generate_tests_node)
    graph.add_node("run_tests_before", run_tests_before_node)
    graph.add_node("repair", repair_node)
    graph.add_node("apply_fix", apply_fix_node)
    graph.add_node("run_tests_after", run_tests_after_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("load_task")

    graph.add_edge("load_task", "generate_tests")
    graph.add_edge("generate_tests", "run_tests_before")
    graph.add_edge("run_tests_before", "repair")
    graph.add_edge("repair", "apply_fix")
    graph.add_edge("apply_fix", "run_tests_after")
    graph.add_edge("run_tests_after", "finalize")

    graph.set_finish_point("finalize")

    return graph.compile()

"""