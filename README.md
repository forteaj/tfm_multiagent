# TFM Multi-Agent Unit Test Generation

This project explores the use of a multi-agent system for the automatic generation and improvement of unit tests for Python code.

The main objective is to evaluate whether a multi-agent workflow can generate more effective unit tests than a baseline based on a single LLM agent.

The system uses:

- LangGraph for multi-agent orchestration.
- OpenAI-compatible language model backends.
- LM Studio for running language models locally.
- Pytest for executing and validating generated tests.
- UnLeakedTestBench as the main evaluation benchmark.
- Toy Python tasks for the initial development and debugging of the workflow.

## Research Goal

The project initially focused on automated software repair. However, the current research direction focuses primarily on unit test generation, as it provides a more manageable and measurable experimental setting.

Software repair is retained as a secondary component, applied mainly to the generated test cases. For example, an additional agent may inspect, correct or improve tests that:

- Contain syntax errors.
- Cannot be executed.
- Import the target function incorrectly.
- Do not properly test the intended behaviour.
- Provide insufficient coverage.
- Fail to detect relevant mutations or faults.

The main research question is:

> Can a multi-agent generation and review workflow produce more effective unit tests than a single-agent baseline using the same language model?

## Current Prototype

The current prototype implements a simple sequential workflow with two main agents:

1. **Test Generator**  
   Analyses a Python function and generates a Pytest test suite.

2. **Test Repair Agent**  
   Receives the generated tests and their execution results, then attempts to correct invalid or failing test cases.

The prototype currently supports:

- Loading Python tasks from local directories.
- Copying the original task into an isolated working directory.
- Generating unit tests with a local LLM.
- Saving the generated tests as Python files.
- Running the tests with Pytest.
- Feeding execution errors back to the repair agent.
- Re-running the corrected test suite.
- Storing execution results and failure reasons in JSONL logs.
- Running multiple tasks through a benchmark script.

Toy tasks are used during development to validate the workflow with known Python problems such as:

- Closures and late binding.
- Mutable default arguments.
- Incorrect arithmetic operations.
- Syntax errors.
- Incorrect function behaviour.

These tasks are only intended for debugging. The final experimental results will be obtained from a recognised benchmark.

## Planned Multi-Agent Architecture

The intended LangGraph workflow is:

```text
Function and Context
        ↓
Test Generator
        ↓
Test Executor
        ↓
Test Reviewer
        ↓
Decision Node
        ├── invalid or weak tests → Test Repairer
        │                               ↓
        │                         Test Executor
        │                               ↓
        └───────────────────────────────┘
        ↓
Final Test Suite