# TFM Multi-Agent Software Repair

This project implements a graph-based multi-agent system for automated software issue resolution.

The system uses:

- LangGraph for multi-agent orchestration.
- A local or OpenAI-compatible language model backend.
- LM Studio as an optional local model server.
- Pytest-based feedback for validating generated patches.
- Toy repositories and benchmark-inspired tasks for evaluation.

## Initial Goal

The first goal is to build a minimal multi-agent workflow capable of fixing simple Python bugs in toy repositories.

## Planned Architecture

```text
Issue Analyzer
    ↓
Code Locator
    ↓
Context Builder
    ↓
Patch Generator
    ↓
Patch Applier
    ↓
Test Runner
    ↓
Reviewer
    ↓
Decision Node
    ├── if tests fail: return to Patch Generator
    └── if tests pass: Final Writer