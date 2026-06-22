from llm import get_llm


def ask_model_for_fix(
    issue: str,
    bug_code: str,
    test_code: str,
    generated_test_code: str,
    test_output: str,
) -> str:
    llm = get_llm()

    prompt = f"""
You are a software repair agent.

Your task is to fix the bug described in the issue.

You will receive:
1. The issue description.
2. The current buggy Python file.
3. The existing test file.
4. The generated test file.
5. The failing pytest output.

Return ONLY valid JSON with this exact structure:

{{
  "explanation": "short explanation of the bug",
  "fixed_code": "complete corrected content of bug1.py"
}}

Rules:
- Only return the corrected content of bug1.py in "fixed_code".
- Do not modify the tests.
- Do not include markdown.
- Do not include ```json.
- Do not include any text outside the JSON.

ISSUE:
{issue}

BUGGY FILE bug1.py:
{bug_code}

EXISTING TEST FILE test_bug1.py:
{test_code}

GENERATED TEST FILE test_generated.py:
{generated_test_code}

PYTEST OUTPUT:
{test_output}
"""

    response = llm.invoke(prompt)
    return response.content