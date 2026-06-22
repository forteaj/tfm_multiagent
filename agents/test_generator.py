from llm import get_llm


def generate_test_cases(issue: str, bug_code: str, module_name: str = "bug1") -> str:
    """
    Generates pytest test cases using the local LLM.
    The response is returned as raw text and parsed outside this function.
    """

    llm = get_llm()

    prompt = f"""
You are a Python test generation agent.

Your task is to generate pytest test cases that reproduce the bug described in the issue.

You will receive:
1. The issue description.
2. The current buggy Python source code.
3. The module name that should be imported in the tests.

Return ONLY valid JSON with this exact structure:

{{
  "test_file_name": "test_generated.py",
  "test_code": "complete pytest code here"
}}

Rules:
- Use pytest.
- Import the required functions from the module named: {module_name}
- The generated tests should check the expected behavior described in the issue.
- The generated tests should fail with the current buggy implementation.
- The generated tests should pass after a correct fix.
- Do not modify the source code.
- Do not include markdown.
- Do not include ```json.
- Do not include any text outside the JSON.

ISSUE:
{issue}

BUGGY SOURCE CODE:
{bug_code}
"""

    response = llm.invoke(prompt)
    return response.content