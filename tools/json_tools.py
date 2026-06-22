import json


def extract_json(text: str) -> dict:
    """
    Tries to extract JSON from a model response.
    Local models sometimes add markdown or extra text.
    """
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model response:\n{text}")

    json_text = text[start:end + 1]
    return json.loads(json_text)