from langchain_openai import ChatOpenAI

from config import (
    LMSTUDIO_BASE_URL,
    LMSTUDIO_API_KEY,
    LMSTUDIO_MODEL,
    TEMPERATURE,
)


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=LMSTUDIO_BASE_URL,
        api_key=LMSTUDIO_API_KEY,
        model=LMSTUDIO_MODEL,
        temperature=TEMPERATURE,
    )