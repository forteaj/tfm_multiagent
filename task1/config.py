import os
from dotenv import load_dotenv

load_dotenv()


LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "local-model")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))