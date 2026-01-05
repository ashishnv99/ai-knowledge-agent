from __future__ import annotations

import os
from typing import Optional
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

DEFAULT_TIMEOUT_SECS = float(os.getenv("OLLAMA_TIMEOUT_SECS", "120"))


def generate(prompt: str, model: Optional[str] = None) -> str:
    """
    Calls Ollama /api/generate (non-streaming) and returns the response text.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT_SECS)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()
