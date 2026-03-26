import os
from pathlib import Path

from dotenv import load_dotenv


def load_env() -> Path | None:
    """Load .env from current or parent folders."""
    p = Path.cwd()
    for _ in range(6):
        candidate = p / ".env"
        if candidate.exists():
            load_dotenv(candidate, override=True)
            return candidate
        p = p.parent
    return None


def validate_env() -> list[str]:
    required = [
        "OPENAI_API_KEY",
        "UPSTASH_VECTOR_REST_URL",
        "UPSTASH_VECTOR_REST_TOKEN",
    ]
    return [k for k in required if not os.getenv(k)]

