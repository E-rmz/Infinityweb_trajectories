"""LLM provider configuration.

Secrets (`REPLICATE_API_TOKEN`) and the model identifier (`REPLICATE_MODEL_SLUG`)
live in `pipeline/.env`. Defaults below are LLM call parameters.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PIPELINE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = PIPELINE_DIR / ".env"

load_dotenv(ENV_PATH)

PROVIDER = os.environ.get("LLM_PROVIDER", "replicate")
# Accept REPLICATE_MODEL (preferred) and fall back to REPLICATE_MODEL_SLUG.
MODEL_SLUG = (
    os.environ.get("REPLICATE_MODEL")
    or os.environ.get("REPLICATE_MODEL_SLUG")
    or ""
)
API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")

TEMPERATURE = 0.0
MAX_TOKENS = 8196
REQUEST_TIMEOUT = 600
BATCH_TIMEOUT = 7200


def require_credentials() -> None:
    """Raise a helpful error if the .env was not filled in."""
    missing: list[str] = []
    if not API_TOKEN:
        missing.append("REPLICATE_API_TOKEN")
    if not MODEL_SLUG:
        missing.append("REPLICATE_MODEL")
    if missing:
        raise RuntimeError(
            f"Missing required env var(s): {', '.join(missing)}. "
            f"Copy pipeline/.env.example to pipeline/.env and fill in the values."
        )
