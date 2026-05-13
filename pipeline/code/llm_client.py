"""Provider-agnostic LLM client. Single Replicate Gemini impl ships."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import replicate

from . import llm_config


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return the model's text completion for `prompt`."""


class ReplicateGeminiClient(LLMClient):
    """Calls `replicate.run(MODEL_SLUG, input={...})` and joins the stream.

    Honors `llm_config.TEMPERATURE`, `MAX_TOKENS`, `REQUEST_TIMEOUT`.
    """

    def __init__(self, model_slug: str | None = None) -> None:
        llm_config.require_credentials()
        os.environ.setdefault("REPLICATE_API_TOKEN", llm_config.API_TOKEN)
        self.model_slug = model_slug or llm_config.MODEL_SLUG

    def complete(self, prompt: str) -> str:
        output = replicate.run(
            self.model_slug,
            input={
                "prompt": prompt,
                "temperature": llm_config.TEMPERATURE,
                "max_output_tokens": llm_config.MAX_TOKENS,
            },
        )
        # replicate.run can return str, list[str], or an iterator of chunks
        if isinstance(output, str):
            return output
        try:
            return "".join(str(chunk) for chunk in output)
        except TypeError:
            return str(output)


def get_client() -> LLMClient:
    if llm_config.PROVIDER == "replicate":
        return ReplicateGeminiClient()
    raise NotImplementedError(f"Unknown LLM_PROVIDER: {llm_config.PROVIDER!r}")
