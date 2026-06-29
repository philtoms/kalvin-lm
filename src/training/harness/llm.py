"""Shared LLM client infrastructure.

Defines the :class:`LLMClient` protocol, the :class:`LLMResponse` data
type, and :class:`OpenAICompatibleClient` (GLM-5.1 via an OpenAI-compatible
API). Shared between two otherwise-independent consumers:

- the LLMSupervisor's reactive pipeline
  (:mod:`training.supervisors.llm_supervisor`) — builds prompts, calls the
  client with tools, extracts scaffolding;
- the Trainer's goal-based curriculum generator
  (:mod:`training.trainer.curriculum_generator`) — a one-shot startup
  transform.

Each consumer owns its own prompts and result handling; this module owns
only the transport: the chat-completion call shape and the structured
response. It has no knowledge of KScript, scaffolding, or training, so
neither consumer carries a dependency on the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMResponse:
    """Response from an LLM chat completion call."""

    content: str | None
    tool_calls: list[dict] | None
    finish_reason: str  # e.g. "stop", "tool_calls"


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for the LLM API client.

    Implementations must support chat completions with optional tool calling.
    """

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse: ...


class OpenAICompatibleClient:
    """Concrete LLMClient for GLM-5.1 via an OpenAI-compatible API.

    Uses the ``openai`` Python package to call the chat completions
    endpoint. This client is intended for production use; tests should
    use mock clients instead.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.z.ai/api/coding/paas/v4",
        model: str = "glm-5.1",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAICompatibleClient. "
                "Install it with: pip install kalvin[trainer]"
            ) from exc

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Call the chat completions endpoint and return a structured LLMResponse."""
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
        }
        if tools is not None:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
        )
