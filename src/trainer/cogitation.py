"""Cogitation module — GLM-5.1 integration for reactive scaffolding.

This module provides the architecture the Trainer calls when it enters
reactive mode (S2/S3 events). It constructs prompts from event context,
sets up tool calling for scaffolding extraction, calls a GLM-5.1-compatible
LLM API, and returns compiled scaffolding (KScript source) plus a confidence
level.

The LLM client is abstracted behind the LLMClient protocol so tests can
inject mock clients without hitting real APIs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from kalvin.events import RationaliseEvent

# ── Module constants ──────────────────────────────────────────────────

ESCALATION_THRESHOLD: float = 0.5
"""Confidence below this value triggers escalation to the supervisor."""


# ── Data types ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MisfitInfo:
    """Misfit diagnosis for a single event.

    Captures whether the expectation vs proposal gap is underfitting
    (signature promises more than nodes deliver) or overfitting
    (nodes carry more than signature captures), along with the
    specific gap/excess bits and human-readable summaries.
    """

    underfit: bool
    overfit: bool
    underfit_gap: int
    overfit_mask: int
    expectation_summary: str
    proposal_summary: str


@dataclass(frozen=True)
class ConversationTurn:
    """A single turn in the supervisor–trainer conversation history."""

    role: str  # "supervisor" or "trainer"
    content: str


@dataclass(frozen=True)
class CogitationRequest:
    """All context the Trainer provides to the cogitation module.

    Contains the S2/S3 events that triggered reactive mode, pre-computed
    misfit diagnoses, curriculum context, conversation history, and budget
    tracking information.

    Supports both the legacy ``curriculum_context`` field and the new
    structured fields (``objective``, ``approach``, ``lesson_prose``).
    When the new fields are provided, ``build_prompt()`` prefers them
    over ``curriculum_context``.
    """

    events: list[RationaliseEvent]
    misfits: list[MisfitInfo]
    curriculum_context: str
    conversation_history: list[ConversationTurn]
    round_number: int
    max_rounds: int = 3
    objective: str = ""
    approach: str = ""
    lesson_prose: str = ""


@dataclass(frozen=True)
class CogitationResult:
    """Result returned from the cogitation module to the Trainer.

    Contains the scaffolding KScript source (or None if generation failed),
    a confidence score (0.0–1.0), the LLM's reasoning explanation, and
    the raw LLM response text for debugging.
    """

    scaffolding: str | None
    confidence: float
    reasoning: str
    raw_response: str | None


# ── LLM client interface ─────────────────────────────────────────────


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


# ── Prompt construction ──────────────────────────────────────────────


_SYSTEM_PROMPT = """\
You are a KScript scaffolding generator. Analyze the misfit between \
expectations and proposals, and write KScript that bridges the gap.

KScript syntax overview:
- Identity: `NAME`  (uppercase identifiers, e.g. M, H, MH)
- Relationship: `NAME > N1 N2`  (nodes listed after >)
- Countersign: `SIG == N1 N2`  (bidirectional mapping)
- Canonize: `SIG => N1 N2`  (unidirectional mapping)
- Assignment: `SIG = N1 N2`  (undersign/expansion)

All identifiers are UPPERCASE LETTERS ONLY (A–Z). Never use hex literals \
(0x...) or numbers — KScript only accepts uppercase names. Each line \
defines one construct. Comments use parenthesised syntax: (this is a comment).

The only valid operators are: == (countersign), => (canonize), \
= (undersign), and > (relationship). Do not use ~>, <-, ->, or any \
other operators.

Use the submit_scaffolding tool to return your generated KScript, your \
confidence level (0.0–1.0), and a brief explanation of your reasoning.
"""


def _classify_significance(significance: int) -> str:
    """Classify a raw significance value into a band label (S1–S4).

    Uses the same boundary constants as the KAgent expand module
    so the LLM sees a meaningful band label instead of a raw 64-bit int.
    """
    from kalvin.expand import boundaries
    from kalvin.expand import classify as _classify

    s12, s23, s34 = boundaries()
    return _classify(significance, s12, s23, s34)


def build_prompt(request: CogitationRequest) -> list[dict]:
    """Construct the chat message list for the LLM.

    Returns a list of message dicts with "role" and "content" keys,
    suitable for passing to an OpenAI-compatible chat completions API.
    """
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
    ]

    # Event context — one block per event/misfit pair
    for i, (event, misfit) in enumerate(zip(request.events, request.misfits)):
        sig_level = _classify_significance(event.significance)

        _log.debug(
            "build_prompt: event %d significance=%#x → band=%s",
            i,
            event.significance,
            sig_level,
        )

        if misfit.underfit and misfit.overfit:
            misfit_type = "dual (underfitting + overfitting)"
        elif misfit.underfit:
            misfit_type = "underfitting"
        elif misfit.overfit:
            misfit_type = "overfitting"
        else:
            misfit_type = "none"

        lines = [
            f"--- Event {i + 1} (significance: {sig_level}) ---",
            f"Misfit type: {misfit_type}",
            f"Expectation: {misfit.expectation_summary}",
            f"Proposal:    {misfit.proposal_summary}",
        ]
        if misfit.underfit:
            lines.append(
                f"Underfit gap: 0x{misfit.underfit_gap:X} (bits in signature not covered by nodes)"
            )
        if misfit.overfit:
            lines.append(
                f"Overfit excess: 0x{misfit.overfit_mask:X} "
                "(bits in nodes not captured by signature)"
            )

        messages.append({"role": "user", "content": "\n".join(lines)})

    # Curriculum context — prefer structured fields, fall back to legacy
    has_structured = bool(request.objective or request.approach or request.lesson_prose)
    if has_structured:
        parts: list[str] = []
        if request.objective:
            parts.append(f"Objective: {request.objective}")
        if request.approach:
            parts.append(f"Approach: {request.approach}")
        if request.lesson_prose:
            parts.append(f"Current lesson context: {request.lesson_prose}")
        messages.append(
            {
                "role": "user",
                "content": "\n".join(parts),
            }
        )
    elif request.curriculum_context:
        messages.append(
            {
                "role": "user",
                "content": f"Current curriculum context: {request.curriculum_context}",
            }
        )

    # Conversation history
    for turn in request.conversation_history:
        messages.append({"role": turn.role, "content": turn.content})

    # Round budget awareness
    remaining = request.max_rounds - request.round_number
    messages.append(
        {
            "role": "user",
            "content": (
                f"Round {request.round_number} of {request.max_rounds}. "
                f"{remaining} rounds remaining."
            ),
        }
    )

    return messages


# ── Tool definitions ─────────────────────────────────────────────────


def build_tool_definitions() -> list[dict]:
    """Return the tool schema for scaffolding extraction.

    Defines a single tool `submit_scaffolding` with parameters for
    KScript source, confidence, and reasoning. The schema follows
    the OpenAI-compatible function tool definition format.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "submit_scaffolding",
                "description": (
                    "Submit generated KScript scaffolding along with "
                    "a confidence score and reasoning explanation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kscript_source": {
                            "type": "string",
                            "description": "The KScript source code for reactive scaffolding.",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence level from 0.0 to 1.0.",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of the reasoning behind this scaffolding.",
                        },
                    },
                    "required": ["kscript_source", "confidence", "reasoning"],
                },
            },
        },
    ]


# ── Result extraction ────────────────────────────────────────────────

# Minimal pattern to detect KScript-like lines
_KSCRIPT_LINE_PREFIXES = (
    "0x",
    "#",
    "-> ",
    "=> ",
    "~> ",
    "<- ",
)


_log = logging.getLogger(__name__)


def extract_result(response: LLMResponse) -> CogitationResult:
    """Parse the LLM response into a CogitationResult.

    Strategy:
    1. If tool_calls contains a submit_scaffolding call, extract structured data.
    2. If only text content is present, attempt to extract KScript lines and
       set low confidence (force escalation for unstructured responses).
    3. Otherwise return a failed result.
    """
    _log.debug(
        "extract_result: content=%r, tool_calls=%s, finish_reason=%r",
        response.content[:200] if response.content else None,
        [tc.get("function", {}).get("name") for tc in (response.tool_calls or [])],
        response.finish_reason,
    )

    # Strategy 1: structured tool call
    if response.tool_calls:
        for tc in response.tool_calls:
            fn = tc.get("function", {})
            if fn.get("name") == "submit_scaffolding":
                args_raw = fn.get("arguments", "{}")
                try:
                    if isinstance(args_raw, str):
                        args = json.loads(args_raw)
                    else:
                        args = args_raw
                except json.JSONDecodeError as e:
                    _log.error(
                        "extract_result: failed to parse tool call arguments: %s — raw: %r",
                        e,
                        args_raw[:200] if isinstance(args_raw, str) else args_raw,
                    )
                    return CogitationResult(
                        scaffolding=None,
                        confidence=0.0,
                        reasoning=f"Failed to parse tool arguments: {e}",
                        raw_response=response.content,
                    )

                _log.info(
                    "extract_result: strategy 1 (tool call) — kscript=%r, confidence=%.2f",
                    args.get("kscript_source", "")[:80],
                    float(args.get("confidence", 0.0)),
                )
                return CogitationResult(
                    scaffolding=args.get("kscript_source", ""),
                    confidence=float(args.get("confidence", 0.0)),
                    reasoning=args.get("reasoning", ""),
                    raw_response=response.content,
                )

        _log.warning(
            "extract_result: tool_calls present but no submit_scaffolding found — tool names: %s",
            [tc.get("function", {}).get("name") for tc in response.tool_calls],
        )

    # Strategy 2: plain text — try to extract KScript
    if response.content:
        kscript_lines = [
            line
            for line in response.content.splitlines()
            if line.strip() and any(line.strip().startswith(p) for p in _KSCRIPT_LINE_PREFIXES)
        ]
        if kscript_lines:
            scaffolding = "\n".join(kscript_lines)
            _log.info(
                "extract_result: strategy 2 (text extraction) — %d KScript lines found",
                len(kscript_lines),
            )
            return CogitationResult(
                scaffolding=scaffolding,
                confidence=ESCALATION_THRESHOLD - 0.1,
                reasoning="Extracted from unstructured text response",
                raw_response=response.content,
            )
        else:
            _log.warning(
                "extract_result: strategy 2 failed — no KScript lines found in %d chars of content",
                len(response.content),
            )

    # Strategy 3: nothing usable
    _log.warning("extract_result: strategy 3 — no usable content in LLM response")
    return CogitationResult(
        scaffolding=None,
        confidence=0.0,
        reasoning="No valid response from LLM",
        raw_response=response.content,
    )


# ── Cogitator class ──────────────────────────────────────────────────


def _strip_hash_comments(source: str) -> str:
    """Remove ``#``-style comment lines from scaffolding source.

    KScript uses parenthesised comments ``(…)``, not ``#``.  LLMs
    sometimes produce ``#`` comments despite the system prompt, so
    this strips them as a defensive measure before compilation.
    Blank lines are also removed to keep the source compact.
    """
    lines = source.splitlines()
    stripped = [line for line in lines if line.strip() and not line.strip().startswith("#")]
    return "\n".join(stripped)


class Cogitator:
    """Main entry point for reactive scaffolding generation.

    Holds an LLM client and model identifier. The ``cogitate`` method
    builds prompts, invokes the LLM, extracts results, and validates
    that the generated scaffolding compiles.
    """

    def __init__(self, client: LLMClient, model: str = "glm-5.1") -> None:
        self._client = client
        self._model = model

    def cogitate(self, request: CogitationRequest) -> CogitationResult:
        """Generate reactive scaffolding for the given request.

        1. Build prompt from request context.
        2. Build tool definitions for structured output.
        3. Call the LLM.
        4. Extract the result from the response.
        5. Validate scaffolding compiles (lower confidence on failure).
        """
        messages = build_prompt(request)
        tools = build_tool_definitions()

        _log.info(
            "Cogitator.cogitate: calling LLM with %d messages, %d tools",
            len(messages),
            len(tools),
        )
        for i, msg in enumerate(messages):
            _log.debug(
                "  msg[%d]: role=%s, content=%s",
                i,
                msg["role"],
                msg["content"][:200] if msg.get("content") else "(none)",
            )

        response = self._client.complete(messages, tools=tools)

        _log.info(
            "Cogitator.cogitate: LLM response — content=%r, tool_calls=%s, finish_reason=%r",
            response.content[:200] if response.content else None,
            [tc.get("function", {}).get("name") for tc in (response.tool_calls or [])],
            response.finish_reason,
        )

        result = extract_result(response)

        # Validate scaffolding compiles
        if result.scaffolding is not None:
            # Strip # comments that LLMs may produce despite the prompt
            sanitised = _strip_hash_comments(result.scaffolding)
            if sanitised != result.scaffolding:
                _log.info(
                    "Cogitator.cogitate: stripped # comments from scaffolding (%d → %d chars)",
                    len(result.scaffolding),
                    len(sanitised),
                )
                result = CogitationResult(
                    scaffolding=sanitised or None,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    raw_response=result.raw_response,
                )
                if result.scaffolding is None:
                    _log.warning(
                        "Cogitator.cogitate: scaffolding was all comments — nothing to compile",
                    )
                    return result

            try:
                from ks.compiler import compile_source

                compile_source(result.scaffolding)
                _log.info(
                    "Cogitator.cogitate: scaffolding compiled OK (%d chars)",
                    len(result.scaffolding),
                )
            except Exception as exc:
                _log.error(
                    "Cogitator.cogitate: scaffolding compilation failed: %s",
                    exc,
                )
                result = CogitationResult(
                    scaffolding=None,
                    confidence=0.0,
                    reasoning=f"{result.reasoning} [compilation failed: {exc}]",
                    raw_response=result.raw_response,
                )

        return result

    @staticmethod
    def should_escalate(result: CogitationResult) -> bool:
        """Return True if the result warrants escalation to the supervisor."""
        return result.confidence < ESCALATION_THRESHOLD or result.scaffolding is None


# ── Concrete LLM client ──────────────────────────────────────────────


class OpenAICompatibleClient:
    """Concrete LLMClient for GLM-5.1 via OpenAI-compatible API.

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

        # Extract tool calls if present
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
