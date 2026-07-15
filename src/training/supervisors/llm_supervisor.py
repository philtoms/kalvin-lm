"""LLMSupervisor — an LLM decider supervisor participant, and its pipeline.

A WebSocket client participant that registers as role ``supervisor`` and
resolves reactive decisions (``ratify_request`` frames) via an LLM. It is a
peer of the TUI, Slack, and CLI supervisors — same decision contract
(`@specs/supervisor-decision.md`), differing only in the process that
produces an answer (an LLM rather than a human or pi).

This module owns the LLMSupervisor's reasoning pipeline (the SD-16…21
contract): the system prompt, prompt construction, the
``submit_scaffolding`` tool schema, response extraction, ``#``-comment
sanitisation, and the :class:`Cogitator` entry point. The shared LLM
transport (the :class:`~training.harness.llm.LLMClient` protocol,
:class:`~training.harness.llm.LLMResponse`, and
:class:`~training.harness.llm.OpenAICompatibleClient`) lives in
:mod:`training.harness.llm`, shared with the Trainer's curriculum
generator; neither pipeline imports from the other.

On a ``ratify_request`` the LLMSupervisor builds a
:class:`CogitationRequest` from the frame's ``misfit`` and
``curriculum_context`` (decompiling the proposal/query klines to
human-readable KScript so the LLM has name context), calls the
:class:`Cogitator`, and emits a ``supervisor_decision`` answer:
``scaffold`` when the LLM produced adequate scaffolding, otherwise
``continue`` (the decider cannot help — there is no higher authority to
escalate to, so the run advances past the gap).

Routine observation frames (``progress``, ``event`` relay) are observed but
not acted on — the LLMSupervisor is a decider, not a logger.

Spec ref: specs/supervisor-decision.md §LLMSupervisor (SD-2),
          §LLMSupervisor Pipeline (SD-16…21)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import websockets

from kalvin.events import RationaliseEvent
from kalvin.kline import KLine, kline_display
from kalvin.kvalue import KValue
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from training.harness.constants import TRAINER_ROLE
from training.harness.llm import LLMClient, LLMResponse, OpenAICompatibleClient

logger = logging.getLogger(__name__)


# ── Module constants ─────────────────────────────────────────────────


ESCALATION_THRESHOLD: float = 0.5
"""Confidence below this value triggers escalation to the supervisor."""


# ── Data types ───────────────────────────────────────────────────────


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
    """All context the LLMSupervisor consumes to build a prompt.

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
    """Result returned from the Cogitator.

    Contains the scaffolding KScript source (or None if generation failed),
    a confidence score (0.0–1.0), the LLM's reasoning explanation, and
    the raw LLM response text for debugging.
    """

    scaffolding: str | None
    confidence: float
    reasoning: str
    raw_response: str | None


# ── Prompt construction ──────────────────────────────────────────────


_SYSTEM_PROMPT = """\
You are a KScript scaffolding generator. Analyze the misfit between \
expectations and proposals, and write KScript that bridges the gap.

KScript syntax overview:
- Identity: `NAME`  (uppercase identifiers, e.g. M, H, MH)
- Relationship: `NAME > N1 N2`  (nodes listed after >)
- Countersign: `SIG == N1 N2`  (bidirectional mapping)
- Canonize: `SIG => N1 N2`  (unidirectional mapping)
- Denote: `SIG = N1 N2`  (objective mapping — SIG denotes each node)
- Connote: `SIG > N1 N2`  (unidirectional mapping — SIG connotes each node)

All identifiers are UPPERCASE LETTERS ONLY (A–Z). Never use hex literals \
(0x...) or numbers — KScript only accepts uppercase names. Each line \
defines one construct. Comments use parenthesised syntax: (this is a comment).

The only valid operators are: == (countersign), => (canonize), \
= (denote), and > (connote). Do not use ~>, <-, ->, or any \
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
        sig_level = _classify_significance(event.proposal.significance)

        logger.debug(
            "build_prompt: event %d significance=%#x → band=%s",
            i,
            event.proposal.significance,
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

    # Curriculum context — prefer structured fields, else the flat context string
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


_KSCRIPT_LINE_PREFIXES = (
    "0x",
    "#",
    "-> ",
    "=> ",
    "~> ",
    "<- ",
)


def extract_result(response: LLMResponse) -> CogitationResult:
    """Parse the LLM response into a CogitationResult.

    Strategy:
    1. If tool_calls contains a submit_scaffolding call, extract structured data.
    2. If only text content is present, attempt to extract KScript lines and
       set low confidence (force escalation for unstructured responses).
    3. Otherwise return a failed result.
    """
    logger.debug(
        "extract_result: content=%r, tool_calls=%s, finish_reason=%r",
        response.content[:200] if response.content else None,
        [tc.get("function", {}).get("name") for tc in (response.tool_calls or [])],
        response.finish_reason,
    )

    # 1. Structured tool call
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
                    logger.error(
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

                logger.info(
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

        logger.warning(
            "extract_result: tool_calls present but no submit_scaffolding found — tool names: %s",
            [tc.get("function", {}).get("name") for tc in response.tool_calls],
        )

    # 2. Plain text — try to extract KScript
    if response.content:
        kscript_lines = [
            line
            for line in response.content.splitlines()
            if line.strip() and any(line.strip().startswith(p) for p in _KSCRIPT_LINE_PREFIXES)
        ]
        if kscript_lines:
            scaffolding = "\n".join(kscript_lines)
            logger.info(
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
            logger.warning(
                "extract_result: strategy 2 failed — no KScript lines found in %d chars of content",
                len(response.content),
            )

    # 3. Nothing usable
    logger.warning("extract_result: strategy 3 — no usable content in LLM response")
    return CogitationResult(
        scaffolding=None,
        confidence=0.0,
        reasoning="No valid response from LLM",
        raw_response=response.content,
    )


# ── Cogitator ────────────────────────────────────────────────────────


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

        logger.info(
            "Cogitator.cogitate: calling LLM with %d messages, %d tools",
            len(messages),
            len(tools),
        )
        for i, msg in enumerate(messages):
            logger.debug(
                "  msg[%d]: role=%s, content=%s",
                i,
                msg["role"],
                msg["content"][:200] if msg.get("content") else "(none)",
            )

        response = self._client.complete(messages, tools=tools)

        logger.info(
            "Cogitator.cogitate: LLM response — content=%r, tool_calls=%s, finish_reason=%r",
            response.content[:200] if response.content else None,
            [tc.get("function", {}).get("name") for tc in (response.tool_calls or [])],
            response.finish_reason,
        )

        result = extract_result(response)

        if result.scaffolding is not None:
            sanitised = _strip_hash_comments(result.scaffolding)
            if sanitised != result.scaffolding:
                logger.info(
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
                    logger.warning(
                        "Cogitator.cogitate: scaffolding was all comments — nothing to compile",
                    )
                    return result

            try:
                from ks.compiler import compile_source

                compile_source(result.scaffolding)
                logger.info(
                    "Cogitator.cogitate: scaffolding compiled OK (%d chars)",
                    len(result.scaffolding),
                )
            except Exception as exc:
                logger.error(
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


# ── LLMSupervisor participant ────────────────────────────────────────


@lru_cache(maxsize=1)
def _display_tokenizer() -> NLPTokenizer:
    """Lazily-built kalvin tokenizer for kline display (cached; data required)."""
    return NLPTokenizer()


@lru_cache(maxsize=1)
def _display_signifier() -> NLPSignifier:
    """Lazily-built kalvin signifier for kline display (cached)."""
    return NLPSignifier()


class LLMSupervisor:
    """An LLM decider supervisor participant.

    Parameters
    ----------
    harness_url:
        WebSocket URL of the harness server (e.g. ``ws://localhost:8765``).
    cogitator:
        A configured :class:`Cogitator` (LLM client bound).
        Injected so tests can pass a fake without touching real APIs.
    tokenizer / signifier:
        Optional display helpers for decompiling klines to KScript. Defaults
        are lazily built from kalvin data (cached). Inject fakes in tests.
    """

    def __init__(
        self,
        harness_url: str,
        cogitator: Cogitator,
        *,
        tokenizer: Any | None = None,
        signifier: Any | None = None,
    ) -> None:
        self._url = harness_url
        self._cogitator = cogitator
        self._tokenizer = tokenizer
        self._signifier = signifier
        self._connected: bool = False
        self._ws: websockets.asyncio.client.ClientConnection | None = None

    # -- decision core (unit-testable, no WebSocket) ------------------------

    def decide(self, message: dict) -> dict:
        """Resolve a ``ratify_request`` message into a ``supervisor_decision`` payload.

        Builds a :class:`CogitationRequest` from the frame's enriched fields,
        calls the Cogitator, and maps the result:

        - scaffolding produced with adequate confidence → ``scaffold`` (carry
          the sanitised KScript).
        - otherwise → ``continue`` (the decider cannot help; the run advances
          past the gap). There is no higher authority to escalate to.

        Returns a payload shaped for the ``supervisor_decision`` answer
        message: ``{decision, proposal, text?}``. ``proposal`` is passed
        through verbatim from the inbound frame so the Trainer countersigns
        the exact kline Kalvin proposed.
        """
        proposal = message.get("proposal")
        request = self._build_request(message)
        result = self._cogitator.cogitate(request)

        if not Cogitator.should_escalate(result):
            # Adequate scaffolding — submit it.
            logger.info(
                "LLMSupervisor: scaffolding (confidence=%.2f): %s",
                result.confidence,
                (result.scaffolding or "")[:80],
            )
            return {
                "decision": "scaffold",
                "proposal": proposal,
                "text": result.scaffolding,
            }

        # Cannot produce adequate scaffolding — continue past the gap. The
        # LLMSupervisor is the terminal decider; there is no escalation path.
        logger.info(
            "LLMSupervisor: no adequate scaffolding (confidence=%.2f, reason=%s) — continuing",
            result.confidence,
            result.reasoning[:80] if result.reasoning else "(none)",
        )
        return {"decision": "continue", "proposal": proposal}

    def _build_request(self, message: dict) -> CogitationRequest:
        """Build a CogitationRequest from an enriched ratify_request frame.

        Reconstructs the proposal/query as KValues, decompiles them to
        human-readable KScript for the LLM (RS-2: misfit summaries are
        decompiled, falling back to repr on failure), and lifts the
        curriculum context into the request's structured fields.
        """
        proposal_kv = _to_kvalue(message.get("proposal"))
        query_kv = _to_kvalue(message.get("query"))
        event = RationaliseEvent(
            kind="frame",
            query=query_kv,
            proposal=proposal_kv,
        )

        tok = self._tokenizer or _display_tokenizer()
        sig = self._signifier or _display_signifier()
        query_src = _display_kline(query_kv.kline, tok, sig)
        proposal_src = _display_kline(proposal_kv.kline, tok, sig)

        misfit_field = message.get("misfit") or {}
        misfit_info = MisfitInfo(
            underfit=bool(misfit_field.get("underfit", False)),
            overfit=bool(misfit_field.get("overfit", False)),
            underfit_gap=int(misfit_field.get("underfit_gap", 0)),
            overfit_mask=int(misfit_field.get("overfit_mask", 0)),
            expectation_summary=query_src,
            proposal_summary=proposal_src,
        )

        ctx = message.get("curriculum_context")
        objective = approach = lesson_prose = ""
        curriculum_context_str = ""
        if isinstance(ctx, dict):
            objective = str(ctx.get("objective", "") or "")
            approach = str(ctx.get("approach", "") or "")
            lesson_prose = str(ctx.get("lesson_prose", "") or "")
        elif isinstance(ctx, str):
            curriculum_context_str = ctx

        return CogitationRequest(
            events=[event],
            misfits=[misfit_info],
            curriculum_context=curriculum_context_str,
            conversation_history=[],
            round_number=1,
            max_rounds=1,  # one decision per request — the Trainer gates between
            objective=objective,
            approach=approach,
            lesson_prose=lesson_prose,
        )

    # -- WebSocket participant plumbing --------------------------------------

    async def connect(self) -> None:
        """Open the WebSocket and register as the ``supervisor`` role."""
        self._ws = await websockets.connect(self._url)
        await self._ws.send(json.dumps({"register": "supervisor"}))
        self._connected = True
        logger.info("LLMSupervisor registered as 'supervisor'")

    async def disconnect(self) -> None:
        """Close the WebSocket."""
        self._connected = False
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def run(self) -> None:
        """Main loop: receive frames, decide on ratify_request, observe the rest."""
        await self.connect()
        try:
            async for raw in self._ws:
                try:
                    frame = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Malformed frame: %s", str(raw)[:200])
                    continue

                action = frame.get("action")
                if action != "ratify_request":
                    # Routine observation (progress, event relay) — observe, do not act.
                    logger.debug("LLMSupervisor observed %s (no action)", action)
                    continue

                message = frame.get("message", {})
                payload = self.decide(message)
                await self._send_frame(TRAINER_ROLE, "supervisor_decision", payload)
        except websockets.ConnectionClosed:
            logger.info("LLMSupervisor: connection closed")
        finally:
            await self.disconnect()

    async def _send_frame(self, role: str, action: str, message: Any) -> None:
        if not self._connected or self._ws is None:
            return
        frame = {"role": role, "action": action, "message": message}
        await self._ws.send(json.dumps(frame))


# ── Module helpers ────────────────────────────────────────────────────


def _to_kvalue(obj: Any) -> KValue:
    """Reconstruct a KValue from a wire dict or pass through a KValue/KLine.

    Accepts the KValue wire shape ``{signature, nodes, significance?}`` and
    the KLine wire shape ``{signature, nodes}`` (significance defaults to 0).
    """
    if isinstance(obj, KValue):
        return obj
    if isinstance(obj, dict):
        return KValue(
            KLine(signature=obj["signature"], nodes=obj.get("nodes", [])),
            obj.get("significance", 0),
        )
    raise TypeError(f"Cannot convert {type(obj).__name__} to KValue")


def _display_kline(kline: KLine, tokenizer: Any, signifier: Any) -> str:
    """Decompile a kline to KScript, falling back to repr on failure (RS-2)."""
    try:
        return kline_display(kline, tokenizer, signifier)
    except Exception:
        return repr(kline)


# ── Standalone entry point ────────────────────────────────────────────


def _build_cogitator_from_env() -> Cogitator:
    """Build a Cogitator from KALVIN_LLM_API_KEY (mirrors the harness wiring)."""
    api_key = os.environ.get("KALVIN_LLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "KALVIN_LLM_API_KEY is required for the LLMSupervisor "
            "(set it to enable the LLM decider participant)."
        )
    client = OpenAICompatibleClient(api_key=api_key)
    return Cogitator(client=client)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="LLMSupervisor — LLM decider participant"
    )
    parser.add_argument(
        "--harness-url",
        default="ws://localhost:8765",
        help="Harness WebSocket URL",
    )
    args = parser.parse_args()

    supervisor = LLMSupervisor(args.harness_url, _build_cogitator_from_env())
    asyncio.run(supervisor.run())
