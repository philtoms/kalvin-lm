"""LLMSupervisor — an LLM decider supervisor participant.

A WebSocket client participant that registers as role ``supervisor`` and
resolves reactive decisions (``ratify_request`` frames) via an LLM. It is a
peer of the TUI, Slack, and CLI supervisors — same decision contract
(`@specs/supervisor-decision.md`), differing only in the process that
produces an answer (an LLM rather than a human or pi).

On a ``ratify_request`` the LLMSupervisor builds a
:class:`~trainer.cogitation.CogitationRequest` from the frame's ``misfit``
and ``curriculum_context`` (decompiling the proposal/query klines to
human-readable KScript so the LLM has name context), calls the
:class:`~trainer.cogitation.Cogitator`, and emits a ``supervisor_decision``
answer: ``scaffold`` when the LLM produced adequate scaffolding, otherwise
``continue`` (the decider cannot help — there is no higher authority to
escalate to, so the run advances past the gap).

Routine observation frames (``progress``, ``event`` relay) are observed but
not acted on — the LLMSupervisor is a decider, not a logger.

Reuses :mod:`training.trainer.cogitation` as a library: the prompt
construction, tool schema, response extraction, and ``#``-comment
sanitisation (the SD-16…21 contract) live there. This module owns only the
participant plumbing (WebSocket loop) and the decision mapping.

Spec ref: specs/supervisor-decision.md §LLMSupervisor (SD-2),
          §LLMSupervisor Pipeline (SD-16…21)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from functools import lru_cache
from typing import Any

import websockets

from kalvin.events import RationaliseEvent
from kalvin.kline import KLine, kline_display
from kalvin.kvalue import KValue
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from training.harness.constants import TRAINER_ROLE
from training.trainer.cogitation import (
    CogitationRequest,
    Cogitator,
    MisfitInfo,
)

logger = logging.getLogger(__name__)


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
        A configured :class:`~trainer.cogitation.Cogitator` (LLM client bound).
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
    from training.trainer.cogitation import OpenAICompatibleClient

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
