# ADR 0001: Addressed Message Bus for Multi-Agent Harness

**Date:** 2026-05-29  
**Status:** Accepted  

## Context

The existing harness is a monolithic TUI that directly owns the KAgent, compiles KScript, submits entries, and handles ratification. This couples the training loop to a single UI framework and a single human interaction mode.

The system needs to support:
- A Trainer agent that drives the training loop autonomously
- Multiple human-in-the-loop modalities (TUI, Slack)
- The KAgent remaining unaware of the training loop

## Decision

The harness becomes a persistent server with a thread-safe addressed message bus. All participants communicate by sending `{address, action, message}` envelopes through the bus. The bus routes by address only.

Key structural choices:
1. **Addressed routing, not typed events.** No message type taxonomy — just address + action. Simpler, and participants interpret actions for themselves.
2. **Single dispatch thread.** Thread-safe queue feeds a single event loop. The KAgent's Cogitator thread sends through the queue; handlers execute on the dispatch thread.
3. **KAgent calls adapter directly.** No internal EventBus — the adapter implements the callback protocol and wraps events into bus messages.
4. **Embedded Trainer, connected clients.** The Trainer lives in-process. The TUI and Slack agent connect via WebSocket.

## Alternatives Considered

### Typed event bus
Separate event types (rationalisation, escalation, progress) with type-based subscription. Rejected because it couples the bus to the domain — adding new participant types requires new event types.

### Direct participant-to-participant communication
Participants address each other directly, no central bus. Rejected because it prevents diagnostic listeners and makes participant lifecycle management harder.

### All participants embedded
No WebSocket; everything in one process. Rejected because the TUI is a Textual app with its own event loop, and the Slack agent needs to receive external webhooks.

## Consequences

- Adding a new participant type requires implementing the `Participant` protocol and listing it in the config — no harness code changes.
- The KAgent loses its internal EventBus, requiring a refactor of event publishing.
- The TUI becomes a thin WebSocket client, losing its direct ownership of the KAgent.
- Wire protocol (WebSocket + JSON) adds a dependency on a WebSocket library.
