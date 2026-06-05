# ADR 0003: File-Mediated Protocol for Auto-Tune Supervisor

**Date:** 2026-06-05  
**Status:** Accepted  

## Context

The auto-tune CLI supervisor needs to communicate with an LLM coding agent (pi). Pi's tool interface is `bash` — each invocation runs a command, captures output, and returns. Pi cannot maintain a persistent stdin/stdout conversation with a running process.

The supervisor must stay connected to the harness WebSocket for the duration of a training session (to receive all events), but pi can only interact with it through discrete bash calls.

## Decision

Pi and the supervisor communicate through files in the session directory:

- **`cmd.json`** — pi writes a single command, supervisor reads and deletes it
- **`events.jsonl`** — supervisor appends one JSON line per event, pi reads with `--after <seq>`
- **`status.json`** — supervisor maintains its current state for pi to poll

The supervisor blocks on `cmd.json` after writing each event, giving pi per-event granularity.

## Alternatives Considered

### Persistent stdin/stdout
Run the supervisor as a foreground process and pipe commands/events through stdin/stdout. Rejected because pi's `bash` tool cannot maintain a persistent conversation — each call is a separate process.

### Pi extension with custom tools
Build a pi extension that provides typed tools (`auto_tune_send`, `auto_tune_events`). Rejected because it couples the project to pi's extension API and the bash CLI is sufficient.

### HTTP API
Run the supervisor as an HTTP server with REST endpoints. Rejected because it adds unnecessary complexity — the communication is strictly between one writer and one reader on the same machine.

## Consequences

- Pi's workflow is a loop of `step`/`events`/`send` bash calls — simple and testable.
- File operations are atomic enough for single-writer/single-reader on a local filesystem.
- The `--after <seq>` mechanism gives pi a natural cursor for event consumption.
- Debugging is straightforward — inspect the files directly.
