"""CLI entry point for the multi-agent harness.

Usage::

    python -m harness --config harness.yaml
    python -m harness --host 0.0.0.0 --port 9000

Loads a YAML configuration file, instantiates embedded participants
(KAgent adapter, Trainer), starts the WebSocket server for client
participants (Slack, TUI), and runs the bus event loop.  SIGTERM/SIGINT
trigger graceful shutdown with Trainer state persistence.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from harness.bus import MessageBus
from harness.server import HarnessServer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thin wrapper to prevent double-subscription
# ---------------------------------------------------------------------------
# Both KAgentAdapter and Trainer call bus.subscribe() in their constructors.
# HarnessServer._setup() also calls bus.subscribe() on the factory result.
# Returning this wrapper avoids double-dispatch of every message.

class _AlreadySubscribed:
    """Wrapper returned by factories for participants that self-subscribe.

    ``HarnessServer._setup()`` calls ``bus.subscribe(address, result.on_message)``
    but the real participant already subscribed during construction.  This
    wrapper's ``on_message`` is a no-op to avoid double-dispatch.
    """

    def __init__(self, participant: Any) -> None:
        self._participant = participant

    @property
    def address(self) -> str:
        return self._participant.address

    @property
    def wrapped(self) -> Any:
        """Access the underlying participant (e.g. for shutdown callbacks)."""
        return self._participant

    def on_message(self, msg: Any) -> None:
        """No-op — the real participant is already subscribed."""
        pass


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Multi-agent harness runtime",
    )
    parser.add_argument(
        "--config",
        default="harness.yaml",
        help="Path to YAML/JSON config file (default: harness.yaml)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override WebSocket server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override WebSocket server port",
    )
    return parser


# ---------------------------------------------------------------------------
# Extra config sections (not parsed by load_config)
# ---------------------------------------------------------------------------


def _load_extras(path: str | Path) -> dict:
    """Load the full YAML config and return non-participants sections.

    ``load_config()`` only parses the ``participants:`` list.  This function
    reads the full file so the CLI can extract ``server:`` and ``trainer:``
    sections.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: load config, wire participants, run harness."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = args.config
    extras = _load_extras(config_path)

    # Extract server settings (CLI args override config file)
    server_cfg = extras.get("server", {})
    host = args.host or server_cfg.get("host", "localhost")
    port = args.port or server_cfg.get("port", 8765)

    # Extract trainer settings
    trainer_cfg = extras.get("trainer", {})
    state_path = trainer_cfg.get("state_path", "trainer_state.json")

    # -- Create bus and server -----------------------------------------------
    bus = MessageBus()
    server = HarnessServer(config_path, bus)

    # -- Shutdown callback list (decoupled) ----------------------------------
    shutdown_callbacks: list = []

    # -- Register embedded participant factories -----------------------------

    # KAgent adapter factory (two-phase wiring to avoid circular dep)
    def kagent_factory(address: str, bus: MessageBus) -> _AlreadySubscribed:
        from harness.adapter import KAgentAdapter
        from kalvin.agent import KAgent

        adapter = KAgentAdapter(bus, address=address)
        kagent = KAgent(adapter=adapter)
        adapter.bind(kagent)
        return _AlreadySubscribed(adapter)

    server.register_participant_class("KAgent", kagent_factory)

    # Trainer factory
    # Capture trainer reference for state persistence on shutdown
    trainer_holder: list = []

    def trainer_factory(address: str, bus: MessageBus) -> _AlreadySubscribed:
        from trainer.curriculum import Curriculum
        from trainer.trainer import Trainer

        curriculum_file = trainer_cfg.get("curriculum_file", "")
        curricula_dir = trainer_cfg.get("curricula_dir", "curricula")

        # Load curriculum from file if specified
        curriculum: Curriculum
        if curriculum_file:
            from trainer.curriculum_document import CurriculumDocument
            doc = CurriculumDocument.from_file(curriculum_file)
            curriculum = Curriculum(doc)
        else:
            curriculum = Curriculum(lessons=[])  # Empty default; loaded via session startup

        # Set up LLM client if configured
        llm_client = _build_llm_client(trainer_cfg)

        trainer = Trainer(
            bus,
            curriculum,
            address=address,
            save_path=state_path,
            curriculum_file=curriculum_file or None,
            curricula_dir=curricula_dir or None,
            llm_client=llm_client,
        )
        trainer_holder.append(trainer)

        # Register state persistence as a shutdown callback
        if state_path:
            shutdown_callbacks.append(
                lambda: _persist_trainer_state(trainer)
            )

        return _AlreadySubscribed(trainer)

    server.register_participant_class("Trainer", trainer_factory)

    # -- Register bus.stop() and server.stop() as shutdown callbacks --------
    shutdown_callbacks.append(bus.stop)
    shutdown_callbacks.append(server.stop)

    # -- Run -----------------------------------------------------------------
    try:
        server.run_sync(host=host, port=port)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — shutting down")
    finally:
        for callback in shutdown_callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Shutdown callback failed")


def _persist_trainer_state(trainer: Any) -> None:
    """Persist Trainer curriculum state for restart recovery."""
    try:
        trainer.state.save()
        logger.info("Trainer state persisted")
    except ValueError:
        # No save path configured — skip silently
        logger.debug("No save path configured — skipping Trainer state persistence")
    except Exception:
        logger.exception("Failed to persist Trainer state")


def _build_llm_client(trainer_cfg: dict) -> Any | None:
    """Build an LLM client from the trainer config, or return None.

    The API key is read from the ``KALVIN_LLM_API_KEY`` environment variable.
    The ``llm`` section in the trainer config is optional and only provides
    ``base_url`` and ``model`` overrides — it should NOT contain ``api_key``.

    Config keys (all optional):
    - ``base_url``: defaults to OpenAI-compatible endpoint
    - ``model``: defaults to "glm-5.1"
    """
    import os

    api_key = os.environ.get("KALVIN_LLM_API_KEY", "").strip()
    if not api_key:
        logger.debug("KALVIN_LLM_API_KEY not set — LLM client disabled")
        return None

    llm_cfg = trainer_cfg.get("llm") or {}

    try:
        from trainer.cogitation import OpenAICompatibleClient

        base_url = llm_cfg.get(
            "base_url", "https://open.bigmodel.cn/api/paas/v4"
        )
        model = llm_cfg.get("model", "glm-5.1")
        client = OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        logger.info("LLM client configured (model=%s, base_url=%s)", model, base_url)
        return client
    except ImportError:
        logger.warning(
            "'openai' package not installed — LLM client unavailable. "
            "Install with: pip install kalvin[trainer]"
        )
        return None
    except Exception:
        logger.exception("Failed to create LLM client")
        return None


if __name__ == "__main__":
    main()
