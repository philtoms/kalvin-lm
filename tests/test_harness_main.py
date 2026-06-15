"""Tests for harness __main__ — _build_llm_client and trainer_factory wiring.

Covers KB-032: LLM client construction from config/env and Trainer wiring.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from harness.__main__ import _build_llm_client, _resolve_llm_wiring  # noqa: E402

# ── TestBuildLLMClient ────────────────────────────────────────────────


class TestBuildLLMClient:
    """KB-032: _build_llm_client constructs OpenAICompatibleClient from config."""

    def test_returns_none_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without KALVIN_LLM_API_KEY env var, returns None."""
        monkeypatch.delenv("KALVIN_LLM_API_KEY", raising=False)
        result = _build_llm_client({"llm": {"base_url": "http://test", "model": "test-model"}})
        assert result is None

    def test_returns_client_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With KALVIN_LLM_API_KEY set, returns an OpenAICompatibleClient."""
        monkeypatch.setenv("KALVIN_LLM_API_KEY", "test-key-123")

        mock_client = MagicMock()
        with patch(
            "trainer.cogitation.OpenAICompatibleClient", return_value=mock_client
        ) as mock_cls:
            result = _build_llm_client({"llm": {"model": "test-model"}})
            assert result is mock_client
            mock_cls.assert_called_once_with(
                api_key="test-key-123",
                base_url="https://api.z.ai/api/coding/paas/v4",
                model="test-model",
            )

    def test_client_uses_config_model_and_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returned client uses config's model and base_url."""
        monkeypatch.setenv("KALVIN_LLM_API_KEY", "test-key-123")

        mock_client = MagicMock()
        mock_client._model = "custom-model"
        with patch(
            "trainer.cogitation.OpenAICompatibleClient", return_value=mock_client
        ) as mock_cls:
            result = _build_llm_client(
                {
                    "llm": {"model": "custom-model", "base_url": "https://custom.api/v1"},
                }
            )
            assert result is mock_client
            mock_cls.assert_called_once_with(
                api_key="test-key-123",
                base_url="https://custom.api/v1",
                model="custom-model",
            )

    def test_client_uses_defaults_without_llm_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no llm config section, uses default model and base_url."""
        monkeypatch.setenv("KALVIN_LLM_API_KEY", "test-key-123")

        mock_client = MagicMock()
        with patch(
            "trainer.cogitation.OpenAICompatibleClient", return_value=mock_client
        ) as mock_cls:
            result = _build_llm_client({})
            assert result is mock_client
            mock_cls.assert_called_once_with(
                api_key="test-key-123",
                base_url="https://api.z.ai/api/coding/paas/v4",
                model="glm-5.1",
            )

    def test_returns_none_on_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When openai package import fails, returns None without crashing."""
        monkeypatch.setenv("KALVIN_LLM_API_KEY", "test-key-123")

        with patch(
            "trainer.cogitation.OpenAICompatibleClient",
            side_effect=ImportError("no openai"),
        ):
            result = _build_llm_client({"llm": {"model": "test"}})
            assert result is None

    def test_empty_llm_config_uses_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_build_llm_client({}) with KALVIN_LLM_API_KEY set uses all defaults."""
        monkeypatch.setenv("KALVIN_LLM_API_KEY", "test-key-123")

        mock_client = MagicMock()
        with patch(
            "trainer.cogitation.OpenAICompatibleClient", return_value=mock_client
        ) as mock_cls:
            result = _build_llm_client({})
            assert result is mock_client
            # Default model is "glm-5.1", default base_url is the GLM endpoint
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["model"] == "glm-5.1"
            assert call_kwargs["base_url"] == "https://api.z.ai/api/coding/paas/v4"

    def test_llm_enabled_defaults_to_true(self) -> None:
        """RD-1: trainer.llm.enabled defaults to True; unset config behaves as today."""
        sentinel = MagicMock()
        with patch("harness.__main__._build_llm_client", return_value=sentinel) as mock_build:
            # No llm section at all
            _, delegate_reactive = _resolve_llm_wiring({})
            assert delegate_reactive is False
            # llm section present but no enabled key
            _, delegate_reactive = _resolve_llm_wiring({"llm": {}})
            assert delegate_reactive is False
            # The default path builds the client (proving today's behaviour)
            assert mock_build.call_count == 2

    def test_llm_enabled_explicitly_true(self) -> None:
        """RD-1: enabled explicitly True → delegate_reactive False, client built."""
        with patch("harness.__main__._build_llm_client", return_value="CLIENT") as mock_build:
            _, delegate_reactive = _resolve_llm_wiring({"llm": {"enabled": True}})
            assert delegate_reactive is False
            mock_build.assert_called_once_with({"llm": {"enabled": True}})


# ── TestTrainerFactoryLLMWiring ───────────────────────────────────────


class TestTrainerFactoryLLMWiring:
    """KB-032: trainer_factory calls _build_llm_client and passes to Trainer."""

    def test_trainer_llm_client_not_none_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When KALVIN_LLM_API_KEY is set, trainer receives a non-None llm_client."""
        monkeypatch.setenv("KALVIN_LLM_API_KEY", "test-key-123")

        from harness.bus import MessageBus
        from trainer.cogitation import LLMClient
        from trainer.curriculum import Curriculum
        from trainer.trainer import Trainer

        mock_llm = MagicMock(spec=LLMClient)
        bus = MessageBus()
        curriculum = Curriculum([])

        trainer = Trainer(
            bus,
            curriculum,
            role="trainer",
            curricula_dir="/tmp/curricula",
            llm_client=mock_llm,
        )
        assert trainer._llm_client is not None

    def test_trainer_llm_client_none_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When KALVIN_LLM_API_KEY is unset, _build_llm_client returns None."""
        monkeypatch.delenv("KALVIN_LLM_API_KEY", raising=False)

        from harness.bus import MessageBus
        from trainer.curriculum import Curriculum
        from trainer.trainer import Trainer

        llm_client = _build_llm_client({"llm": {"model": "test-model"}})
        assert llm_client is None

        bus = MessageBus()
        curriculum = Curriculum([])

        trainer = Trainer(
            bus,
            curriculum,
            role="trainer",
            curricula_dir="/tmp/curricula",
            llm_client=llm_client,
        )
        assert trainer._llm_client is None

        # Trainer still works — just no generation capability
        assert not trainer._session_active

    def test_build_llm_client_and_trainer_integration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_build_llm_client output can be passed directly to Trainer."""
        monkeypatch.setenv("KALVIN_LLM_API_KEY", "test-key-123")

        from harness.bus import MessageBus
        from trainer.curriculum import Curriculum
        from trainer.trainer import Trainer

        mock_client = MagicMock()
        mock_client._model = "test-model"

        with patch("trainer.cogitation.OpenAICompatibleClient", return_value=mock_client):
            llm_client = _build_llm_client({"llm": {"model": "test-model"}})
            assert llm_client is mock_client

            bus = MessageBus()
            curriculum = Curriculum([])

            trainer = Trainer(
                bus,
                curriculum,
                role="trainer",
                curricula_dir="/tmp/curricula",
                llm_client=llm_client,
            )
            assert trainer._llm_client is mock_client

    def test_trainer_auto_wires_cogitate_fn_with_llm_client(self) -> None:
        """KB-125: When llm_client is passed without cogitate_fn, reactor's cogitate_fn is wired."""
        from harness.bus import MessageBus
        from trainer.cogitation import LLMClient
        from trainer.curriculum import Curriculum
        from trainer.trainer import Trainer

        mock_llm = MagicMock(spec=LLMClient)
        bus = MessageBus()
        curriculum = Curriculum([])

        trainer = Trainer(
            bus,
            curriculum,
            role="trainer",
            llm_client=mock_llm,
        )
        assert trainer._reactor._cogitate_fn is not None

    def test_flag_true_with_key_builds_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RD-2: enabled True + API key set → client built, delegate_reactive False."""
        monkeypatch.setenv("KALVIN_LLM_API_KEY", "test-key-123")
        sentinel = MagicMock()
        with patch("harness.__main__._build_llm_client", return_value=sentinel) as mock_build:
            llm_client, delegate_reactive = _resolve_llm_wiring({"llm": {"enabled": True}})
            assert llm_client is sentinel
            assert delegate_reactive is False
            mock_build.assert_called_once()

    def test_flag_false_skips_build_even_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RD-3: enabled False → _build_llm_client never called even with API key set."""
        monkeypatch.setenv("KALVIN_LLM_API_KEY", "test-key-123")
        with patch("harness.__main__._build_llm_client") as mock_build:
            llm_client, delegate_reactive = _resolve_llm_wiring({"llm": {"enabled": False}})
            assert mock_build.call_count == 0
            assert llm_client is None
            assert delegate_reactive is True

    def test_flag_false_no_llm_section(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RD-3 robustness: enabled False without API key still skips the build."""
        monkeypatch.delenv("KALVIN_LLM_API_KEY", raising=False)
        with patch("harness.__main__._build_llm_client") as mock_build:
            llm_client, delegate_reactive = _resolve_llm_wiring({"llm": {"enabled": False}})
            assert mock_build.call_count == 0
            assert llm_client is None
            assert delegate_reactive is True


# ── TestAlreadySubscribedWrapper ──────────────────────────────────────


class TestAlreadySubscribedWrapper:
    """KB-126: _AlreadySubscribed exposes .role (not .address)."""

    def test_role_property_returns_participant_role(self) -> None:
        """_AlreadySubscribed.role delegates to the wrapped participant's .role."""
        from harness.__main__ import _AlreadySubscribed

        mock_participant = MagicMock()
        mock_participant.role = "trainer"

        wrapper = _AlreadySubscribed(mock_participant)
        assert wrapper.role == "trainer"

    def test_trainer_constructed_with_role_keyword(self) -> None:
        """Trainer(bus, curriculum, role='trainer') works — KB-126 contract."""
        from harness.bus import MessageBus
        from trainer.curriculum import Curriculum
        from trainer.trainer import Trainer

        bus = MessageBus()
        curriculum = Curriculum([])
        trainer = Trainer(bus, curriculum, role="trainer", llm_client=None)
        assert trainer.role == "trainer"

    def test_kagent_adapter_constructed_with_role_keyword(self) -> None:
        """KAgentAdapter(bus, role='trainee') works — KB-126 contract."""
        from harness.adapter import KAgentAdapter
        from harness.bus import MessageBus

        bus = MessageBus()
        adapter = KAgentAdapter(bus, role="trainee")
        assert adapter.role == "trainee"


# ── TestStatePathDerivation ───────────────────────────────────────────


class TestStatePathDerivation:
    """KB-128: State file path is derived from curriculum_file (not a config key)."""

    def test_state_path_derived_from_curriculum_file(self, tmp_path: Path) -> None:
        """When curriculum_file is set, save_path follows <curriculum>.json convention."""
        from harness.bus import MessageBus
        from trainer.curriculum import Curriculum
        from trainer.trainer import Trainer

        # Simulate the derivation done in __main__.py trainer_factory
        curriculum_file = str(tmp_path / "curricula" / "first-steps.md")
        curriculum_path = Path(curriculum_file)
        state_path = str(curriculum_path.with_suffix(".json"))

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])

        trainer = Trainer(
            bus,
            curriculum,
            role="trainer",
            save_path=state_path,
            curriculum_file=curriculum_file,
        )

        # Save state and verify file exists at derived path
        trainer.state.save()
        assert Path(state_path).exists()

        # The derived path should be <curriculum_dir>/<name>.json
        assert state_path == str(tmp_path / "curricula" / "first-steps.json")

    def test_no_state_file_when_curriculum_file_empty(self) -> None:
        """When curriculum_file is empty, save_path is None and save raises ValueError."""
        from harness.bus import MessageBus
        from trainer.curriculum import Curriculum
        from trainer.trainer import Trainer

        bus = MessageBus()
        curriculum = Curriculum([])

        trainer = Trainer(
            bus,
            curriculum,
            role="trainer",
            save_path=None,
        )

        # No save_path → save() raises ValueError
        assert trainer.state._save_path is None
        with pytest.raises(ValueError, match="No save path specified"):
            trainer.state.save()
