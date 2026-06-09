"""Tests for dev/nlp/run_corpus.py corpus runner."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure imports work
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "dev" / "nlp"))

from run_corpus import build_parser, load_dataset_texts


# ---------------------------------------------------------------------------
# CLI argument parser tests
# ---------------------------------------------------------------------------

class TestBuildParser:
    """Tests for the CLI argument parser."""

    def test_default_dataset(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.dataset == "stas/openwebtext-10k"

    def test_default_split(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.split == "train"

    def test_default_text_field(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.text_field == "text"

    def test_default_stem(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.stem == "openwebtext"

    def test_default_output(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.output == Path("data/tokenizer")

    def test_default_model(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.model == "en_core_web_trf"

    def test_default_batch_size(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.batch_size == 100

    def test_default_max_samples_none(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.max_samples is None

    def test_default_verbose_false(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.verbose is False

    def test_default_gpu_false(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.gpu is False

    def test_custom_dataset(self):
        parser = build_parser()
        args = parser.parse_args(["--dataset", "wikipedia"])
        assert args.dataset == "wikipedia"

    def test_custom_split(self):
        parser = build_parser()
        args = parser.parse_args(["--split", "test"])
        assert args.split == "test"

    def test_custom_text_field(self):
        parser = build_parser()
        args = parser.parse_args(["--text-field", "content"])
        assert args.text_field == "content"

    def test_custom_stem(self):
        parser = build_parser()
        args = parser.parse_args(["--stem", "wiki"])
        assert args.stem == "wiki"

    def test_custom_output(self):
        parser = build_parser()
        args = parser.parse_args(["--output", "/tmp/out"])
        assert args.output == Path("/tmp/out")

    def test_custom_model(self):
        parser = build_parser()
        args = parser.parse_args(["--model", "en_core_web_sm"])
        assert args.model == "en_core_web_sm"

    def test_gpu_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--gpu"])
        assert args.gpu is True

    def test_batch_size(self):
        parser = build_parser()
        args = parser.parse_args(["--batch-size", "200"])
        assert args.batch_size == 200

    def test_max_samples(self):
        parser = build_parser()
        args = parser.parse_args(["--max-samples", "5"])
        assert args.max_samples == 5

    def test_verbose_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--verbose"])
        assert args.verbose is True


# ---------------------------------------------------------------------------
# load_dataset_texts tests
# ---------------------------------------------------------------------------

class TestLoadDatasetTexts:
    """Tests for the load_dataset_texts function."""

    def _mock_datasets(self, mock_rows):
        """Create a mock for the datasets.load_dataset function.

        Returns a context manager that patches the import inside
        load_dataset_texts by injecting a fake 'datasets' module.
        """
        mock_ds = mock_rows
        mock_module = MagicMock()
        mock_module.load_dataset.return_value = mock_ds
        return patch.dict(sys.modules, {"datasets": mock_module})

    def test_extracts_text_field(self):
        """Should extract the specified text field from each row."""
        rows = [
            {"text": "Hello world", "id": 1},
            {"text": "Another text", "id": 2},
        ]
        with self._mock_datasets(rows):
            texts = load_dataset_texts("some/dataset")
        assert texts == ["Hello world", "Another text"]

    def test_calls_load_dataset_correctly(self):
        """Should call load_dataset with the correct dataset name and split."""
        with self._mock_datasets([]) as mods:
            load_dataset_texts("my/dataset", split="test", verbose=False)
            mods["datasets"].load_dataset.assert_called_once_with(
                "my/dataset", split="test", streaming=False, trust_remote_code=True
            )

    def test_custom_text_field(self):
        """Should extract a custom text field."""
        rows = [
            {"content": "Content A", "text": "Text A"},
            {"content": "Content B", "text": "Text B"},
        ]
        with self._mock_datasets(rows):
            texts = load_dataset_texts("some/dataset", text_field="content")
        assert texts == ["Content A", "Content B"]

    def test_max_samples_limits_output(self):
        """Should limit the output to max_samples rows."""
        rows = [{"text": f"text {i}"} for i in range(100)]
        with self._mock_datasets(rows):
            texts = load_dataset_texts("some/dataset", max_samples=5)
        assert len(texts) == 5
        assert texts[0] == "text 0"
        assert texts[4] == "text 4"

    def test_max_samples_none_returns_all(self):
        """When max_samples is None, return all rows."""
        rows = [{"text": f"text {i}"} for i in range(50)]
        with self._mock_datasets(rows):
            texts = load_dataset_texts("some/dataset", max_samples=None)
        assert len(texts) == 50

    def test_max_samples_larger_than_dataset(self):
        """When max_samples > dataset size, return all rows."""
        rows = [{"text": f"text {i}"} for i in range(10)]
        with self._mock_datasets(rows):
            texts = load_dataset_texts("some/dataset", max_samples=999)
        assert len(texts) == 10

    def test_empty_dataset(self):
        """Should handle an empty dataset."""
        with self._mock_datasets([]):
            texts = load_dataset_texts("some/dataset")
        assert texts == []

    def test_verbose_does_not_crash(self, capsys):
        """Verbose mode should print without errors."""
        rows = [{"text": "hello"}]
        with self._mock_datasets(rows):
            texts = load_dataset_texts("some/dataset", verbose=True)
        assert texts == ["hello"]
        captured = capsys.readouterr()
        assert "some/dataset" in captured.out
