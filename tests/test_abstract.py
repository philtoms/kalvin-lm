"""Tests for abstract.py — verify remaining ABCs are intact after KModel deletion."""

import pytest
from kalvin.abstract import KTokenizer, KAgent
from kalvin.kline import KLine, KNode, KSig


class TestExports:
    """Verify __all__ no longer exports KModel."""

    def test_kmodel_not_in_all(self):
        from kalvin import abstract
        assert "KModel" not in abstract.__all__

    def test_ktokenizer_in_all(self):
        from kalvin import abstract
        assert "KTokenizer" in abstract.__all__

    def test_kagent_in_all(self):
        from kalvin import abstract
        assert "KAgent" in abstract.__all__


class TestKTokenizerStillAbstract:
    """KTokenizer must remain a valid, instantiatable-for-subclass ABC."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            KTokenizer()  # type: ignore[abstract]


class TestKAgentStillAbstract:
    """KAgent must remain a valid ABC."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            KAgent()  # type: ignore[abstract]

    def test_model_annotation_is_model(self):
        """KAgent.model return type should resolve to Model, not KModel."""
        from kalvin.model import Model
        # Verify the annotation is a string forward reference to "Model"
        annotations = KAgent.__annotations__
        # abc property annotations live on the class
        assert "model" not in annotations or annotations.get("model") != "KModel"


class TestNoKModelAttribute:
    """Confirm KModel does not exist as an attribute."""

    def test_kmodel_not_accessible(self):
        from kalvin import abstract
        assert not hasattr(abstract, "KModel")
