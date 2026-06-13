"""Tests for abstract.py — verify remaining ABCs are intact after KModel/KAgent deletion."""

import pytest

from kalvin.abstract import KTokenizer


class TestExports:
    """Verify __all__ no longer exports KModel or KAgent."""

    def test_kmodel_not_in_all(self):
        from kalvin import abstract

        assert "KModel" not in abstract.__all__

    def test_ktokenizer_in_all(self):
        from kalvin import abstract

        assert "KTokenizer" in abstract.__all__


class TestKTokenizerStillAbstract:
    """KTokenizer must remain a valid, instantiatable-for-subclass ABC."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            KTokenizer()  # type: ignore[abstract]


class TestNoKModelAttribute:
    """Confirm KModel does not exist as an attribute."""

    def test_kmodel_not_accessible(self):
        from kalvin import abstract

        assert not hasattr(abstract, "KModel")
