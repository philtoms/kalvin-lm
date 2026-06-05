"""Tests for harness role constants."""

from harness.constants import SUPERVISOR_ROLE, TRAINEE_ROLE, TRAINER_ROLE


def test_role_values():
    """Each constant equals its canonical role string."""
    assert TRAINEE_ROLE == "trainee"
    assert TRAINER_ROLE == "trainer"
    assert SUPERVISOR_ROLE == "supervisor"


def test_constants_are_strings():
    """Every role constant is a str instance."""
    for constant in (TRAINEE_ROLE, TRAINER_ROLE, SUPERVISOR_ROLE):
        assert isinstance(constant, str)


def test_constants_are_distinct():
    """No two role constants share the same value."""
    values = [TRAINEE_ROLE, TRAINER_ROLE, SUPERVISOR_ROLE]
    assert len(set(values)) == len(values)


def test_re_exported_from_package():
    """Constants are importable directly from the harness package."""
    from harness import SUPERVISOR_ROLE as PKG_SUPERVISOR
    from harness import TRAINEE_ROLE as PKG_TRAINEE
    from harness import TRAINER_ROLE as PKG_TRAINER

    assert PKG_TRAINEE == TRAINEE_ROLE
    assert PKG_TRAINER == TRAINER_ROLE
    assert PKG_SUPERVISOR == SUPERVISOR_ROLE
