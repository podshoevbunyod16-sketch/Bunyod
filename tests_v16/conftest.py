"""Shared fixtures for the v0.16 test corpus.

No test touches ``~/.rasm-cim``: every store is built in ``tmp_path`` and
reached through ``RASM_CIM_HOME`` / an explicit ``home=`` argument.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rasm_cim_v16.dialogue_manager import DialogueManager      # noqa: E402
from rasm_cim_v16.knowledge_graph import KnowledgeGraph        # noqa: E402
from rasm_cim_v16.persistence import Store                     # noqa: E402


class FakeClock:
    """Deterministic, monotonically increasing timestamps."""

    def __init__(self, start: int = 0):
        self.n = start

    def __call__(self) -> str:
        self.n += 1
        return f"2026-01-01T00:00:{self.n:02d}"


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def dm(clock):
    """A bare DialogueManager (no storage attached)."""
    return DialogueManager(clock=clock)


@pytest.fixture
def taught(dm):
    """The Newton corpus used across the spec scenarios."""
    for line in [
        "teach Исаак Ньютон — это физик",
        "teach Ньютон — это Исаак Ньютон",
        "teach Исаак Ньютон открыл закон всемирного тяготения",
        "teach Исаак Ньютон сформулировал законы движения",
        "teach Физик — это учёный",
        "teach Учёный — это человек",
    ]:
        dm.say(line)
    return dm


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated storage directory, also exported as RASM_CIM_HOME."""
    d = tmp_path / ".rasm-cim"
    d.mkdir()
    monkeypatch.setenv("RASM_CIM_HOME", str(d))
    return str(d)


@pytest.fixture
def stored(home, clock):
    """DialogueManager wired to a Store on the isolated home."""
    dm = DialogueManager(clock=clock)
    st = Store(home=home, clock=clock)
    st.attach(dm)
    return dm, st


NEWTON_TEACHINGS = [
    "teach Исаак Ньютон — это физик",
    "teach Ньютон — это Исаак Ньютон",
    "teach Исаак Ньютон открыл закон всемирного тяготения",
    "teach Исаак Ньютон сформулировал законы движения",
    "teach Физик — это учёный",
    "teach Учёный — это человек",
]
