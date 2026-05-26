"""AgentCodec — serialization for Agent state.

Two adapters: binary (KAC1 format) and JSON. Both read/write the same
Agent state: frame KLines + activity counter.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from struct import pack, unpack
from typing import Literal, TYPE_CHECKING

from kalvin.kline import KLine
from kalvin.model import Model

if TYPE_CHECKING:
    from kalvin.agent import Agent


def to_bytes(agent: Agent) -> bytes:
    """Serialize agent state to binary (KAC1 format)."""
    klines = [kl for kl in agent.model if kl is not None]
    parts: list[bytes] = []
    parts.append(pack("<I", len(klines)))
    for kl in klines:
        parts.append(pack("<Q", kl.signature))
        parts.append(pack("<I", len(kl.nodes)))
        for n in kl.nodes:
            parts.append(pack("<Q", n))

    activity = getattr(agent, "_activity", Counter())
    parts.append(pack("<I", len(activity)))
    for key, count in activity.items():
        parts.append(pack("<Q", key))
        parts.append(pack("<I", count))

    return b"".join(parts)


def from_bytes(data: bytes, agent_cls: type | None = None) -> Agent:
    """Deserialize agent state from binary.

    Args:
        data: Binary blob produced by to_bytes().
        agent_cls: Agent class to instantiate. Defaults to Agent.
    """
    if agent_cls is None:
        from kalvin.agent import Agent as _Agent
        agent_cls = _Agent

    offset = 0
    kline_count = unpack("<I", data[offset:offset + 4])[0]
    offset += 4

    klines: list[KLine] = []
    for _ in range(kline_count):
        sig = unpack("<Q", data[offset:offset + 8])[0]
        offset += 8
        n_count = unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        nodes = []
        for _ in range(n_count):
            nodes.append(unpack("<Q", data[offset:offset + 8])[0])
            offset += 8
        klines.append(KLine(sig, nodes))

    activity_count = unpack("<I", data[offset:offset + 4])[0]
    offset += 4
    activity: Counter = Counter()
    for _ in range(activity_count):
        key = unpack("<Q", data[offset:offset + 8])[0]
        offset += 8
        count = unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        activity[key] = count

    model = Model()
    for kl in klines:
        model.add(kl)
        model.promote(kl)
    agent = agent_cls(model=model)
    agent._activity = activity
    return agent


def to_dict(agent: Agent) -> dict:
    """Serialize agent state to a dictionary."""
    activity = getattr(agent, "_activity", Counter())
    return {
        "klines": [
            {"signature": kl.signature, "nodes": kl.nodes}
            for kl in agent.model
        ],
        "activity": {str(k): v for k, v in activity.items()},
    }


def from_dict(data: dict, agent_cls: type | None = None) -> Agent:
    """Deserialize agent state from a dictionary.

    Args:
        data: Dictionary produced by to_dict().
        agent_cls: Agent class to instantiate. Defaults to Agent.
    """
    if agent_cls is None:
        from kalvin.agent import Agent as _Agent
        agent_cls = _Agent

    klines = [
        KLine(item["signature"], item["nodes"])
        for item in data.get("klines", [])
    ]
    model = Model()
    for kl in klines:
        model.add(kl)
        model.promote(kl)
    agent = agent_cls(model=model)
    if "activity" in data:
        agent._activity = Counter({int(k): v for k, v in data["activity"].items()})
    return agent


def save(agent: Agent, path: str | Path, format: Literal["bin", "json"] | None = None) -> None:
    """Save agent state to file. Format auto-detected from suffix."""
    path = Path(path)
    if format is None:
        format = "json" if path.suffix.lower() == ".json" else "bin"
    if format == "bin":
        path.write_bytes(to_bytes(agent))
    else:
        path.write_text(json.dumps(to_dict(agent), indent=2))


def load(path: str | Path = "data/agent.bin", format: Literal["bin", "json"] | None = None,
         agent_cls: type | None = None) -> Agent:
    """Load agent state from file. Format auto-detected from suffix."""
    path = Path(path)
    if format is None:
        format = "json" if path.suffix.lower() == ".json" else "bin"
    if format == "bin":
        return from_bytes(path.read_bytes(), agent_cls=agent_cls)
    else:
        return from_dict(json.loads(path.read_text()), agent_cls=agent_cls)
