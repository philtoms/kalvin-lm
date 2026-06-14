"""Kalvin - A learning system that stores and retrieves memory as klines."""

__version__ = "0.1.0"

from kalvin.events import EventBus, RationaliseEvent

# EventBus is kept as a test/dev utility adapter implementing the
# KAgentAdapter protocol.  Production code should use the KAgentAdapter
# from harness.adapter instead.
__all__ = ["EventBus", "RationaliseEvent", "__version__"]
