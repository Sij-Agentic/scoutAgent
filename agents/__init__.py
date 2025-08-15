"""
Agents package for ScoutAgent.

Avoid heavy side-effects at import time. We only expose base classes and
registry utilities here. Individual agent modules (e.g., `scout`, `research_agent`)
should handle their own registration when explicitly imported.
"""

from .base import BaseAgent, AgentInput, AgentOutput, get_registry, register_agent, create_agent

__all__ = [
    'BaseAgent',
    'AgentInput',
    'AgentOutput',
    'get_registry',
    'register_agent',
    'create_agent',
]
