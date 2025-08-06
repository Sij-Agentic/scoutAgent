"""
Agent Registry for ScoutAgent.

This module provides centralized management of all available agents.
"""

from agents.base import get_registry, register_agent, BaseAgent, AgentInput, AgentOutput

# Re-export the main registry functions for convenience
__all__ = [
    'get_registry',
    'register_agent', 
    'BaseAgent',
    'AgentInput',
    'AgentOutput'
]

# Import and register built-in agents
from agents.research_agent import ResearchAgent
from agents.analysis_agent import AnalysisAgent

# Register built-in agents
register_agent(ResearchAgent)
register_agent(AnalysisAgent)