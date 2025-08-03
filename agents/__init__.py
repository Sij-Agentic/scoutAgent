"""
Agents module for ScoutAgent.

This module contains all specialized agents for the ScoutAgent system.
"""

from .base import BaseAgent, AgentInput, AgentOutput, get_registry, register_agent, create_agent
from .research_agent import ResearchAgent
from .code_agent import CodeAgent
from .analysis_agent import AnalysisAgent

# Re-export base classes
__all__ = [
    'BaseAgent',
    'AgentInput', 
    'AgentOutput',
    'get_registry',
    'register_agent',
    'ResearchAgent',
    'CodeAgent',
    'AnalysisAgent'
]

# Register all built-in agents
register_agent(ResearchAgent)
register_agent(CodeAgent)
register_agent(AnalysisAgent)
