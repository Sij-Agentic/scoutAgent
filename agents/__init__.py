"""
Agents module for ScoutAgent.

This module contains all specialized agents for the ScoutAgent system.
"""

from .base import BaseAgent, AgentInput, AgentOutput, get_registry, register_agent, create_agent
from .research_agent import ResearchAgent
from .code_agent import CodeAgent
from .analysis_agent import AnalysisAgent
from .gap_finder import GapFinderAgent
from .builder import BuilderAgent
from .writer import WriterAgent
from .memory_agent import MemoryAgent

# Re-export base classes
__all__ = [
    'BaseAgent',
    'AgentInput', 
    'AgentOutput',
    'get_registry',
    'register_agent',
    'ResearchAgent',
    'CodeAgent',
    'AnalysisAgent',
    'GapFinderAgent',
    'BuilderAgent',
    'WriterAgent',
    'MemoryAgent'
]

# Register all built-in agents
register_agent(ResearchAgent)
register_agent(CodeAgent)
register_agent(AnalysisAgent)
register_agent(GapFinderAgent)
register_agent(BuilderAgent)
register_agent(WriterAgent)
register_agent(MemoryAgent)
