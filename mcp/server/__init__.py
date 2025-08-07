"""
MCP Server package for ScoutAgent.

This package provides the core server components for the Model Context Protocol (MCP).
"""

from .base import Server, Context
from .sse import SseServerTransport

__all__ = ["Server", "Context", "SseServerTransport"]
