"""
MCP Client package for ScoutAgent.

This package provides client components for connecting to MCP servers.
"""

from .base import ClientSession
from .sse import sse_client

__all__ = ["ClientSession", "sse_client"]
