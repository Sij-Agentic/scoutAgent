"""
Memory Service Package.

This package provides memory storage and retrieval capabilities as a registered service.
"""

from services.agents.memory.service import MemoryService, get_memory_service

__all__ = ['MemoryService', 'get_memory_service']
