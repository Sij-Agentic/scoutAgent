"""
Memory Service Package.

This package provides memory storage and retrieval capabilities as a registered service.
"""

from .service import MemoryService, get_memory_service

__all__ = ['MemoryService', 'get_memory_service']
