"""
Logging Service Package.

This package provides logging functionality as a registered service.
"""

from services.logging.service import LoggingService, get_logging_service

__all__ = ['LoggingService', 'get_logging_service']
