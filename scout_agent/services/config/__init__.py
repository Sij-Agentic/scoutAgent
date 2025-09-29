"""
Config Service Package.

This package provides configuration management as a registered service.
"""

from .service import ConfigService, get_config_service

__all__ = ['ConfigService', 'get_config_service']
