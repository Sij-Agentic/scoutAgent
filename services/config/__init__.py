"""
Config Service Package.

This package provides configuration management as a registered service.
"""

from services.config.service import ConfigService, get_config_service

__all__ = ['ConfigService', 'get_config_service']
