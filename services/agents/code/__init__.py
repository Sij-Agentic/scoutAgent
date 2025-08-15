"""
Code Execution Service Package.

This package provides code execution capabilities as a registered service.
"""

from .service import CodeExecutionService, get_code_execution_service

__all__ = ['CodeExecutionService', 'get_code_execution_service']
