"""
Multi-Backend LLM Abstraction Layer for ScoutAgent.

This module provides a unified interface for interacting with multiple LLM backends
including Ollama, OpenAI, Claude (Anthropic), and Gemini.
"""

from .base import LLMBackend, LLMRequest, LLMResponse, LLMConfig
from .manager import LLMManager, get_llm_manager, initialize_llm_backends
from .backends import OllamaBackend, OpenAIBackend, ClaudeBackend, GeminiBackend, DeepSeekBackend

__all__ = [
    'LLMBackend',
    'LLMRequest', 
    'LLMResponse',
    'LLMConfig',
    'LLMManager',
    'get_llm_manager',
    'initialize_llm_backends',
    'OllamaBackend',
    'OpenAIBackend', 
    'ClaudeBackend',
    'GeminiBackend',
    'DeepSeekBackend'
]
