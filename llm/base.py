"""
Base classes and interfaces for the LLM abstraction layer.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum

from scout_agent.custom_logging import get_logger


class LLMBackendType(Enum):
    """Supported LLM backend types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"


@dataclass
class LLMConfig:
    """Configuration for LLM backends."""
    backend_type: LLMBackendType
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMRequest:
    """Request structure for LLM calls."""
    messages: List[Dict[str, str]]
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response structure from LLM calls."""
    content: str
    model: str
    backend_type: str
    usage: Dict[str, Any]
    metadata: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None
    response_time: float = 0.0
    raw_response: Optional[Any] = None


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    def __init__(self, config: LLMConfig):
        """Initialize the backend with configuration."""
        self.config = config
        self.logger = get_logger(f"llm.{config.backend_type.value}")
        self._client = None
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the backend (setup client, validate credentials, etc.)."""
        pass
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def stream_generate(self, request: LLMRequest):
        """Generate a streaming response from the LLM."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is healthy and available."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this backend."""
        pass
    
    async def cleanup(self):
        """Cleanup resources (close connections, etc.)."""
        if hasattr(self._client, 'close'):
            await self._client.close()
        self._initialized = False
    
    def _create_response(self, content: str, model: str, usage: Dict[str, Any] = None,
                        metadata: Dict[str, Any] = None, success: bool = True,
                        error: str = None, response_time: float = 0.0,
                        raw_response: Any = None) -> LLMResponse:
        """Helper method to create standardized LLM responses."""
        return LLMResponse(
            content=content,
            model=model,
            backend_type=self.config.backend_type.value,
            usage=usage or {},
            metadata=metadata or {},
            success=success,
            error=error,
            response_time=response_time,
            raw_response=raw_response
        )
    
    def _prepare_messages(self, request: LLMRequest) -> List[Dict[str, str]]:
        """Prepare messages for the specific backend format."""
        messages = request.messages.copy()
        
        # Add system prompt if provided
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})
        
        return messages
    
    def _get_effective_config(self, request: LLMRequest) -> Dict[str, Any]:
        """Get effective configuration merging request and backend config."""
        config = {
            "temperature": request.temperature or self.config.temperature,
            "max_tokens": request.max_tokens or self.config.max_tokens,
        }
        
        # Merge extra params
        config.update(self.config.extra_params)
        config.update(request.extra_params)
        
        return config
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.retry_attempts - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All {self.config.retry_attempts} attempts failed")
        
        raise last_exception


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMBackendError(LLMError):
    """Exception for backend-specific errors."""
    pass


class LLMConfigError(LLMError):
    """Exception for configuration errors."""
    pass


class LLMTimeoutError(LLMError):
    """Exception for timeout errors."""
    pass


class LLMRateLimitError(LLMError):
    """Exception for rate limit errors."""
    pass
