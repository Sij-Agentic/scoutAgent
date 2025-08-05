"""
DeepSeek backend implementation.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

from ..base import LLMBackend, LLMRequest, LLMResponse, LLMConfig, LLMBackendError, LLMTimeoutError, LLMRateLimitError


class DeepSeekBackend(LLMBackend):
    """DeepSeek backend using OpenAI-compatible API."""
    
    def __init__(self, config: LLMConfig):
        """Initialize DeepSeek backend."""
        super().__init__(config)
        
        if not OPENAI_AVAILABLE:
            raise LLMBackendError("OpenAI library not available. Install with: pip install openai")
        
        if not config.api_key:
            raise LLMBackendError("DeepSeek API key is required")
        
        # DeepSeek uses OpenAI-compatible API
        self.base_url = config.base_url or "https://api.deepseek.com/v1"
        self._client: Optional[AsyncOpenAI] = None
    
    async def initialize(self) -> bool:
        """Initialize the DeepSeek backend."""
        try:
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.base_url,
                timeout=self.config.timeout
            )
            
            # Test the connection with a simple request
            await self._test_connection()
            
            self._initialized = True
            self.logger.info(f"DeepSeek backend initialized with model: {self.config.model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DeepSeek backend: {e}")
            self._client = None
            return False
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using DeepSeek."""
        if not self._initialized or not self._client:
            raise LLMBackendError("DeepSeek backend not initialized")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Prepare the request
            messages = self._prepare_messages(request)
            config = self._get_effective_config(request)
            
            # Build DeepSeek request parameters
            params = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": config["temperature"],
                "max_tokens": config["max_tokens"],
                "stream": False
            }
            
            # Add optional parameters
            if "top_p" in config:
                params["top_p"] = config["top_p"]
            if "frequency_penalty" in config:
                params["frequency_penalty"] = config["frequency_penalty"]
            if "presence_penalty" in config:
                params["presence_penalty"] = config["presence_penalty"]
            
            # Add tools if provided
            if request.tools:
                params["tools"] = request.tools
                if request.tool_choice:
                    params["tool_choice"] = request.tool_choice
            
            # Make the request with retry logic
            response = await self._retry_with_backoff(
                self._client.chat.completions.create,
                **params
            )
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            # Extract content
            content = ""
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if choice.message.content:
                    content = choice.message.content
                elif choice.message.tool_calls:
                    # Handle tool calls
                    content = json.dumps([
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in choice.message.tool_calls
                    ])
            
            return self._create_response(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                },
                metadata={
                    "finish_reason": response.choices[0].finish_reason if response.choices else None,
                    "system_fingerprint": getattr(response, "system_fingerprint", None)
                },
                response_time=response_time,
                raw_response=response
            )
            
        except openai.RateLimitError as e:
            raise LLMRateLimitError(f"DeepSeek rate limit exceeded: {e}")
        except openai.APITimeoutError as e:
            raise LLMTimeoutError(f"DeepSeek request timed out: {e}")
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise LLMBackendError(f"DeepSeek generation failed: {e}")
    
    async def stream_generate(self, request: LLMRequest):
        """Generate a streaming response using DeepSeek."""
        if not self._initialized or not self._client:
            raise LLMBackendError("DeepSeek backend not initialized")
        
        try:
            # Prepare the request
            messages = self._prepare_messages(request)
            config = self._get_effective_config(request)
            
            # Build DeepSeek request parameters
            params = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": config["temperature"],
                "max_tokens": config["max_tokens"],
                "stream": True
            }
            
            # Add optional parameters
            if "top_p" in config:
                params["top_p"] = config["top_p"]
            if "frequency_penalty" in config:
                params["frequency_penalty"] = config["frequency_penalty"]
            if "presence_penalty" in config:
                params["presence_penalty"] = config["presence_penalty"]
            
            # Make the streaming request
            stream = await self._client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
                        
        except openai.RateLimitError as e:
            raise LLMRateLimitError(f"DeepSeek rate limit exceeded: {e}")
        except openai.APITimeoutError as e:
            raise LLMTimeoutError(f"DeepSeek streaming request timed out: {e}")
        except Exception as e:
            self.logger.error(f"Error streaming response: {e}")
            raise LLMBackendError(f"DeepSeek streaming failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if DeepSeek API is healthy and available."""
        try:
            if not self._client:
                return False
            
            # Make a minimal request to test the API
            response = await self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                timeout=10
            )
            
            return response is not None
            
        except Exception as e:
            self.logger.debug(f"DeepSeek health check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available DeepSeek models."""
        return [
            "deepseek-chat",
            "deepseek-coder",
            "deepseek-reasoner",
            "deepseek-v3"
        ]
    
    async def cleanup(self):
        """Cleanup DeepSeek backend resources."""
        if self._client:
            await self._client.close()
            self._client = None
        await super().cleanup()
    
    async def _test_connection(self):
        """Test the DeepSeek connection."""
        try:
            # Make a minimal request to verify the API key and connection
            response = await self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                timeout=10
            )
            
            if not response:
                raise LLMBackendError("Invalid response from DeepSeek API")
                
        except openai.AuthenticationError:
            raise LLMBackendError("Invalid DeepSeek API key")
        except openai.NotFoundError:
            raise LLMBackendError(f"Model {self.config.model_name} not found")
        except Exception as e:
            raise LLMBackendError(f"DeepSeek connection test failed: {e}")
