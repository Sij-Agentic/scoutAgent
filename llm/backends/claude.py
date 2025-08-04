"""
Claude (Anthropic) backend implementation.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional

try:
    import anthropic
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    AsyncAnthropic = None

from ..base import LLMBackend, LLMRequest, LLMResponse, LLMConfig, LLMBackendError, LLMTimeoutError, LLMRateLimitError


class ClaudeBackend(LLMBackend):
    """Claude (Anthropic) backend for Claude models."""
    
    def __init__(self, config: LLMConfig):
        """Initialize Claude backend."""
        super().__init__(config)
        
        if not ANTHROPIC_AVAILABLE:
            raise LLMBackendError("Anthropic library not available. Install with: pip install anthropic")
        
        if not config.api_key:
            raise LLMBackendError("Anthropic API key is required")
        
        self._client: Optional[AsyncAnthropic] = None
    
    async def initialize(self) -> bool:
        """Initialize the Claude backend."""
        try:
            self._client = AsyncAnthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
            
            # Test the connection
            await self._test_connection()
            
            self._initialized = True
            self.logger.info(f"Claude backend initialized with model: {self.config.model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Claude backend: {e}")
            self._client = None
            return False
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using Claude."""
        if not self._initialized or not self._client:
            raise LLMBackendError("Claude backend not initialized")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Prepare the request
            messages = self._prepare_messages(request)
            config = self._get_effective_config(request)
            
            # Extract system message if present
            system_message = None
            filtered_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    filtered_messages.append(msg)
            
            # Build Claude request parameters
            params = {
                "model": self.config.model_name,
                "messages": filtered_messages,
                "max_tokens": config["max_tokens"],
                "temperature": config["temperature"],
                "stream": False
            }
            
            # Add system message if present
            if system_message:
                params["system"] = system_message
            
            # Add optional parameters
            if "top_p" in config:
                params["top_p"] = config["top_p"]
            if "top_k" in config:
                params["top_k"] = config["top_k"]
            
            # Add tools if provided
            if request.tools:
                params["tools"] = request.tools
                if request.tool_choice:
                    params["tool_choice"] = request.tool_choice
            
            # Make the request with retry logic
            response = await self._retry_with_backoff(
                self._client.messages.create,
                **params
            )
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            # Extract content
            content = ""
            if response.content:
                # Claude returns content as a list of content blocks
                content_blocks = []
                for block in response.content:
                    if hasattr(block, 'text'):
                        content_blocks.append(block.text)
                    elif hasattr(block, 'tool_use'):
                        # Handle tool use
                        tool_use = {
                            "id": block.tool_use.id,
                            "type": "tool_use",
                            "name": block.tool_use.name,
                            "input": block.tool_use.input
                        }
                        content_blocks.append(json.dumps(tool_use))
                
                content = "".join(content_blocks)
            
            return self._create_response(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                    "completion_tokens": response.usage.output_tokens if response.usage else 0,
                    "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0
                },
                metadata={
                    "stop_reason": response.stop_reason,
                    "stop_sequence": getattr(response, "stop_sequence", None)
                },
                response_time=response_time,
                raw_response=response
            )
            
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Claude rate limit exceeded: {e}")
        except anthropic.APITimeoutError as e:
            raise LLMTimeoutError(f"Claude request timed out: {e}")
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise LLMBackendError(f"Claude generation failed: {e}")
    
    async def stream_generate(self, request: LLMRequest):
        """Generate a streaming response using Claude."""
        if not self._initialized or not self._client:
            raise LLMBackendError("Claude backend not initialized")
        
        try:
            # Prepare the request
            messages = self._prepare_messages(request)
            config = self._get_effective_config(request)
            
            # Extract system message if present
            system_message = None
            filtered_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    filtered_messages.append(msg)
            
            # Build Claude request parameters
            params = {
                "model": self.config.model_name,
                "messages": filtered_messages,
                "max_tokens": config["max_tokens"],
                "temperature": config["temperature"],
                "stream": True
            }
            
            # Add system message if present
            if system_message:
                params["system"] = system_message
            
            # Add optional parameters
            if "top_p" in config:
                params["top_p"] = config["top_p"]
            if "top_k" in config:
                params["top_k"] = config["top_k"]
            
            # Make the streaming request
            stream = await self._client.messages.create(**params)
            
            async for chunk in stream:
                if chunk.type == "content_block_delta":
                    if hasattr(chunk.delta, 'text'):
                        yield chunk.delta.text
                        
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Claude rate limit exceeded: {e}")
        except anthropic.APITimeoutError as e:
            raise LLMTimeoutError(f"Claude streaming request timed out: {e}")
        except Exception as e:
            self.logger.error(f"Error streaming response: {e}")
            raise LLMBackendError(f"Claude streaming failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if Claude API is healthy and available."""
        try:
            if not self._client:
                return False
            
            # Make a minimal request to test the API
            response = await self._client.messages.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                timeout=10
            )
            
            return response is not None
            
        except Exception as e:
            self.logger.debug(f"Claude health check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available Claude models."""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]
    
    async def cleanup(self):
        """Cleanup Claude backend resources."""
        if self._client:
            await self._client.close()
            self._client = None
        await super().cleanup()
    
    async def _test_connection(self):
        """Test the Claude connection."""
        try:
            # Make a minimal request to verify the API key and connection
            response = await self._client.messages.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                timeout=10
            )
            
            if not response:
                raise LLMBackendError("Invalid response from Claude API")
                
        except anthropic.AuthenticationError:
            raise LLMBackendError("Invalid Anthropic API key")
        except anthropic.NotFoundError:
            raise LLMBackendError(f"Model {self.config.model_name} not found")
        except Exception as e:
            raise LLMBackendError(f"Claude connection test failed: {e}")
