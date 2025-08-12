"""
Ollama backend implementation for local LLM inference.
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional

from ..base import LLMBackend, LLMRequest, LLMResponse, LLMConfig, LLMBackendError, LLMTimeoutError


class OllamaBackend(LLMBackend):
    """Ollama backend for local LLM inference."""
    
    def __init__(self, config: LLMConfig):
        """Initialize Ollama backend."""
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> bool:
        """Initialize the Ollama backend."""
        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            
            # Test connection and pull model if needed
            await self._ensure_model_available()
            
            self._initialized = True
            self.logger.info(f"Ollama backend initialized with model: {self.config.model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama backend: {e}")
            if self._session:
                await self._session.close()
                self._session = None
            return False
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using Ollama."""
        if not self._initialized or not self._session:
            raise LLMBackendError("Ollama backend not initialized")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Prepare the request
            messages = self._prepare_messages(request)
            config = self._get_effective_config(request)
            
            # Convert messages to Ollama format
            prompt = self._messages_to_prompt(messages)
            
            model_name = request.extra_params.get("model_name_override", self.config.model_name) if request and request.extra_params else self.config.model_name
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config["temperature"],
                    "num_predict": config["max_tokens"],
                }
            }
            
            # Add extra parameters
            if "top_p" in config:
                payload["options"]["top_p"] = config["top_p"]
            if "top_k" in config:
                payload["options"]["top_k"] = config["top_k"]
            
            # Make the request
            async with self._session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMBackendError(f"Ollama API error {response.status}: {error_text}")
                
                result = await response.json()
                
                response_time = asyncio.get_event_loop().time() - start_time
                
                return self._create_response(
                    content=result.get("response", ""),
                    model=self.config.model_name,
                    usage={
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "completion_tokens": result.get("eval_count", 0),
                        "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                    },
                    metadata={
                        "eval_duration": result.get("eval_duration", 0),
                        "load_duration": result.get("load_duration", 0),
                        "prompt_eval_duration": result.get("prompt_eval_duration", 0)
                    },
                    response_time=response_time,
                    raw_response=result
                )
                
        except asyncio.TimeoutError:
            raise LLMTimeoutError(f"Ollama request timed out after {self.config.timeout}s")
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise LLMBackendError(f"Ollama generation failed: {e}")
    
    async def stream_generate(self, request: LLMRequest):
        """Generate a streaming response using Ollama."""
        if not self._initialized or not self._session:
            raise LLMBackendError("Ollama backend not initialized")
        
        try:
            # Prepare the request
            messages = self._prepare_messages(request)
            config = self._get_effective_config(request)
            
            # Convert messages to Ollama format
            prompt = self._messages_to_prompt(messages)
            
            model_name = request.extra_params.get("model_name_override", self.config.model_name) if request and request.extra_params else self.config.model_name
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": config["temperature"],
                    "num_predict": config["max_tokens"],
                }
            }
            
            # Make the streaming request
            async with self._session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMBackendError(f"Ollama API error {response.status}: {error_text}")
                
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if chunk.get("response"):
                                yield chunk["response"]
                            
                            # Check if this is the final chunk
                            if chunk.get("done", False):
                                break
                                
                        except json.JSONDecodeError:
                            continue
                            
        except asyncio.TimeoutError:
            raise LLMTimeoutError(f"Ollama streaming request timed out after {self.config.timeout}s")
        except Exception as e:
            self.logger.error(f"Error streaming response: {e}")
            raise LLMBackendError(f"Ollama streaming failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if Ollama is healthy and available."""
        temp_session: Optional[aiohttp.ClientSession] = None
        try:
            session = self._session
            if session is None:
                temp_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10)
                )
                session = temp_session
            async with session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            self.logger.debug(f"Ollama health check failed: {e}")
            return False
        finally:
            if temp_session is not None:
                try:
                    await temp_session.close()
                except Exception:
                    pass
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        # This would need to be implemented as an async method in practice
        # For now, return common Ollama models
        return [
            "llama2", "llama2:7b", "llama2:13b", "llama2:70b",
            "codellama", "codellama:7b", "codellama:13b", "codellama:34b",
            "mistral", "mistral:7b",
            "neural-chat", "starling-lm",
            "phi", "orca-mini"
        ]
    
    async def cleanup(self):
        """Cleanup Ollama backend resources."""
        if self._session:
            await self._session.close()
            self._session = None
        await super().cleanup()
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a single prompt for Ollama."""
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add final prompt for assistant response
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    async def _ensure_model_available(self):
        """Ensure the specified model is available, pull if necessary."""
        try:
            # Check if model exists
            async with self._session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    result = await response.json()
                    models = [model["name"] for model in result.get("models", [])]
                    
                    if self.config.model_name not in models:
                        self.logger.info(f"Model {self.config.model_name} not found, attempting to pull...")
                        await self._pull_model()
                else:
                    # If we can't check models, try to pull anyway
                    await self._pull_model()
                    
        except Exception as e:
            self.logger.warning(f"Could not verify model availability: {e}")
    
    async def _pull_model(self):
        """Pull a model from Ollama registry."""
        try:
            payload = {"name": self.config.model_name}
            
            async with self._session.post(
                f"{self.base_url}/api/pull",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMBackendError(f"Failed to pull model: {error_text}")
                
                # Stream the pull progress
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if chunk.get("status"):
                                self.logger.debug(f"Pull status: {chunk['status']}")
                        except json.JSONDecodeError:
                            continue
                
                self.logger.info(f"Successfully pulled model: {self.config.model_name}")
                
        except Exception as e:
            raise LLMBackendError(f"Failed to pull model {self.config.model_name}: {e}")
