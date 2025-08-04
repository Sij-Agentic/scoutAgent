"""
Gemini (Google) backend implementation.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

from ..base import LLMBackend, LLMRequest, LLMResponse, LLMConfig, LLMBackendError, LLMTimeoutError, LLMRateLimitError


class GeminiBackend(LLMBackend):
    """Gemini (Google) backend for Gemini models."""
    
    def __init__(self, config: LLMConfig):
        """Initialize Gemini backend."""
        super().__init__(config)
        
        if not GEMINI_AVAILABLE:
            raise LLMBackendError("Google GenerativeAI library not available. Install with: pip install google-generativeai")
        
        if not config.api_key:
            raise LLMBackendError("Gemini API key is required")
        
        self._model = None
    
    async def initialize(self) -> bool:
        """Initialize the Gemini backend."""
        try:
            # Configure the API key
            genai.configure(api_key=self.config.api_key)
            
            # Initialize the model
            generation_config = genai.types.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )
            
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            self._model = genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Test the connection
            await self._test_connection()
            
            self._initialized = True
            self.logger.info(f"Gemini backend initialized with model: {self.config.model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini backend: {e}")
            self._model = None
            return False
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using Gemini."""
        if not self._initialized or not self._model:
            raise LLMBackendError("Gemini backend not initialized")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Prepare the request
            messages = self._prepare_messages(request)
            config = self._get_effective_config(request)
            
            # Convert messages to Gemini format
            prompt = self._messages_to_prompt(messages)
            
            # Update generation config if needed
            if config["temperature"] != self.config.temperature or config["max_tokens"] != self.config.max_tokens:
                generation_config = genai.types.GenerationConfig(
                    temperature=config["temperature"],
                    max_output_tokens=config["max_tokens"],
                )
                
                # Create a new model instance with updated config
                model = genai.GenerativeModel(
                    model_name=self.config.model_name,
                    generation_config=generation_config,
                    safety_settings=self._model._safety_settings
                )
            else:
                model = self._model
            
            # Make the request with retry logic
            response = await self._retry_with_backoff(
                self._generate_async,
                model,
                prompt
            )
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            # Extract content
            content = ""
            if response.text:
                content = response.text
            
            # Extract usage information
            usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                }
            
            return self._create_response(
                content=content,
                model=self.config.model_name,
                usage=usage,
                metadata={
                    "finish_reason": getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None,
                    "safety_ratings": [
                        {
                            "category": rating.category.name,
                            "probability": rating.probability.name
                        }
                        for rating in getattr(response.candidates[0], 'safety_ratings', [])
                    ] if response.candidates else []
                },
                response_time=response_time,
                raw_response=response
            )
            
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                raise LLMRateLimitError(f"Gemini rate limit exceeded: {e}")
            elif "timeout" in str(e).lower():
                raise LLMTimeoutError(f"Gemini request timed out: {e}")
            else:
                self.logger.error(f"Error generating response: {e}")
                raise LLMBackendError(f"Gemini generation failed: {e}")
    
    async def stream_generate(self, request: LLMRequest):
        """Generate a streaming response using Gemini."""
        if not self._initialized or not self._model:
            raise LLMBackendError("Gemini backend not initialized")
        
        try:
            # Prepare the request
            messages = self._prepare_messages(request)
            config = self._get_effective_config(request)
            
            # Convert messages to Gemini format
            prompt = self._messages_to_prompt(messages)
            
            # Update generation config if needed
            if config["temperature"] != self.config.temperature or config["max_tokens"] != self.config.max_tokens:
                generation_config = genai.types.GenerationConfig(
                    temperature=config["temperature"],
                    max_output_tokens=config["max_tokens"],
                )
                
                # Create a new model instance with updated config
                model = genai.GenerativeModel(
                    model_name=self.config.model_name,
                    generation_config=generation_config,
                    safety_settings=self._model._safety_settings
                )
            else:
                model = self._model
            
            # Make the streaming request
            response = await self._stream_generate_async(model, prompt)
            
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                raise LLMRateLimitError(f"Gemini rate limit exceeded: {e}")
            elif "timeout" in str(e).lower():
                raise LLMTimeoutError(f"Gemini streaming request timed out: {e}")
            else:
                self.logger.error(f"Error streaming response: {e}")
                raise LLMBackendError(f"Gemini streaming failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if Gemini API is healthy and available."""
        try:
            if not self._model:
                return False
            
            # Make a minimal request to test the API
            response = await self._generate_async(self._model, "test")
            return response is not None
            
        except Exception as e:
            self.logger.debug(f"Gemini health check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available Gemini models."""
        return [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ]
    
    async def cleanup(self):
        """Cleanup Gemini backend resources."""
        self._model = None
        await super().cleanup()
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a single prompt for Gemini."""
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
        
        return "\n\n".join(prompt_parts)
    
    async def _generate_async(self, model, prompt):
        """Async wrapper for Gemini generate_content."""
        # Run the synchronous method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, model.generate_content, prompt)
    
    async def _stream_generate_async(self, model, prompt):
        """Async wrapper for Gemini stream generation."""
        # Run the synchronous method in a thread pool
        loop = asyncio.get_event_loop()
        
        def _stream_sync():
            return model.generate_content(prompt, stream=True)
        
        stream = await loop.run_in_executor(None, _stream_sync)
        
        # Convert synchronous iterator to async iterator
        for chunk in stream:
            yield chunk
    
    async def _test_connection(self):
        """Test the Gemini connection."""
        try:
            # Make a minimal request to verify the API key and connection
            response = await self._generate_async(self._model, "Hello")
            
            if not response or not response.text:
                raise LLMBackendError("Invalid response from Gemini API")
                
        except Exception as e:
            if "api_key" in str(e).lower() or "authentication" in str(e).lower():
                raise LLMBackendError("Invalid Gemini API key")
            elif "not found" in str(e).lower():
                raise LLMBackendError(f"Model {self.config.model_name} not found")
            else:
                raise LLMBackendError(f"Gemini connection test failed: {e}")
