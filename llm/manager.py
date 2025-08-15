"""
LLM Manager for handling multiple backends and routing requests.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from .base import LLMBackend, LLMRequest, LLMResponse, LLMConfig, LLMBackendType, LLMError
from scout_agent.custom_logging import get_logger
from scout_agent.config import get_config


@dataclass
class BackendHealth:
    """Health status of a backend."""
    backend_type: str
    is_healthy: bool
    last_check: float
    error_count: int
    avg_response_time: float


class LLMManager:
    """
    Central manager for LLM backends.
    
    Handles backend registration, health monitoring, load balancing,
    and request routing.
    """
    
    def __init__(self):
        """Initialize the LLM manager."""
        self.logger = get_logger("llm.manager")
        self.backends: Dict[str, LLMBackend] = {}
        self.health_status: Dict[str, BackendHealth] = {}
        self.config = get_config()
        self._default_backend: Optional[str] = None
        self._health_check_interval = 300  # 5 minutes
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def register_backend(self, backend: LLMBackend, is_default: bool = False) -> bool:
        """
        Register a new LLM backend.
        
        Args:
            backend: LLM backend instance
            is_default: Whether this should be the default backend
            
        Returns:
            bool: True if registration successful
        """
        backend_name = backend.config.backend_type.value
        
        try:
            # Initialize the backend
            if not await backend.initialize():
                self.logger.error(f"Failed to initialize backend: {backend_name}")
                return False
            
            # Register the backend
            self.backends[backend_name] = backend
            
            # Initialize health status
            self.health_status[backend_name] = BackendHealth(
                backend_type=backend_name,
                is_healthy=True,
                last_check=asyncio.get_event_loop().time(),
                error_count=0,
                avg_response_time=0.0
            )
            
            # Set as default if requested or if it's the first backend
            if is_default or not self._default_backend:
                self._default_backend = backend_name
            
            self.logger.info(f"Registered LLM backend: {backend_name}")
            
            # Start health monitoring if this is the first backend
            if len(self.backends) == 1:
                await self._start_health_monitoring()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register backend {backend_name}: {e}")
            return False
    
    async def unregister_backend(self, backend_type: Union[str, LLMBackendType]) -> bool:
        """
        Unregister an LLM backend.
        
        Args:
            backend_type: Backend type to unregister
            
        Returns:
            bool: True if unregistration successful
        """
        backend_name = backend_type.value if isinstance(backend_type, LLMBackendType) else backend_type
        
        if backend_name not in self.backends:
            self.logger.warning(f"Backend not registered: {backend_name}")
            return False
        
        try:
            # Cleanup backend
            await self.backends[backend_name].cleanup()
            
            # Remove from registry
            del self.backends[backend_name]
            del self.health_status[backend_name]
            
            # Update default backend if necessary
            if self._default_backend == backend_name:
                self._default_backend = next(iter(self.backends.keys())) if self.backends else None
            
            self.logger.info(f"Unregistered LLM backend: {backend_name}")
            
            # Stop health monitoring if no backends left
            if not self.backends:
                await self._stop_health_monitoring()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister backend {backend_name}: {e}")
            return False
    
    async def generate(self, request: LLMRequest, backend_type: Optional[str] = None) -> LLMResponse:
        """
        Generate a response using the specified or default backend.
        
        Args:
            request: LLM request
            backend_type: Specific backend to use (optional)
            
        Returns:
            LLMResponse: Response from the LLM
        """
        # Determine which backend to use
        target_backend = backend_type or self._default_backend
        
        if not target_backend:
            raise LLMError("No LLM backends available")
        
        if target_backend not in self.backends:
            raise LLMError(f"Backend not available: {target_backend}")
        
        # Check backend health
        if not self.health_status[target_backend].is_healthy:
            # Try to find a healthy fallback
            healthy_backends = [
                name for name, health in self.health_status.items()
                if health.is_healthy and name != target_backend
            ]
            
            if healthy_backends:
                target_backend = healthy_backends[0]
                self.logger.warning(f"Using fallback backend: {target_backend}")
            else:
                self.logger.warning(f"Using unhealthy backend: {target_backend}")
        
        # Execute the request
        backend = self.backends[target_backend]

        # Ensure model compatibility with the selected backend (handles fallback cases)
        try:
            available = set(backend.get_available_models()) if hasattr(backend, "get_available_models") else set()
            if available and request and getattr(request, "extra_params", None):
                override = request.extra_params.get("model_name_override")
                if override and override not in available:
                    request.extra_params["model_name_override"] = backend.config.model_name
                    self.logger.warning(
                        f"[stream] Remapped unsupported model '{override}' for backend '{target_backend}' to default '{backend.config.model_name}'"
                    )
        except Exception as _e:
            self.logger.debug(f"[stream] Model compatibility check skipped: {_e}")

        # Ensure model compatibility with the selected backend (handles fallback cases)
        try:
            available = set(backend.get_available_models()) if hasattr(backend, "get_available_models") else set()
            # Extract current model (override or backend default)
            current_model = None
            if request and getattr(request, "extra_params", None):
                current_model = request.extra_params.get("model_name_override")
            if not current_model:
                current_model = getattr(backend.config, "model_name", None)
            # If override exists and is incompatible, remap to backend default
            if available and request and getattr(request, "extra_params", None):
                override = request.extra_params.get("model_name_override")
                if override and override not in available:
                    # Remap to backend's configured default model
                    request.extra_params["model_name_override"] = backend.config.model_name
                    self.logger.warning(
                        f"Remapped unsupported model '{override}' for backend '{target_backend}' to default '{backend.config.model_name}'"
                    )
        except Exception as _e:
            # Do not fail generation due to compatibility check
            self.logger.debug(f"Model compatibility check skipped: {_e}")
        start_time = asyncio.get_event_loop().time()
        
        try:
            response = await backend.generate(request)
            
            # Update health metrics
            response_time = asyncio.get_event_loop().time() - start_time
            self._update_health_metrics(target_backend, True, response_time)
            
            return response
            
        except Exception as e:
            # Update health metrics
            self._update_health_metrics(target_backend, False, 0.0)
            self.logger.error(f"Error generating response with {target_backend}: {e}")
            raise
    
    async def stream_generate(self, request: LLMRequest, backend_type: Optional[str] = None):
        """
        Generate a streaming response using the specified or default backend.
        
        Args:
            request: LLM request
            backend_type: Specific backend to use (optional)
            
        Yields:
            Streaming response chunks
        """
        target_backend = backend_type or self._default_backend
        
        if not target_backend:
            raise LLMError("No LLM backends available")
        
        if target_backend not in self.backends:
            raise LLMError(f"Backend not available: {target_backend}")
        
        backend = self.backends[target_backend]
        start_time = asyncio.get_event_loop().time()
        
        try:
            async for chunk in backend.stream_generate(request):
                yield chunk
            
            # Update health metrics
            response_time = asyncio.get_event_loop().time() - start_time
            self._update_health_metrics(target_backend, True, response_time)
            
        except Exception as e:
            # Update health metrics
            self._update_health_metrics(target_backend, False, 0.0)
            self.logger.error(f"Error streaming response with {target_backend}: {e}")
            raise
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backend types."""
        return list(self.backends.keys())
    
    def get_healthy_backends(self) -> List[str]:
        """Get list of healthy backend types."""
        return [
            name for name, health in self.health_status.items()
            if health.is_healthy
        ]
    
    def get_backend_health(self, backend_type: str) -> Optional[BackendHealth]:
        """Get health status for a specific backend."""
        return self.health_status.get(backend_type)
    
    def get_all_health_status(self) -> Dict[str, BackendHealth]:
        """Get health status for all backends."""
        return self.health_status.copy()
    
    def set_default_backend(self, backend_type: str) -> bool:
        """
        Set the default backend.
        
        Args:
            backend_type: Backend type to set as default
            
        Returns:
            bool: True if successful
        """
        if backend_type not in self.backends:
            self.logger.error(f"Cannot set default backend: {backend_type} not registered")
            return False
        
        self._default_backend = backend_type
        self.logger.info(f"Set default backend to: {backend_type}")
        return True
    
    def get_default_backend(self) -> Optional[str]:
        """Get the current default backend."""
        return self._default_backend
    
    async def cleanup(self):
        """Cleanup all backends and stop monitoring."""
        await self._stop_health_monitoring()
        
        for backend in self.backends.values():
            try:
                await backend.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up backend: {e}")
        
        self.backends.clear()
        self.health_status.clear()
        self._default_backend = None
    
    def _update_health_metrics(self, backend_type: str, success: bool, response_time: float):
        """Update health metrics for a backend."""
        if backend_type not in self.health_status:
            return
        
        health = self.health_status[backend_type]
        
        if success:
            health.error_count = max(0, health.error_count - 1)
            # Update average response time (simple moving average)
            if health.avg_response_time == 0:
                health.avg_response_time = response_time
            else:
                health.avg_response_time = (health.avg_response_time + response_time) / 2
        else:
            health.error_count += 1
        
        # Update health status based on error count
        health.is_healthy = health.error_count < 5  # Threshold for unhealthy
        health.last_check = asyncio.get_event_loop().time()
    
    async def _start_health_monitoring(self):
        """Start periodic health monitoring."""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_monitor_loop())
            self.logger.info("Started LLM backend health monitoring")
    
    async def _stop_health_monitoring(self):
        """Stop periodic health monitoring."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            self.logger.info("Stopped LLM backend health monitoring")
    
    async def _health_monitor_loop(self):
        """Periodic health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._check_all_backends_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
    
    async def _check_all_backends_health(self):
        """Check health of all registered backends."""
        for backend_name, backend in self.backends.items():
            try:
                is_healthy = await backend.health_check()
                
                if backend_name in self.health_status:
                    health = self.health_status[backend_name]
                    health.is_healthy = is_healthy
                    health.last_check = asyncio.get_event_loop().time()
                    
                    if not is_healthy:
                        health.error_count += 1
                    else:
                        health.error_count = max(0, health.error_count - 1)
                
            except Exception as e:
                self.logger.error(f"Health check failed for {backend_name}: {e}")
                if backend_name in self.health_status:
                    self.health_status[backend_name].is_healthy = False
                    self.health_status[backend_name].error_count += 1


# Global LLM manager instance
_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance."""
    global _manager
    if _manager is None:
        _manager = LLMManager()
    return _manager


async def initialize_llm_backends():
    """Initialize LLM backends based on configuration."""
    from .backends import OllamaBackend, OpenAIBackend, ClaudeBackend, GeminiBackend, DeepSeekBackend
    
    manager = get_llm_manager()
    config = get_config()
    logger = get_logger("llm.init")
    
    # Initialize OpenAI backend if API key is available
    if config.api.openai_api_key:
        try:
            openai_config = LLMConfig(
                backend_type=LLMBackendType.OPENAI,
                model_name="gpt-4",
                api_key=config.api.openai_api_key,
                temperature=0.7,
                max_tokens=4096
            )
            openai_backend = OpenAIBackend(openai_config)
            await manager.register_backend(openai_backend, is_default=True)
            logger.info("Initialized OpenAI backend")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI backend: {e}")
    
    # Initialize Claude backend if API key is available
    if config.api.anthropic_api_key:
        try:
            claude_config = LLMConfig(
                backend_type=LLMBackendType.CLAUDE,
                model_name="claude-3-5-sonnet-20241022",
                api_key=config.api.anthropic_api_key,
                temperature=0.7,
                max_tokens=4096
            )
            claude_backend = ClaudeBackend(claude_config)
            await manager.register_backend(claude_backend)
            logger.info("Initialized Claude backend")
        except Exception as e:
            logger.error(f"Failed to initialize Claude backend: {e}")
    
    # Initialize Gemini backend if API key is available
    if config.api.gemini_api_key:
        try:
            gemini_config = LLMConfig(
                backend_type=LLMBackendType.GEMINI,
                model_name="gemini-2.5-flash",
                api_key=config.api.gemini_api_key,
                temperature=0.7,
                max_tokens=4096
            )
            gemini_backend = GeminiBackend(gemini_config)
            await manager.register_backend(gemini_backend)
            logger.info("Initialized Gemini backend")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini backend: {e}")
    
    # Initialize DeepSeek backend if API key is available
    if config.api.deepseek_api_key:
        try:
            deepseek_config = LLMConfig(
                backend_type=LLMBackendType.DEEPSEEK,
                model_name="deepseek-chat",
                api_key=config.api.deepseek_api_key,
                temperature=0.7,
                max_tokens=4096
            )
            deepseek_backend = DeepSeekBackend(deepseek_config)
            await manager.register_backend(deepseek_backend)
            logger.info("Initialized DeepSeek backend")
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek backend: {e}")
    
    # Initialize Ollama backend (local, no API key needed)
    try:
        ollama_config = LLMConfig(
            backend_type=LLMBackendType.OLLAMA,
            model_name="phi4-mini:latest",
            base_url="http://localhost:11434",
            temperature=0.7,
            max_tokens=4096
        )
        ollama_backend = OllamaBackend(ollama_config)
        # Only register if Ollama is actually running
        if await ollama_backend.health_check():
            await manager.register_backend(ollama_backend)
            logger.info("Initialized Ollama backend")
        else:
            logger.info("Ollama not available, skipping registration")
    except Exception as e:
        logger.error(f"Failed to initialize Ollama backend: {e}")
    
    available_backends = manager.get_available_backends()

    # Respect configured default backend if specified and available
    try:
        routing = getattr(config, 'llm_routing', None)
        if routing and routing.default_backend and routing.default_backend in available_backends:
            manager.set_default_backend(routing.default_backend)
            logger.info(f"Configured default backend set to: {routing.default_backend}")
    except Exception as e:
        logger.warning(f"Could not set configured default backend: {e}")
    if available_backends:
        logger.info(f"LLM initialization complete. Available backends: {available_backends}")
        logger.info(f"Default backend: {manager.get_default_backend()}")
    else:
        logger.warning("No LLM backends were successfully initialized")
