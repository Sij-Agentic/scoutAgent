"""
Test suite for Multi-Backend LLM Abstraction Layer.

This test suite validates:
1. LLM backend initialization and configuration
2. Basic generation and streaming capabilities
3. Error handling and fallbacks
4. Health monitoring
5. Agent integration
6. Backend switching
"""

import asyncio
import pytest
import json
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from config import init_config, get_config
from llm.base import LLMConfig, LLMBackendType, LLMRequest, LLMResponse
from llm.manager import LLMManager, get_llm_manager, initialize_llm_backends
from llm.backends import OllamaBackend, OpenAIBackend, ClaudeBackend, GeminiBackend
from llm.utils import LLMAgentMixin, AgentPrompt, quick_llm_generate, llm_summarize
from agents.gap_finder_enhanced import EnhancedGapFinderAgent
from agents.gap_finder import GapFinderInput
from agents.base import AgentInput
from custom_logging import get_logger

logger = get_logger("llm_test")


class TestLLMBase:
    """Test base LLM functionality."""
    
    def test_llm_config_creation(self):
        """Test LLM configuration creation."""
        config = LLMConfig(
            backend_type=LLMBackendType.OPENAI,
            model_name="gpt-4",
            api_key="test-key",
            temperature=0.7,
            max_tokens=1000
        )
        
        assert config.backend_type == LLMBackendType.OPENAI
        assert config.model_name == "gpt-4"
        assert config.api_key == "test-key"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
    
    def test_llm_request_creation(self):
        """Test LLM request creation."""
        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.5,
            max_tokens=100
        )
        
        assert len(request.messages) == 1
        assert request.messages[0]["role"] == "user"
        assert request.temperature == 0.5
        assert request.max_tokens == 100
    
    def test_llm_response_creation(self):
        """Test LLM response creation."""
        response = LLMResponse(
            content="test response",
            model="gpt-4",
            backend_type="openai",
            usage={"total_tokens": 50},
            metadata={"test": True},
            success=True,
            response_time=1.5
        )
        
        assert response.content == "test response"
        assert response.model == "gpt-4"
        assert response.backend_type == "openai"
        assert response.success is True
        assert response.response_time == 1.5


class TestLLMManager:
    """Test LLM manager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh LLM manager for testing."""
        return LLMManager()
    
    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend for testing."""
        backend = Mock()
        backend.config = LLMConfig(
            backend_type=LLMBackendType.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        )
        backend.initialize = Mock(return_value=True)
        backend.generate = Mock()
        backend.health_check = Mock(return_value=True)
        backend.cleanup = Mock()
        return backend
    
    async def test_backend_registration(self, manager, mock_backend):
        """Test backend registration."""
        success = await manager.register_backend(mock_backend, is_default=True)
        
        assert success is True
        assert "openai" in manager.get_available_backends()
        assert manager.get_default_backend() == "openai"
        mock_backend.initialize.assert_called_once()
    
    async def test_backend_unregistration(self, manager, mock_backend):
        """Test backend unregistration."""
        await manager.register_backend(mock_backend)
        success = await manager.unregister_backend("openai")
        
        assert success is True
        assert "openai" not in manager.get_available_backends()
        mock_backend.cleanup.assert_called_once()
    
    async def test_generate_with_backend(self, manager, mock_backend):
        """Test generation with specific backend."""
        # Setup mock response
        mock_response = LLMResponse(
            content="test response",
            model="gpt-4",
            backend_type="openai",
            usage={},
            metadata={},
            success=True
        )
        mock_backend.generate.return_value = mock_response
        
        await manager.register_backend(mock_backend)
        
        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}]
        )
        
        response = await manager.generate(request, backend_type="openai")
        
        assert response.content == "test response"
        assert response.success is True
        mock_backend.generate.assert_called_once()


class TestLLMBackends:
    """Test individual LLM backends."""
    
    def test_ollama_backend_creation(self):
        """Test Ollama backend creation."""
        config = LLMConfig(
            backend_type=LLMBackendType.OLLAMA,
            model_name="llama2",
            base_url="http://localhost:11434"
        )
        
        backend = OllamaBackend(config)
        assert backend.config.backend_type == LLMBackendType.OLLAMA
        assert backend.base_url == "http://localhost:11434"
    
    def test_openai_backend_creation(self):
        """Test OpenAI backend creation."""
        config = LLMConfig(
            backend_type=LLMBackendType.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        )
        
        try:
            backend = OpenAIBackend(config)
            assert backend.config.api_key == "test-key"
        except Exception as e:
            # Expected if OpenAI library not installed
            assert "not available" in str(e)
    
    def test_claude_backend_creation(self):
        """Test Claude backend creation."""
        config = LLMConfig(
            backend_type=LLMBackendType.CLAUDE,
            model_name="claude-3-sonnet-20240229",
            api_key="test-key"
        )
        
        try:
            backend = ClaudeBackend(config)
            assert backend.config.api_key == "test-key"
        except Exception as e:
            # Expected if Anthropic library not installed
            assert "not available" in str(e)
    
    def test_gemini_backend_creation(self):
        """Test Gemini backend creation."""
        config = LLMConfig(
            backend_type=LLMBackendType.GEMINI,
            model_name="gemini-pro",
            api_key="test-key"
        )
        
        try:
            backend = GeminiBackend(config)
            assert backend.config.api_key == "test-key"
        except Exception as e:
            # Expected if Google GenerativeAI library not installed
            assert "not available" in str(e)


class TestLLMUtils:
    """Test LLM utility functions and classes."""
    
    def test_agent_prompt_creation(self):
        """Test AgentPrompt creation and message conversion."""
        prompt = AgentPrompt(
            system_prompt="You are a helpful assistant",
            user_prompt="What is AI?",
            examples=[
                {"user": "What is ML?", "assistant": "Machine Learning is..."}
            ]
        )
        
        messages = prompt.to_messages()
        
        assert len(messages) == 4  # system + example user + example assistant + main user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "What is AI?"
    
    def test_llm_agent_mixin(self):
        """Test LLM agent mixin functionality."""
        class TestAgent(LLMAgentMixin):
            def __init__(self):
                self.name = "test_agent"
                super().__init__()
        
        agent = TestAgent()
        assert hasattr(agent, 'llm_logger')
        assert hasattr(agent, 'llm_generate')
        assert hasattr(agent, 'llm_stream_generate')
        assert hasattr(agent, 'get_llm_status')


class TestAgentIntegration:
    """Test LLM integration with agents."""
    
    @pytest.fixture
    def sample_gap_finder_input(self):
        """Create sample input for gap finder agent."""
        return GapFinderInput(
            validated_pain_points=[
                {
                    "description": "Difficulty tracking inventory manually",
                    "severity": "high",
                    "frequency": "daily"
                },
                {
                    "description": "Complex expense reporting process",
                    "severity": "medium",
                    "frequency": "weekly"
                }
            ],
            market_context="Small business software market",
            analysis_scope="focused"
        )
    
    def test_enhanced_gap_finder_creation(self):
        """Test enhanced gap finder agent creation."""
        agent = EnhancedGapFinderAgent()
        
        assert agent.name == "enhanced_gap_finder"
        assert hasattr(agent, 'llm_generate')
        assert hasattr(agent, 'plan')
        assert hasattr(agent, 'think')
        assert hasattr(agent, 'act')
    
    async def test_enhanced_gap_finder_plan(self, sample_gap_finder_input):
        """Test enhanced gap finder planning phase."""
        agent = EnhancedGapFinderAgent()
        agent_input = AgentInput(
            data=sample_gap_finder_input,
            metadata={"test": True}
        )
        
        plan = await agent.plan(agent_input)
        
        assert "phases" in plan
        assert "llm_pain_point_analysis" in plan["phases"]
        assert plan["use_llm"] is True
        assert plan["pain_point_count"] == 2


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    async def test_manager_with_no_backends(self):
        """Test manager behavior with no registered backends."""
        manager = LLMManager()
        
        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}]
        )
        
        with pytest.raises(Exception) as exc_info:
            await manager.generate(request)
        
        assert "No LLM backends available" in str(exc_info.value)
    
    async def test_invalid_backend_request(self):
        """Test request with invalid backend."""
        manager = LLMManager()
        
        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}]
        )
        
        with pytest.raises(Exception) as exc_info:
            await manager.generate(request, backend_type="invalid_backend")
        
        assert "Backend not available" in str(exc_info.value)
    
    def test_invalid_prompt_format(self):
        """Test invalid prompt format handling."""
        class TestAgent(LLMAgentMixin):
            def __init__(self):
                self.name = "test"
                super().__init__()
        
        agent = TestAgent()
        
        # This should raise an error for invalid prompt type
        with pytest.raises(ValueError):
            asyncio.run(agent.llm_generate(123))  # Invalid prompt type


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_simulation(self):
        """Test a complete workflow simulation."""
        # Initialize config
        init_config()
        
        # This test would require actual API keys to run fully
        # For now, we'll test the structure and mock responses
        
        manager = LLMManager()
        
        # Mock a backend
        mock_backend = Mock()
        mock_backend.config = LLMConfig(
            backend_type=LLMBackendType.OPENAI,
            model_name="gpt-4",
            api_key="test"
        )
        mock_backend.initialize = Mock(return_value=True)
        mock_backend.health_check = Mock(return_value=True)
        mock_backend.cleanup = Mock()
        
        # Mock successful generation
        mock_response = LLMResponse(
            content='{"pain_point_clusters": [{"theme": "inventory", "pain_points": ["manual tracking"]}]}',
            model="gpt-4",
            backend_type="openai",
            usage={"total_tokens": 100},
            metadata={},
            success=True,
            response_time=1.0
        )
        mock_backend.generate = Mock(return_value=mock_response)
        
        # Register mock backend
        await manager.register_backend(mock_backend)
        
        # Test that the workflow structure is correct
        assert manager.get_default_backend() == "openai"
        assert "openai" in manager.get_available_backends()
        
        # Cleanup
        await manager.cleanup()


def run_basic_tests():
    """Run basic tests that don't require async."""
    print("Running basic LLM tests...")
    
    # Test configuration
    test_base = TestLLMBase()
    test_base.test_llm_config_creation()
    test_base.test_llm_request_creation()
    test_base.test_llm_response_creation()
    print("✓ Base LLM tests passed")
    
    # Test backends
    test_backends = TestLLMBackends()
    test_backends.test_ollama_backend_creation()
    test_backends.test_openai_backend_creation()
    test_backends.test_claude_backend_creation()
    test_backends.test_gemini_backend_creation()
    print("✓ Backend creation tests passed")
    
    # Test utilities
    test_utils = TestLLMUtils()
    test_utils.test_agent_prompt_creation()
    test_utils.test_llm_agent_mixin()
    print("✓ Utility tests passed")
    
    # Test agent integration
    test_integration = TestAgentIntegration()
    test_integration.test_enhanced_gap_finder_creation()
    print("✓ Agent integration tests passed")
    
    print("All basic tests completed successfully!")


async def run_async_tests():
    """Run async tests."""
    print("Running async LLM tests...")
    
    # Test manager functionality
    manager = LLMManager()
    mock_backend = Mock()
    mock_backend.config = LLMConfig(
        backend_type=LLMBackendType.OPENAI,
        model_name="gpt-4",
        api_key="test"
    )
    mock_backend.initialize = Mock(return_value=True)
    mock_backend.health_check = Mock(return_value=True)
    mock_backend.cleanup = Mock()
    
    # Test registration
    success = await manager.register_backend(mock_backend)
    assert success is True
    print("✓ Backend registration test passed")
    
    # Test unregistration
    success = await manager.unregister_backend("openai")
    assert success is True
    print("✓ Backend unregistration test passed")
    
    print("All async tests completed successfully!")


if __name__ == "__main__":
    """Run the test suite."""
    print("Multi-Backend LLM Abstraction Layer Test Suite")
    print("=" * 50)
    
    try:
        # Run basic tests
        run_basic_tests()
        
        # Run async tests
        asyncio.run(run_async_tests())
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("\nTo run with pytest:")
        print("pip install pytest pytest-asyncio")
        print("pytest test_llm_integration.py -v")
        
    except Exception as e:
        print(f"Test failed: {e}")
        logger.error(f"Test execution error: {e}")
        raise
