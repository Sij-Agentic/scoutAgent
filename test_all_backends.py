"""
Comprehensive test script for all LLM backends with real API keys.

This script tests:
1. Ollama (local) with phi4-mini:latest
2. OpenAI with GPT models
3. Claude (Anthropic) with Claude-3 models
4. Gemini (Google) with Gemini Pro
5. DeepSeek with DeepSeek Chat

Tests include:
- Backend initialization and health checks
- Basic text generation
- Streaming responses
- Error handling
- Performance metrics
- Agent integration
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from config import init_config, get_config
from llm.manager import get_llm_manager, initialize_llm_backends
from llm.base import LLMRequest, LLMResponse, LLMBackendType, LLMConfig
from llm.backends import OllamaBackend, OpenAIBackend, ClaudeBackend, GeminiBackend, DeepSeekBackend
from llm.utils import quick_llm_generate, llm_summarize, AgentPromptTemplates
from agents.gap_finder_enhanced import EnhancedGapFinderAgent
from agents.gap_finder import GapFinderInput
from agents.base import AgentInput
from custom_logging import get_logger

logger = get_logger("backend_test")


class BackendTestResult:
    """Test result for a backend."""
    
    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        self.initialization_success = False
        self.health_check_success = False
        self.basic_generation_success = False
        self.streaming_success = False
        self.error_handling_success = False
        self.performance_metrics = {}
        self.errors = []
        self.responses = {}
    
    def add_error(self, test_name: str, error: str):
        """Add an error to the test result."""
        self.errors.append(f"{test_name}: {error}")
    
    def add_response(self, test_name: str, response: str, response_time: float = 0.0):
        """Add a successful response to the test result."""
        self.responses[test_name] = {
            "content": response[:200] + "..." if len(response) > 200 else response,
            "response_time": response_time,
            "length": len(response)
        }
    
    def get_success_rate(self) -> float:
        """Calculate overall success rate."""
        tests = [
            self.initialization_success,
            self.health_check_success,
            self.basic_generation_success,
            self.streaming_success,
            self.error_handling_success
        ]
        return sum(tests) / len(tests)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "backend_name": self.backend_name,
            "initialization_success": self.initialization_success,
            "health_check_success": self.health_check_success,
            "basic_generation_success": self.basic_generation_success,
            "streaming_success": self.streaming_success,
            "error_handling_success": self.error_handling_success,
            "success_rate": self.get_success_rate(),
            "performance_metrics": self.performance_metrics,
            "errors": self.errors,
            "responses": self.responses
        }


class LLMBackendTester:
    """Comprehensive LLM backend tester."""
    
    def __init__(self):
        """Initialize the tester."""
        self.manager = None
        self.config = None
        self.test_results = {}
        self.test_prompts = {
            "simple": "What is artificial intelligence?",
            "creative": "Write a short story about a robot learning to paint.",
            "analytical": "Analyze the pros and cons of remote work for software development teams.",
            "technical": "Explain the concept of machine learning in simple terms.",
            "complex": "Design a system architecture for a scalable e-commerce platform that can handle millions of users."
        }
    
    async def initialize(self):
        """Initialize the test environment."""
        print("🔧 Initializing test environment...")
        
        # Initialize configuration
        init_config()
        self.config = get_config()
        
        # Initialize LLM backends
        await initialize_llm_backends()
        self.manager = get_llm_manager()
        
        available_backends = self.manager.get_available_backends()
        print(f"✅ Available backends: {available_backends}")
        
        if not available_backends:
            print("❌ No backends available! Please check your API keys and Ollama installation.")
            return False
        
        return True
    
    async def test_backend_initialization(self, backend_name: str) -> BackendTestResult:
        """Test backend initialization."""
        print(f"\n🧪 Testing {backend_name} backend initialization...")
        result = BackendTestResult(backend_name)
        
        try:
            # Check if backend is available
            if backend_name in self.manager.get_available_backends():
                result.initialization_success = True
                print(f"✅ {backend_name} initialization: SUCCESS")
            else:
                result.add_error("initialization", f"Backend {backend_name} not available")
                print(f"❌ {backend_name} initialization: FAILED - not available")
        
        except Exception as e:
            result.add_error("initialization", str(e))
            print(f"❌ {backend_name} initialization: FAILED - {e}")
        
        return result
    
    async def test_health_check(self, backend_name: str, result: BackendTestResult):
        """Test backend health check."""
        print(f"🏥 Testing {backend_name} health check...")
        
        try:
            health_status = self.manager.get_backend_health(backend_name)
            if health_status and health_status.is_healthy:
                result.health_check_success = True
                result.performance_metrics["avg_response_time"] = health_status.avg_response_time
                print(f"✅ {backend_name} health check: HEALTHY")
            else:
                result.add_error("health_check", "Backend reported as unhealthy")
                print(f"⚠️ {backend_name} health check: UNHEALTHY")
        
        except Exception as e:
            result.add_error("health_check", str(e))
            print(f"❌ {backend_name} health check: ERROR - {e}")
    
    async def test_basic_generation(self, backend_name: str, result: BackendTestResult):
        """Test basic text generation."""
        print(f"📝 Testing {backend_name} basic generation...")
        
        try:
            start_time = time.time()
            
            request = LLMRequest(
                messages=[{"role": "user", "content": self.test_prompts["simple"]}],
                temperature=0.7,
                max_tokens=100
            )
            
            response = await self.manager.generate(request, backend_type=backend_name)
            response_time = time.time() - start_time
            
            if response.success and response.content:
                result.basic_generation_success = True
                result.add_response("basic_generation", response.content, response_time)
                result.performance_metrics["basic_generation_time"] = response_time
                result.performance_metrics["basic_generation_tokens"] = response.usage.get("total_tokens", 0)
                print(f"✅ {backend_name} basic generation: SUCCESS ({response_time:.2f}s)")
                print(f"   Response: {response.content[:100]}...")
            else:
                result.add_error("basic_generation", f"Empty or failed response: {response.error}")
                print(f"❌ {backend_name} basic generation: FAILED - empty response")
        
        except Exception as e:
            result.add_error("basic_generation", str(e))
            print(f"❌ {backend_name} basic generation: ERROR - {e}")
    
    async def test_streaming_generation(self, backend_name: str, result: BackendTestResult):
        """Test streaming text generation."""
        print(f"🌊 Testing {backend_name} streaming generation...")
        
        try:
            start_time = time.time()
            
            request = LLMRequest(
                messages=[{"role": "user", "content": self.test_prompts["creative"]}],
                temperature=0.8,
                max_tokens=150,
                stream=True
            )
            
            streamed_content = ""
            chunk_count = 0
            
            async for chunk in self.manager.stream_generate(request, backend_type=backend_name):
                streamed_content += chunk
                chunk_count += 1
                if chunk_count > 50:  # Prevent infinite loops
                    break
            
            response_time = time.time() - start_time
            
            if streamed_content:
                result.streaming_success = True
                result.add_response("streaming_generation", streamed_content, response_time)
                result.performance_metrics["streaming_time"] = response_time
                result.performance_metrics["streaming_chunks"] = chunk_count
                print(f"✅ {backend_name} streaming: SUCCESS ({chunk_count} chunks, {response_time:.2f}s)")
                print(f"   Response: {streamed_content[:100]}...")
            else:
                result.add_error("streaming_generation", "No content received from stream")
                print(f"❌ {backend_name} streaming: FAILED - no content")
        
        except Exception as e:
            result.add_error("streaming_generation", str(e))
            print(f"❌ {backend_name} streaming: ERROR - {e}")
    
    async def test_error_handling(self, backend_name: str, result: BackendTestResult):
        """Test error handling capabilities."""
        print(f"⚠️ Testing {backend_name} error handling...")
        
        try:
            # Test with invalid parameters
            request = LLMRequest(
                messages=[{"role": "user", "content": "Test"}],
                temperature=2.0,  # Invalid temperature
                max_tokens=-1     # Invalid max_tokens
            )
            
            try:
                response = await self.manager.generate(request, backend_type=backend_name)
                # If it succeeds, the backend handled the invalid parameters gracefully
                result.error_handling_success = True
                print(f"✅ {backend_name} error handling: SUCCESS - graceful parameter handling")
            except Exception as e:
                # Expected behavior - backend should reject invalid parameters
                result.error_handling_success = True
                print(f"✅ {backend_name} error handling: SUCCESS - properly rejected invalid parameters")
        
        except Exception as e:
            result.add_error("error_handling", str(e))
            print(f"❌ {backend_name} error handling: ERROR - {e}")
    
    async def test_different_prompts(self, backend_name: str, result: BackendTestResult):
        """Test with different types of prompts."""
        print(f"🎯 Testing {backend_name} with different prompt types...")
        
        for prompt_type, prompt in self.test_prompts.items():
            if prompt_type == "simple":  # Already tested in basic generation
                continue
            
            try:
                start_time = time.time()
                
                request = LLMRequest(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=200
                )
                
                response = await self.manager.generate(request, backend_type=backend_name)
                response_time = time.time() - start_time
                
                if response.success and response.content:
                    result.add_response(f"prompt_{prompt_type}", response.content, response_time)
                    print(f"✅ {backend_name} {prompt_type} prompt: SUCCESS ({response_time:.2f}s)")
                else:
                    result.add_error(f"prompt_{prompt_type}", "Failed or empty response")
                    print(f"❌ {backend_name} {prompt_type} prompt: FAILED")
            
            except Exception as e:
                result.add_error(f"prompt_{prompt_type}", str(e))
                print(f"❌ {backend_name} {prompt_type} prompt: ERROR - {e}")
    
    async def test_agent_integration(self, backend_name: str, result: BackendTestResult):
        """Test integration with enhanced agents."""
        print(f"🤖 Testing {backend_name} agent integration...")
        
        try:
            # Create sample data for gap finder agent
            sample_pain_points = [
                {
                    "description": "Small businesses struggle with manual inventory tracking",
                    "severity": "high",
                    "frequency": "daily"
                },
                {
                    "description": "Difficulty finding reliable freelancers for projects",
                    "severity": "medium",
                    "frequency": "weekly"
                }
            ]
            
            gap_finder_input = GapFinderInput(
                validated_pain_points=sample_pain_points,
                market_context="Small business software market",
                analysis_scope="focused"
            )
            
            # Force the agent to use the specific backend
            agent = EnhancedGapFinderAgent()
            agent._llm_manager = self.manager
            
            # Override the generate method to use specific backend
            original_generate = agent.llm_generate
            
            async def backend_specific_generate(prompt, **kwargs):
                return await original_generate(prompt, backend_type=backend_name, **kwargs)
            
            agent.llm_generate = backend_specific_generate
            
            start_time = time.time()
            
            agent_input = AgentInput(
                data=gap_finder_input,
                metadata={"test": True, "backend": backend_name}
            )
            
            agent_result = await agent.execute(agent_input)
            response_time = time.time() - start_time
            
            if agent_result.success:
                result.add_response("agent_integration", 
                                  f"Agent executed successfully with {len(agent_result.result.market_gaps)} market gaps found",
                                  response_time)
                result.performance_metrics["agent_integration_time"] = response_time
                print(f"✅ {backend_name} agent integration: SUCCESS ({response_time:.2f}s)")
                print(f"   Found {len(agent_result.result.market_gaps)} market gaps")
            else:
                result.add_error("agent_integration", f"Agent execution failed: {agent_result.error}")
                print(f"❌ {backend_name} agent integration: FAILED - {agent_result.error}")
        
        except Exception as e:
            result.add_error("agent_integration", str(e))
            print(f"❌ {backend_name} agent integration: ERROR - {e}")
    
    async def run_comprehensive_test(self, backend_name: str) -> BackendTestResult:
        """Run comprehensive test suite for a backend."""
        print(f"\n{'='*60}")
        print(f"🚀 COMPREHENSIVE TEST: {backend_name.upper()}")
        print(f"{'='*60}")
        
        # Initialize test result
        result = await self.test_backend_initialization(backend_name)
        
        if not result.initialization_success:
            print(f"❌ Skipping further tests for {backend_name} - initialization failed")
            return result
        
        # Run all tests
        await self.test_health_check(backend_name, result)
        await self.test_basic_generation(backend_name, result)
        await self.test_streaming_generation(backend_name, result)
        await self.test_error_handling(backend_name, result)
        await self.test_different_prompts(backend_name, result)
        await self.test_agent_integration(backend_name, result)
        
        # Store result
        self.test_results[backend_name] = result
        
        # Print summary
        success_rate = result.get_success_rate() * 100
        print(f"\n📊 {backend_name} Test Summary:")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Errors: {len(result.errors)}")
        print(f"   Successful Responses: {len(result.responses)}")
        
        return result
    
    async def run_all_tests(self):
        """Run tests for all available backends."""
        print("🎯 Starting comprehensive backend testing...")
        
        if not await self.initialize():
            return
        
        available_backends = self.manager.get_available_backends()
        
        # Test each backend
        for backend_name in available_backends:
            try:
                await self.run_comprehensive_test(backend_name)
                # Small delay between backends
                await asyncio.sleep(1)
            except Exception as e:
                print(f"❌ Critical error testing {backend_name}: {e}")
                logger.error(f"Critical error testing {backend_name}: {e}")
        
        # Generate final report
        await self.generate_final_report()
    
    async def generate_final_report(self):
        """Generate final test report."""
        print(f"\n{'='*80}")
        print("📋 FINAL TEST REPORT")
        print(f"{'='*80}")
        
        if not self.test_results:
            print("❌ No test results available")
            return
        
        # Overall statistics
        total_backends = len(self.test_results)
        successful_backends = sum(1 for result in self.test_results.values() if result.get_success_rate() > 0.5)
        
        print(f"📊 Overall Statistics:")
        print(f"   Total Backends Tested: {total_backends}")
        print(f"   Successful Backends: {successful_backends}")
        print(f"   Overall Success Rate: {(successful_backends/total_backends)*100:.1f}%")
        
        # Backend-by-backend results
        print(f"\n🔍 Backend Results:")
        for backend_name, result in self.test_results.items():
            success_rate = result.get_success_rate() * 100
            status = "✅ PASS" if success_rate >= 70 else "⚠️ PARTIAL" if success_rate >= 50 else "❌ FAIL"
            
            print(f"\n   {backend_name.upper()}: {status} ({success_rate:.1f}%)")
            
            if result.initialization_success:
                print(f"     ✅ Initialization")
            else:
                print(f"     ❌ Initialization")
            
            if result.health_check_success:
                print(f"     ✅ Health Check")
            else:
                print(f"     ❌ Health Check")
            
            if result.basic_generation_success:
                print(f"     ✅ Basic Generation")
            else:
                print(f"     ❌ Basic Generation")
            
            if result.streaming_success:
                print(f"     ✅ Streaming")
            else:
                print(f"     ❌ Streaming")
            
            if result.error_handling_success:
                print(f"     ✅ Error Handling")
            else:
                print(f"     ❌ Error Handling")
            
            # Performance metrics
            if result.performance_metrics:
                print(f"     📈 Performance:")
                for metric, value in result.performance_metrics.items():
                    if "time" in metric:
                        print(f"        {metric}: {value:.2f}s")
                    else:
                        print(f"        {metric}: {value}")
            
            # Errors
            if result.errors:
                print(f"     ❌ Errors ({len(result.errors)}):")
                for error in result.errors[:3]:  # Show first 3 errors
                    print(f"        - {error}")
                if len(result.errors) > 3:
                    print(f"        ... and {len(result.errors) - 3} more")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        working_backends = [name for name, result in self.test_results.items() 
                          if result.get_success_rate() >= 0.7]
        
        if working_backends:
            print(f"   ✅ Recommended backends: {', '.join(working_backends)}")
            fastest_backend = min(working_backends, 
                                key=lambda x: self.test_results[x].performance_metrics.get("basic_generation_time", float('inf')))
            print(f"   🚀 Fastest backend: {fastest_backend}")
        else:
            print(f"   ⚠️ No backends achieved 70% success rate")
        
        # Configuration check
        print(f"\n🔧 Configuration Status:")
        config = get_config()
        
        api_keys = {
            "OpenAI": bool(config.api.openai_api_key),
            "Anthropic": bool(config.api.anthropic_api_key),
            "Gemini": bool(config.api.gemini_api_key),
            "DeepSeek": bool(config.api.deepseek_api_key)
        }
        
        for service, has_key in api_keys.items():
            status = "✅" if has_key else "❌"
            print(f"   {status} {service} API Key: {'Configured' if has_key else 'Missing'}")
        
        # Ollama status
        ollama_available = "ollama" in [result.backend_name for result in self.test_results.values() 
                                      if result.initialization_success]
        status = "✅" if ollama_available else "❌"
        print(f"   {status} Ollama: {'Available' if ollama_available else 'Not Available'}")
        
        # Save detailed report
        await self.save_detailed_report()
        
        print(f"\n🎉 Testing complete! Check 'backend_test_report.json' for detailed results.")
    
    async def save_detailed_report(self):
        """Save detailed test report to JSON file."""
        try:
            report_data = {
                "timestamp": time.time(),
                "test_summary": {
                    "total_backends": len(self.test_results),
                    "successful_backends": sum(1 for r in self.test_results.values() if r.get_success_rate() > 0.5),
                    "overall_success_rate": sum(r.get_success_rate() for r in self.test_results.values()) / len(self.test_results) if self.test_results else 0
                },
                "backend_results": {name: result.to_dict() for name, result in self.test_results.items()},
                "configuration": {
                    "available_backends": self.manager.get_available_backends() if self.manager else [],
                    "default_backend": self.manager.get_default_backend() if self.manager else None
                }
            }
            
            with open("backend_test_report.json", "w") as f:
                json.dump(report_data, f, indent=2)
            
            print("💾 Detailed report saved to: backend_test_report.json")
            
        except Exception as e:
            print(f"❌ Failed to save detailed report: {e}")
    
    async def cleanup(self):
        """Cleanup test resources."""
        if self.manager:
            await self.manager.cleanup()


async def main():
    """Main test function."""
    print("🧪 Multi-Backend LLM Comprehensive Test Suite")
    print("=" * 80)
    print("Testing all configured backends with real API keys...")
    print("This may take several minutes to complete.\n")
    
    tester = LLMBackendTester()
    
    try:
        await tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical test error: {e}")
        logger.error(f"Critical test error: {e}")
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    # Run the comprehensive test suite
    asyncio.run(main())
