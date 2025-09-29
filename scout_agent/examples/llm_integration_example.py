"""
Example demonstrating the Multi-Backend LLM Abstraction Layer integration.

This example shows how to:
1. Initialize LLM backends
2. Use the enhanced agents with LLM capabilities
3. Switch between different LLM backends
4. Handle errors and fallbacks
"""

import asyncio
import json
from typing import Dict, Any

from config import init_config
from llm.manager import get_llm_manager, initialize_llm_backends
from llm.base import LLMConfig, LLMBackendType, LLMRequest
from llm.utils import quick_llm_generate, llm_summarize, AgentPromptTemplates
from agents.gap_finder_enhanced import EnhancedGapFinderAgent
from agents.gap_finder import GapFinderInput
from agents.base import AgentInput
from custom_logging import get_logger

logger = get_logger("llm_example")


async def demonstrate_basic_llm_usage():
    """Demonstrate basic LLM functionality."""
    print("\n=== Basic LLM Usage Demo ===")
    
    # Initialize LLM backends
    await initialize_llm_backends()
    
    manager = get_llm_manager()
    available_backends = manager.get_available_backends()
    
    print(f"Available LLM backends: {available_backends}")
    print(f"Default backend: {manager.get_default_backend()}")
    
    if not available_backends:
        print("No LLM backends available. Please configure API keys in your .env file.")
        return
    
    # Test basic generation
    prompt = "Explain what a market gap is in 2-3 sentences."
    
    for backend in available_backends:
        try:
            print(f"\nTesting {backend} backend:")
            response = await quick_llm_generate(prompt, backend_type=backend, max_tokens=100)
            print(f"Response: {response[:200]}...")
        except Exception as e:
            print(f"Error with {backend}: {e}")


async def demonstrate_agent_integration():
    """Demonstrate LLM integration with enhanced agents."""
    print("\n=== Enhanced Agent Integration Demo ===")
    
    # Create sample pain points data
    sample_pain_points = [
        {
            "description": "Small businesses struggle with manual inventory tracking",
            "severity": "high",
            "frequency": "daily",
            "affected_users": 1000
        },
        {
            "description": "Difficulty finding reliable freelancers for short-term projects",
            "severity": "medium", 
            "frequency": "weekly",
            "affected_users": 500
        },
        {
            "description": "Complex expense reporting processes for remote teams",
            "severity": "medium",
            "frequency": "monthly", 
            "affected_users": 800
        }
    ]
    
    # Create input for the enhanced gap finder agent
    gap_finder_input = GapFinderInput(
        validated_pain_points=sample_pain_points,
        market_context="Small to medium businesses (SMBs) in the digital transformation space",
        analysis_scope="comprehensive",
        include_competitive_analysis=True,
        include_market_sizing=True
    )
    
    # Create and execute the enhanced agent
    agent = EnhancedGapFinderAgent()
    agent_input = AgentInput(
        data=gap_finder_input,
        metadata={"example": True, "timestamp": "2024-01-01"}
    )
    
    try:
        print("Executing enhanced gap finder agent with LLM integration...")
        result = await agent.execute(agent_input)
        
        if result.success:
            print(f"\nAnalysis completed successfully!")
            print(f"Execution time: {result.execution_time:.2f} seconds")
            print(f"Market gaps found: {len(result.result.market_gaps)}")
            
            # Display key results
            output = result.result
            print(f"\nKey Insights:")
            for insight in output.result.get("key_insights", [])[:3]:
                print(f"- {insight}")
            
            print(f"\nTop Market Gap:")
            if output.market_gaps:
                top_gap = output.market_gaps[0]
                print(f"- {top_gap.gap_description}")
                print(f"- Opportunity Score: {top_gap.opportunity_score}")
                print(f"- Competition Level: {top_gap.competition_level}")
            
            print(f"\nTop Recommendations:")
            for rec in output.recommendations[:3]:
                print(f"- {rec}")
                
        else:
            print(f"Analysis failed: {result.error}")
            
    except Exception as e:
        print(f"Error executing enhanced agent: {e}")
        logger.error(f"Agent execution error: {e}")


async def demonstrate_backend_switching():
    """Demonstrate switching between different LLM backends."""
    print("\n=== Backend Switching Demo ===")
    
    manager = get_llm_manager()
    available_backends = manager.get_available_backends()
    
    if len(available_backends) < 2:
        print("Need at least 2 backends to demonstrate switching")
        return
    
    prompt = "What are the key factors to consider when analyzing market opportunities?"
    
    # Test the same prompt with different backends
    for backend in available_backends[:2]:  # Test first 2 backends
        try:
            print(f"\nUsing {backend} backend:")
            
            # Create request
            request = LLMRequest(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )
            
            response = await manager.generate(request, backend_type=backend)
            
            print(f"Model: {response.model}")
            print(f"Response time: {response.response_time:.2f}s")
            print(f"Tokens used: {response.usage.get('total_tokens', 'N/A')}")
            print(f"Response: {response.content[:200]}...")
            
        except Exception as e:
            print(f"Error with {backend}: {e}")


async def demonstrate_prompt_templates():
    """Demonstrate using prompt templates for common operations."""
    print("\n=== Prompt Templates Demo ===")
    
    # Market analysis template
    pain_points = [
        "Difficult to track project progress across teams",
        "Manual time tracking is error-prone",
        "Poor visibility into resource allocation"
    ]
    
    market_context = "Project management software for mid-size companies"
    
    prompt = AgentPromptTemplates.market_analysis(pain_points, market_context)
    
    try:
        print("Using market analysis template...")
        response = await quick_llm_generate(
            prompt.to_messages()[0]["content"] + "\n" + prompt.to_messages()[1]["content"],
            temperature=0.4,
            max_tokens=300
        )
        
        print(f"Market analysis result:")
        print(response[:400] + "..." if len(response) > 400 else response)
        
    except Exception as e:
        print(f"Error with prompt template: {e}")


async def demonstrate_health_monitoring():
    """Demonstrate LLM backend health monitoring."""
    print("\n=== Health Monitoring Demo ===")
    
    manager = get_llm_manager()
    
    # Get health status for all backends
    health_status = manager.get_all_health_status()
    
    print("Backend Health Status:")
    for backend_name, health in health_status.items():
        print(f"\n{backend_name}:")
        print(f"  Healthy: {health.is_healthy}")
        print(f"  Error count: {health.error_count}")
        print(f"  Avg response time: {health.avg_response_time:.2f}s")
        print(f"  Last check: {health.last_check}")
    
    # Test health check
    healthy_backends = manager.get_healthy_backends()
    print(f"\nHealthy backends: {healthy_backends}")


async def demonstrate_error_handling():
    """Demonstrate error handling and fallbacks."""
    print("\n=== Error Handling Demo ===")
    
    manager = get_llm_manager()
    
    # Test with invalid backend
    try:
        request = LLMRequest(
            messages=[{"role": "user", "content": "Test message"}],
            max_tokens=50
        )
        
        response = await manager.generate(request, backend_type="nonexistent_backend")
        print(f"Unexpected success: {response.content}")
        
    except Exception as e:
        print(f"Expected error with invalid backend: {e}")
    
    # Test with very long prompt (should handle gracefully)
    try:
        long_prompt = "Analyze this: " + "very long text " * 1000
        response = await quick_llm_generate(long_prompt, max_tokens=50)
        print(f"Long prompt handled successfully: {len(response)} chars")
        
    except Exception as e:
        print(f"Error with long prompt: {e}")


async def main():
    """Main demonstration function."""
    print("Multi-Backend LLM Abstraction Layer Demo")
    print("=" * 50)
    
    # Initialize configuration
    init_config()
    
    try:
        # Run all demonstrations
        await demonstrate_basic_llm_usage()
        await demonstrate_backend_switching()
        await demonstrate_prompt_templates()
        await demonstrate_health_monitoring()
        await demonstrate_error_handling()
        await demonstrate_agent_integration()
        
    except Exception as e:
        print(f"Demo error: {e}")
        logger.error(f"Demo execution error: {e}")
    
    finally:
        # Cleanup
        manager = get_llm_manager()
        await manager.cleanup()
        print("\n=== Demo Complete ===")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
