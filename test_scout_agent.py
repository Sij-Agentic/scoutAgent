"""
Test harness for ScoutAgent prompt-driven workflow
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add project root to sys.path to resolve imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from agents.scout import ScoutAgent, ScoutInput
from llm.base import LLMConfig, LLMBackendType
from llm.manager import LLMManager
from llm.backends.ollama import OllamaBackend
from custom_logging import get_logger


async def test_scout_agent():
    """Test ScoutAgent with prompt templates."""
    # Configure logging
    logger = get_logger("scout_agent_test")
    logger.info("Testing ScoutAgent with prompt templates...")
    
    # Create ScoutInput
    scout_input = ScoutInput(
        target_market="SaaS productivity tools",
        research_scope="focused",
        max_pain_points=5,
        sources=["reddit", "product_reviews", "blogs"],
        keywords=["pain point", "frustration", "difficult", "complicated", "confusing"]
    )
    
    # Initialize ScoutAgent
    agent = ScoutAgent("scout-test")
    
    # Initialize LLM Backend
    # Create the LLM configuration for the Ollama backend
    config = LLMConfig(
        backend_type=LLMBackendType.OLLAMA,
        model_name="phi4-mini:latest",  # Using a smaller model for testing
        base_url="http://localhost:11434",
        timeout=180  # 3 minute timeout for complex queries
    )
    
    # Create and register the LLM backend
    llm_manager = LLMManager()
    backend = OllamaBackend(config)
    await llm_manager.register_backend(backend)
    
    # Assign LLM manager to agent
    agent._llm_manager = llm_manager
    
    try:
        # Execute plan phase
        logger.info("Running plan phase...")
        plan_start = datetime.now()
        plan = await agent.plan(scout_input)
        plan_time = (datetime.now() - plan_start).total_seconds()
        logger.info(f"Plan complete: {len(plan.get('phases', []))} phases identified in {plan_time:.2f}s")
        
        # Execute think phase
        logger.info("Running think phase...")
        think_start = datetime.now()
        think_result = await agent.think(scout_input)
        think_time = (datetime.now() - think_start).total_seconds()
        logger.info(f"Think complete: Analysis with {think_result.get('pain_points_found', 0)} pain points in {think_time:.2f}s")
        
        # Execute act phase
        logger.info("Running act phase...")
        act_start = datetime.now()
        act_result = await agent.act(scout_input)
        act_time = (datetime.now() - act_start).total_seconds()
        
        # Log results
        logger.info(f"Act complete: Pain point discovery completed in {act_time:.2f}s")
        logger.info(f"Market summary: {act_result.market_summary}")
        logger.info(f"Discovered pain points: {len(act_result.pain_points)}")
        logger.info(f"Confidence score: {act_result.confidence_score}")
        
        logger.info("\n===== TEST SUCCESSFUL =====\n")
        logger.info(f"ScoutAgent completed discovery in {plan_time + think_time + act_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error testing ScoutAgent: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(test_scout_agent())
