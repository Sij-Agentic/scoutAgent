"""
Test harness for ScreenerAgent prompt-driven workflow
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add project root to sys.path to resolve imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from agents.screener import ScreenerAgent, ScreenerInput, ScreeningCriteria
from llm.base import LLMConfig, LLMBackendType
from llm.manager import LLMManager
from llm.backends.ollama import OllamaBackend
from custom_logging import get_logger


async def test_screener_agent():
    """Test ScreenerAgent with prompt templates."""
    # Configure logging
    logger = get_logger("screener_agent_test")
    logger.info("Testing ScreenerAgent with prompt templates...")
    
    # Create sample pain points
    pain_points = [
        {
            "id": "pp1",
            "description": "Difficulty integrating with third-party tools",
            "severity": "high",
            "impact_score": 8.5,
            "frequency": 12,
            "tags": ["integration", "technical"]
        },
        {
            "id": "pp2",
            "description": "Poor documentation for API endpoints",
            "severity": "medium",
            "impact_score": 6.2,
            "frequency": 8,
            "tags": ["documentation", "technical"]
        },
        {
            "id": "pp3",
            "description": "Slow response times for complex queries",
            "severity": "high",
            "impact_score": 7.8,
            "frequency": 15,
            "tags": ["performance", "technical"]
        },
        {
            "id": "pp4",
            "description": "Minor UI inconsistencies",
            "severity": "low",
            "impact_score": 3.2,
            "frequency": 2,
            "tags": ["ui"]
        },
        {
            "id": "pp5",
            "description": "Limited customization options",
            "severity": "medium",
            "impact_score": 5.5,
            "frequency": 7,
            "tags": ["customization", "features"]
        }
    ]
    
    # Create screening criteria
    criteria = ScreeningCriteria(
        min_severity="medium",
        min_impact_score=5.0,
        min_frequency=3
    )
    
    # Create ScreenerInput
    screener_input = ScreenerInput(
        data=pain_points,  # pain_points will be extracted from data in __post_init__
        metadata={"source": "test_harness"},
        criteria=criteria,
        market_focus="SaaS productivity tools"
    )
    
    # Initialize ScreenerAgent
    agent = ScreenerAgent("screener-test")
    
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
        plan = await agent.plan(screener_input)
        plan_time = (datetime.now() - plan_start).total_seconds()
        logger.info(f"Plan complete: {len(plan.get('phases', []))} phases identified in {plan_time:.2f}s")
        
        # Execute think phase
        logger.info("Running think phase...")
        think_start = datetime.now()
        think_result = await agent.think(screener_input)
        think_time = (datetime.now() - think_start).total_seconds()
        logger.info(f"Think complete: Analysis with strategy '{think_result.get('filtering_strategy', 'unknown')}' in {think_time:.2f}s")
        
        # Execute act phase
        logger.info("Running act phase...")
        act_start = datetime.now()
        act_result = await agent.act(screener_input)
        act_time = (datetime.now() - act_start).total_seconds()
        
        # Log results
        logger.info(f"Act complete: Screening completed in {act_time:.2f}s")
        logger.info(f"Filtered pain points: {len(act_result.filtered_pain_points)}")
        logger.info(f"Rejected pain points: {len(act_result.rejected_pain_points)}")
        logger.info(f"Summary: {act_result.summary}")
        logger.info(f"Confidence score: {act_result.confidence_score}")
        
        logger.info("\n===== TEST SUCCESSFUL =====\n")
        logger.info(f"ScreenerAgent completed screening in {plan_time + think_time + act_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error testing ScreenerAgent: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(test_screener_agent())
