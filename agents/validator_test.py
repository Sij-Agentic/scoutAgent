"""
Test harness for ValidatorAgent prompt-driven workflow
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scout_agent.agents.validator import ValidatorAgent, ValidatorInput
from scout_agent.llm.base import LLMConfig, LLMBackendType
from scout_agent.llm.manager import LLMManager
from scout_agent.llm.backends.ollama import OllamaBackend


async def test_validator_agent():
    """Test ValidatorAgent with prompt templates."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("validator_agent_test")
    logger.info("Testing ValidatorAgent with prompt templates...")
    
    # Sample pain points data
    pain_points = [
        {
            "id": "pp1",
            "description": "Small businesses struggle with project tracking and team coordination",
            "pain_point": "Project management complexity",
            "market_segment": "SMB",
            "severity": "high",
            "tags": ["productivity", "collaboration"],
            "impact_score": 8.5
        },
        {
            "id": "pp2",
            "description": "E-commerce businesses lack integrated analytics for sales and marketing",
            "pain_point": "Fragmented data analytics",
            "market_segment": "E-commerce",
            "severity": "medium",
            "tags": ["analytics", "marketing"],
            "impact_score": 7.2
        }
    ]
    
    # Create ValidatorInput
    validator_input = ValidatorInput(
        pain_points=pain_points,
        validation_depth="moderate",
        market_context="SaaS market for small to medium businesses with focus on productivity tools",
        include_user_interviews=True,
        include_competitor_analysis=True
    )
    
    # Initialize ValidatorAgent
    agent = ValidatorAgent("validator-test")
    
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
        plan = await agent.plan(validator_input)
        plan_time = (datetime.now() - plan_start).total_seconds()
        logger.info(f"Plan complete: {len(plan.get('phases', []))} phases identified in {plan_time:.2f}s")
        
        # Execute think phase
        logger.info("Running think phase...")
        think_start = datetime.now()
        think_result = await agent.think(validator_input)
        think_time = (datetime.now() - think_start).total_seconds()
        logger.info(f"Think complete: Validation strategy developed in {think_time:.2f}s")
        
        # Execute act phase
        logger.info("Running act phase...")
        act_start = datetime.now()
        act_result = await agent.act(validator_input)
        act_time = (datetime.now() - act_start).total_seconds()
        
        # Log results
        logger.info(f"Act complete: Validation completed in {act_time:.2f}s")
        logger.info(f"Validation summary: {act_result.validation_summary}")
        logger.info(f"Validated pain points: {len(act_result.validated_pain_points)}")
        logger.info(f"Invalid pain points: {len(act_result.invalid_pain_points)}")
        logger.info(f"Recommendations: {len(act_result.recommendations)}")
        
        logger.info("\n===== TEST SUCCESSFUL =====\n")
        logger.info(f"ValidatorAgent completed validation in {plan_time + think_time + act_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error testing ValidatorAgent: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(test_validator_agent())
