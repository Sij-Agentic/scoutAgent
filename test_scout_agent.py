"""
Test harness for ScoutAgent prompt-driven workflow
"""

import asyncio
import json
import sys
import os
from datetime import datetime
import traceback

# Add project root to sys.path to resolve imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from agents.scout import ScoutAgent, ScoutInput
from llm.base import LLMConfig, LLMBackendType
from llm.manager import LLMManager
from custom_logging import get_logger


async def test_scout_agent():
    """Test ScoutAgent with complete workflow (plan→collect→think→act)."""
    # Configure logging
    logger = get_logger("scout_agent_test")
    logger.info("Testing ScoutAgent with complete workflow (plan→collect→think→act)...")
    
    # Generate a unique run_id for this test
    run_id = f"scout_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Using run_id: {run_id}")
    
    # Create ScoutInput
    scout_input = ScoutInput(
        target_market="SaaS productivity tools",
        research_scope="focused",
        max_pain_points=5,
        sources=["reddit"],
        keywords=["pain point", "frustration", "difficult", "complicated", "confusing"]
    )
    
    # Initialize ScoutAgent with the run_id
    agent = ScoutAgent("scout-test", run_id=run_id)
    
    # Set up LLM backend based on environment variables or default to a local model
    llm_backend = os.environ.get("LLM_BACKEND", "deepseek")
    llm_model = os.environ.get("LLM_MODEL", "deepseek-chat")
    
    logger.info(f"Using LLM backend: {llm_backend}, model: {llm_model}")
    
    # Initialize LLM Backend
    config = LLMConfig(
        backend_type=LLMBackendType(llm_backend),
        model_name=llm_model,
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434"),
        timeout=180  # 3 minute timeout for complex queries
    )
    
    # Create and register the LLM backend
    llm_manager = LLMManager()
    
    if llm_backend == "ollama":
        from llm.backends.ollama import OllamaBackend
        backend = OllamaBackend(config)
    elif llm_backend == "deepseek":
        from llm.backends.deepseek import DeepseekBackend
        backend = DeepseekBackend(config)
    else:
        logger.warning(f"Unknown backend {llm_backend}, falling back to Ollama")
        from llm.backends.ollama import OllamaBackend
        backend = OllamaBackend(config)
    
    await llm_manager.register_backend(backend)
    
    # Assign LLM manager to agent
    agent._llm_manager = llm_manager
    
    total_time = 0
    
    try:
        # Execute plan phase
        logger.info("Running plan phase...")
        plan_start = datetime.now()
        plan = await agent.plan(scout_input)
        plan_time = (datetime.now() - plan_start).total_seconds()
        total_time += plan_time
        logger.info(f"Plan complete: {len(plan.get('phases', []))} phases identified in {plan_time:.2f}s")
        
        # Execute collect phase
        logger.info("Running collect phase...")
        collect_start = datetime.now()
        collect_result = await agent.collect(scout_input, plan)
        collect_time = (datetime.now() - collect_start).total_seconds()
        total_time += collect_time
        logger.info(f"Collect complete: {collect_result.get('threads_collected', 0)} threads collected in {collect_time:.2f}s")
        
        # Execute think phase
        logger.info("Running think phase...")
        think_start = datetime.now()
        think_result = await agent.think(scout_input, plan)
        think_time = (datetime.now() - think_start).total_seconds()
        total_time += think_time
        logger.info(f"Think complete: Analysis with {len(think_result.get('pains', []))} pain points in {think_time:.2f}s")
        
        # Execute act phase
        logger.info("Running act phase...")
        act_start = datetime.now()
        act_result = await agent.act(scout_input, plan, think_result)
        act_time = (datetime.now() - act_start).total_seconds()
        total_time += act_time
        
        # Log results
        logger.info(f"Act complete: Pain point discovery completed in {act_time:.2f}s")
        logger.info(f"Market summary: {act_result.market_summary}")
        logger.info(f"Discovered pain points: {len(act_result.pain_points)}")
        logger.info(f"Confidence score: {act_result.confidence_score}")
        
        # Print a sample of the discovered pain points
        if act_result.pain_points:
            logger.info("\nSample pain points discovered:")
            for i, pp in enumerate(act_result.pain_points[:3]):  # Show up to 3 pain points
                logger.info(f"  {i+1}. {pp.description} (Severity: {pp.severity}, Impact: {pp.impact_score})")
                if pp.evidence:
                    logger.info(f"     Evidence: {pp.evidence[0][:100]}...")
        
        logger.info("\n===== TEST SUCCESSFUL =====\n")
        logger.info(f"ScoutAgent completed full workflow in {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error testing ScoutAgent: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(test_scout_agent())
