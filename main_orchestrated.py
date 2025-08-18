#!/usr/bin/env python3
"""
Main script for running the ScoutAgent workflow using the AgentOrchestrator.

This script demonstrates how to use the AgentOrchestrator to manage multiple agents
with varying lifecycle stages, integrating their workflows into a unified DAG engine.
"""

import argparse
import asyncio
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from scout_agent.agents.scout import ScoutAgent
from scout_agent.agents.base import AgentInput
from scout_agent.orchestration import AgentOrchestrator, create_orchestrator, AGENT_STAGE_CONFIGS
from scout_agent.custom_logging.logger import get_logger


async def run_orchestrated_workflow(
    target_market: str,
    sources: List[str] = None,
    keywords: List[str] = None,
    research_scope: str = "comprehensive",
    max_pain_points: int = 10,
    run_id: Optional[str] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Run the ScoutAgent workflow using the AgentOrchestrator.
    
    Args:
        target_market: The target market to research
        sources: List of sources to use for research
        keywords: List of keywords to search for
        research_scope: Scope of research (quick, focused, comprehensive)
        max_pain_points: Maximum number of pain points to discover
        run_id: Optional run ID for the workflow
        debug: Enable debug logging
    
    Returns:
        Dict containing the workflow results
    """
    logger = get_logger("orchestrated_workflow", level=logging.DEBUG if debug else logging.INFO)
    logger.info(f"Starting orchestrated workflow for market: {target_market}")
    
    # Generate run_id if not provided
    if not run_id:
        run_id = f"scout_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Default sources and keywords if not provided
    if sources is None:
        sources = ["reddit", "twitter", "forums", "reviews", "blogs"]
    if keywords is None:
        keywords = ["pain point", "problem", "frustration", "issue"]
    
    # Create the agent input
    agent_input = AgentInput(
        data={
            "target_market": target_market,
            "research_scope": research_scope,
            "max_pain_points": max_pain_points,
            "sources": sources,
            "keywords": keywords
        },
        run_id=run_id
    )
    
    # Create the orchestrator
    orchestrator = create_orchestrator(run_id=run_id)
    
    # Register the ScoutAgent
    scout_agent = ScoutAgent(agent_id="scout")
    orchestrator.register_agent(scout_agent, AGENT_STAGE_CONFIGS["scout"])
    
    # Initialize the orchestrator
    await orchestrator.initialize(agent_input)
    
    try:
        # Execute the workflow
        logger.info("Executing workflow...")
        results = await orchestrator.execute()
        
        # Log completion
        logger.info(f"Workflow completed with status: {results.get('status', 'unknown')}")
        
        # Return the results
        return results
    except Exception as e:
        logger.error(f"Error executing workflow: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e)}


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run ScoutAgent with orchestration")
    parser.add_argument("--target-market", type=str, required=True, help="Target market to research")
    parser.add_argument("--sources", type=str, nargs="+", help="Sources to use for research")
    parser.add_argument("--keywords", type=str, nargs="+", help="Keywords to search for")
    parser.add_argument("--research-scope", type=str, default="comprehensive", 
                        choices=["quick", "focused", "comprehensive"],
                        help="Scope of research")
    parser.add_argument("--max-pain-points", type=int, default=10, 
                        help="Maximum number of pain points to discover")
    parser.add_argument("--run-id", type=str, help="Run ID for the workflow")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Run the workflow
    results = asyncio.run(run_orchestrated_workflow(
        target_market=args.target_market,
        sources=args.sources,
        keywords=args.keywords,
        research_scope=args.research_scope,
        max_pain_points=args.max_pain_points,
        run_id=args.run_id,
        debug=args.debug
    ))
    
    # Print results summary
    print(json.dumps(results, indent=2))
    
    # Return exit code based on workflow status
    return 0 if results.get("status") == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
