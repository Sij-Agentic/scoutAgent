#!/usr/bin/env python3
"""
Main script for running the ScoutAgent and ScreenerAgent workflow using the AgentOrchestrator.

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
from scout_agent.agents.screener import ScreenerAgent
from scout_agent.agents.validator import ValidatorAgent
from scout_agent.agents.gap_finder import GapFinderAgent
from scout_agent.agents.builder import BuilderAgent
from scout_agent.agents.writer import WriterAgent
from scout_agent.agents.base import AgentInput
from scout_agent.orchestration import AgentOrchestrator, create_orchestrator, AGENT_STAGE_CONFIGS
from scout_agent.orchestration.enhanced_main_orchestrator import EnhancedMainOrchestrator
from scout_agent.custom_logging.logger import get_logger


os.environ.setdefault("SCOUT_LLM_DEFAULT_BACKEND", "deepseek")
os.environ.setdefault("SCOUT_LLM_DEFAULT_MODEL", "deepseek-chat")

async def run_orchestrated_workflow(
    target_market: str,
    sources: List[str] = None,
    keywords: List[str] = None,
    subreddits: List[str] = None,
    research_scope: str = "comprehensive",
    max_pain_points: int = 10,
    top_k: int = 5,
    per_query_limit: int = 50,
    include_comments: bool = True,
    comment_depth: int = 2,
    comment_limit: int = 200,
    run_id: Optional[str] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Run the ScoutAgent and ScreenerAgent workflow using the AgentOrchestrator.
    
    Args:
        target_market: The target market to research
        sources: List of sources to use for research
        keywords: List of keywords to search for
        subreddits: List of subreddits to search
        research_scope: Scope of research (quick, focused, comprehensive)
        max_pain_points: Maximum number of pain points to discover
        top_k: Number of top pain points to return from screener
        per_query_limit: Maximum number of threads per query
        include_comments: Include comments in thread collection
        comment_depth: Maximum depth of comment tree to collect
        comment_limit: Maximum number of comments per thread
        run_id: Optional run ID for the workflow
        debug: Enable debug logging
    
    Returns:
        Dict containing the workflow results
    """
    logger = get_logger("orchestrated_workflow")
    logger.info(f"Starting orchestrated workflow for market: {target_market}")
    
    # Generate run_id if not provided
    if not run_id:
        run_id = f"scout_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Default sources and keywords if not provided
    if sources is None:
        sources = ["reddit"]
    if keywords is None:
        keywords = ["pain point", "problem", "frustration", "issue"]
    if subreddits is None:
        subreddits = []
    
    # Create the agent input
    agent_input = AgentInput(
        data={
            "target_market": target_market,
            "research_scope": research_scope,
            "max_pain_points": max_pain_points,
            "sources": sources,
            "keywords": keywords,
            "subreddits": subreddits,
            "per_query_limit": per_query_limit,
            "include_comments": include_comments,
            "comment_depth": comment_depth,
            "comment_limit": comment_limit
        },
        metadata={
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat()
        },
        context={
            "target_market": target_market,
            "top_k": top_k
        }
    )
    
    # Create the enhanced orchestrator
    orchestrator = EnhancedMainOrchestrator(run_id=run_id)
    
    # Register agents in dependency order
    # ScoutAgent (no dependencies)
    scout_agent = ScoutAgent(agent_id="scout")
    orchestrator.register_agent("scout", AGENT_STAGE_CONFIGS["scout"], scout_agent)
    
    # ScreenerAgent (depends on scout)
    screener_agent = ScreenerAgent(agent_id="screener")
    orchestrator.register_agent("screener", AGENT_STAGE_CONFIGS["screener"], screener_agent)

    # ValidatorAgent (depends on scout)
    validator_agent = ValidatorAgent(agent_id="validator")
    orchestrator.register_agent("validator", AGENT_STAGE_CONFIGS["validator"], validator_agent)

    # GapFinderAgent (depends on scout, screener, validator)
    gap_finder_agent = GapFinderAgent(agent_id="gap_finder")
    orchestrator.register_agent("gap_finder", AGENT_STAGE_CONFIGS["gap_finder"], gap_finder_agent)

    # BuilderAgent (depends on gap_finder)
    builder_agent = BuilderAgent(agent_id="builder")
    orchestrator.register_agent("builder", AGENT_STAGE_CONFIGS["builder"], builder_agent)

    # WriterAgent (depends on builder and gap_finder)
    writer_agent = WriterAgent(agent_id="writer")
    orchestrator.register_agent("writer", AGENT_STAGE_CONFIGS["writer"], writer_agent)
    
    # Initialize the orchestrator
    await orchestrator.initialize(agent_input)
    
    try:
        # Execute the workflow
        logger.info("Executing workflow...")
        results = await orchestrator.execute(agent_input)
        
        # Log completion
        if hasattr(results, 'status'):
            logger.info(f"Workflow completed with status: {results.status}")
        else:
            logger.info(f"Workflow completed with results type: {type(results)}")
        
        # Convert ExecutionState to dictionary if needed
        if hasattr(results, '__dict__'):
            # If it's an ExecutionState object, convert to dict
            if hasattr(results, 'status') and hasattr(results, 'duration'):
                results_dict = {
                    "run_id": run_id,
                    "status": getattr(results, 'status', 'completed'),
                    "execution_time": getattr(results, 'duration', 0),
                    "completed_nodes": getattr(results, 'completed_nodes', 0),
                    "failed_nodes": getattr(results, 'failed_nodes', 0),
                    "total_nodes": getattr(results, 'total_nodes', 0),
                    "progress": getattr(results, 'progress', 0.0)
                }
                return results_dict
        
        # Return the results (should already be a dict from AgentOrchestrator)
        return results
    except Exception as e:
        logger.error(f"Error executing workflow: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e)}


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run ScoutAgent and ScreenerAgent with orchestration")
    parser.add_argument("--target-market", type=str, required=True, help="Target market to research")
    parser.add_argument("--keywords", type=str, required=True, help="Comma-separated keywords to search for")
    parser.add_argument("--subreddits", type=str, default="", help="Comma-separated subreddits to search")
    parser.add_argument("--sources", type=str, nargs="+", help="Sources to use for research")
    parser.add_argument("--research-scope", type=str, default="comprehensive", 
                        choices=["quick", "focused", "comprehensive"],
                        help="Scope of research")
    parser.add_argument("--max-pain-points", type=int, default=10, 
                        help="Maximum number of pain points to discover")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top pain points to return from screener")
    parser.add_argument("--per-query-limit", type=int, default=50, help="Maximum number of threads per query")
    parser.add_argument("--include-comments", action="store_true", help="Include comments in thread collection")
    parser.add_argument("--comment-depth", type=int, default=2, help="Maximum depth of comment tree to collect")
    parser.add_argument("--comment-limit", type=int, default=200, help="Maximum number of comments per thread")
    parser.add_argument("--run-id", type=str, help="Run ID for the workflow")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Parse keywords and subreddits from command line
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    subreddits = [s.strip() for s in args.subreddits.split(",") if s.strip()] if args.subreddits else []
    
    # Run the workflow
    results = asyncio.run(run_orchestrated_workflow(
        target_market=args.target_market,
        sources=args.sources,
        keywords=keywords,
        subreddits=subreddits,
        research_scope=args.research_scope,
        max_pain_points=args.max_pain_points,
        top_k=args.top_k,
        per_query_limit=args.per_query_limit,
        include_comments=args.include_comments,
        comment_depth=args.comment_depth,
        comment_limit=args.comment_limit,
        run_id=args.run_id,
        debug=args.debug
    ))
    
    # Convert ExecutionState to dictionary if needed for JSON serialization
    try:
        # Test if results is JSON serializable
        json.dumps(results)
    except (TypeError, ValueError) as e:
        print(f"DEBUG: Converting non-serializable results to dict: {type(results)}")
        # Convert to a safe dictionary format
        if hasattr(results, '__dict__'):
            results_dict = {
                "run_id": str(getattr(results, 'workflow_id', getattr(results, 'run_id', 'unknown'))),
                "status": str(getattr(results, 'status', 'completed')),
                "execution_time": float(getattr(results, 'duration', getattr(results, 'execution_time', 0))),
                "completed_nodes": int(getattr(results, 'completed_nodes', 0)),
                "failed_nodes": int(getattr(results, 'failed_nodes', 0)),
                "total_nodes": int(getattr(results, 'total_nodes', 0)),
                "progress": float(getattr(results, 'progress', 0.0)),
                "type": str(type(results).__name__)
            }
            results = results_dict
        else:
            results = {"status": "completed", "message": f"Non-serializable result: {type(results)}"}
    
    # Print results summary
    print(json.dumps(results, indent=2))
    
    # Return exit code based on workflow status
    return 0 if (isinstance(results, dict) and results.get("status") == "completed") else 1


if __name__ == "__main__":
    sys.exit(main())