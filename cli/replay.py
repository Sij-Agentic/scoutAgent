#!/usr/bin/env python3
"""
Checkpoint replay tool for ScoutAgent.

This script allows replaying a workflow from a checkpoint, skipping stages,
or replaying until a specific stage.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

from scout_agent.orchestration.replay_orchestrator import ReplayOrchestrator
from scout_agent.agents.base import register_all_agents
from scout_agent.custom_logging.logger import get_logger

# Set default LLM backend
os.environ.setdefault("SCOUT_LLM_DEFAULT_BACKEND", "deepseek")
os.environ.setdefault("SCOUT_LLM_DEFAULT_MODEL", "deepseek-chat")

logger = get_logger("replay")


async def replay_from_checkpoint(
    run_id: str,
    from_stage: Optional[str] = None,
    to_stage: Optional[str] = None,
    skip_stages: Optional[List[str]] = None,
    new_run_id: Optional[str] = None
) -> dict:
    """
    Replay a workflow from a checkpoint.
    
    Args:
        run_id: ID of the run to replay
        from_stage: Optional stage to start replay from
        to_stage: Optional stage to replay until
        skip_stages: Optional list of stages to skip
        new_run_id: Optional new run ID for the replay
        
    Returns:
        Results of the replay execution
    """
    logger.info(f"Replaying run {run_id}")
    
    # Create replay orchestrator
    orchestrator = ReplayOrchestrator(
        run_id=run_id,
        from_stage=from_stage,
        to_stage=to_stage,
        skip_stages=skip_stages,
        new_run_id=new_run_id
    )
    
    # Register all agents
    register_all_agents(orchestrator)
    
    # Initialize and execute
    await orchestrator.initialize()
    results = await orchestrator.execute()
    
    return results


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Replay a workflow from a checkpoint")
    parser.add_argument("--run-id", required=True, help="ID of the run to replay")
    parser.add_argument("--from-stage", help="Stage to start replay from (inclusive)")
    parser.add_argument("--to-stage", help="Stage to replay until (inclusive)")
    parser.add_argument("--skip-stages", help="Comma-separated list of stages to skip")
    parser.add_argument("--new-run-id", help="Optional new run ID for the replay")
    
    args = parser.parse_args()
    
    # Parse skip_stages
    skip_stages = None
    if args.skip_stages:
        skip_stages = [s.strip() for s in args.skip_stages.split(",") if s.strip()]
    
    # Run the replay
    results = asyncio.run(replay_from_checkpoint(
        run_id=args.run_id,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        skip_stages=skip_stages,
        new_run_id=args.new_run_id
    ))
    
    # Print results
    print(f"Replay completed with status: {results.get('status', 'unknown')}")
    print(f"Execution time: {results.get('execution_time', 0):.2f}s")
    print(f"Completed nodes: {results.get('completed_nodes', 0)}")
    print(f"Failed nodes: {results.get('failed_nodes', 0)}")
    
    # Return exit code based on status
    return 0 if results.get("status") == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
