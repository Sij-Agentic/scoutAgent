#!/usr/bin/env python3
"""
Enhanced development tool for ScoutAgent.

This script provides a more robust way to develop new agent stages
using data from existing runs with improved data extraction and passing.
"""

import argparse
import asyncio
import os
import sys
import importlib
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

from scout_agent.orchestration.enhanced_development_orchestrator import EnhancedDevelopmentOrchestrator
from scout_agent.orchestration.agent_orchestrator import AgentStageConfig
from scout_agent.custom_logging.logger import get_logger

# Set default LLM backend
os.environ.setdefault("SCOUT_LLM_DEFAULT_BACKEND", "deepseek")
os.environ.setdefault("SCOUT_LLM_DEFAULT_MODEL", "deepseek-chat")

logger = get_logger("enhanced_develop_stage")


async def develop_stages(
    source_run_id: str,
    agent_id: str,
    stages: List[str],
    new_run_id: Optional[str] = None,
    debug_mode: bool = False
) -> dict:
    """
    Develop specific stages of an agent using data from a source run.
    
    Args:
        source_run_id: ID of the source run to use for data
        agent_id: ID of the agent being developed
        stages: List of stages to develop
        new_run_id: Optional ID for the new development run
        debug_mode: Enable additional debugging output
        
    Returns:
        Results of the development execution
    """
    logger.info(f"Developing {agent_id} stages: {stages}")
    logger.info(f"Using source run: {source_run_id}")
    
    # Create enhanced development orchestrator
    orchestrator = EnhancedDevelopmentOrchestrator(
        source_run_id=source_run_id,
        agent_id=agent_id,
        stages=stages,
        new_run_id=new_run_id,
        debug_mode=debug_mode
    )
    
    # Register the agent being developed
    try:
        # Import the agent module dynamically
        module_name = f"scout_agent.agents.{agent_id}"
        agent_module = importlib.import_module(module_name)
        
        # Get the agent class (assuming it follows the naming convention)
        agent_class_name = f"{agent_id.capitalize()}Agent"
        if hasattr(agent_module, agent_class_name):
            agent_class = getattr(agent_module, agent_class_name)
            agent_instance = agent_class(agent_id=agent_id)
            
            # Get the full agent config from AGENT_STAGE_CONFIGS
            from scout_agent.orchestration.agent_orchestrator import AGENT_STAGE_CONFIGS
            agent_config = AGENT_STAGE_CONFIGS.get(agent_id)
            
            if agent_config:
                # Register with development orchestrator
                orchestrator.register_agent_for_development(agent_id, agent_config, agent_instance)
            else:
                # Create a new config if not found
                logger.warning(f"No configuration found for agent: {agent_id}")
                agent_config = AgentStageConfig(stages=stages)
                orchestrator.register_agent_for_development(agent_id, agent_config, agent_instance)
        else:
            logger.error(f"Agent class {agent_class_name} not found in module {module_name}")
            return {"status": "failed", "error": f"Agent class {agent_class_name} not found"}
    except ImportError:
        logger.error(f"Failed to import agent module: {module_name}")
        return {"status": "failed", "error": f"Failed to import agent module: {module_name}"}
    except Exception as e:
        logger.error(f"Error registering agent: {str(e)}")
        return {"status": "failed", "error": f"Error registering agent: {str(e)}"}
    
    # Register dependency agents
    try:
        # Get the dependency map
        dependency_map = {
            "scout": [],
            "screener": ["scout"],
            "validator": ["scout", "screener"],
            "gap_finder": ["scout", "screener", "validator"],
            "builder": ["scout", "screener", "validator", "gap_finder"],
            "writer": ["scout", "screener", "validator", "gap_finder", "builder"]
        }
        
        # Register dependencies
        dependencies = dependency_map.get(agent_id, [])
        for dep_agent_id in dependencies:
            try:
                # Import the dependency agent module
                dep_module_name = f"scout_agent.agents.{dep_agent_id}"
                dep_agent_module = importlib.import_module(dep_module_name)
                
                # Get the dependency agent class
                dep_agent_class_name = f"{dep_agent_id.capitalize()}Agent"
                if hasattr(dep_agent_module, dep_agent_class_name):
                    dep_agent_class = getattr(dep_agent_module, dep_agent_class_name)
                    dep_agent_instance = dep_agent_class(agent_id=dep_agent_id)
                    
                    # Get the dependency agent config
                    from scout_agent.orchestration.agent_orchestrator import AGENT_STAGE_CONFIGS
                    dep_agent_config = AGENT_STAGE_CONFIGS.get(dep_agent_id)
                    
                    if dep_agent_config:
                        # Register the dependency agent
                        orchestrator.register_agent(dep_agent_id, dep_agent_config, dep_agent_instance)
                        logger.info(f"Registered dependency agent: {dep_agent_id}")
            except Exception as e:
                logger.warning(f"Failed to register dependency agent {dep_agent_id}: {str(e)}")
    except Exception as e:
        logger.warning(f"Error registering dependency agents: {str(e)}")
    
    # Initialize and execute
    await orchestrator.initialize()
    results = await orchestrator.execute()
    
    # Dump debug info if enabled
    if debug_mode:
        orchestrator.dump_debug_info()
    
    return results


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Enhanced development tool for agent stages")
    parser.add_argument("--source-run", required=True, help="Source run ID to use for data")
    parser.add_argument("--agent", required=True, help="Agent ID to develop")
    parser.add_argument("--stages", required=True, help="Comma-separated list of stages to develop")
    parser.add_argument("--run-id", help="Optional run ID for the development run")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Parse stages
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    
    if not stages:
        print("Error: No stages specified")
        return 1
    
    # Run the development
    results = asyncio.run(develop_stages(
        source_run_id=args.source_run,
        agent_id=args.agent,
        stages=stages,
        new_run_id=args.run_id,
        debug_mode=args.debug
    ))
    
    # Print results
    print(f"Development completed with status: {results.get('status', 'unknown')}")
    print(f"Execution time: {results.get('execution_time', 0):.2f}s")
    print(f"Completed nodes: {results.get('completed_nodes', 0)}")
    print(f"Failed nodes: {results.get('failed_nodes', 0)}")
    
    # Return exit code based on status
    return 0 if results.get("status") == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
