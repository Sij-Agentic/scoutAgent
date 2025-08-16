#!/usr/bin/env python3
"""
Script to run only the Scout agent's ACT stage.
This consolidates and validates pain points from the THINK stage analysis.
"""

import os
import asyncio
import argparse
import json
import time
import datetime
from pathlib import Path

from scout_agent.custom_logging import get_logger
from scout_agent.agents.scout import ScoutAgent
from scout_agent.memory.manifest_manager import ManifestManager
import traceback

logger = get_logger("scripts.act_only")


def ensure_env_defaults():
    # Keep defaults lightweight; can be overridden by env
    os.environ.setdefault("SCOUT_LLM_DEFAULT_BACKEND", "deepseek")
    os.environ.setdefault("SCOUT_LLM_DEFAULT_MODEL", "deepseek-chat")


def resolve_run_dir(run_id: str) -> Path:
    # project root at ScoutAgent/ (not scout_agent/)
    # The script is in scout_agent/scripts/ so we need to go up 2 levels to get to ScoutAgent/
    root = Path(__file__).resolve().parents[2]
    run_dir = root / "data" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run only the Scout act stage")
    ap.add_argument("--run-id", required=True, help="Run id whose manifest to use under data/runs/{run_id}")
    ap.add_argument("--plan-path", default=None, help="Optional explicit path to manifest (run_manifest.json preferred)")
    return ap.parse_args()


async def main_async():
    ensure_env_defaults()
    args = parse_args()
    
    start_time = time.time()
    
    # Resolve paths
    run_dir = resolve_run_dir(args.run_id)
    plan_path = Path(args.plan_path) if args.plan_path else run_dir / "run_manifest.json"
    
    logger.info(f"Executing act stage using manifest: {plan_path}")
    
    if not plan_path.exists():
        logger.error(f"Manifest file not found: {plan_path}")
        return
    
    # Load manifest
    manifest_manager = ManifestManager(plan_path)
    manifest = manifest_manager.get_manifest()
    
    # Ensure run_metadata exists
    if "run_metadata" not in manifest:
        manifest["run_metadata"] = {}
    
    # Add or update run_id in run_metadata
    manifest["run_metadata"]["run_id"] = args.run_id
    
    # Extract target market and research scope from the manifest
    target_market = manifest.get("target_market", "")
    research_scope = manifest.get("research_scope", "")
    
    if not target_market:
        logger.warning("No target_market found in manifest, using default")
        target_market = "default target market"
    
    # Get the think stage analysis from the manifest
    think_analysis = manifest_manager.get_node_output("think")
    if not think_analysis:
        logger.error("No think stage analysis found in manifest. Run think stage first.")
        return
    
    # Create a minimal agent input
    agent_input = {
        "target_market": target_market,
        "research_scope": research_scope
    }
    
    # Initialize ScoutAgent
    scout = ScoutAgent()
    
    # Set the run_id in the agent's state
    scout.state.run_id = args.run_id
    scout.state.analysis = think_analysis  # Set the think stage analysis
    
    # Execute the act stage
    output = await scout.act(agent_input=agent_input, plan=manifest, thoughts=think_analysis)
    
    # Update manifest with act stage status
    try:
        # Find the act node ID from the DAG
        act_node_id = "act"  # Default
        if "dag" in manifest and "nodes" in manifest["dag"]:
            for node in manifest["dag"]["nodes"]:
                if node.get("type") == "agent" and node.get("agent_task") == "act":
                    act_node_id = node.get("id", "act")
                    break
        
        # Store the act output data in the manifest
        output_dict = {
            "pain_points": [pp.to_dict() for pp in output.pain_points],
            "total_discovered": output.total_discovered,
            "market_summary": output.market_summary,
            "confidence_score": output.confidence_score,
            "sources_used": output.sources_used,
            "research_duration": output.research_duration
        }
        
        manifest_manager.store_node_output(
            node_id=act_node_id,
            data=output_dict
        )
        
        # Update the act node status
        manifest_manager.update_node_status(
            node_id=act_node_id,
            state="completed",
            duration_seconds=time.time() - start_time
        )
        
        # Record metrics if available
        metrics = {}
        if hasattr(scout, "last_metrics") and scout.last_metrics:
            metrics = scout.last_metrics
        elif hasattr(scout, "llm") and hasattr(scout.llm, "last_usage"):
            metrics = {
                "tokens_used": getattr(scout.llm.last_usage, "total_tokens", 0),
                "cost": getattr(scout.llm.last_usage, "cost", 0.0),
                "backend": getattr(scout.llm, "backend_type", None),
                "model": getattr(scout.llm, "model", None)
            }
        
        if metrics:
            manifest_manager.record_metrics(act_node_id, metrics)
        
        logger.info(f"Act stage updated in manifest: {plan_path} (node: {act_node_id})")
        
        # Print a summary of the consolidated results
        summary = {
            "manifest": str(plan_path),
            "node_id": act_node_id,
            "pain_points_consolidated": len(output.pain_points),
            "total_discovered": output.total_discovered,
            "confidence_score": output.confidence_score,
            "execution_time": time.time() - start_time
        }
        print(json.dumps(summary, indent=2))
        
    except Exception as e:
        logger.error(f"Failed to update manifest: {e}")
        print(json.dumps({
            "manifest": str(plan_path), 
            "error": str(e),
            "output": output_dict if 'output_dict' in locals() else {}
        }, indent=2))


def run():
    asyncio.run(main_async())


if __name__ == "__main__":
    run()
