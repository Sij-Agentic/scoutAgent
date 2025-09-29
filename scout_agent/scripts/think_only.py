#!/usr/bin/env python3
import os
import json
import argparse
import asyncio
import time
import datetime
from pathlib import Path

from scout_agent.custom_logging import get_logger
from scout_agent.agents.scout import ScoutAgent
from scout_agent.memory.manifest_manager import ManifestManager
import traceback

logger = get_logger("scripts.think_only")


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
    ap = argparse.ArgumentParser(description="Run only the Scout think stage")
    ap.add_argument("--run-id", required=True, help="Run id whose manifest to use under data/runs/{run_id}")
    ap.add_argument("--plan-path", default=None, help="Optional explicit path to manifest (run_manifest.json preferred)")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    return ap.parse_args()


async def main_async():
    ensure_env_defaults()
    args = parse_args()
    
    # Record start time for execution metrics
    start_time = time.time()

    # Resolve manifest path
    if args.plan_path:
        plan_path = Path(args.plan_path)
    else:
        run_dir = resolve_run_dir(args.run_id)
        # Prefer consolidated manifest
        preferred = run_dir / "run_manifest.json"
        legacy1 = run_dir / "scout_plan.json"
        legacy2 = run_dir / "plan.json"
        plan_path = preferred if preferred.exists() else (legacy1 if legacy1.exists() else legacy2)

    if not plan_path.exists():
        raise FileNotFoundError(f"Manifest not found at: {plan_path}")

    logger.info(f"Executing think stage using manifest: {plan_path}")

    # Load the manifest to get the plan and input data
    manifest_manager = ManifestManager(plan_path)
    manifest = manifest_manager.get_manifest()
    
    # Ensure run_id is in the manifest's run_metadata
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
    
    # Create a minimal agent input
    agent_input = {
        "target_market": target_market,
        "research_scope": research_scope
    }
    
    # Initialize ScoutAgent
    scout = ScoutAgent()
    
    # Set the run_id in the agent's state
    scout.state.run_id = args.run_id
    
    # Execute the think stage
    analysis = await scout.think(agent_input=agent_input, plan=manifest)
    
    # Update manifest with think stage status
    try:
        # Find the think node ID from the DAG
        think_node_id = "think"  # Default
        if "dag" in manifest and "nodes" in manifest["dag"]:
            for node in manifest["dag"]["nodes"]:
                if node.get("type") == "agent" and node.get("agent_task") == "think":
                    think_node_id = node.get("id", "think")
                    break
        
        # Store the analysis data in the manifest
        manifest_manager.store_node_output(
            node_id=think_node_id,
            data=analysis
        )
        
        # Update the think node status
        manifest_manager.update_node_status(
            node_id=think_node_id,
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
            manifest_manager.record_metrics(think_node_id, metrics)
        
        logger.info(f"Think stage updated in manifest: {plan_path} (node: {think_node_id})")
        
        # Print a summary of the analysis
        summary = {
            "manifest": str(plan_path),
            "node_id": think_node_id,
            "threads_analyzed": len(analysis.get("per_thread_summaries", [])),
            "pains_found": len(analysis.get("pains", [])),
            "themes_identified": len(analysis.get("themes", [])),
            "execution_time": time.time() - start_time
        }
        print(json.dumps(summary, indent=2))
        
    except Exception as e:
        logger.error(f"Failed to update manifest: {e}")
        print(json.dumps({
            "manifest": str(plan_path), 
            "error": str(e),
            "analysis": analysis
        }, indent=2))


def run():
    asyncio.run(main_async())


if __name__ == "__main__":
    run()
