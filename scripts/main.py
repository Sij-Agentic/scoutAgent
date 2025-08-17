import os
import json
import argparse
import asyncio
import time
import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from scout_agent.custom_logging import get_logger
from scout_agent.config import get_config
from scout_agent.agents.scout import ScoutAgent, ScoutInput
from scout_agent.mcp_integration.client.multi import MultiMCPClient
from scout_agent.memory.manifest_manager import ManifestManager

logger = get_logger("scripts.main")


def ensure_env_defaults():
    # Respect existing env; set sensible defaults otherwise
    os.environ.setdefault("SCOUT_LLM_DEFAULT_BACKEND", "deepseek")
    os.environ.setdefault("SCOUT_LLM_DEFAULT_MODEL", "deepseek-chat")


def ensure_run_dirs(run_id: str) -> Path:
    # Project root at ScoutAgent/ (not scout_agent/)
    # The script is in scout_agent/scripts/ so we need to go up 2 levels to get to ScoutAgent/
    root = Path(__file__).resolve().parents[2]
    run_dir = root / "data" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using run directory: {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Scout workflow end-to-end")
    ap.add_argument("--target-market", required=True, help="Target market description")
    ap.add_argument("--keywords", required=True, help="Comma-separated keywords")
    ap.add_argument("--subreddits", default="", help="Comma-separated subreddits")
    ap.add_argument("--run-id", default=f"scout_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                   help="Run id for artifacts under data/runs/")
    ap.add_argument("--research-scope", default="focused", choices=["quick", "focused", "comprehensive"],
                   help="Research scope: quick, focused, or comprehensive")
    ap.add_argument("--max-pain-points", type=int, default=10, help="Maximum number of pain points to discover")
    ap.add_argument("--per-query-limit", type=int, default=15, help="Maximum number of threads per query")
    ap.add_argument("--include-comments", action="store_true", help="Include comments in thread collection")
    ap.add_argument("--comment-depth", type=int, default=2, help="Maximum depth of comment tree to collect")
    ap.add_argument("--comment-limit", type=int, default=50, help="Maximum number of comments per thread")
    ap.add_argument("--use-api", action="store_true", help="Use API-backed reddit tool; default cache tool")
    ap.add_argument("--skip-stages", default="", help="Comma-separated list of stages to skip (plan,collect,think,act)")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return ap.parse_args()


# The run_collect function is no longer needed as we're using scout.collect() directly
# which handles sandboxed execution of the tool nodes in the plan


async def main_async():
    """Main async function to run the complete Scout workflow"""
    # Set up environment and parse arguments
    ensure_env_defaults()
    args = parse_args()
    run_dir = ensure_run_dirs(args.run_id)
    
    # Track overall execution time
    overall_start_time = time.time()
    
    # Parse stages to skip
    skip_stages = [s.strip() for s in args.skip_stages.split(",") if s.strip()] if args.skip_stages else []
    logger.info(f"Skipping stages: {skip_stages if skip_stages else 'None'}")
    
    # Initialize ScoutAgent
    scout = ScoutAgent(agent_id=args.run_id)
    # Store run_id in the agent's state for later use
    setattr(scout.state, "run_id", args.run_id)
    
    # Create ScoutInput from arguments
    scout_input = ScoutInput(
        target_market=args.target_market,
        research_scope=args.research_scope,
        max_pain_points=args.max_pain_points,
        keywords=[s.strip() for s in args.keywords.split(",") if s.strip()],
        sources=["reddit"],
    )
    
    # Initialize manifest path
    manifest_path = run_dir / "run_manifest.json"
    
    # Create a ManifestManager instance
    manifest_manager = None
    
    # Track stage results
    plan_result = None
    collect_result = None
    think_result = None
    act_result = None
    
    try:
        # PLAN STAGE
        if "plan" not in skip_stages:
            logger.info("=== STARTING PLAN STAGE ===")
            plan_start = time.time()
            
            # Execute plan phase
            plan_result = await scout.plan(scout_input)
            plan_time = time.time() - plan_start
            
            # Initialize manifest manager after plan stage
            manifest_manager = ManifestManager(manifest_path)
            
            # Add metadata to manifest
            manifest = manifest_manager.get_manifest()
            manifest["target_market"] = args.target_market
            manifest["research_scope"] = args.research_scope
            manifest["keywords"] = [s.strip() for s in args.keywords.split(",") if s.strip()]
            manifest["subreddits"] = [s.strip() for s in args.subreddits.split(",") if s.strip()] if args.subreddits else []
            manifest["limits"] = {
                "per_query_limit": args.per_query_limit,
                "comment_depth": args.comment_depth,
                "comment_limit": args.comment_limit
            }
            
            # Store the plan output in the manifest
            manifest_manager.store_node_output("plan", plan_result)
            
            # Update the plan node status
            manifest_manager.update_node_status(
                node_id="plan",
                state="completed",
                duration_seconds=plan_time
            )
            
            manifest_manager._save()
            
            logger.info(f"Plan complete: {len(plan_result.get('phases', []))} phases identified in {plan_time:.2f}s")
        else:
            logger.info("Skipping plan stage...")
            # Initialize manifest manager from existing manifest
            manifest_manager = ManifestManager(manifest_path)
            plan_result = manifest_manager.get_manifest()
        
        # COLLECT STAGE
        if "collect" not in skip_stages:
            logger.info("=== STARTING COLLECT STAGE ===")
            collect_start = time.time()
            
            # Ensure the plan has a collect node with the Reddit tool
            dag_nodes = plan_result.get("dag", {}).get("nodes", [])
            has_collect_node = any(n.get("id") == "collect_reddit" for n in dag_nodes)
            
            if not has_collect_node:
                logger.info("Adding collect_reddit node to the DAG")
                # Create a collect node with the Reddit tool
                keywords = [s.strip() for s in args.keywords.split(",") if s.strip()]
                subreddits = [s.strip() for s in args.subreddits.split(",") if s.strip()] if args.subreddits else []
                
                tool_name = "reddit_api_search_and_fetch_threads" if args.use_api else "reddit_search_and_fetch_threads"
                
                # Create the code for the collect node
                code = f'result = mcp_call(tool="{tool_name}", params={{"keywords": {keywords}, "subreddits": {subreddits}, "per_query_limit": {args.per_query_limit}, "include_comments": {bool(args.include_comments)}, "comment_depth": {args.comment_depth}, "comment_limit": {args.comment_limit}, "use_cache": True}}); save_to_manifest("stages.collect_reddit", result)'
                
                collect_node = {
                    "id": "collect_reddit",
                    "type": "tool",
                    "tool": tool_name,
                    "params": {
                        "keywords": keywords,
                        "subreddits": subreddits,
                        "per_query_limit": args.per_query_limit,
                        "include_comments": bool(args.include_comments),
                        "comment_depth": args.comment_depth,
                        "comment_limit": args.comment_limit,
                        "use_cache": True
                    },
                    "code": code,
                    "inputs": {},
                    "outputs": ["stages.collect_reddit"],
                    "deps": ["plan"]
                }
                
                # Add the collect node to the DAG
                dag_nodes.append(collect_node)
                plan_result["dag"]["nodes"] = dag_nodes
                
                # Update the manifest with the modified plan
                manifest_manager.store_node_output("plan", plan_result)
                manifest_manager._save()
            
            try:
                # Use scout's collect method with sandboxed execution
                # This will execute the tool nodes in the plan using the CodeExecutionService
                collect_result = await scout.collect(plan=plan_result, run_id=args.run_id)
                
                # Log the results
                logger.info(f"Collect stage completed with result: {collect_result}")
                
                # Get the actual data from the manifest for logging
                manifest = manifest_manager.get_manifest()
                collect_data = manifest.get("stages", {}).get("collect_reddit", {}).get("data", {})
                threads_collected = collect_data.get("execution_details", {}).get("total_threads_collected", 0)
                comments_collected = collect_data.get("execution_details", {}).get("total_comments_collected", 0)
                logger.info(f"Collect complete: {threads_collected} threads, {comments_collected} comments collected")
                
                # Calculate and log execution time
                collect_duration = time.time() - collect_start
                logger.info(f"Collect stage execution time: {collect_duration:.2f} seconds")
            except Exception as e:
                logger.error(f"Error in collect stage: {e}")
                collect_result = {"error": str(e), "completed": [], "failed": ["collect_reddit"]}
        else:
            logger.info("Skipping collect stage...")
            # Get collect result from manifest if available
            collect_node_id = "collect_reddit"
            collect_result = manifest_manager.get_node_output(collect_node_id) or {}
        
        # THINK STAGE
        if "think" not in skip_stages:
            logger.info("=== STARTING THINK STAGE ===")
            think_start = time.time()
            
            # Make sure the collect data is available in the state for the think stage
            # This is important because the think stage expects to find the Reddit data in the state
            collect_data = manifest_manager.get_manifest().get("stages", {}).get("collect_reddit", {}).get("data", {})
            if collect_data:
                logger.info("Found collect data in manifest, setting it in scout state")
                # Store the collect data in the scout state for the think stage to access
                setattr(scout.state, "collect_data", collect_data)
            else:
                logger.warning("No collect data found in manifest stages.collect_reddit")
                
                # Try to find collect data in node outputs as fallback
                collect_data = manifest_manager.get_node_output("collect_reddit")
                if collect_data:
                    logger.info("Found collect data in node outputs, setting it in scout state")
                    setattr(scout.state, "collect_data", collect_data)
                else:
                    logger.warning("No collect data found in node outputs either")
            
            # Execute think phase
            think_result = await scout.think(agent_input=scout_input, plan=plan_result)
            think_time = time.time() - think_start
            
            # Store the think output in the manifest
            manifest_manager.store_node_output("think", think_result)
            
            # Also store in the stages section for compatibility with the act stage
            stages = manifest_manager.get_manifest().setdefault("stages", {})
            stages["think"] = {
                "data": think_result,
                "status": "completed",
                "updated_at": datetime.datetime.now().isoformat()
            }
            manifest_manager._save()
            
            # Update the think node status
            manifest_manager.update_node_status(
                node_id="think",
                state="completed",
                duration_seconds=think_time
            )
            
            logger.info(f"Think complete: Analysis with {len(think_result.get('pains', []))} pain points in {think_time:.2f}s")
        else:
            logger.info("Skipping think stage...")
            # Get think result from manifest if available
            think_result = manifest_manager.get_node_output("think") or {}
        
        # ACT STAGE
        if "act" not in skip_stages:
            logger.info("=== STARTING ACT STAGE ===")
            act_start = time.time()
            
            # Make sure the think data is available in the state for the act stage
            # This is important because the act stage expects to find the think data in the state
            if think_result:
                logger.info("Setting think result in scout state for act stage")
                setattr(scout.state, "analysis", think_result)
            else:
                logger.warning("No think result available for act stage")
                
                # Try to get think data from manifest as fallback
                think_data = manifest_manager.get_manifest().get("stages", {}).get("think", {}).get("data", {})
                if think_data:
                    logger.info("Found think data in manifest, setting it in scout state")
                    setattr(scout.state, "analysis", think_data)
                else:
                    logger.warning("No think data found in manifest stages.think")
            
            # Execute act phase
            act_result = await scout.act(agent_input=scout_input, plan=plan_result, thoughts=think_result)
            act_time = time.time() - act_start
            
            # Convert dataclass output to dict
            try:
                act_output_dict = {
                    "pain_points": [pp.to_dict() for pp in act_result.pain_points],
                    "total_discovered": act_result.total_discovered,
                    "market_summary": act_result.market_summary,
                    "confidence_score": act_result.confidence_score,
                }
                
                # Write final output to a dedicated file
                (run_dir / "scout_output.json").write_text(json.dumps(act_output_dict, indent=2))
                
                # Store in manifest
                manifest_manager.store_node_output("act", act_output_dict)
                manifest_manager.update_node_status(
                    node_id="act",
                    state="completed",
                    duration_seconds=act_time
                )
                
                # Update run status
                manifest_manager.update_run_status("completed")
                manifest_manager.update_run_metadata({"progress": 100})
            except Exception as e:
                logger.error(f"Error processing act result: {e}")
                act_output_dict = {"error": str(e)}
                
            # Print a sample of the discovered pain points
            if hasattr(act_result, 'pain_points') and act_result.pain_points:
                logger.info("\nSample pain points discovered:")
                for i, pp in enumerate(act_result.pain_points[:3]):  # Show up to 3 pain points
                    logger.info(f"  {i+1}. {pp.description} (Severity: {pp.severity}, Impact: {pp.impact_score})")
                    if pp.evidence:
                        logger.info(f"     Evidence: {pp.evidence[0][:100]}...")
        else:
            logger.info("Skipping act stage...")
            # Get act result from manifest if available
            act_dict = manifest_manager.get_node_output("act") or {}
            logger.info(f"Found {len(act_dict.get('pain_points', []))} pain points in manifest")
        
        # Calculate total execution time
        total_time = time.time() - overall_start_time
        logger.info(f"\n===== SCOUT WORKFLOW COMPLETE =====\n")
        logger.info(f"Total execution time: {total_time:.2f}s")
        logger.info(f"Results stored in: {run_dir}")
        
        # Return final manifest path
        return manifest_path
        
    except Exception as e:
        logger.error(f"Error in Scout workflow: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def run():
    asyncio.run(main_async())


if __name__ == "__main__":
    run()
