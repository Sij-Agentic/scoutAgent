import os
import json
import argparse
import asyncio
from pathlib import Path

from scout_agent.custom_logging import get_logger
from scout_agent.agents.scout import ScoutAgent
from scout_agent.memory.manifest_manager import ManifestManager
import time, datetime

logger = get_logger("scripts.collect_only")


def ensure_env_defaults():
    # Keep defaults lightweight; can be overridden by env
    os.environ.setdefault("SCOUT_LLM_DEFAULT_BACKEND", "gemini")
    os.environ.setdefault("SCOUT_LLM_DEFAULT_MODEL", "gemini-2.5-flash")


def resolve_run_dir(run_id: str) -> Path:
    # project root at scout_agent/
    root = Path(__file__).resolve().parents[2]
    run_dir = root / "data" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run only the Scout collect (non-agent DAG) stage")
    ap.add_argument("--run-id", required=True, help="Run id whose plan.json to execute under data/runs/{run_id}")
    ap.add_argument("--plan-path", default=None, help="Optional explicit path to manifest (run_manifest.json preferred)")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    return ap.parse_args()


async def main_async():
    ensure_env_defaults()
    args = parse_args()
    
    # Record start time for execution metrics
    start_time = time.time()

    # Resolve manifest/plan path
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
        raise FileNotFoundError(f"plan.json not found at: {plan_path}")

    logger.info(f"Executing collect stage using manifest: {plan_path}")

    scout = ScoutAgent()
    summary = await scout.collect(plan_path=str(plan_path))

    # Write a simple summary artifact
    out_dir = plan_path.parent
    # Update manifest stage status using ManifestManager
    try:
        # Create a ManifestManager instance for the manifest file
        manifest_manager = ManifestManager(plan_path)
        
        # Get the manifest to find the correct collect node ID
        manifest = manifest_manager.get_manifest()
        collect_node_id = "collect_reddit"  # Default to collect_reddit
        
        # Find the actual collect node ID from the DAG
        if "dag" in manifest and "nodes" in manifest["dag"]:
            for node in manifest["dag"]["nodes"]:
                if node.get("type") == "tool" and node.get("tool") == "reddit_search_and_fetch_threads":
                    collect_node_id = node.get("id", "collect_reddit")
                    
                    # Clean up duplicate keywords and subreddits in the node
                    if "inputs" in node and "params" in node:
                        # Keep only one copy of keywords and subreddits in inputs
                        if "keywords" in node["inputs"] and "keywords" in node["params"]:
                            node["params"].pop("keywords", None)
                        if "subreddits" in node["inputs"] and "subreddits" in node["params"]:
                            node["params"].pop("subreddits", None)
                    break
        
        # Create detailed output data with useful information about what happened
        # Parse the summary to extract meaningful information
        completed_items = summary.get("completed", [])
        failed_items = summary.get("failed", [])
        
        # Count threads and comments collected
        thread_count = 0
        comment_count = 0
        keywords_found = set()
        subreddits_found = set()
        
        # Extract detailed information from completed items
        for item in completed_items:
            if isinstance(item, dict):
                if "thread_count" in item:
                    thread_count += item.get("thread_count", 0)
                if "comment_count" in item:
                    comment_count += item.get("comment_count", 0)
                if "keyword" in item:
                    keywords_found.add(item.get("keyword"))
                if "subreddit" in item:
                    subreddits_found.add(item.get("subreddit"))
        
        # Actually perform Reddit data collection
        # Get the keywords and subreddits from the manifest
        keywords = manifest.get("keywords", [])
        subreddits = manifest.get("subreddits", [])
        
        # Import the Reddit client
        from scout_agent.sources.reddit_client import RedditClient
        
        # Initialize the Reddit client
        try:
            reddit_client = RedditClient()
            
            # Collect data for each keyword and subreddit combination
            collected_threads = []
            collected_comments = []
            successful_queries = []
            failed_queries = []
            
            # Use a small subset for testing if there are many combinations
            test_keywords = keywords[:2] if len(keywords) > 2 else keywords
            test_subreddits = subreddits[:2] if len(subreddits) > 2 else subreddits
            
            # Get limits from manifest
            per_query_limit = manifest.get("limits", {}).get("per_query_limit", 5)
            comment_depth = manifest.get("limits", {}).get("comment_depth", 2)
            comment_limit = manifest.get("limits", {}).get("comment_limit", 10)
            
            # Use the fetch_conversations method which handles both posts and comments
            try:
                conversations = reddit_client.fetch_conversations(
                    keywords=test_keywords,
                    subreddits=test_subreddits,
                    per_query_limit=per_query_limit,
                    include_comments=True,
                    comment_depth=comment_depth,
                    comment_limit=comment_limit,
                    use_cache=True
                )
                
                # Process the results
                for conversation in conversations:
                    if conversation.get("type") == "reddit_thread":
                        thread_count += 1
                        collected_threads.append(conversation.get("post", {}))
                        comments = conversation.get("comments", [])
                        comment_count += len(comments)
                        collected_comments.extend(comments)
                        
                        # Track which keywords and subreddits had results
                        post = conversation.get("post", {})
                        if post:
                            subreddit = post.get("subreddit")
                            if subreddit:
                                subreddits_found.add(subreddit)
                            
                            # Try to determine which keyword matched this post
                            title = post.get("title", "").lower()
                            selftext = post.get("selftext", "").lower()
                            for kw in test_keywords:
                                if kw.lower() in title or kw.lower() in selftext:
                                    keywords_found.add(kw)
                                    break
                
                # Record successful queries
                if conversations:
                    for kw in test_keywords:
                        for sr in test_subreddits:
                            successful_queries.append({"keyword": kw, "subreddit": sr, "thread_count": thread_count})
                else:
                    for kw in test_keywords:
                        for sr in test_subreddits:
                            failed_queries.append({"keyword": kw, "subreddit": sr, "reason": "No results found"})
                            
            except Exception as e:
                logger.error(f"Error fetching conversations: {e}")
                failed_queries.append({"reason": f"Error fetching conversations: {str(e)}"})
        except Exception as e:
            logger.error(f"Error initializing Reddit client: {e}")
            failed_queries.append({"reason": f"Error initializing Reddit client: {str(e)}"})
        
        # Create detailed output
        detailed_output = {
            "summary": {
                "completed": successful_queries,
                "failed": failed_queries
            },
            "execution_details": {
                "total_threads_collected": thread_count,
                "total_comments_collected": comment_count,
                "keywords_with_results": list(keywords_found) if 'keywords_found' in locals() else [],
                "subreddits_with_results": list(subreddits_found) if 'subreddits_found' in locals() else [],
                "successful_queries": len(successful_queries),
                "failed_queries": len(failed_queries),
                "execution_time": time.time() - start_time,
                "timestamp": datetime.datetime.now().isoformat()
            },
            "threads": collected_threads if 'collected_threads' in locals() else [],
            "comments": collected_comments if 'collected_comments' in locals() else []
        }
        
        # Store the detailed collect stage output data
        manifest_manager.store_node_output(collect_node_id, detailed_output)
        
        # Update the collect node status
        manifest_manager.update_node_status(
            node_id=collect_node_id,
            state="completed",
            duration_seconds=time.time() - start_time
        )
        
        # Update metrics if available
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
            manifest_manager.record_metrics(collect_node_id, metrics)
            
        # Remove any duplicate collect node if it exists and is different from collect_reddit
        if "collect" in manifest["stages"] and collect_node_id != "collect":
            # Get a copy of the stages without the generic collect node
            stages = {k: v for k, v in manifest["stages"].items() if k != "collect"}
            # Update the manifest with the cleaned stages
            manifest["stages"] = stages
            # Save the manifest
            manifest_manager._save()
            
        logger.info(f"Collect stage updated in manifest: {plan_path} (node: {collect_node_id})")
        print(json.dumps({
            "manifest": str(plan_path), 
            "node_id": collect_node_id,
            "threads_collected": detailed_output["execution_details"]["total_threads_collected"],
            "comments_collected": detailed_output["execution_details"]["total_comments_collected"],
            "execution_time": detailed_output["execution_details"]["execution_time"]
        }))
    except Exception as e:
        logger.error(f"Failed to update manifest: {e}")
        print(json.dumps({"manifest": str(plan_path), "summary": summary, "error": str(e)}))



def run():
    asyncio.run(main_async())


if __name__ == "__main__":
    run()
