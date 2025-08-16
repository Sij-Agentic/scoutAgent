import json, sys, os
from pathlib import Path
from scout_agent.sources.reddit_client import RedditClient
from typing import Optional

# Load .env if available so SCOUT_REDDIT_* vars are present when running directly
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

run_id = 'scout_smoke_run'
root = Path.cwd()  # project root: scout_agent/
run_dir = root / "data" / "runs" / run_id
plan_path = run_dir / "scout_plan.json"
if not plan_path.exists():
    plan_path = run_dir / "plan.json"
if not plan_path.exists():
    raise FileNotFoundError(f"No plan file found at {plan_path}")

plan = json.loads(plan_path.read_text())
nodes = (plan.get("dag") or {}).get("nodes", [])
collect = next((n for n in nodes if n.get("id") == "collect_reddit"), None)
if not collect:
    raise RuntimeError("collect_reddit node not found in plan")

inputs = (collect.get("inputs") or collect.get("params") or {})
keywords = inputs.get("keywords") or plan.get("keywords") or []
subreddits = inputs.get("subreddits") or plan.get("subreddits") or None
per_query_limit = int(inputs.get("per_query_limit", (plan.get("limits") or {}).get("per_query_limit", 15)))
include_comments = bool(inputs.get("include_comments", True))
comment_depth = int(inputs.get("comment_depth", (plan.get("limits") or {}).get("comment_depth", 2)))
comment_limit = int(inputs.get("comment_limit", (plan.get("limits") or {}).get("comment_limit", 50)))
use_cache = bool(inputs.get("use_cache", True))

print(f"Plan: {plan_path}")
print(f"Keywords: {keywords}")
print(f"Subreddits: {subreddits}")
print(f"per_query_limit={per_query_limit}, include_comments={include_comments}, depth={comment_depth}, comment_limit={comment_limit}, use_cache={use_cache}")

client = RedditClient()
results = client.fetch_conversations(
    keywords=keywords,
    subreddits=subreddits,
    per_query_limit=per_query_limit,
    include_comments=include_comments,
    comment_depth=comment_depth,
    comment_limit=comment_limit,
    use_cache=False,
)

print(f"Warmed cache. Conversations fetched: {len(results)}")
# Optionally: write a simple index to the run dir, useful for quick inspection
out = run_dir / "reddit_index_preview.json"
out.write_text(json.dumps({"count": len(results), "sample": results[:3]}, indent=2))
print(f"Wrote preview: {out}")