import os
import json
import argparse
import asyncio
from pathlib import Path
from typing import List, Optional

from scout_agent.custom_logging import get_logger
from scout_agent.config import get_config
from scout_agent.agents.scout import ScoutAgent, ScoutInput
from scout_agent.mcp_integration.client.multi import MultiMCPClient

logger = get_logger("scripts.main")


def ensure_env_defaults():
    # Respect existing env; set sensible defaults otherwise
    os.environ.setdefault("SCOUT_LLM_DEFAULT_BACKEND", "gemini")
    os.environ.setdefault("SCOUT_LLM_DEFAULT_MODEL", "gemini-2.5-flash")


def ensure_run_dirs(run_id: str) -> Path:
    root = Path(__file__).resolve().parents[2]  # project root at scout_agent/
    run_dir = root / "data" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Scout workflow end-to-end")
    ap.add_argument("--target-market", required=True, help="Target market description")
    ap.add_argument("--keywords", required=True, help="Comma-separated keywords")
    ap.add_argument("--subreddits", default="", help="Comma-separated subreddits")
    ap.add_argument("--run-id", default="dev_run", help="Run id for artifacts under data/runs/")
    ap.add_argument("--per-query-limit", type=int, default=15)
    ap.add_argument("--include-comments", action="store_true")
    ap.add_argument("--comment-depth", type=int, default=2)
    ap.add_argument("--comment-limit", type=int, default=50)
    ap.add_argument("--use-api", action="store_true", help="Use API-backed reddit tool; default cache tool")
    return ap.parse_args()


async def run_collect(mcp: MultiMCPClient, args: argparse.Namespace) -> dict:
    # Choose tool
    tools = mcp.get_all_tools() if hasattr(mcp, "get_all_tools") else []
    prefer_api = args.use_api
    tool_name = "reddit_api_search_and_fetch_threads" if prefer_api else "reddit_search_and_fetch_threads"
    # Fallback if chosen tool not available
    names = [t.name for t in tools] if tools else []
    if names and tool_name not in names:
        tool_name = names[0] if names else tool_name

    keywords: List[str] = [s.strip() for s in args.keywords.split(",") if s.strip()]
    subreddits: Optional[List[str]] = [s.strip() for s in args.subreddits.split(",") if s.strip()] if args.subreddits else None

    res = await mcp.call_tool(tool_name, {
        "keywords": keywords,
        "subreddits": subreddits,
        "per_query_limit": args.per_query_limit,
        "include_comments": bool(args.include_comments),
        "comment_depth": args.comment_depth,
        "comment_limit": args.comment_limit,
        "use_cache": True,
    })
    # MCP result content
    payload = {}
    try:
        content = res.content[0].text if getattr(res, "content", None) else "{}"
        payload = json.loads(content)
    except Exception:
        logger.warning("Failed to parse MCP tool response; using empty payload")
    return payload


async def main_async():
    ensure_env_defaults()
    args = parse_args()
    run_dir = ensure_run_dirs(args.run_id)

    # Initialize ScoutAgent
    scout = ScoutAgent()
    scout_input = ScoutInput(
        target_market=args.target_market,
        keywords=[s.strip() for s in args.keywords.split(",") if s.strip()],
        sources=["reddit"],
    )

    # Plan
    plan = await scout.plan(scout_input)
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=2))
    logger.info(f"Plan written: {run_dir / 'plan.json'}")

    # MCP client for Reddit server
    mcp = MultiMCPClient([
        {"id": "reddit", "url": "http://127.0.0.1:8001/sse", "description": "Reddit MCP"}
    ])
    await mcp.initialize()
    try:
        # Collect via MCP
        collect_payload = await run_collect(mcp, args)
        (run_dir / "reddit_index.json").write_text(json.dumps(collect_payload, indent=2))
        logger.info(f"Collect written: {run_dir / 'reddit_index.json'}")
    finally:
        await mcp.shutdown()

    # Think
    analysis = await scout.think(scout_input)
    (run_dir / "scout_think.json").write_text(json.dumps(analysis, indent=2))
    logger.info(f"Think written: {run_dir / 'scout_think.json'}")

    # Act
    output = await scout.act(scout_input)
    # Convert dataclass output to dict if needed
    try:
        out_dict = {
            "pain_points": [pp.to_dict() for pp in output.pain_points],
            "total_discovered": output.total_discovered,
            "market_summary": output.market_summary,
            "confidence_score": output.confidence_score,
            "sources_used": output.sources_used,
            "research_duration": output.research_duration,
        }
    except Exception:
        out_dict = output if isinstance(output, dict) else {"raw": str(output)}
    (run_dir / "scout_output.json").write_text(json.dumps(out_dict, indent=2))
    logger.info(f"Output written: {run_dir / 'scout_output.json'}")


def run():
    asyncio.run(main_async())


if __name__ == "__main__":
    run()
