import os
import json
import argparse
import asyncio
from pathlib import Path

from scout_agent.custom_logging import get_logger
from scout_agent.agents.scout import ScoutAgent, ScoutInput

logger = get_logger("scripts.plan_only")


def ensure_env_defaults():
    # Keep defaults lightweight; can be overridden by env
    os.environ.setdefault("SCOUT_LLM_DEFAULT_BACKEND", "gemini")
    os.environ.setdefault("SCOUT_LLM_DEFAULT_MODEL", "gemini-2.5-flash")


def ensure_run_dirs(run_id: str) -> Path:
    root = Path(__file__).resolve().parents[2]  # project root at scout_agent/
    run_dir = root / "data" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run only the Scout plan phase")
    ap.add_argument("--target-market", required=True, help="Target market description")
    ap.add_argument("--keywords", required=True, help="Comma-separated keywords")
    ap.add_argument("--run-id", default="plan_only_run", help="Run id for artifacts under data/runs/")
    ap.add_argument("--research-scope", default="comprehensive")
    ap.add_argument("--max-pain-points", type=int, default=10)
    return ap.parse_args()


async def main_async():
    ensure_env_defaults()
    args = parse_args()
    run_dir = ensure_run_dirs(args.run_id)

    # Initialize ScoutAgent
    scout = ScoutAgent()
    scout_input = ScoutInput(
        target_market=args.target_market,
        research_scope=args.research_scope,
        max_pain_points=args.max_pain_points,
        keywords=[s.strip() for s in args.keywords.split(",") if s.strip()],
        sources=["reddit"],
    )

    # Plan
    plan = await scout.plan(scout_input)
    out_path = run_dir / "plan.json"
    out_path.write_text(json.dumps(plan, indent=2))
    logger.info(f"Plan written: {out_path}")
    print(str(out_path))


def run():
    asyncio.run(main_async())


if __name__ == "__main__":
    run()
