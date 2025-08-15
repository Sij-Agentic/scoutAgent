import os
import json
import argparse
import asyncio
from pathlib import Path

from scout_agent.custom_logging import get_logger
from scout_agent.agents.scout import ScoutAgent

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
    ap.add_argument("--plan-path", default=None, help="Optional explicit path to plan (scout_plan.json or plan.json)")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    return ap.parse_args()


async def main_async():
    ensure_env_defaults()
    args = parse_args()

    # Resolve plan path
    if args.plan_path:
        plan_path = Path(args.plan_path)
    else:
        run_dir = resolve_run_dir(args.run_id)
        # Prefer agent-specific plan file name
        preferred = run_dir / "scout_plan.json"
        fallback = run_dir / "plan.json"
        plan_path = preferred if preferred.exists() else fallback

    if not plan_path.exists():
        raise FileNotFoundError(f"plan.json not found at: {plan_path}")

    logger.info(f"Executing collect stage using plan: {plan_path}")

    scout = ScoutAgent()
    summary = await scout.collect(plan_path=str(plan_path))

    # Write a simple summary artifact
    out_dir = plan_path.parent
    (out_dir / "collect_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Collect summary written: {out_dir / 'collect_summary.json'}")
    print(json.dumps(summary))


def run():
    asyncio.run(main_async())


if __name__ == "__main__":
    run()
