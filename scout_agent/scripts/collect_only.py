import os
import json
import argparse
import asyncio
from pathlib import Path

from scout_agent.custom_logging import get_logger
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
        # Log candidate existence for easier debugging
        logger.info(
            "Manifest candidates => run_manifest:%s scout_plan:%s plan:%s",
            str(preferred if preferred.exists() else "MISSING"),
            str(legacy1 if legacy1.exists() else "MISSING"),
            str(legacy2 if legacy2.exists() else "MISSING"),
        )
        plan_path = preferred if preferred.exists() else (legacy1 if legacy1.exists() else legacy2)

    if not plan_path.exists():
        raise FileNotFoundError(
            f"Manifest not found. Expected consolidated manifest at {preferred if 'preferred' in locals() else '[unknown]'} "
            f"or provide --plan-path to an existing run_manifest.json."
        )

    logger.info(f"Executing collect stage using manifest: {plan_path}")
    
    # Delegate to ScoutAgent's collect executor (runs non-agent DAG nodes via CodeExecutionService)
    try:
        from scout_agent.agents.scout import ScoutAgent
        agent = ScoutAgent()
        result = await agent.collect(plan_path=str(plan_path))
        logger.info(f"Collect stage finished. Completed: {result.get('completed')}, Failed: {result.get('failed')}")
        print(json.dumps({
            "manifest": str(plan_path),
            "completed": result.get("completed", []),
            "failed": result.get("failed", []),
            "execution_time": time.time() - start_time
        }))
    except Exception as e:
        logger.error(f"Failed to run collect stage: {e}")
        print(json.dumps({"manifest": str(plan_path), "error": str(e)}))



def run():
    asyncio.run(main_async())


if __name__ == "__main__":
    run()
