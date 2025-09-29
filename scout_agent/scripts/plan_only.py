import os
import json
import argparse
import asyncio
from pathlib import Path

from scout_agent.custom_logging import get_logger
from scout_agent.agents.scout import ScoutAgent, ScoutInput
from scout_agent.memory.manifest_manager import ManifestManager

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
    # Embed run_id into plan for cross-stage consistency
    dag = dict(plan.get("dag") or {})
    dag["run_id"] = args.run_id
    plan["dag"] = dag
    plan["run_id"] = args.run_id

    # Consolidate into a single manifest file using ManifestManager
    manifest_path = run_dir / "run_manifest.json"
    
    try:
        # Create a ManifestManager instance for the manifest file
        manifest_manager = ManifestManager(manifest_path, create_if_missing=True)
        
        # Get the current manifest
        manifest = manifest_manager.get_manifest()
        
        # Update with new plan content
        for key, value in plan.items():
            if key != "stages" and key != "run_metadata":
                manifest[key] = value
        
        # Store plan data in stages section
        manifest_manager.store_node_output("plan", plan)
        
        # Update the plan node status
        manifest_manager.update_node_status(
            node_id="plan",
            state="completed"
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
            manifest_manager.record_metrics("plan", metrics)
        
        # Start the run
        manifest_manager.update_run_status("running")
        
        logger.info(f"Manifest written: {manifest_path}")
    except Exception as e:
        logger.error(f"Failed to update manifest: {e}")
        # Fallback to direct file write
        manifest_path.write_text(json.dumps(plan, indent=2))
    print(str(manifest_path))


def run():
    asyncio.run(main_async())


if __name__ == "__main__":
    run()
