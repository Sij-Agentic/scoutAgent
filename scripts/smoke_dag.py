import asyncio
import os
import sys
import time

# Ensure both package root (scout_agent) and project root are on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))  # .../scout_agent
PROJECT_ROOT = os.path.dirname(ROOT)  # .../
for p in (PROJECT_ROOT, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm.manager import get_llm_manager
from config import get_config

# Import agents (prefer package-qualified to support relative imports inside modules)
try:
    from scout_agent.agents.screener import ScreenerAgent
except Exception as e1:
    print(f"[IMPORT][screener via scout_agent.agents] failed: {e1}")
    from agents.screener import ScreenerAgent  # fallback

try:
    from scout_agent.agents.validator import ValidatorAgent
except Exception as e1:
    print(f"[IMPORT][validator via scout_agent.agents] failed: {e1}")
    from agents.validator import ValidatorAgent  # fallback

try:
    from scout_agent.agents.gap_finder import GapFinderAgent
except Exception as e1:
    print(f"[IMPORT][gap_finder via scout_agent.agents] failed: {e1}")
    from agents.gap_finder import GapFinderAgent  # fallback

try:
    from scout_agent.agents.builder import BuilderAgent
except Exception as e1:
    print(f"[IMPORT][builder via scout_agent.agents] failed: {e1}")
    from agents.builder import BuilderAgent  # fallback

try:
    from scout_agent.agents.writer import WriterAgent
except Exception as e1:
    print(f"[IMPORT][writer via scout_agent.agents] failed: {e1}")
    from agents.writer import WriterAgent  # fallback

from agents.base import AgentInput


async def main():
    # Ensure defaults use Ollama unless overridden by env/config
    os.environ.setdefault("SCOUT_LLM_DEFAULT_BACKEND", "ollama")

    cfg = get_config()
    print("DAG Smoke Test - Starting")
    print(f"Default backend from config: {getattr(cfg.llm_routing, 'default_backend', None)}")

    screener = ScreenerAgent()
    validator = ValidatorAgent()
    gapfinder = GapFinderAgent()
    builder = BuilderAgent()
    writer = WriterAgent()

    # Step 1: Screener
    screen_input = AgentInput(
        data=[{"id": "p1", "description": "Slow CI builds", "severity": "high", "market": "devtools"}],
        metadata={"source": "smoke_dag", "stage": "screener"},
        context={"criteria": {"min_severity": "low"}},
    )
    print("[DAG] Screener -> execute")
    s_out = await screener.execute(screen_input)
    print(f"[DAG] Screener success={s_out.success}")

    # Step 2: Validator
    val_input = AgentInput(
        data={"pain_points": [
            {"id": "p1", "description": "Slow CI builds", "market": "devtools"}
        ]},
        metadata={"source": "smoke_dag", "stage": "validator"},
    )
    print("[DAG] Validator -> execute")
    v_out = await validator.execute(val_input)
    print(f"[DAG] Validator success={v_out.success}")

    # Step 3: GapFinder
    gap_input = AgentInput(
        data={
            "validated_pain_points": [
                {"id": "p1", "description": "Slow CI builds", "market": "devtools"}
            ],
            "market_context": "devtools",
        },
        metadata={"source": "smoke_dag", "stage": "gap_finder"},
    )
    print("[DAG] GapFinder -> plan/think subset (execute may expect structured inputs)")
    # Many agents define custom input dataclasses; here we just ensure LLM routes are exercised
    try:
        # If execute signature fits AgentInput, try it
        gf_out = await gapfinder.execute(gap_input)  # type: ignore
        print(f"[DAG] GapFinder success={gf_out.success}")
    except Exception as e:
        print(f"[DAG][GapFinder][WARN] execute failed in smoke path: {e}")

    # Step 4: Builder (minimal path)
    builder_input = AgentInput(
        data={"market_gaps": [{"gap": "CI speed", "opportunity": 0.7}]},
        metadata={"source": "smoke_dag", "stage": "builder"},
    )
    print("[DAG] Builder -> execute")
    try:
        b_out = await builder.execute(builder_input)  # type: ignore
        print(f"[DAG] Builder success={b_out.success}")
    except Exception as e:
        print(f"[DAG][Builder][WARN] execute failed in smoke path: {e}")

    # Step 5: Writer
    writer_input = AgentInput(
        data={"workflow_data": {"summary": "Demo"}},
        metadata={"source": "smoke_dag", "stage": "writer"},
    )
    print("[DAG] Writer -> execute")
    try:
        w_out = await writer.execute(writer_input)  # type: ignore
        print(f"[DAG] Writer success={w_out.success}")
    except Exception as e:
        print(f"[DAG][Writer][WARN] execute failed in smoke path: {e}")

    # Cleanup LLM resources
    print("[DAG] Cleaning up LLM manager...")
    await get_llm_manager().cleanup()
    print("DAG Smoke Test - Done")


if __name__ == "__main__":
    asyncio.run(main())
