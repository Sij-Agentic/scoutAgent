#!/usr/bin/env python3
import asyncio
import os
import sys
from typing import Optional

# Ensure both package root (scout_agent) and project root are on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))  # .../scout_agent
PROJECT_ROOT = os.path.dirname(ROOT)  # .../
for p in (PROJECT_ROOT, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm.manager import initialize_llm_backends, get_llm_manager
from config import get_config
from custom_logging import get_logger

# Import agents (prefer package-qualified to support relative imports inside modules)
try:
    from scout_agent.agents.screener import ScreenerAgent
except Exception as e1:
    print(f"[IMPORT][screener via scout_agent.agents] failed: {e1}")
    try:
        from agents.screener import ScreenerAgent  # fallback
    except Exception as e2:
        print(f"[IMPORT][screener via agents] failed: {e2}")
        ScreenerAgent = None

try:
    from scout_agent.agents.validator import ValidatorAgent
except Exception as e1:
    print(f"[IMPORT][validator via scout_agent.agents] failed: {e1}")
    try:
        from agents.validator import ValidatorAgent  # fallback
    except Exception as e2:
        print(f"[IMPORT][validator via agents] failed: {e2}")
        ValidatorAgent = None

logger = get_logger("scripts.smoke_agents")


def ensure_env_defaults():
    # Do not override user's env; only set if missing
    os.environ.setdefault("SCOUT_LLM_DEFAULT_BACKEND", "ollama")
    os.environ.setdefault("SCOUT_LLM_DEFAULT_MODEL", "phi4-mini:latest")


async def run_agent_generate(agent, name: str, prompt: str):
    logger.info(f"[GEN] Running {name}.llm_generate() ...")
    try:
        text = await agent.llm_generate(prompt, temperature=0.2, max_tokens=64)
        print(f"[{name}][GEN] -> {text[:200]}\n")
    except Exception as e:
        print(f"[{name}][GEN][ERROR] {e}")


async def run_agent_stream(agent, name: str, prompt: str):
    logger.info(f"[STREAM] Running {name}.llm_stream_generate() ...")
    try:
        chunks = []
        async for chunk in agent.llm_stream_generate(prompt, temperature=0.2, max_tokens=64):
            chunks.append(chunk)
            if len(chunks) >= 5:  # limit output
                break
        print(f"[{name}][STREAM] -> {''.join(chunks)[:200]}\n")
    except Exception as e:
        print(f"[{name}][STREAM][ERROR] {e}")


async def main():
    ensure_env_defaults()

    cfg = get_config()
    logger.info(f"Default backend from config: {getattr(getattr(cfg, 'llm_routing', None), 'default_backend', None)}")
    # Debug path
    print("sys.path[0:3]=", sys.path[:3])

    # Initialize LLM backends
    await initialize_llm_backends()

    mgr = get_llm_manager()
    print("Backends:", mgr.get_available_backends())
    print("Default backend:", mgr.get_default_backend())

    # Instantiate available agents
    agents = []

    if ScreenerAgent is not None:
        try:
            screener = ScreenerAgent()  # constructor expects optional agent_id
            agents.append((screener, "ScreenerAgent"))
        except Exception as e:
            print(f"[INIT][ScreenerAgent][ERROR] {e}")

    if ValidatorAgent is not None:
        try:
            validator = ValidatorAgent()
            agents.append((validator, "ValidatorAgent"))
        except Exception as e:
            print(f"[INIT][ValidatorAgent][ERROR] {e}")

    if not agents:
        print("No agents available to test. Ensure ScreenerAgent and ValidatorAgent exist and import paths are correct.")
        return

    # Simple prompts
    gen_prompt = "Say 'ok' and identify yourself in one short sentence."
    stream_prompt = "Stream a very short acknowledgment and your backend preference in <10 words.>"

    # Run tests for each agent
    for agent, name in agents:
        await run_agent_generate(agent, name, gen_prompt)
        await run_agent_stream(agent, name, stream_prompt)

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
