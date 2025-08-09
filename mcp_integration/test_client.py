import asyncio
import json
from typing import Any

from mcp_integration.client.multi import MultiMCPClient
from mcp_integration.config import load_server_configs


async def main() -> None:
    servers = load_server_configs()
    client = MultiMCPClient(servers)
    await client.initialize()

    tools = [t.name for t in client.get_all_tools()]
    print("TOOLS:", tools)

    # Call web search if present
    if "search_web" in tools:
        res: Any = await client.call_tool("search_web", {"query": "model context protocol", "limit": 3})
        try:
            content = getattr(res, "content", [])[0].text
            parsed = json.loads(content)
            print("SEARCH_WEB_RESULT:", parsed)
        except Exception:
            print("SEARCH_WEB_RAW:", res)

    # Call reddit top if present
    if "reddit_top" in tools:
        res: Any = await client.call_tool("reddit_top", {"subreddit": "python", "limit": 3, "timeframe": "day"})
        try:
            content = getattr(res, "content", [])[0].text
            parsed = json.loads(content)
            print("REDDIT_TOP_RESULT:", parsed)
        except Exception:
            print("REDDIT_TOP_RAW:", res)

    await client.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
