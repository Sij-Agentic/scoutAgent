import json
from typing import Dict, Any, List

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

import httpx

mcp = FastMCP("reddit-api", host="127.0.0.1", port=8001)


async def _fetch_top(subreddit: str, limit: int = 5, t: str = "day") -> List[Dict[str, Any]]:
    """
    Fetch top posts from a subreddit using Reddit's public JSON endpoint (no auth).
    Note: This uses the public .json endpoints and is subject to rate limiting.
    """
    url = f"https://www.reddit.com/r/{subreddit}/top.json?limit={limit}&t={t}"
    headers = {"User-Agent": "mcp-reddit/1.0"}
    items: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=10, headers=headers) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            items.append(
                {
                    "id": d.get("id"),
                    "title": d.get("title"),
                    "url": d.get("url_overridden_by_dest") or d.get("url"),
                    "permalink": f"https://www.reddit.com{d.get('permalink')}",
                    "score": d.get("score"),
                    "author": d.get("author"),
                    "num_comments": d.get("num_comments"),
                }
            )
    return items


@mcp.tool()
async def reddit_top(subreddit: str, limit: int = 5, timeframe: str = "day") -> Dict[str, Any]:
    """Get top posts from a subreddit (public JSON, no auth)."""
    try:
        posts = await _fetch_top(subreddit=subreddit, limit=limit, t=timeframe)
        payload = {"results": posts}
    except Exception as e:
        payload = {"error": str(e), "results": []}
    return {"content": [TextContent(type="text", text=json.dumps(payload))]}


if __name__ == "__main__":
    # SSE transport; uses SDK defaults for host/port and serves inspector if enabled by SDK
    mcp.run(transport="sse")
