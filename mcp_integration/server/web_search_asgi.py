import json
from typing import Dict, Any, List

from mcp.types import TextContent
from .base import MCPServer

import httpx
from urllib.parse import quote_plus
from bs4 import BeautifulSoup  # Requires beautifulsoup4

server = MCPServer(name="web-search")


async def _ddg_search(query: str, limit: int = 5) -> List[Dict[str, str]]:
    q = quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={q}"
    headers = {"User-Agent": "mcp-web-search/1.0"}

    results: List[Dict[str, str]] = []
    async with httpx.AsyncClient(timeout=10, headers=headers) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.select("a.result__a"):
            title = a.get_text(strip=True)
            href = a.get("href")
            if title and href:
                results.append({"title": title, "url": href})
                if len(results) >= limit:
                    break
    return results


@server.tool()
async def search_web(query: str, limit: int = 5) -> Dict[str, Any]:
    """Search the web and return a list of results with title and url."""
    try:
        rows = await _ddg_search(query, limit)
        payload = {"results": rows}
    except Exception as e:
        payload = {"error": str(e), "results": []}
    return {"content": [TextContent(type="text", text=json.dumps(payload))]}


# Expose ASGI app; run with: uvicorn mcp_integration.server.web_search_asgi:app --host 127.0.0.1 --port 8002
app = server.asgi_app()
