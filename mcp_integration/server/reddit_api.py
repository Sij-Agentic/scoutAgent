import json
from typing import Dict, Any, List

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

import httpx
import os
import glob
import time
from datetime import datetime, timedelta
from ...sources.reddit_client import RedditClient

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


#@mcp.tool()
async def reddit_top(subreddit: str, limit: int = 5, timeframe: str = "day") -> Dict[str, Any]:
    """Get top posts from a subreddit (public JSON, no auth)."""
    try:
        posts = await _fetch_top(subreddit=subreddit, limit=limit, t=timeframe)
        payload = {"results": posts}
    except Exception as e:
        payload = {"error": str(e), "results": []}
    return {"content": [TextContent(type="text", text=json.dumps(payload))]}


# --- Cache-backed Reddit collection tool (no live fetch) ---

def _cache_root() -> str:
    # Two levels up from this file to reach scout_agent/
    here = os.path.dirname(__file__)  # .../mcp_integration/server
    scout_root = os.path.dirname(os.path.dirname(here))  # .../scout_agent
    return os.path.join(scout_root, "data", "reddit_cache")


def _load_cached_threads(threads_dir: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in glob.glob(os.path.join(threads_dir, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # minimal validation
            if "post" in data:
                items.append(data)
        except Exception:
            continue
    return items


def _months_to_seconds(window: str) -> float:
    if window == "all":
        return float("inf")
    try:
        if window.endswith("m"):
            months = int(window[:-1])
            # approximate month length
            return months * 30 * 24 * 3600
        elif window.endswith("d"):
            days = int(window[:-1])
            return days * 24 * 3600
    except Exception:
        pass
    # default 12m
    return 12 * 30 * 24 * 3600


def _filter_cached(
    items: List[Dict[str, Any]],
    keywords: List[str],
    subreddits: List[str],
    time_window: str,
) -> List[Dict[str, Any]]:
    now = time.time()
    max_age = _months_to_seconds(time_window)
    subs = set([s.lower() for s in (subreddits or [])])
    keys = [k.lower() for k in (keywords or [])]

    def ok_time(created_utc: float) -> bool:
        if not created_utc:
            return True
        if max_age == float("inf"):
            return True
        return (now - float(created_utc)) <= max_age

    out: List[Dict[str, Any]] = []
    for it in items:
        post = it.get("post", {})
        sr = (post.get("subreddit") or "").lower()
        title = (post.get("title") or "").lower()
        selftext = (post.get("selftext") or "").lower()
        created = post.get("created_utc")

        if subs and sr not in subs:
            continue
        if not ok_time(created):
            continue
        if keys:
            text = title + "\n" + selftext
            if not any(k in text for k in keys):
                continue
        out.append(it)
    return out


@mcp.tool()
async def reddit_search_and_fetch_threads(
    keywords: List[str],
    subreddits: List[str] = None,
    time_window: str = "12m",
    per_query_limit: int = 15,
    include_comments: bool = True,
    comment_depth: int = 2,
    comment_limit: int = 50,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Return cached reddit threads matching keywords/subreddits/time_window from data/reddit_cache.
    This is cache-only (no live API calls). Results are normalized like reddit_thread objects.
    """
    try:
        cache_root = _cache_root()
        threads_dir = os.path.join(cache_root, "threads")
        all_items = _load_cached_threads(threads_dir)
        # Normalize subreddit names to bare names (strip any leading 'r/') to match cached post.subreddit
        subs_norm = [
            (s[2:] if isinstance(s, str) and s.lower().startswith("r/") else s)
            for s in (subreddits or [])
        ]
        filtered = _filter_cached(all_items, keywords=keywords, subreddits=subs_norm, time_window=time_window)

        # Fan-out limit per keyword: build per-keyword buckets
        buckets: Dict[str, List[Dict[str, Any]]] = {k: [] for k in (keywords or ["_"])}
        for k in buckets.keys():
            kl = k.lower()
            for it in filtered:
                post = it.get("post", {})
                text = ((post.get("title") or "") + "\n" + (post.get("selftext") or "")).lower()
                if kl in text:
                    buckets[k].append(it)

        # Apply per_query_limit per keyword and dedupe by post id
        seen = set()
        results: List[Dict[str, Any]] = []
        for k, items_k in buckets.items():
            count = 0
            for it in items_k:
                pid = (it.get("post", {}).get("id") or "")
                if pid in seen:
                    continue
                seen.add(pid)
                obj = {
                    "type": "reddit_thread" if include_comments else "reddit_post",
                    "source": it.get("post", {}).get("permalink"),
                    "post": it.get("post"),
                }
                if include_comments:
                    obj["comments"] = it.get("comments", [])
                results.append(obj)
                count += 1
                if count >= max(1, int(per_query_limit)):
                    break

        payload = {"results": results}
    except Exception as e:
        payload = {"error": str(e), "results": []}
    return {"content": [TextContent(type="text", text=json.dumps(payload))]}


#@mcp.tool()
async def reddit_api_search_and_fetch_threads(
    keywords: List[str],
    subreddits: List[str] = None,
    per_query_limit: int = 20,
    include_comments: bool = True,
    comment_depth: int = 2,
    comment_limit: int = 50,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    API-backed fetch using RedditClient with on-disk cache. Mirrors scripts/fetch_reddit.py behavior.
    """
    try:
        client = RedditClient()
        results = client.fetch_conversations(
            keywords=keywords,
            subreddits=subreddits,
            per_query_limit=per_query_limit,
            include_comments=include_comments,
            comment_depth=comment_depth,
            comment_limit=comment_limit,
            use_cache=use_cache,
        )

        normalized: List[Dict[str, Any]] = []
        for r in results:
            obj = {
                "type": "reddit_thread" if include_comments else "reddit_post",
                "source": r.get("post", {}).get("permalink"),
                "post": r.get("post"),
            }
            if include_comments:
                obj["comments"] = r.get("comments", [])
            normalized.append(obj)

        payload = {"results": normalized}
    except Exception as e:
        payload = {"error": str(e), "results": []}
    return {"content": [TextContent(type="text", text=json.dumps(payload))]}


if __name__ == "__main__":
    # SSE transport; uses SDK defaults for host/port and serves inspector if enabled by SDK
    mcp.run(transport="sse")
