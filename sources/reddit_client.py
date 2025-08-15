import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from scout_agent.config import get_config
from scout_agent.custom_logging import get_logger
from scout_agent.data_cache.file_cache import FileCache

logger = get_logger("sources.reddit_client")


@dataclass
class RedditAuth:
    client_id: str
    client_secret: str
    user_agent: str


class RedditClient:
    """Minimal Reddit API client using OAuth2 client_credentials for read-only access.

    Endpoints (OAuth):
      - search:   GET https://oauth.reddit.com/search
      - comments: GET https://oauth.reddit.com/comments/{id}
    """

    TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
    API_BASE = "https://oauth.reddit.com"

    def __init__(self) -> None:
        cfg = get_config()
        auth = RedditAuth(
            client_id=(cfg.api.__dict__.get("reddit_client_id") or
                       getenv_default("SCOUT_REDDIT_CLIENT_ID")),
            client_secret=(cfg.api.__dict__.get("reddit_client_secret") or
                           getenv_default("SCOUT_REDDIT_CLIENT_SECRET")),
            user_agent=(getenv_default("SCOUT_REDDIT_USER_AGENT") or cfg.search.user_agent),
        )
        if not (auth.client_id and auth.client_secret):
            raise RuntimeError("Reddit API credentials missing: set SCOUT_REDDIT_CLIENT_ID and SCOUT_REDDIT_CLIENT_SECRET")
        self.auth = auth
        self.token: Optional[str] = None
        self.token_expiry: float = 0.0

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.auth.user_agent})

        self.cache = FileCache(base_subdir="reddit_cache")
        self.cfg = cfg

    # --- Auth helpers ---
    def _ensure_token(self) -> None:
        now = time.time()
        if self.token and now < self.token_expiry - 30:
            return
        data = {
            "grant_type": "client_credentials",
        }
        auth = (self.auth.client_id, self.auth.client_secret)
        headers = {"User-Agent": self.auth.user_agent}
        resp = requests.post(self.TOKEN_URL, data=data, auth=auth, headers=headers, timeout=self.cfg.search.timeout_seconds)
        resp.raise_for_status()
        payload = resp.json()
        self.token = payload.get("access_token")
        expires_in = int(payload.get("expires_in", 3600))
        self.token_expiry = now + expires_in
        logger.debug("Obtained Reddit access token (client_credentials)")

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._ensure_token()
        url = f"{self.API_BASE}{path}"
        headers = {"Authorization": f"Bearer {self.token}", "User-Agent": self.auth.user_agent}
        # retries
        attempts = self.cfg.search.max_retries
        backoff = self.cfg.search.backoff_factor
        for i in range(attempts):
            try:
                resp = self.session.request(method, url, params=params, headers=headers, timeout=self.cfg.search.timeout_seconds)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"{resp.status_code} {resp.text}")
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if i == attempts - 1:
                    raise
                delay = backoff * (2 ** i)
                logger.warning(f"Reddit request failed (attempt {i+1}/{attempts}): {e}. Retrying in {delay:.2f}s")
                time.sleep(delay)
        raise RuntimeError("Unreachable")

    # --- Cache keys ---
    @staticmethod
    def _hash_key(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    # --- Public API ---
    def search_posts(
        self,
        query: str,
        subreddits: Optional[List[str]] = None,
        sort: str = "relevance",
        time_filter: str = "year",
        limit: int = 50,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search posts and return a normalized list of posts.
        """
        q = query
        if subreddits:
            # Restrict search to subreddits using OR, e.g., (subreddit:foo OR subreddit:bar)
            subs = " OR ".join([f"subreddit:{s}" for s in subreddits])
            q = f"({q}) AND ({subs})"
        params = {
            "q": q,
            "sort": sort,
            "t": time_filter,
            "limit": min(limit, self.cfg.search.max_results),
            "type": "link",
            "include_over_18": "on",
            "restrict_sr": False,
        }
        cache_key = self._hash_key(json.dumps({"q": q, "sort": sort, "t": time_filter, "limit": limit}, sort_keys=True))
        ttl = self.cfg.agent.cache_ttl if (use_cache and self.cfg.agent.enable_caching) else None
        if use_cache:
            cached = self.cache.load("search", cache_key, ttl)
            if cached is not None:
                return cached
        data = self._request("GET", "/search", params=params)
        posts = []
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            posts.append({
                "id": d.get("id"),
                "subreddit": d.get("subreddit"),
                "title": d.get("title"),
                "author": d.get("author"),
                "created_utc": d.get("created_utc"),
                "score": d.get("score"),
                "num_comments": d.get("num_comments"),
                "url": d.get("url"),
                "permalink": f"https://www.reddit.com{d.get('permalink', '')}",
                "selftext": d.get("selftext", ""),
            })
        self.cache.save("search", cache_key, posts)
        return posts

    def get_post_with_comments(
        self,
        post_id: str,
        depth: int = 2,
        limit: int = 50,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Fetch a post and comments (flattened) for a given post id."""
        cache_key = f"{post_id}-d{depth}-l{limit}"
        ttl = self.cfg.agent.cache_ttl if (use_cache and self.cfg.agent.enable_caching) else None
        if use_cache:
            cached = self.cache.load("threads", cache_key, ttl)
            if cached is not None:
                return cached
        params = {"depth": depth, "limit": limit, "sort": "top"}
        data = self._request("GET", f"/comments/{post_id}", params=params)
        # data is a list: [post_listing, comments_listing]
        post_listing = (data[0] if data else {})
        post_data = (post_listing.get("data", {}).get("children", [{}])[0].get("data", {}))
        comments_listing = (data[1] if len(data) > 1 else {})
        comments = []
        stack = comments_listing.get("data", {}).get("children", [])
        while stack:
            item = stack.pop(0)
            kind = item.get("kind")
            d = item.get("data", {})
            if kind == "t1":  # comment
                comments.append({
                    "id": d.get("id"),
                    "author": d.get("author"),
                    "body": d.get("body"),
                    "created_utc": d.get("created_utc"),
                    "score": d.get("score"),
                    "parent_id": d.get("parent_id"),
                    "depth": d.get("depth"),
                })
                # enqueue replies
                replies = d.get("replies")
                if isinstance(replies, dict):
                    stack.extend(replies.get("data", {}).get("children", []))
            # ignore more/other kinds
        normalized = {
            "post": {
                "id": post_data.get("id"),
                "subreddit": post_data.get("subreddit"),
                "title": post_data.get("title"),
                "author": post_data.get("author"),
                "created_utc": post_data.get("created_utc"),
                "score": post_data.get("score"),
                "num_comments": post_data.get("num_comments"),
                "url": post_data.get("url"),
                "permalink": f"https://www.reddit.com{post_data.get('permalink', '')}",
                "selftext": post_data.get("selftext", ""),
            },
            "comments": comments,
            "fetched_at": int(time.time()),
            "schema": "v1",
        }
        self.cache.save("threads", cache_key, normalized)
        return normalized

    def fetch_conversations(
        self,
        keywords: List[str],
        subreddits: Optional[List[str]] = None,
        per_query_limit: int = 20,
        include_comments: bool = True,
        comment_depth: int = 2,
        comment_limit: int = 50,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """High-level helper: search per keyword and pull threads (optionally with comments)."""
        results: List[Dict[str, Any]] = []
        for q in keywords:
            posts = self.search_posts(q, subreddits=subreddits, limit=per_query_limit, use_cache=use_cache)
            for p in posts:
                if include_comments and p.get("id"):
                    thread = self.get_post_with_comments(p["id"], depth=comment_depth, limit=comment_limit, use_cache=use_cache)
                    results.append({"type": "reddit_thread", "post": thread["post"], "comments": thread["comments"], "source": p.get("permalink")})
                else:
                    results.append({"type": "reddit_post", "post": p, "source": p.get("permalink")})
                # politeness delay
                time.sleep(self.cfg.search.rate_limit_delay)
        return results


def getenv_default(k: str) -> Optional[str]:
    import os
    return os.getenv(k) or None
