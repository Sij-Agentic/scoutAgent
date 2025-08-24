import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from scout_agent.config import get_config
from scout_agent.custom_logging import get_logger
from scout_agent.data_cache.file_cache import FileCache

logger = get_logger("sources.hn_client")


@dataclass
class HNAlgoliaAuth:
    api_key: str
    app_id: str = "IX7ESRH38J"  # This is the public app ID for HN search


class HNClient:
    """Hacker News client using the Algolia API.
    
    Endpoints:
      - search: GET https://hn.algolia.com/api/v1/search
    """

    API_BASE = "https://hn.algolia.com/api/v1"
    SEARCH_INDEX = "search"

    def __init__(self) -> None:
        cfg = get_config()
        # The official HN Algolia API doesn't require authentication
        self.auth = None
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
        })
        
        self.cache = FileCache(base_subdir="hn_cache")
        self.cfg = cfg

    def _getenv_default(self, k: str) -> Optional[str]:
        import os
        return os.getenv(k) or None

    @staticmethod
    def _hash_key(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, 
                 data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.API_BASE}/{path}"
        
        # retries
        attempts = self.cfg.search.max_retries
        backoff = self.cfg.search.backoff_factor
        
        for i in range(attempts):
            try:
                resp = self.session.request(
                    method, 
                    url, 
                    params=params, 
                    json=data, 
                    timeout=self.cfg.search.timeout_seconds
                )
                
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"{resp.status_code} {resp.text}")
                
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if i == attempts - 1:
                    raise
                delay = backoff * (2 ** i)
                logger.warning(f"HN request failed (attempt {i+1}/{attempts}): {e}. Retrying in {delay:.2f}s")
                time.sleep(delay)
        
        raise RuntimeError("Unreachable")

    def search_posts(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        sort_by: str = "relevance",
        time_range: str = "year",
        limit: int = 50,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search HN posts and return a normalized list of results.
        
        Args:
            query: Search query
            tags: Optional list of tags to filter by (e.g., ["story", "comment", "poll"])
            sort_by: Sort by "relevance" or "date"
            time_range: Time range to search in ("day", "week", "month", "year", "all")
            limit: Maximum number of results to return
            use_cache: Whether to use the cache
            
        Returns:
            List of normalized HN posts
        """
        # Build request params
        params = {
            "query": query,
            "hitsPerPage": min(limit, self.cfg.search.max_results),
        }
        
        # Add tag filters if present
        if tags:
            tag_param = ",".join(tags)
            params["tags"] = tag_param
        
        # Add sort
        if sort_by == "date":
            params["sortBy"] = "created_at_i"
            params["sortOrder"] = "desc"
        
        # Add time range filter
        if time_range != "all":
            now = int(time.time())
            if time_range == "day":
                params["numericFilters"] = f"created_at_i>{now-86400}"
            elif time_range == "week":
                params["numericFilters"] = f"created_at_i>{now-604800}"
            elif time_range == "month":
                params["numericFilters"] = f"created_at_i>{now-2592000}"
            elif time_range == "year":
                params["numericFilters"] = f"created_at_i>{now-31536000}"
        
        # Generate cache key
        cache_key = self._hash_key(json.dumps({
            "query": query,
            "tags": tags,
            "sort_by": sort_by,
            "time_range": time_range,
            "limit": limit,
        }, sort_keys=True))
        
        # Try cache first
        ttl = self.cfg.agent.cache_ttl if (use_cache and self.cfg.agent.enable_caching) else None
        if use_cache:
            cached = self.cache.load("search", cache_key, ttl)
            if cached is not None:
                return cached
        
        # Make request
        data = self._request("GET", f"{self.SEARCH_INDEX}", params=params)
        
        # Normalize results
        posts = []
        for hit in data.get("hits", []):
            posts.append({
                "id": hit.get("objectID"),
                "title": hit.get("title", ""),
                "text": hit.get("story_text", ""),
                "comment_text": hit.get("comment_text", ""),
                "points": hit.get("points", 0),
                "num_comments": hit.get("num_comments", 0),
                "author": hit.get("author", ""),
                "created_at": hit.get("created_at"),
                "created_at_i": hit.get("created_at_i", 0),
                "url": hit.get("url", ""),
                "type": hit.get("_tags", [""])[0] if hit.get("_tags") else "",
                "parent_id": hit.get("parent_id"),
                "story_id": hit.get("story_id"),
                "hn_url": f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            })
        
        # Save to cache
        self.cache.save("search", cache_key, posts)
        return posts

    def get_item_with_comments(
        self,
        item_id: str,
        limit: int = 50,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Get an item and its comments.
        
        Args:
            item_id: The ID of the item to fetch
            limit: Maximum number of comments to return
            use_cache: Whether to use the cache
            
        Returns:
            Dictionary with the item and its comments
        """
        cache_key = f"{item_id}-l{limit}"
        ttl = self.cfg.agent.cache_ttl if (use_cache and self.cfg.agent.enable_caching) else None
        
        # Try cache first
        if use_cache:
            cached = self.cache.load("threads", cache_key, ttl)
            if cached is not None:
                return cached
        
        # Get the main item
        item_data = self._request("GET", f"items/{item_id}")
        
        # Get comments
        comments = []
        if item_data.get("children", []):
            # Process comments recursively
            def extract_comments(comment_list, depth=0, count=0):
                result = []
                for comment in comment_list:
                    if count >= limit:
                        break
                    
                    if comment.get("text"):  # Only include valid comments
                        result.append({
                            "id": comment.get("id"),
                            "text": comment.get("text", ""),
                            "author": comment.get("author", ""),
                            "created_at": comment.get("created_at"),
                            "created_at_i": comment.get("created_at_i", 0),
                            "parent_id": comment.get("parent_id") or item_id,
                            "story_id": item_id,
                            "points": comment.get("points", 0),
                            "depth": depth,
                            "hn_url": f"https://news.ycombinator.com/item?id={comment.get('id')}"
                        })
                        count += 1
                    
                    # Process children recursively
                    if comment.get("children") and count < limit:
                        child_comments, child_count = extract_comments(
                            comment.get("children", []), 
                            depth + 1, 
                            count
                        )
                        result.extend(child_comments)
                        count = child_count
                
                return result, count
            
            comments, _ = extract_comments(item_data.get("children", []))
        
        # Build normalized result
        normalized = {
            "item": {
                "id": item_data.get("id"),
                "title": item_data.get("title", ""),
                "text": item_data.get("text", ""),
                "points": item_data.get("points", 0),
                "num_comments": len(comments),
                "author": item_data.get("author", ""),
                "created_at": item_data.get("created_at"),
                "created_at_i": item_data.get("created_at_i", 0),
                "url": item_data.get("url", ""),
                "type": item_data.get("type", ""),
                "hn_url": f"https://news.ycombinator.com/item?id={item_data.get('id')}"
            },
            "comments": comments,
            "fetched_at": int(time.time()),
            "schema": "v1",
        }
        
        # Save to cache
        self.cache.save("threads", cache_key, normalized)
        return normalized

    def fetch_conversations(
        self,
        keywords: List[str],
        tags: Optional[List[str]] = None,
        per_query_limit: int = 20,
        include_comments: bool = True,
        comment_limit: int = 50,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """High-level helper: search per keyword and pull threads (optionally with comments).
        
        Args:
            keywords: List of keywords to search for
            tags: Optional list of tags to filter by (e.g., ["story", "comment", "poll"])
            per_query_limit: Maximum number of results per keyword
            include_comments: Whether to include comments
            comment_limit: Maximum number of comments per item
            use_cache: Whether to use the cache
            
        Returns:
            List of HN items with optional comments
        """
        results: List[Dict[str, Any]] = []
        
        for q in keywords:
            posts = self.search_posts(q, tags=tags, limit=per_query_limit, use_cache=use_cache)
            
            for p in posts:
                if include_comments and p.get("id"):
                    thread = self.get_item_with_comments(p["id"], limit=comment_limit, use_cache=use_cache)
                    results.append({
                        "type": "hn_thread", 
                        "item": thread["item"], 
                        "comments": thread["comments"],
                        "source": p.get("hn_url")
                    })
                else:
                    results.append({
                        "type": "hn_item", 
                        "item": p, 
                        "source": p.get("hn_url")
                    })
                
                # politeness delay
                time.sleep(self.cfg.search.rate_limit_delay)
                
        return results
