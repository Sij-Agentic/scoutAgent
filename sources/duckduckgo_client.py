import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import requests

from scout_agent.config import get_config
from scout_agent.custom_logging import get_logger
from scout_agent.data_cache.file_cache import FileCache

logger = get_logger("sources.duckduckgo_client")


@dataclass
class DuckDuckGoAuth:
    # DuckDuckGo doesn't require authentication for basic search
    pass


class DuckDuckGoClient:
    """DuckDuckGo search client for web search functionality.
    
    This client uses the DuckDuckGo search API to retrieve search results.
    Results are cached locally to avoid repeated API calls during development.
    """
    
    API_BASE = "https://api.duckduckgo.com/"
    
    def __init__(self) -> None:
        cfg = get_config()
        
        self.auth = DuckDuckGoAuth()
        self.session = requests.Session()
        self.cache = FileCache(base_subdir="duckduckgo_cache")
        self.cfg = cfg
    
    def _getenv_default(self, k: str) -> Optional[str]:
        import os
        return os.getenv(k) or None
    
    @staticmethod
    def _hash_key(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()
    
    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        url = self.API_BASE
        
        # retries
        attempts = self.cfg.search.max_retries
        backoff = self.cfg.search.backoff_factor
        
        for i in range(attempts):
            try:
                resp = self.session.get(
                    url, 
                    params=params, 
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
                logger.warning(f"DuckDuckGo API request failed (attempt {i+1}/{attempts}): {e}. Retrying in {delay:.2f}s")
                time.sleep(delay)
        
        raise RuntimeError("Unreachable")
    
    def search(
        self,
        query: str,
        num_results: int = 10,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Search DuckDuckGo and return results.
        
        Args:
            query: Search query
            num_results: Number of results to return
            use_cache: Whether to use the cache
            
        Returns:
            Dictionary containing search results
        """
        # Create cache key
        cache_key = self._hash_key(f"search:{query}:{num_results}")
        
        # Check cache
        if use_cache:
            cached = self.cache.get("search", cache_key)
            if cached:
                logger.info(f"Using cached DuckDuckGo search results for '{query}'")
                return cached
        
        # Prepare parameters
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "no_redirect": "1",
            "skip_disambig": "1",
        }
        
        # Make request
        logger.info(f"Searching DuckDuckGo for '{query}'")
        response = self._request(params)
        
        # Process results to match expected format
        results = {
            "organic_results": [],
            "related_searches": [],
            "answer_box": response.get("AbstractText", ""),
            "knowledge_graph": {},
        }
        
        # Extract results from response
        if "Results" in response:
            for i, result in enumerate(response.get("Results", [])):
                if i >= num_results:
                    break
                    
                results["organic_results"].append({
                    "position": i + 1,
                    "title": result.get("Text", ""),
                    "link": result.get("FirstURL", ""),
                    "snippet": result.get("Text", ""),
                })
        
        # Extract related searches
        if "RelatedTopics" in response:
            for topic in response.get("RelatedTopics", []):
                if "Name" in topic:
                    # This is a category
                    continue
                    
                results["related_searches"].append({
                    "query": topic.get("Text", ""),
                })
        
        # Cache results
        if use_cache:
            self.cache.set("search", cache_key, results)
        
        return results
    
    def get_top_results(
        self,
        query: str,
        num_results: int = 10,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get top search results for a query.
        
        Args:
            query: Search query
            num_results: Number of results to return
            use_cache: Whether to use the cache
            
        Returns:
            List of top search results
        """
        results = self.search(query, num_results=num_results, use_cache=use_cache)
        return results.get("organic_results", [])