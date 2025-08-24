import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import requests

from scout_agent.config import get_config
from scout_agent.custom_logging import get_logger
from scout_agent.data_cache.file_cache import FileCache

logger = get_logger("sources.serp_client")


@dataclass
class SerpApiAuth:
    api_key: str


class SerpApiClient:
    """SERP API client for Google search, trends, and more.
    
    Endpoints:
      - search: GET https://serpapi.com/search
      - google_trends: GET https://serpapi.com/google-trends
      - autocomplete: GET https://serpapi.com/search_autocomplete
    """
    
    API_BASE = "https://serpapi.com"
    
    def __init__(self) -> None:
        cfg = get_config()
        api_key = (cfg.api.__dict__.get("serpapi_key") or 
                   self._getenv_default("SCOUT_SERPAPI_KEY"))
        
        if not api_key:
            raise RuntimeError("SERP API key missing: set SCOUT_SERPAPI_KEY")
        
        self.auth = SerpApiAuth(api_key=api_key)
        self.session = requests.Session()
        self.cache = FileCache(base_subdir="serp_cache")
        self.cfg = cfg
    
    def _getenv_default(self, k: str) -> Optional[str]:
        import os
        return os.getenv(k) or None
    
    @staticmethod
    def _hash_key(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()
    
    def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.API_BASE}/{endpoint}"
        
        # Add API key to params
        params["api_key"] = self.auth.api_key
        
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
                logger.warning(f"SERP API request failed (attempt {i+1}/{attempts}): {e}. Retrying in {delay:.2f}s")
                time.sleep(delay)
        
        raise RuntimeError("Unreachable")
    
    def search_google(
        self,
        query: str,
        location: str = "United States",
        language: str = "en",
        num_results: int = 10,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Search Google and return results.
        
        Args:
            query: Search query
            location: Location to search from
            language: Language to search in
            num_results: Number of results to return
            use_cache: Whether to use the cache
            
        Returns:
            Dictionary of search results
        """
        params = {
            "q": query,
            "location": location,
            "hl": language,
            "num": min(num_results, self.cfg.search.max_results),
            "engine": "google",
        }
        
        # Generate cache key
        cache_key = self._hash_key(json.dumps(params, sort_keys=True))
        
        # Try cache first
        ttl = self.cfg.agent.cache_ttl if (use_cache and self.cfg.agent.enable_caching) else None
        if use_cache:
            cached = self.cache.load("search", cache_key, ttl)
            if cached is not None:
                return cached
        
        # Make request
        data = self._request("search", params)
        
        # Save to cache
        self.cache.save("search", cache_key, data)
        return data
    
    def get_google_trends(
        self,
        query: str = None,
        date: str = "now 7-d",
        geo: str = "US",
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Get Google Trends data.
        
        Args:
            query: Optional search query to get related trends
            date: Time range (e.g., "now 1-d", "now 7-d", "today 12-m")
            geo: Geography code (e.g., "US", "GB")
            use_cache: Whether to use the cache
            
        Returns:
            Dictionary of trends data
        """
        params = {
            "engine": "google_trends",
            "date": date,
            "geo": geo,
        }
        
        if query:
            params["q"] = query
        
        # Generate cache key
        cache_key = self._hash_key(json.dumps(params, sort_keys=True))
        
        # Try cache first
        ttl = self.cfg.agent.cache_ttl if (use_cache and self.cfg.agent.enable_caching) else None
        if use_cache:
            cached = self.cache.load("trends", cache_key, ttl)
            if cached is not None:
                return cached
        
        # Make request
        data = self._request("search", params)
        
        # Save to cache
        self.cache.save("trends", cache_key, data)
        return data
    
    def get_autocomplete(
        self,
        query: str,
        engine: str = "google",
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Get search autocomplete suggestions.
        
        Args:
            query: Partial search query
            engine: Search engine to use (google, bing, yahoo)
            use_cache: Whether to use the cache
            
        Returns:
            Dictionary of autocomplete suggestions
        """
        params = {
            "q": query,
            "engine": f"{engine}_autocomplete",
        }
        
        # Generate cache key
        cache_key = self._hash_key(json.dumps(params, sort_keys=True))
        
        # Try cache first
        ttl = self.cfg.agent.cache_ttl if (use_cache and self.cfg.agent.enable_caching) else None
        if use_cache:
            cached = self.cache.load("autocomplete", cache_key, ttl)
            if cached is not None:
                return cached
        
        # Make request
        data = self._request("search", params)
        
        # Save to cache
        self.cache.save("autocomplete", cache_key, data)
        return data
    
    def search_news(
        self,
        query: str,
        location: str = "United States",
        language: str = "en",
        time_period: str = None,  # "h" (hour), "d" (day), "w" (week), "m" (month), "y" (year)
        num_results: int = 10,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Search Google News and return results.
        
        Args:
            query: Search query
            location: Location to search from
            language: Language to search in
            time_period: Time period to search in
            num_results: Number of results to return
            use_cache: Whether to use the cache
            
        Returns:
            Dictionary of news search results
        """
        params = {
            "q": query,
            "location": location,
            "hl": language,
            "num": min(num_results, self.cfg.search.max_results),
            "engine": "google_news",
        }
        
        if time_period:
            params["tbs"] = f"qdr:{time_period}"
        
        # Generate cache key
        cache_key = self._hash_key(json.dumps(params, sort_keys=True))
        
        # Try cache first
        ttl = self.cfg.agent.cache_ttl if (use_cache and self.cfg.agent.enable_caching) else None
        if use_cache:
            cached = self.cache.load("news", cache_key, ttl)
            if cached is not None:
                return cached
        
        # Make request
        data = self._request("search", params)
        
        # Save to cache
        self.cache.save("news", cache_key, data)
        return data
    
    def search_site(
        self,
        query: str,
        site: str,
        num_results: int = 10,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Search within a specific site using Google.
        
        Args:
            query: Search query
            site: Site to search within (e.g., "g2.com")
            num_results: Number of results to return
            use_cache: Whether to use the cache
            
        Returns:
            Dictionary of search results
        """
        site_query = f"{query} site:{site}"
        
        params = {
            "q": site_query,
            "num": min(num_results, self.cfg.search.max_results),
            "engine": "google",
        }
        
        # Generate cache key
        cache_key = self._hash_key(json.dumps(params, sort_keys=True))
        
        # Try cache first
        ttl = self.cfg.agent.cache_ttl if (use_cache and self.cfg.agent.enable_caching) else None
        if use_cache:
            cached = self.cache.load("site_search", cache_key, ttl)
            if cached is not None:
                return cached
        
        # Make request
        data = self._request("search", params)
        
        # Save to cache
        self.cache.save("site_search", cache_key, data)
        return data
    
    def normalize_search_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize search results to a common format.
        
        Args:
            data: Raw search results from SERP API
            
        Returns:
            List of normalized search results
        """
        results = []
        
        # Process organic results
        for result in data.get("organic_results", []):
            results.append({
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "source": "organic",
                "position": result.get("position"),
                "displayed_link": result.get("displayed_link", ""),
            })
        
        # Process news results
        for result in data.get("news_results", []):
            results.append({
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "source": "news",
                "date": result.get("date", ""),
                "source_name": result.get("source", ""),
                "thumbnail": result.get("thumbnail", ""),
            })
        
        # Process knowledge graph
        if "knowledge_graph" in data:
            kg = data["knowledge_graph"]
            results.append({
                "title": kg.get("title", ""),
                "type": kg.get("type", ""),
                "description": kg.get("description", ""),
                "source": "knowledge_graph",
                "attributes": kg.get("attributes", {}),
            })
        
        return results
    
    def normalize_news_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize news results to a common format.
        
        Args:
            data: Raw news results from SERP API
            
        Returns:
            List of normalized news results
        """
        results = []
        
        # Process news results
        for result in data.get("news_results", []):
            results.append({
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "source": "news",
                "date": result.get("date", ""),
                "source_name": result.get("source", ""),
                "thumbnail": result.get("thumbnail", ""),
            })
        
        return results
    
    def normalize_trends_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize trends results to a common format.
        
        Args:
            data: Raw trends results from SERP API
            
        Returns:
            Dictionary of normalized trends results
        """
        normalized = {
            "interest_over_time": [],
            "related_topics": [],
            "related_queries": [],
        }
        
        # Process interest over time
        if "interest_over_time" in data:
            normalized["interest_over_time"] = data["interest_over_time"]
        
        # Process related topics
        if "related_topics" in data:
            topics = []
            for category in ["top", "rising"]:
                if category in data["related_topics"]:
                    for topic in data["related_topics"][category]:
                        topics.append({
                            "title": topic.get("topic_title", ""),
                            "type": topic.get("topic_type", ""),
                            "value": topic.get("value", 0),
                            "category": category,
                        })
            normalized["related_topics"] = topics
        
        # Process related queries
        if "related_queries" in data:
            queries = []
            for category in ["top", "rising"]:
                if category in data["related_queries"]:
                    for query in data["related_queries"][category]:
                        queries.append({
                            "query": query.get("query", ""),
                            "value": query.get("value", 0),
                            "category": category,
                        })
            normalized["related_queries"] = queries
        
        return normalized
