import json
import logging
import time
from typing import Dict, List, Optional, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scout_agent.config import get_config

logger = logging.getLogger(__name__)

class SerperApiClient:
    """Client for Serper API - Google Search API alternative."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Serper API client.
        
        Args:
            api_key: Serper API key (optional, will use config if not provided)
        """
        if api_key is None:
            cfg = get_config()
            api_key = (cfg.api.__dict__.get("serper_key") or 
                      cfg.api.__dict__.get("scout_serper_api_key") or
                      cfg.api.__dict__.get("serper_api_key"))  # Use serper_api_key instead of serpapi_key
            
        if not api_key:
            raise RuntimeError("Serper API key missing: set SCOUT_SERPER_API_KEY, SERPER_KEY or SERPAPI_KEY in config")
            
        self.api_key = api_key
        self.base_url = "https://google.serper.dev"
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def _request(self, endpoint: str, payload: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """Make a request to Serper API with retry logic.
        
        Args:
            endpoint: API endpoint (e.g., 'search', 'news')
            payload: Request payload
            max_retries: Maximum number of retries
            
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Making Serper API request to {endpoint} (attempt {attempt + 1})")
                response = self.session.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Serper API request failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    logger.error(f"Serper API request failed after {max_retries + 1} attempts")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def search_google(self, query: str, location: Optional[str] = None, 
                     num_results: int = 10, **kwargs) -> Dict[str, Any]:
        """Search Google using Serper API.
        
        Args:
            query: Search query
            location: Location for search (optional)
            num_results: Number of results to return (default: 10)
            **kwargs: Additional parameters
            
        Returns:
            Search results
        """
        payload = {
            "q": query,
            "num": num_results
        }
        
        if location:
            payload["location"] = location
            
        # Filter out non-API parameters before adding to payload
        api_kwargs = {k: v for k, v in kwargs.items() if k not in ['use_cache']}
        payload.update(api_kwargs)
        
        return self._request("search", payload)
    
    def get_google_trends(self, query: str, location: Optional[str] = None, 
                         timeframe: str = "now 7-d", **kwargs) -> Dict[str, Any]:
        """Get Google Trends data using Serper API.
        
        Note: Serper API may not have direct trends endpoint, 
        this is a placeholder for compatibility.
        
        Args:
            query: Search query
            location: Location for trends
            timeframe: Time frame for trends
            **kwargs: Additional parameters
            
        Returns:
            Trends data (may need to use search with trends-related query)
        """
        # Serper API doesn't have direct trends endpoint
        # We'll search for trend-related queries instead
        trends_query = f"{query} trends"
        payload = {
            "q": trends_query
        }
        
        if location:
            payload["location"] = location
            
        # Filter out non-API parameters before adding to payload
        api_kwargs = {k: v for k, v in kwargs.items() if k not in ['use_cache']}
        payload.update(api_kwargs)
        
        return self._request("search", payload)
    
    def get_autocomplete(self, query: str, **kwargs) -> Dict[str, Any]:
        """Get autocomplete suggestions.
        
        Note: Serper API may not have direct autocomplete endpoint,
        this is a placeholder for compatibility.
        
        Args:
            query: Partial search query
            **kwargs: Additional parameters
            
        Returns:
            Autocomplete suggestions
        """
        # Serper API doesn't have direct autocomplete endpoint
        # Return empty results for compatibility
        return {"suggestions": []}
    
    def search_news(self, query: str, location: Optional[str] = None,
                   num_results: int = 10, **kwargs) -> Dict[str, Any]:
        """Search Google News using Serper API.
        
        Args:
            query: Search query
            location: Location for search
            num_results: Number of results to return
            **kwargs: Additional parameters
            
        Returns:
            News search results
        """
        payload = {
            "q": query,
            "num": num_results
        }
        
        if location:
            payload["location"] = location
            
        # Filter out non-API parameters before adding to payload
        api_kwargs = {k: v for k, v in kwargs.items() if k not in ['use_cache']}
        payload.update(api_kwargs)
        
        return self._request("news", payload)
    
    def search_site(self, query: str, site: str, **kwargs) -> Dict[str, Any]:
        """Search within a specific site using Serper API.
        
        Args:
            query: Search query
            site: Site to search within
            **kwargs: Additional parameters
            
        Returns:
            Site search results
        """
        site_query = f"site:{site} {query}"
        payload = {
            "q": site_query
        }
        
        # Filter out non-API parameters before adding to payload
        api_kwargs = {k: v for k, v in kwargs.items() if k not in ['use_cache']}
        payload.update(api_kwargs)
        
        return self._request("search", payload)
    
    def normalize_search_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize Serper API search results to match SerpAPI format.
        
        Args:
            results: Raw Serper API results
            
        Returns:
            Normalized results list
        """
        normalized = []
        
        # Extract organic results
        organic_results = results.get("organic", [])
        
        for result in organic_results:
            normalized_result = {
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "position": result.get("position", 0)
            }
            normalized.append(normalized_result)
        
        return normalized
    
    def normalize_news_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize Serper API news results to match SerpAPI format.
        
        Args:
            results: Raw Serper API news results
            
        Returns:
            Normalized news results list
        """
        normalized = []
        
        # Extract news results
        news_results = results.get("news", [])
        
        for result in news_results:
            normalized_result = {
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "date": result.get("date", ""),
                "source": result.get("source", ""),
                "position": result.get("position", 0)
            }
            normalized.append(normalized_result)
        
        return normalized
    
    def normalize_trends_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize trends results to match SerpAPI format.
        
        Args:
            results: Raw trends results
            
        Returns:
            Normalized trends data
        """
        # Since Serper doesn't have direct trends, we return search results
        # formatted as trends data for compatibility
        return {
            "trends": self.normalize_search_results(results)
        }