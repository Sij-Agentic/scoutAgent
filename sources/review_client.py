import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from scout_agent.config import get_config
from scout_agent.custom_logging import get_logger
from scout_agent.data_cache.file_cache import FileCache
from scout_agent.sources.serp_client import SerpApiClient

logger = get_logger("sources.review_client")


class ReviewSitesClient:
    """Client for fetching reviews from various review sites via SERP API.
    
    This client uses the SERP API to search within specific review sites
    like G2, Capterra, TrustPilot, etc. using site: search queries.
    """
    
    # Common review sites
    REVIEW_SITES = {
        "g2": "g2.com",
        "capterra": "capterra.com",
        "trustpilot": "trustpilot.com",
        "getapp": "getapp.com",
        "producthunt": "producthunt.com",
        "alternativeto": "alternativeto.net",
        "trustradius": "trustradius.com",
        "softwareadvice": "softwareadvice.com",
    }
    
    def __init__(self) -> None:
        self.cfg = get_config()
        self.serp_client = SerpApiClient()
        self.cache = FileCache(base_subdir="reviews_cache")
    
    @staticmethod
    def _hash_key(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()
    
    def search_reviews(
        self,
        product: str,
        site: str = "g2",
        num_results: int = 10,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Search for reviews of a product on a specific review site.
        
        Args:
            product: Product name to search for
            site: Review site to search on (g2, capterra, trustpilot, etc.)
            num_results: Number of results to return
            use_cache: Whether to use the cache
            
        Returns:
            Dictionary of search results
        """
        # Get the domain for the review site
        if site in self.REVIEW_SITES:
            domain = self.REVIEW_SITES[site]
        else:
            domain = site  # Assume the site is a domain
        
        # Generate cache key
        cache_key = self._hash_key(json.dumps({
            "product": product,
            "site": site,
            "num_results": num_results,
        }, sort_keys=True))
        
        # Try cache first
        ttl = self.cfg.agent.cache_ttl if (use_cache and self.cfg.agent.enable_caching) else None
        if use_cache:
            cached = self.cache.load("search", cache_key, ttl)
            if cached is not None:
                return cached
        
        # Search the review site
        results = self.serp_client.search_site(
            query=f"{product} reviews",
            site=domain,
            num_results=num_results,
            use_cache=use_cache,
        )
        
        # Normalize and save to cache
        normalized = self._normalize_review_results(results, site, product)
        self.cache.save("search", cache_key, normalized)
        return normalized
    
    def compare_products(
        self,
        product1: str,
        product2: str,
        site: str = "g2",
        num_results: int = 10,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Search for comparisons between two products on a review site.
        
        Args:
            product1: First product name
            product2: Second product name
            site: Review site to search on
            num_results: Number of results to return
            use_cache: Whether to use the cache
            
        Returns:
            Dictionary of search results
        """
        # Get the domain for the review site
        if site in self.REVIEW_SITES:
            domain = self.REVIEW_SITES[site]
        else:
            domain = site  # Assume the site is a domain
        
        # Generate cache key
        cache_key = self._hash_key(json.dumps({
            "product1": product1,
            "product2": product2,
            "site": site,
            "num_results": num_results,
        }, sort_keys=True))
        
        # Try cache first
        ttl = self.cfg.agent.cache_ttl if (use_cache and self.cfg.agent.enable_caching) else None
        if use_cache:
            cached = self.cache.load("compare", cache_key, ttl)
            if cached is not None:
                return cached
        
        # Search the review site for comparisons
        results = self.serp_client.search_site(
            query=f"{product1} vs {product2}",
            site=domain,
            num_results=num_results,
            use_cache=use_cache,
        )
        
        # Normalize and save to cache
        normalized = self._normalize_review_results(results, site, f"{product1} vs {product2}")
        self.cache.save("compare", cache_key, normalized)
        return normalized
    
    def get_product_alternatives(
        self,
        product: str,
        site: str = "alternativeto",
        num_results: int = 10,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Search for alternatives to a product on a review site.
        
        Args:
            product: Product name to find alternatives for
            site: Review site to search on (default: alternativeto)
            num_results: Number of results to return
            use_cache: Whether to use the cache
            
        Returns:
            Dictionary of search results
        """
        # Get the domain for the review site
        if site in self.REVIEW_SITES:
            domain = self.REVIEW_SITES[site]
        else:
            domain = site  # Assume the site is a domain
        
        # Generate cache key
        cache_key = self._hash_key(json.dumps({
            "product": product,
            "site": site,
            "num_results": num_results,
        }, sort_keys=True))
        
        # Try cache first
        ttl = self.cfg.agent.cache_ttl if (use_cache and self.cfg.agent.enable_caching) else None
        if use_cache:
            cached = self.cache.load("alternatives", cache_key, ttl)
            if cached is not None:
                return cached
        
        # Search the review site for alternatives
        results = self.serp_client.search_site(
            query=f"{product} alternatives",
            site=domain,
            num_results=num_results,
            use_cache=use_cache,
        )
        
        # Normalize and save to cache
        normalized = self._normalize_review_results(results, site, f"{product} alternatives")
        self.cache.save("alternatives", cache_key, normalized)
        return normalized
    
    def _normalize_review_results(
        self, 
        data: Dict[str, Any], 
        site: str,
        query: str,
    ) -> Dict[str, Any]:
        """Normalize review search results to a common format.
        
        Args:
            data: Raw search results from SERP API
            site: Review site that was searched
            query: Original search query
            
        Returns:
            Dictionary of normalized review results
        """
        # Extract the domain
        if site in self.REVIEW_SITES:
            domain = self.REVIEW_SITES[site]
        else:
            domain = site
        
        # Create normalized structure
        normalized = {
            "query": query,
            "site": site,
            "domain": domain,
            "results": [],
            "fetched_at": int(time.time()),
        }
        
        # Process organic results
        for result in data.get("organic_results", []):
            normalized["results"].append({
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "position": result.get("position"),
                "displayed_link": result.get("displayed_link", ""),
            })
        
        return normalized
    
    def search_multiple_sites(
        self,
        product: str,
        sites: List[str] = None,
        num_results: int = 5,
        use_cache: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Search for a product across multiple review sites.
        
        Args:
            product: Product name to search for
            sites: List of review sites to search on (defaults to g2, capterra, trustpilot)
            num_results: Number of results per site
            use_cache: Whether to use the cache
            
        Returns:
            Dictionary of search results by site
        """
        if sites is None:
            sites = ["g2", "capterra", "trustpilot"]
        
        results = {}
        
        for site in sites:
            site_results = self.search_reviews(
                product=product,
                site=site,
                num_results=num_results,
                use_cache=use_cache,
            )
            results[site] = site_results
            
            # politeness delay
            time.sleep(self.cfg.search.rate_limit_delay)
        
        return results
    
    def fetch_product_reviews(
        self,
        products: List[str],
        sites: List[str] = None,
        num_results: int = 5,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """High-level helper: fetch reviews for multiple products across sites.
        
        Args:
            products: List of product names to search for
            sites: List of review sites to search on
            num_results: Number of results per site
            use_cache: Whether to use the cache
            
        Returns:
            List of review results
        """
        if sites is None:
            sites = ["g2", "capterra"]
        
        results = []
        
        for product in products:
            for site in sites:
                site_results = self.search_reviews(
                    product=product,
                    site=site,
                    num_results=num_results,
                    use_cache=use_cache,
                )
                
                results.append({
                    "type": "product_reviews",
                    "product": product,
                    "site": site,
                    "results": site_results,
                })
                
                # politeness delay
                time.sleep(self.cfg.search.rate_limit_delay)
        
        return results
