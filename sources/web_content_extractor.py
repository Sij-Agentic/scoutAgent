import hashlib
import json
from typing import Dict, Optional, Any

from trafilatura import fetch_url, extract
from scout_agent.data_cache.file_cache import FileCache
from scout_agent.config import get_config
from scout_agent.custom_logging import get_logger

logger = get_logger("sources.web_content_extractor")


class WebContentExtractor:
    """Tool for extracting content from web pages using trafilatura with caching.
    
    This class provides methods to fetch and extract text content from web pages,
    with built-in caching to avoid redundant downloads and processing.
    """
    
    def __init__(self) -> None:
        self.cfg = get_config()
        self.cache = FileCache(base_subdir="web_content_cache")
    
    @staticmethod
    def _hash_key(s: str) -> str:
        """Create a hash key for caching."""
        return hashlib.sha1(s.encode("utf-8")).hexdigest()
    
    def extract_content(self, 
                       url: str, 
                       use_cache: bool = True,
                       include_comments: bool = False,
                       include_tables: bool = True,
                       include_links: bool = True,
                       include_images: bool = False) -> Dict[str, Any]:
        """Extract content from a web page with caching.
        
        Args:
            url: The URL of the web page to extract content from
            use_cache: Whether to use cached content if available
            include_comments: Whether to include comments in the extraction
            include_tables: Whether to include tables in the extraction
            include_links: Whether to include links in the extraction
            include_images: Whether to include images in the extraction
            
        Returns:
            Dictionary containing the extracted content and metadata
        """
        # Generate cache key based on URL and extraction parameters
        cache_params = {
            "url": url,
            "include_comments": include_comments,
            "include_tables": include_tables,
            "include_links": include_links,
            "include_images": include_images
        }
        cache_key = self._hash_key(json.dumps(cache_params, sort_keys=True))
        
        # Try cache first
        ttl = self.cfg.agent.cache_ttl if (use_cache and self.cfg.agent.enable_caching) else None
        if use_cache:
            cached = self.cache.load("content", cache_key, ttl)
            if cached is not None:
                logger.info(f"Using cached content for URL: {url}")
                return cached
        
        # Fetch and extract content
        try:
            logger.info(f"Fetching content from URL: {url}")
            downloaded = fetch_url(url)
            if not downloaded:
                error_result = {
                    "url": url,
                    "success": False,
                    "error": "Failed to download content",
                    "content": None
                }
                return error_result
            
            # Extract content
            content = extract(
                downloaded,
                include_comments=include_comments,
                include_tables=include_tables,
                include_links=include_links,
                include_images=include_images,
                output_format="text"
            )
            
            # Prepare result
            result = {
                "url": url,
                "success": content is not None,
                "content": content,
                "error": None if content else "Failed to extract content"
            }
            
            # Save to cache
            if result["success"]:
                self.cache.save("content", cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            error_result = {
                "url": url,
                "success": False,
                "error": str(e),
                "content": None
            }
            return error_result