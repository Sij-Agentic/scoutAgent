import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
import time

from trafilatura import extract, fetch_url
from trafilatura.spider import focused_crawler

from scout_agent.data_cache.file_cache import FileCache
from scout_agent.config import get_config
from scout_agent.custom_logging import get_logger

logger = get_logger("sources.web_crawler")


class WebCrawler:
    """Tool for crawling websites and extracting content from all pages within the same domain.
    
    This class provides methods to crawl a website, extract content from all discovered pages,
    and cache the results for future use.
    """
    
    def __init__(self) -> None:
        self.cfg = get_config()
        self.cache = FileCache(base_subdir="web_crawler_cache")
    
    @staticmethod
    def _hash_key(s: str) -> str:
        """Create a hash key for caching."""
        return hashlib.sha1(s.encode("utf-8")).hexdigest()
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing trailing slashes and query parameters."""
        # Remove query parameters
        url = url.split("?")[0]
        # Remove trailing slash
        if url.endswith("/"):
            url = url[:-1]
        return url
    
    def crawl_website(self,
                      url: str,
                      max_pages: int = 20,
                      use_cache: bool = True,
                      include_comments: bool = False,
                      include_tables: bool = True,
                      include_links: bool = True,
                      include_images: bool = False) -> Dict[str, Any]:
        """Crawl a website and extract content from all discovered pages.
        
        Args:
            url: The starting URL for crawling
            max_pages: Maximum number of pages to crawl and extract content from
            use_cache: Whether to use cached content if available
            include_comments: Whether to include comments in the extraction
            include_tables: Whether to include tables in the extraction
            include_links: Whether to include links in the extraction
            include_images: Whether to include images in the extraction
            
        Returns:
            Dictionary containing the extracted content from all pages and metadata
        """
        # Generate cache key based on URL and extraction parameters
        cache_params = {
            "url": url,
            "max_pages": max_pages,
            "include_comments": include_comments,
            "include_tables": include_tables,
            "include_links": include_links,
            "include_images": include_images
        }
        cache_key = self._hash_key(json.dumps(cache_params, sort_keys=True))
        
        # Try cache first
        ttl = self.cfg.agent.cache_ttl if (use_cache and self.cfg.agent.enable_caching) else None
        if use_cache:
            cached = self.cache.load("crawl", cache_key, ttl)
            if cached is not None:
                logger.info(f"Using cached crawl results for URL: {url}")
                return cached
        
        # Perform crawling
        try:
            logger.info(f"Starting crawl from URL: {url}")
            
            # Use trafilatura's focused_crawler to discover pages
            to_visit, known_links = focused_crawler(url, max_seen_urls=max_pages)
            
            # Extract content from discovered pages
            results = {
                "base_url": url,
                "pages": [],
                "success": True,
                "total_pages": len(known_links),
                "extracted_pages": 0,
                "error": None
            }
            
            # Process discovered pages (up to max_pages)
            pages_processed = 0
            for page_url in known_links:
                if pages_processed >= max_pages:
                    break
                
                try:
                    # Fetch and extract content
                    logger.info(f"Fetching content from: {page_url}")
                    downloaded = fetch_url(page_url)
                    if not downloaded:
                        logger.warning(f"Failed to download content from: {page_url}")
                        continue
                    
                    # Extract content
                    content = extract(
                        downloaded,
                        include_comments=include_comments,
                        include_tables=include_tables,
                        include_links=include_links,
                        include_images=include_images,
                        output_format="text"
                    )
                    
                    if content:
                        # Add to results
                        results["pages"].append({
                            "url": page_url,
                            "content": content,
                            "success": True,
                            "error": None
                        })
                        results["extracted_pages"] += 1
                    else:
                        logger.warning(f"No content extracted from: {page_url}")
                    
                    pages_processed += 1
                    
                    # Be polite - add a small delay between requests
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_url}: {str(e)}")
                    # Add failed page to results
                    results["pages"].append({
                        "url": page_url,
                        "content": None,
                        "success": False,
                        "error": str(e)
                    })
                    pages_processed += 1
            
            # Save to cache if we extracted any content
            if results["extracted_pages"] > 0:
                self.cache.save("crawl", cache_key, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error crawling website {url}: {str(e)}")
            error_result = {
                "base_url": url,
                "pages": [],
                "success": False,
                "total_pages": 0,
                "extracted_pages": 0,
                "error": str(e)
            }
            return error_result
    
    def crawl_website_advanced(self,
                             url: str,
                             max_pages: int = 20,
                             max_depth: int = 2,
                             use_cache: bool = True,
                             include_comments: bool = False,
                             include_tables: bool = True,
                             include_links: bool = True,
                             include_images: bool = False) -> Dict[str, Any]:
        """Advanced website crawling with depth control and more options.
        
        Args:
            url: The starting URL for crawling
            max_pages: Maximum number of pages to crawl and extract content from
            max_depth: Maximum link depth to crawl
            use_cache: Whether to use cached content if available
            include_comments: Whether to include comments in the extraction
            include_tables: Whether to include tables in the extraction
            include_links: Whether to include links in the extraction
            include_images: Whether to include images in the extraction
            
        Returns:
            Dictionary containing the extracted content from all pages and metadata
        """
        # Generate cache key based on URL and extraction parameters
        cache_params = {
            "url": url,
            "max_pages": max_pages,
            "max_depth": max_depth,
            "include_comments": include_comments,
            "include_tables": include_tables,
            "include_links": include_links,
            "include_images": include_images
        }
        cache_key = self._hash_key(json.dumps(cache_params, sort_keys=True))
        
        # Try cache first
        ttl = self.cfg.agent.cache_ttl if (use_cache and self.cfg.agent.enable_caching) else None
        if use_cache:
            cached = self.cache.load("crawl_advanced", cache_key, ttl)
            if cached is not None:
                logger.info(f"Using cached advanced crawl results for URL: {url}")
                return cached
        
        # Perform crawling
        try:
            logger.info(f"Starting advanced crawl from URL: {url} with max depth {max_depth}")
            
            # Initialize crawl state
            to_visit = [url]
            known_links: Set[str] = set()
            visited: Set[str] = set()
            pages_by_depth: Dict[int, List[str]] = {0: [url]}
            
            # Extract domain from URL to stay within the same website
            from urllib.parse import urlparse
            base_domain = urlparse(url).netloc
            
            # Results structure
            results = {
                "base_url": url,
                "pages": [],
                "success": True,
                "total_pages": 0,
                "extracted_pages": 0,
                "error": None
            }
            
            # Crawl up to max_depth
            for current_depth in range(max_depth + 1):
                if current_depth not in pages_by_depth:
                    break
                
                logger.info(f"Crawling at depth {current_depth}, found {len(pages_by_depth[current_depth])} pages")
                
                # Process pages at current depth
                next_depth_pages = []
                
                for page_url in pages_by_depth[current_depth]:
                    if page_url in visited:
                        continue
                    
                    if len(visited) >= max_pages:
                        break
                    
                    visited.add(page_url)
                    
                    try:
                        # Fetch and extract content
                        logger.info(f"Fetching content from: {page_url}")
                        downloaded = fetch_url(page_url)
                        if not downloaded:
                            logger.warning(f"Failed to download content from: {page_url}")
                            continue
                        
                        # Extract content
                        content = extract(
                            downloaded,
                            include_comments=include_comments,
                            include_tables=include_tables,
                            include_links=include_links,
                            include_images=include_images,
                            output_format="text"
                        )
                        
                        if content:
                            # Add to results
                            results["pages"].append({
                                "url": page_url,
                                "depth": current_depth,
                                "content": content,
                                "success": True,
                                "error": None
                            })
                            results["extracted_pages"] += 1
                        
                        # If we're not at max depth, discover links for next depth
                        if current_depth < max_depth:
                            # Use trafilatura to get links from this page
                            page_to_visit, page_known_links = focused_crawler(
                                page_url, 
                                max_seen_urls=1,  # We only want to analyze this page
                                max_known_urls=100  # Limit number of links to discover
                            )
                            
                            # Filter links to stay within the same domain
                            for link in page_known_links:
                                if urlparse(link).netloc == base_domain and link not in known_links:
                                    known_links.add(link)
                                    next_depth_pages.append(link)
                        
                        # Be polite - add a small delay between requests
                        time.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing page {page_url}: {str(e)}")
                        # Add failed page to results
                        results["pages"].append({
                            "url": page_url,
                            "depth": current_depth,
                            "content": None,
                            "success": False,
                            "error": str(e)
                        })
                
                # Store pages for next depth
                if next_depth_pages:
                    pages_by_depth[current_depth + 1] = next_depth_pages
                
                # Check if we've reached the maximum number of pages
                if len(visited) >= max_pages:
                    logger.info(f"Reached maximum number of pages ({max_pages})")
                    break
            
            # Update total pages count
            results["total_pages"] = len(known_links)
            
            # Save to cache if we extracted any content
            if results["extracted_pages"] > 0:
                self.cache.save("crawl_advanced", cache_key, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in advanced crawling of website {url}: {str(e)}")
            error_result = {
                "base_url": url,
                "pages": [],
                "success": False,
                "total_pages": 0,
                "extracted_pages": 0,
                "error": str(e)
            }
            return error_result