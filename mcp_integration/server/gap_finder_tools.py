import json
import os
import hashlib
from typing import Dict, Any, List, Optional
from pathlib import Path

from mcp.types import TextContent

from scout_agent.mcp_integration.server.base import MCPServer
from scout_agent.sources.serp_client import SerpApiClient
from scout_agent.sources.web_content_extractor import WebContentExtractor
from scout_agent.data_cache.file_cache import FileCache
from scout_agent.custom_logging import get_logger
from scout_agent.config import init_config, get_config
from scout_agent.llm.manager import LLMManager
from scout_agent.llm.base import LLMRequest, LLMConfig, LLMBackendType

# Initialize config and logger
init_config()
logger = get_logger("gap_finder_tools")

# Create server instance
server = MCPServer(name="gap-finder-tools")


@server.tool()
async def ping() -> Dict[str, Any]:
    """Health-check tool returning a simple pong payload."""
    payload = {"ok": True, "message": "pong"}
    return {
        "content": [
            TextContent(type="text", text=json.dumps(payload))
        ]
    }


@server.tool()
async def search_links(
    queries: List[str],
    pain_point_id: str = "pp1",
    pain_point_title: str = "Unknown Pain Point",
    num_results: int = 2,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Search for links using the provided search queries.
    
    Args:
        queries: List of search queries
        pain_point_id: ID of the pain point
        pain_point_title: Title of the pain point
        num_results: Number of results to return per query
        use_cache: Whether to use cached search results
        
    Returns:
        Dictionary with search results
    """
    # Initialize SerpAPI client
    client = SerpApiClient()
    
    # Process each query
    pain_point_results = {
        "pain_point_id": pain_point_id,
        "pain_point_title": pain_point_title,
        "query_results": []
    }
    
    for query in queries:
        # Search for the query
        logger.info(f"Searching Google for '{query}'")
        raw_results = client.search_google(query=query, num_results=num_results, use_cache=use_cache)
        search_results = client.normalize_search_results(raw_results)[:num_results]
        
        # Add to results
        query_result = {
            "query": query,
            "results": search_results
        }
        
        pain_point_results["query_results"].append(query_result)
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps(pain_point_results))
        ]
    }


@server.tool()
async def extract_content(
    urls: List[str],
    use_cache: bool = True,
    include_comments: bool = False,
    include_tables: bool = True,
    include_links: bool = True,
    include_images: bool = False
) -> Dict[str, Any]:
    """
    Extract content from the provided URLs.
    
    Args:
        urls: List of URLs to extract content from
        use_cache: Whether to use cached content
        include_comments: Whether to include comments in the extracted content
        include_tables: Whether to include tables in the extracted content
        include_links: Whether to include links in the extracted content
        include_images: Whether to include images in the extracted content
        
    Returns:
        Dictionary with extracted content
    """
    # Initialize Web Content Extractor
    extractor = WebContentExtractor()
    
    # Extract content for each URL
    contents = []
    
    for url in urls:
        # Extract content from the URL
        logger.info(f"Extracting content from: {url}")
        content_result = extractor.extract_content(
            url=url,
            use_cache=use_cache,
            include_comments=include_comments,
            include_tables=include_tables,
            include_links=include_links,
            include_images=include_images
        )
        
        # Add to results
        content_entry = {
            "url": url,
            "success": content_result["success"],
            "content": content_result["content"],
            "error": content_result["error"]
        }
        contents.append(content_entry)
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps({"contents": contents}))
        ]
    }


class ContentTriager:
    def __init__(self, cache_dir: str = None):
        self.llm_manager = LLMManager()
        
        # Set up cache
        if cache_dir is None:
            cache_dir = os.path.join(get_config().cache_dir, "content_triage")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = FileCache(cache_dir)
        
        # Load prompt
        prompt_path = os.path.join(get_config().prompts_dir, "think_triage.prompt")
        with open(prompt_path, "r") as f:
            self.prompt_template = f.read()
    
    def classify_content(self, url: str, content: str, use_cache: bool = True) -> Dict[str, Any]:
        # Generate cache key
        key = hashlib.md5(f"{url}:{content}".encode()).hexdigest()
        
        # Check cache
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                logger.info(f"Using cached classification for {url}")
                return cached
        
        # Prepare input for LLM
        prompt = self.prompt_template.replace("{{CONTENT}}", content)
        
        # Call LLM
        try:
            llm_request = LLMRequest(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1000
            )
            response = self.llm_manager.call_llm(llm_request)
            
            # Parse response
            try:
                classification = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    try:
                        classification = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        return {"success": False, "error": "Failed to parse LLM response as JSON"}
                else:
                    return {"success": False, "error": "Failed to parse LLM response as JSON"}
            
            # Cache result
            result = {"success": True, "classification": classification}
            if use_cache:
                self.cache.set(key, result)
            
            return result
        except Exception as e:
            error_msg = f"Error classifying content: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}


@server.tool()
async def triage_content(
    contents: List[Dict[str, Any]],
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Triage content to determine its relevance and quality.
    
    Args:
        contents: List of content items with URL and content text
        use_cache: Whether to use cached triage results
        
    Returns:
        Dictionary with triage results
    """
    # Initialize Content Triager
    triager = ContentTriager()
    
    # Triage each content item
    triage_results = []
    
    for item in contents:
        url = item.get("url", "")
        content = item.get("content", "")
        
        if not content or not item.get("success", False):
            triage_results.append({
                "url": url,
                "success": False,
                "error": "No content to triage or content extraction failed"
            })
            continue
        
        # Triage the content
        logger.info(f"Triaging content from: {url}")
        triage_result = triager.classify_content(
            url=url,
            content=content,
            use_cache=use_cache
        )
        
        # Add to results
        triage_entry = {
            "url": url,
            "success": triage_result["success"]
        }
        
        if triage_result["success"]:
            triage_entry["classification"] = triage_result["classification"]
        else:
            triage_entry["error"] = triage_result["error"]
        
        triage_results.append(triage_entry)
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps({"triage_results": triage_results}))
        ]
    }


class VendorIdentifier:
    def __init__(self, cache_dir: str = None):
        self.llm_manager = LLMManager()
        
        # Set up cache
        if cache_dir is None:
            cache_dir = os.path.join(get_config().cache_dir, "vendor_identification")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = FileCache(cache_dir)
        
        # Load prompt
        prompt_path = os.path.join(get_config().prompts_dir, "think_vendorid.prompt")
        with open(prompt_path, "r") as f:
            self.prompt_template = f.read()
    
    def identify_vendors(self, url: str, content: str, use_cache: bool = True) -> Dict[str, Any]:
        # Generate cache key
        key = hashlib.md5(f"{url}:{content}".encode()).hexdigest()
        
        # Check cache
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                logger.info(f"Using cached vendor identification for {url}")
                return cached
        
        # Prepare input for LLM
        prompt = self.prompt_template.replace("{{CONTENT}}", content)
        
        # Call LLM
        try:
            llm_request = LLMRequest(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1000
            )
            response = self.llm_manager.call_llm(llm_request)
            
            # Parse response
            try:
                vendors = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    try:
                        vendors = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        return {"success": False, "error": "Failed to parse LLM response as JSON"}
                else:
                    return {"success": False, "error": "Failed to parse LLM response as JSON"}
            
            # Cache result
            result = {"success": True, "vendors": vendors}
            if use_cache:
                self.cache.set(key, result)
            
            return result
        except Exception as e:
            error_msg = f"Error identifying vendors: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}


@server.tool()
async def identify_vendors(
    contents: List[Dict[str, Any]],
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Identify vendors mentioned in the content.
    
    Args:
        contents: List of content items with URL and content text
        use_cache: Whether to use cached vendor identification results
        
    Returns:
        Dictionary with vendor identification results
    """
    # Initialize Vendor Identifier
    identifier = VendorIdentifier()
    
    # Identify vendors in each content item
    vendor_results = []
    
    for item in contents:
        url = item.get("url", "")
        content = item.get("content", "")
        
        if not content or not item.get("success", False):
            vendor_results.append({
                "url": url,
                "success": False,
                "error": "No content to analyze or content extraction failed"
            })
            continue
        
        # Identify vendors in the content
        logger.info(f"Identifying vendors in content from: {url}")
        vendor_result = identifier.identify_vendors(
            url=url,
            content=content,
            use_cache=use_cache
        )
        
        # Add to results
        vendor_entry = {
            "url": url,
            "success": vendor_result["success"]
        }
        
        if vendor_result["success"]:
            vendor_entry["vendors"] = vendor_result["vendors"]
        else:
            vendor_entry["error"] = vendor_result["error"]
        
        vendor_results.append(vendor_entry)
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps({"vendor_results": vendor_results}))
        ]
    }