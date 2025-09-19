import json
import os
import hashlib
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from scout_agent.mcp_integration.server.base import MCPServer

from scout_agent.sources.serper_client import SerperApiClient
from scout_agent.sources.web_content_extractor import WebContentExtractor
from scout_agent.data_cache.file_cache import FileCache
from scout_agent.custom_logging import get_logger
from scout_agent.config import init_config, get_config
from scout_agent.llm.manager import LLMManager, get_llm_manager, initialize_llm_backends
from scout_agent.llm.base import LLMRequest, LLMConfig, LLMBackendType
from scout_agent.sources.scripts.vendor_research_tool import VendorResearchTool

# Initialize config and logger
init_config()
logger = get_logger("gap_finder_tools")

# Create server instance directly using FastMCP like reddit_api.py
# Use underscore instead of hyphen in the server name to match the client's expectations
mcp = FastMCP("gap_finder", host="127.0.0.1", port=8000)

# Create server variable for import by run_gap_finder_tools_server.py
server = MCPServer(name="gap_finder")
# Use the same mcp instance for both
server._mcp = mcp


@mcp.tool()
async def ping() -> Dict[str, Any]:
    """Health-check tool returning a simple pong payload."""
    payload = {"ok": True, "message": "pong"}
    return {
        "content": [
            TextContent(type="text", text=json.dumps(payload))
        ]
    }


@mcp.tool()
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
    client = SerperApiClient()
    
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


@mcp.tool()
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
            cache_dir = os.path.join(get_config().data_dir, "content_triage_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = FileCache(cache_dir)
        
        # Load prompt
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "prompts", "gap_finder_agent", "think_triage.prompt")
        with open(prompt_path, "r") as f:
            self.prompt_template = f.read()
    
    async def _initialize_llm_backends(self) -> None:
        """Initialize LLM backends directly."""
        from scout_agent.llm.backends import DeepSeekBackend
        logger.info("Starting direct LLM backend initialization in ContentTriager")
        config = get_config()
        
        # Only try to register DeepSeek backend
        if config.api.deepseek_api_key:
            try:
                logger.info("Initializing DeepSeek backend")
                # Create LLMConfig as a dataclass instance
                deepseek_config = LLMConfig(
                    backend_type=LLMBackendType.DEEPSEEK,
                    model_name="deepseek-chat",
                    api_key=config.api.deepseek_api_key,
                    temperature=0.7,
                    max_tokens=4096
                )
                
                deepseek_backend = DeepSeekBackend(deepseek_config)
                await self.llm_manager.register_backend(deepseek_backend, is_default=True)
                logger.info("DeepSeek backend registered successfully")
                return  # Return early if we successfully registered a backend
            except Exception as e:
                logger.error(f"Failed to initialize DeepSeek backend: {e}")
        
        logger.warning("No LLM backends were successfully initialized")
    
    async def classify_content(self, url: str, content: str, use_cache: bool = True) -> Dict[str, Any]:
        # Generate cache key
        key = hashlib.md5(f"{url}:{content}".encode()).hexdigest()
        
        # Check cache
        if use_cache:
            cached = self.cache.load("classification", key, None)
            if cached is not None:
                logger.info(f"Using cached classification for {url}")
                return cached
        
        # Prepare input for LLM (format like standalone version)
        input_text = f"URL: {url}\nTitle: {url.split('/')[-1] if url else 'Unknown'}\nContent: {content}..."  # Truncate content to avoid token limits
        
        # Call LLM
        try:
            llm_request = LLMRequest(
                messages=[{"role": "user", "content": input_text}],
                system_prompt=self.prompt_template,
                temperature=0.2,
                max_tokens=500
            )
            loop = asyncio.get_event_loop()
            
            # Add timeout to prevent indefinite waiting
            try:
                task = loop.create_task(self.llm_manager.generate(llm_request))
                response_obj = await asyncio.wait_for(task, timeout=30.0)  # 30 second timeout
                response = response_obj.content if response_obj.success else ""
            except asyncio.TimeoutError:
                logger.error(f"LLM request timed out for URL: {url}")
                return {"success": False, "error": "LLM request timed out after 30 seconds"}
            
            if not response_obj.success:
                logger.error(f"LLM request failed: {response_obj.error if hasattr(response_obj, 'error') else 'Unknown error'}")
                return {"success": False, "error": f"LLM request failed: {response_obj.error if hasattr(response_obj, 'error') else 'Unknown error'}"}
            
            # Parse response
            try:
                classification = json.loads(response)
            except json.JSONDecodeError as e:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    try:
                        classification = json.loads(json_match.group(0))
                    except json.JSONDecodeError as e2:
                        return {"success": False, "error": "Failed to parse LLM response as JSON"}
                else:
                    return {"success": False, "error": "Failed to parse LLM response as JSON"}
            
            # Cache result
            result = {"success": True, "classification": classification}
            if use_cache:
                self.cache.save("classification", key, result)
            
            return result
        except Exception as e:
            error_msg = f"Error classifying content: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}


@mcp.tool()
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
    # Use global triager to avoid conflicts during rapid calls
    global global_triager
    if global_triager is None:
        global_triager = ContentTriager()
        await global_triager._initialize_llm_backends()
    
    triager = global_triager
    
    # Triage each content item
    triage_results = []
    
    for item in contents:
        url = item.get("url", "")
        content = item.get("content", "")
        
        if not content:
            triage_results.append({
                "url": url,
                "success": False,
                "error": "No content to triage or content extraction failed"
            })
            continue
            
        # If the item was directly provided (not from extract_content), assume success
        if "success" not in item:
            item["success"] = True
        
        # Triage the content
        logger.info(f"Triaging content from: {url}")
        triage_result = await triager.classify_content(
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
            cache_dir = os.path.join(get_config().data_dir, "vendor_identification_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = FileCache(cache_dir)
        
        # Load prompt
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "prompts", "gap_finder_agent", "think_vendorid.prompt")
        with open(prompt_path, "r") as f:
            self.prompt_template = f.read()
    
    async def _initialize_llm_backends(self) -> None:
        """Initialize LLM backends directly."""
        from scout_agent.llm.backends import DeepSeekBackend
        logger.info("Starting direct LLM backend initialization in VendorIdentifier")
        config = get_config()
        
        # Only try to register DeepSeek backend
        if config.api.deepseek_api_key:
            try:
                logger.info("Initializing DeepSeek backend")
                # Create LLMConfig as a dataclass instance
                deepseek_config = LLMConfig(
                    backend_type=LLMBackendType.DEEPSEEK,
                    model_name="deepseek-chat",
                    api_key=config.api.deepseek_api_key,
                    temperature=0.7,
                    max_tokens=4096
                )
                
                deepseek_backend = DeepSeekBackend(deepseek_config)
                await self.llm_manager.register_backend(deepseek_backend, is_default=True)
                logger.info("DeepSeek backend registered successfully")
                return  # Return early if we successfully registered a backend
            except Exception as e:
                logger.error(f"Failed to initialize DeepSeek backend: {e}")
        
        logger.warning("No LLM backends were successfully initialized")
    
    async def identify_vendors(self, url: str, content: str, use_cache: bool = True) -> Dict[str, Any]:
        # Generate cache key
        key = hashlib.md5(f"{url}:{content}".encode()).hexdigest()
        
        logger.info(f"[VENDOR_IDENTIFIER] Starting vendor identification for URL: {url[:100]}...")
        logger.info(f"[VENDOR_IDENTIFIER] Content length: {len(content)} chars, Cache key: {key[:16]}...")
        logger.info(f"[VENDOR_IDENTIFIER] Use cache: {use_cache}")
        
        # Check cache
        if use_cache:
            logger.debug(f"[VENDOR_IDENTIFIER] Checking cache for key: {key}")
            cached = self.cache.load("vendors", key, None)
            if cached is not None:
                logger.info(f"[VENDOR_IDENTIFIER] Using cached vendor identification for {url}")
                logger.debug(f"[VENDOR_IDENTIFIER] Cached result success: {cached.get('success', False)}")
                if cached.get('success') and 'vendors' in cached:
                    vendors_data = cached['vendors']
                    vendor_count = vendors_data.get('vendor_count', 0) if isinstance(vendors_data, dict) else len(vendors_data) if isinstance(vendors_data, list) else 0
                    logger.info(f"[VENDOR_IDENTIFIER] Cached result contains {vendor_count} vendors")
                return cached
            else:
                logger.debug(f"[VENDOR_IDENTIFIER] No cached result found")
        
        # Ensure LLM backends are initialized
        logger.debug(f"[VENDOR_IDENTIFIER] Ensuring LLM backends are initialized")
        await self._initialize_llm_backends()
        
        # Prepare input for LLM - use content directly like triage_content
        input_text = content[:3000]
        logger.info(f"[VENDOR_IDENTIFIER] Prepared LLM input text length: {len(input_text)} chars (truncated from {len(content)})")
        logger.debug(f"[VENDOR_IDENTIFIER] Input text preview: {input_text[:200]}...")
        
        # Call LLM
        try:
            logger.info(f"[VENDOR_IDENTIFIER] Creating LLM request")
            llm_request = LLMRequest(
                messages=[{"role": "user", "content": input_text}],
                system_prompt=self.prompt_template,
                temperature=0.2,
                max_tokens=4000  # Increased from 1000 to handle more comprehensive vendor identification
            )
            loop = asyncio.get_event_loop()
            
            logger.info(f"[VENDOR_IDENTIFIER] Sending request to LLM with 30s timeout")
            # Add timeout to prevent indefinite waiting
            try:
                task = loop.create_task(self.llm_manager.generate(llm_request))
                response_obj = await asyncio.wait_for(task, timeout=30.0)  # 30 second timeout
                response = response_obj.content if response_obj.success else ""
                logger.info(f"[VENDOR_IDENTIFIER] LLM request completed successfully")
                logger.debug(f"[VENDOR_IDENTIFIER] Response length: {len(response)} chars")
                logger.debug(f"[VENDOR_IDENTIFIER] Response preview: {response[:300]}...")
            except asyncio.TimeoutError:
                logger.error(f"[VENDOR_IDENTIFIER] LLM request timed out for URL: {url}")
                return {"success": False, "error": "LLM request timed out after 30 seconds"}
            
            if not response_obj.success:
                error_msg = response_obj.error if hasattr(response_obj, 'error') else 'Unknown error'
                logger.error(f"[VENDOR_IDENTIFIER] LLM request failed: {error_msg}")
                return {"success": False, "error": f"LLM request failed: {error_msg}"}
            
            # Parse response
            logger.info(f"[VENDOR_IDENTIFIER] Parsing LLM response as JSON")
            try:
                vendors = json.loads(response)
                logger.info(f"[VENDOR_IDENTIFIER] JSON parsing successful")
                logger.debug(f"[VENDOR_IDENTIFIER] Parsed vendors keys: {list(vendors.keys()) if isinstance(vendors, dict) else 'not dict'}")
                
                # Log vendor details if available
                if isinstance(vendors, dict):
                    vendor_count = vendors.get('vendor_count', 0)
                    vendors_found = vendors.get('vendors_found', False)
                    vendor_list = vendors.get('vendors', [])
                    
                    logger.info(f"[VENDOR_IDENTIFIER] Vendors found: {vendors_found}")
                    logger.info(f"[VENDOR_IDENTIFIER] Vendor count: {vendor_count}")
                    logger.info(f"[VENDOR_IDENTIFIER] Vendor list length: {len(vendor_list)}")
                    
                    if vendor_list and len(vendor_list) > 0:
                        sample_vendors = [v.get('name', 'Unknown') for v in vendor_list[:3]]
                        logger.info(f"[VENDOR_IDENTIFIER] Sample vendor names: {sample_vendors}")
                        
            except json.JSONDecodeError as e:
                logger.warning(f"[VENDOR_IDENTIFIER] Initial JSON parsing failed: {e}")
                logger.info(f"[VENDOR_IDENTIFIER] Attempting to extract JSON from response using regex")
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    logger.debug(f"[VENDOR_IDENTIFIER] Found JSON match: {json_match.group(0)[:200]}...")
                    try:
                        vendors = json.loads(json_match.group(0))
                        logger.info(f"[VENDOR_IDENTIFIER] Regex-extracted JSON parsing successful")
                    except json.JSONDecodeError as e2:
                        logger.error(f"[VENDOR_IDENTIFIER] Regex-extracted JSON parsing also failed: {e2}")
                        return {"success": False, "error": "Failed to parse LLM response as JSON"}
                else:
                    logger.error(f"[VENDOR_IDENTIFIER] No JSON pattern found in response")
                    logger.debug(f"[VENDOR_IDENTIFIER] Raw response: {response}")
                    return {"success": False, "error": "Failed to parse LLM response as JSON"}
            
            # Cache result
            result = {"success": True, "vendors": vendors}
            logger.info(f"[VENDOR_IDENTIFIER] Vendor identification successful, preparing to cache")
            
            if use_cache:
                logger.debug(f"[VENDOR_IDENTIFIER] Saving result to cache with key: {key}")
                self.cache.save("vendors", key, result)
                logger.info(f"[VENDOR_IDENTIFIER] Result cached successfully")
            else:
                logger.debug(f"[VENDOR_IDENTIFIER] Caching disabled, skipping cache save")
            
            logger.info(f"[VENDOR_IDENTIFIER] Vendor identification completed successfully for {url}")
            return result
            
        except Exception as e:
            error_msg = f"Error identifying vendors: {str(e)}"
            logger.error(f"[VENDOR_IDENTIFIER] Exception during vendor identification: {error_msg}")
            logger.error(f"[VENDOR_IDENTIFIER] Exception details", exc_info=True)
            return {"success": False, "error": error_msg}


@mcp.tool()
async def identify_vendors(
    contents: List[Dict[str, Any]],
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Identify vendors mentioned in the content.
    
    Args:
        contents: List of content items with URL and content text, or extract_content output format
        use_cache: Whether to use cached vendor identification results
        
    Returns:
        Dictionary with vendor identification results
    """
    try:
        logger.info(f"[IDENTIFY_VENDORS] Starting tool call with {len(contents)} content items, use_cache={use_cache}")
        logger.debug(f"[IDENTIFY_VENDORS] Input contents structure: {[{k: type(v).__name__ for k, v in item.items()} for item in contents[:3]]}")
        
        # Use global identifier to avoid conflicts during rapid calls
        global global_identifier
        logger.info("[IDENTIFY_VENDORS] Checking global identifier instance")
        
        if global_identifier is None:
            logger.info("[IDENTIFY_VENDORS] Creating new VendorIdentifier instance")
            global_identifier = VendorIdentifier()
            logger.info("[IDENTIFY_VENDORS] Initializing LLM backends")
            await global_identifier._initialize_llm_backends()
            logger.info("[IDENTIFY_VENDORS] LLM backends initialized successfully")
        else:
            logger.info("[IDENTIFY_VENDORS] Using existing global identifier instance")
        
        identifier = global_identifier
        logger.info("[IDENTIFY_VENDORS] Identifier ready, starting content processing")
        
        # Parse extract_content output format if needed
        processed_contents = []
        logger.info(f"[IDENTIFY_VENDORS] Parsing input format for {len(contents)} items")
        
        for idx, item in enumerate(contents):
            logger.debug(f"[IDENTIFY_VENDORS] Processing input item {idx+1}: keys={list(item.keys())}")
            
            if "content" in item and isinstance(item["content"], list):
                logger.info(f"[IDENTIFY_VENDORS] Item {idx+1} detected as extract_content output format")
                # This is extract_content output format - parse the JSON content
                for content_item in item["content"]:
                    if content_item.get("type") == "text":
                        try:
                            # Parse the JSON string to get individual content items
                            content_data = json.loads(content_item["text"])
                            logger.debug(f"[IDENTIFY_VENDORS] Parsed JSON content keys: {list(content_data.keys()) if isinstance(content_data, dict) else 'not dict'}")
                            
                            if "contents" in content_data:
                                processed_contents.extend(content_data["contents"])
                                logger.info(f"[IDENTIFY_VENDORS] Extended with {len(content_data['contents'])} content items")
                            elif isinstance(content_data, list):
                                processed_contents.extend(content_data)
                                logger.info(f"[IDENTIFY_VENDORS] Extended with {len(content_data)} list items")
                            else:
                                processed_contents.append(content_data)
                                logger.info(f"[IDENTIFY_VENDORS] Appended single content data item")
                        except json.JSONDecodeError as e:
                            logger.warning(f"[IDENTIFY_VENDORS] JSON decode error for item {idx+1}: {e}")
                            # If not JSON, treat as plain text content
                            processed_contents.append({
                                "url": item.get("url", ""),
                                "content": content_item["text"]
                            })
            else:
                logger.info(f"[IDENTIFY_VENDORS] Item {idx+1} detected as standard format")
                # Standard format with url and content fields
                processed_contents.append(item)
        
        logger.info(f"[IDENTIFY_VENDORS] Content parsing complete. Processed {len(processed_contents)} content items from {len(contents)} input items")
        
        # Identify vendors in each content item
        vendor_results = []
        logger.info(f"[IDENTIFY_VENDORS] Starting vendor identification for {len(processed_contents)} processed content items")
        
        for i, item in enumerate(processed_contents):
            try:
                url = item.get("url", "")
                content = item.get("content", "")
                content_preview = content[:200] + "..." if len(content) > 200 else content
                
                logger.info(f"[IDENTIFY_VENDORS] Processing item {i+1}/{len(processed_contents)}")
                logger.info(f"[IDENTIFY_VENDORS] Item {i+1} - URL: {url}")
                logger.info(f"[IDENTIFY_VENDORS] Item {i+1} - Content length: {len(content)} chars")
                logger.debug(f"[IDENTIFY_VENDORS] Item {i+1} - Content preview: {content_preview}")
                
                if not content:
                    logger.warning(f"[IDENTIFY_VENDORS] Skipping item {i+1} - no content text")
                    vendor_results.append({
                        "url": url,
                        "success": False,
                        "error": "No content to analyze or content extraction failed"
                    })
                    continue
                    
                # If the item was directly provided (not from extract_content), assume success
                if "success" not in item:
                    item["success"] = True
                
                # Identify vendors in the content
                logger.info(f"[IDENTIFY_VENDORS] Calling identifier.identify_vendors for item {i+1}")
                
                vendor_result = await identifier.identify_vendors(
                    url=url,
                    content=content,
                    use_cache=use_cache
                )
                
                logger.info(f"[IDENTIFY_VENDORS] Item {i+1} - Vendor identification completed")
                logger.debug(f"[IDENTIFY_VENDORS] Item {i+1} - Result success: {vendor_result.get('success', False)}")
                
                if vendor_result.get("success"):
                    vendors_data = vendor_result.get("vendors", {})
                    vendor_count = vendors_data.get("vendor_count", 0) if isinstance(vendors_data, dict) else len(vendors_data) if isinstance(vendors_data, list) else 0
                    logger.info(f"[IDENTIFY_VENDORS] Item {i+1} - Found {vendor_count} vendors")
                    
                    if vendor_count > 0:
                        vendor_names = []
                        if isinstance(vendors_data, dict) and "vendors" in vendors_data:
                            vendor_names = [v.get("name", "Unknown") for v in vendors_data["vendors"][:3]]
                        elif isinstance(vendors_data, list):
                            vendor_names = [v.get("name", "Unknown") for v in vendors_data[:3]]
                        logger.info(f"[IDENTIFY_VENDORS] Item {i+1} - Sample vendors: {vendor_names}")
                else:
                    logger.warning(f"[IDENTIFY_VENDORS] Item {i+1} - Vendor identification failed: {vendor_result.get('error', 'Unknown error')}")
                
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
                logger.info(f"[IDENTIFY_VENDORS] Item {i+1} - Added to results (total: {len(vendor_results)})")
                
            except Exception as e:
                logger.error(f"[IDENTIFY_VENDORS] Error processing item {i+1} ({url}): {e}")
                logger.error(f"[IDENTIFY_VENDORS] Item {i+1} - Exception details", exc_info=True)
                # Continue processing other items even if one fails
                vendor_results.append({
                    "url": url,
                    "success": False,
                    "error": f"Processing error: {str(e)}"
                })
                continue
        
        # Summary logging
        successful_results = [r for r in vendor_results if r.get("success", False)]
        failed_results = [r for r in vendor_results if not r.get("success", False)]
        total_vendors_found = 0
        
        for result in successful_results:
            vendors_data = result.get("vendors", {})
            if isinstance(vendors_data, dict):
                total_vendors_found += vendors_data.get("vendor_count", 0)
            elif isinstance(vendors_data, list):
                total_vendors_found += len(vendors_data)
        
        logger.info(f"[IDENTIFY_VENDORS] Processing complete - Summary:")
        logger.info(f"[IDENTIFY_VENDORS] - Total items processed: {len(processed_contents)}")
        logger.info(f"[IDENTIFY_VENDORS] - Successful identifications: {len(successful_results)}")
        logger.info(f"[IDENTIFY_VENDORS] - Failed identifications: {len(failed_results)}")
        logger.info(f"[IDENTIFY_VENDORS] - Total vendors found: {total_vendors_found}")
        
        if failed_results:
            logger.warning(f"[IDENTIFY_VENDORS] Failed URLs: {[r['url'] for r in failed_results[:5]]}")
        
        final_result = {
            "content": [
                TextContent(type="text", text=json.dumps({"vendor_results": vendor_results}))
            ]
        }
        logger.info(f"[IDENTIFY_VENDORS] Returning final result with {len(vendor_results)} vendor results")
        return final_result
        
    except Exception as e:
        logger.error(f"[IDENTIFY_VENDORS] Unexpected error in identify_vendors: {e}")
        logger.error(f"[IDENTIFY_VENDORS] Exception details", exc_info=True)
        return {
            "content": [
                TextContent(type="text", text=json.dumps({
                    "error": f"Tool execution failed: {str(e)}",
                    "success": False
                }))
            ]
        }


@mcp.tool()
async def vendor_research(
    vendor_name: str,
    pain_point: str,
    url: str = None,
) -> Dict[str, Any]:
    """
    Conduct deep research on a vendor including their offerings, features, and reviews.
    
    Args:
        vendor_name: The name of the vendor to research
        pain_point: The specific pain point or use case to focus the research on
        url: Optional URL of the vendor's website
        
    Returns:
        Dictionary with vendor research results
    """
    # Initialize Vendor Research Tool
    research_tool = VendorResearchTool()
    
    # Prepare the input as a dictionary for the tool
    tool_input = {
        'vendor_name': vendor_name,
        'pain_point': pain_point
    }
    
    if url:
        tool_input['url'] = url
    
    # Call the tool asynchronously
    logger.info(f"Researching vendor: {vendor_name} for pain point: {pain_point}")
    result = await research_tool.forward(**tool_input)
    
    # Parse the result to ensure it's a proper JSON object
    try:
        result_json = json.loads(result)
        return result_json
    except json.JSONDecodeError as e:
        error_msg = f"Error parsing vendor research result: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
async def vendor_research_batch(
    vendors_list: List[Dict[str, Any]],
    pain_point: str,
    deduplicate: bool = True
) -> Dict[str, Any]:
    """
    Conduct research on multiple vendors with deduplication support.
    
    Args:
        vendors_list: List of vendor dictionaries with 'name' and optional 'url' fields
        pain_point: The specific pain point or use case to focus the research on
        deduplicate: Whether to deduplicate vendors by name before research
        
    Returns:
        Dictionary with batch vendor research results
    """
    logger.info(f"[VENDOR_RESEARCH_BATCH] Starting batch vendor research")
    logger.info(f"[VENDOR_RESEARCH_BATCH] Input vendors_list length: {len(vendors_list)}")
    logger.info(f"[VENDOR_RESEARCH_BATCH] Input vendors_list content: {vendors_list}")
    logger.info(f"[VENDOR_RESEARCH_BATCH] Pain point: {pain_point}")
    logger.info(f"[VENDOR_RESEARCH_BATCH] Deduplicate: {deduplicate}")
    
    # Validate input
    if not vendors_list:
        logger.warning(f"[VENDOR_RESEARCH_BATCH] Empty vendors_list received")
        return {
            "research_results": [],
            "summary": {
                "total_vendors": 0,
                "successful_research": 0,
                "failed_research": 0,
                "deduplication_enabled": deduplicate
            },
            "pain_point": pain_point,
            "error": "Empty vendors list provided"
        }
    
    # Deduplicate vendors if requested
    if deduplicate:
        logger.info(f"[VENDOR_RESEARCH_BATCH] Starting deduplication process")
        seen_names = set()
        unique_vendors = []
        for i, vendor in enumerate(vendors_list):
            # Handle both string and dictionary vendor inputs
            if isinstance(vendor, str):
                vendor_name = vendor.lower().strip()
                logger.debug(f"[VENDOR_RESEARCH_BATCH] Processing string vendor {i}: '{vendor}' -> name: '{vendor_name}'")
            elif isinstance(vendor, dict):
                vendor_name = vendor.get('name', '').lower().strip()
                logger.debug(f"[VENDOR_RESEARCH_BATCH] Processing dict vendor {i}: {vendor} -> name: '{vendor_name}'")
            else:
                logger.warning(f"[VENDOR_RESEARCH_BATCH] Unexpected vendor type {type(vendor)} at index {i}: {vendor}")
                continue
                
            if vendor_name and vendor_name not in seen_names:
                seen_names.add(vendor_name)
                unique_vendors.append(vendor)
                logger.debug(f"[VENDOR_RESEARCH_BATCH] Added unique vendor: {vendor_name}")
            else:
                logger.debug(f"[VENDOR_RESEARCH_BATCH] Skipped duplicate/empty vendor: {vendor_name}")
        
        logger.info(f"[VENDOR_RESEARCH_BATCH] Deduplicated {len(vendors_list)} vendors to {len(unique_vendors)} unique vendors")
        logger.info(f"[VENDOR_RESEARCH_BATCH] Unique vendors: {unique_vendors}")
        vendors_list = unique_vendors
    
    # TEMPORARY: Limit to 2 vendors to reduce costs during testing
    if len(vendors_list) > 2:
        logger.info(f"[VENDOR_RESEARCH_BATCH] COST LIMITING: Reducing {len(vendors_list)} vendors to 2 for cost control during testing")
        vendors_list = vendors_list[:10]
        logger.info(f"[VENDOR_RESEARCH_BATCH] Limited vendor list: {vendors_list}")
    
    # Initialize Vendor Research Tool
    logger.info(f"[VENDOR_RESEARCH_BATCH] Initializing VendorResearchTool")
    research_tool = VendorResearchTool()
    logger.info(f"[VENDOR_RESEARCH_BATCH] VendorResearchTool initialized successfully")
    
    # Research each vendor
    research_results = []
    successful_research = 0
    failed_research = 0
    
    logger.info(f"[VENDOR_RESEARCH_BATCH] Starting individual vendor research for {len(vendors_list)} vendors")
    
    for i, vendor in enumerate(vendors_list):
        logger.info(f"[VENDOR_RESEARCH_BATCH] Processing vendor {i+1}/{len(vendors_list)}: {vendor}")
        try:
            # Handle both string and dictionary vendor inputs
            if isinstance(vendor, str):
                vendor_name = vendor
                vendor_url = None
                logger.debug(f"[VENDOR_RESEARCH_BATCH] String vendor: name='{vendor_name}', url=None")
            elif isinstance(vendor, dict):
                vendor_name = vendor.get('name', '')
                vendor_url = vendor.get('url')
                logger.debug(f"[VENDOR_RESEARCH_BATCH] Dict vendor: name='{vendor_name}', url='{vendor_url}'")
            else:
                logger.error(f"[VENDOR_RESEARCH_BATCH] Invalid vendor type {type(vendor)}: {vendor}")
                failed_research += 1
                continue
            
            logger.info(f"[VENDOR_RESEARCH_BATCH] Vendor {i+1} - name: '{vendor_name}', url: '{vendor_url}'")
            
            if not vendor_name:
                logger.warning(f"[VENDOR_RESEARCH_BATCH] Vendor {i+1} - Skipping vendor with no name: {vendor}")
                continue
            
            # Prepare the input for the research tool
            tool_input = {
                'vendor_name': vendor_name,
                'pain_point': pain_point
            }
            
            if vendor_url:
                tool_input['url'] = vendor_url
            
            logger.info(f"[VENDOR_RESEARCH_BATCH] Vendor {i+1} - Tool input prepared: {tool_input}")
            
            # Call the tool asynchronously with timeout and retry logic
            logger.info(f"[VENDOR_RESEARCH_BATCH] Vendor {i+1} - Starting research for: {vendor_name}")
            result = await _execute_vendor_research_with_retry(research_tool, tool_input, vendor_name, i+1)
            
            if result is None:
                logger.error(f"[VENDOR_RESEARCH_BATCH] Vendor {i+1} - Research failed after retries for: {vendor_name}")
                research_results.append({
                    "vendor_name": vendor_name,
                    "error": "Research failed after multiple retry attempts",
                    "success": False
                })
                failed_research += 1
                continue
                
            logger.info(f"[VENDOR_RESEARCH_BATCH] Vendor {i+1} - Research completed, result length: {len(result) if result else 0}")
            logger.debug(f"[VENDOR_RESEARCH_BATCH] Vendor {i+1} - Raw result: {result[:500]}..." if result and len(result) > 500 else f"[VENDOR_RESEARCH_BATCH] Vendor {i+1} - Raw result: {result}")
            
            # Parse the result
            try:
                result_json = json.loads(result)
                logger.info(f"[VENDOR_RESEARCH_BATCH] Vendor {i+1} - JSON parsing successful")
                logger.debug(f"[VENDOR_RESEARCH_BATCH] Vendor {i+1} - Parsed result keys: {list(result_json.keys()) if isinstance(result_json, dict) else 'Not a dict'}")
                
                research_results.append({
                    "vendor_name": vendor_name,
                    "research_data": result_json,
                    "success": True
                })
                successful_research += 1
                logger.info(f"[VENDOR_RESEARCH_BATCH] Vendor {i+1} - Successfully added to results (total successful: {successful_research})")
                
            except json.JSONDecodeError as e:
                logger.error(f"[VENDOR_RESEARCH_BATCH] Vendor {i+1} - JSON parsing error for {vendor_name}: {str(e)}")
                logger.error(f"[VENDOR_RESEARCH_BATCH] Vendor {i+1} - Raw result that failed parsing: {result}")
                research_results.append({
                    "vendor_name": vendor_name,
                    "error": f"JSON parsing error: {str(e)}",
                    "success": False
                })
                failed_research += 1
                
        except Exception as e:
            logger.error(f"[VENDOR_RESEARCH_BATCH] Vendor {i+1} - Exception during research for {vendor.get('name', 'unknown')}: {str(e)}")
            logger.error(f"[VENDOR_RESEARCH_BATCH] Vendor {i+1} - Exception details", exc_info=True)
            research_results.append({
                "vendor_name": vendor.get('name', 'unknown'),
                "error": str(e),
                "success": False
            })
            failed_research += 1
    
    # Prepare final result
    final_result = {
        "research_results": research_results,
        "summary": {
            "total_vendors": len(vendors_list),
            "successful_research": successful_research,
            "failed_research": failed_research,
            "deduplication_enabled": deduplicate
        },
        "pain_point": pain_point
    }
    
    logger.info(f"[VENDOR_RESEARCH_BATCH] Research completed - Summary:")
    logger.info(f"[VENDOR_RESEARCH_BATCH] - Total vendors processed: {len(vendors_list)}")
    logger.info(f"[VENDOR_RESEARCH_BATCH] - Successful research: {successful_research}")
    logger.info(f"[VENDOR_RESEARCH_BATCH] - Failed research: {failed_research}")
    logger.info(f"[VENDOR_RESEARCH_BATCH] - Research results count: {len(research_results)}")
    logger.info(f"[VENDOR_RESEARCH_BATCH] - Final result keys: {list(final_result.keys())}")
    logger.debug(f"[VENDOR_RESEARCH_BATCH] - Complete final result: {final_result}")
    
    return final_result


async def _execute_vendor_research_with_retry(research_tool, tool_input: Dict[str, Any], vendor_name: str, vendor_index: int, max_retries: int = 3, timeout_seconds: int = 480*4):
    """Execute vendor research with enhanced timeout and retry logic.
    
    Increased timeout from 240s to 480s (8 minutes) to accommodate:
    - Web scraping operations (10-30s)
    - LLM analysis calls (30-90s)
    - Review searches (20-60s)
    - Network latency and retries
    - Complex vendor analysis requiring multiple API calls
    """
    import asyncio
    from httpx import ReadTimeout, ConnectTimeout
    
    for attempt in range(max_retries):
        try:
            logger.info(f"[VENDOR_RESEARCH_BATCH] Vendor {vendor_index} - Executing research for {vendor_name} (attempt {attempt + 1}/{max_retries}, timeout: {timeout_seconds}s)")
            
            # Execute with extended timeout to prevent premature connection closure
            result = await asyncio.wait_for(
                research_tool.forward(**tool_input),
                timeout=timeout_seconds
            )
            
            logger.info(f"[VENDOR_RESEARCH_BATCH] Vendor {vendor_index} - Successfully executed research for {vendor_name} on attempt {attempt + 1}")
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"[VENDOR_RESEARCH_BATCH] Vendor {vendor_index} - Timeout executing research for {vendor_name} (attempt {attempt + 1}/{max_retries}) after {timeout_seconds}s")
            if attempt == max_retries - 1:
                logger.error(f"[VENDOR_RESEARCH_BATCH] Vendor {vendor_index} - Research for {vendor_name} timed out after {max_retries} attempts")
                # Return partial result instead of None to maintain progress
                return {
                    "vendor": {"canonical_name": vendor_name, "website": tool_input.get('url', ''), "disambiguation_notes": "Research timed out"},
                    "business_profile": {"summary": "Analysis unavailable - operation timed out", "value_proposition": "Analysis unavailable - operation timed out", "features": [], "offerings": [], "pricing": "Not available", "faqs": [], "target_customers": []},
                    "pain_point_alignment": {"given_pain_point": tool_input.get('pain_point', ''), "how_addressed": "Analysis unavailable - operation timed out", "notable_gaps": "Analysis unavailable - operation timed out"},
                    "reviews_and_complaints": {"sources": [], "overall_sentiment": "Not available"},
                    "evidence": [{"type": "timeout_error", "title": f"Research timeout for {vendor_name}", "url": tool_input.get('url', ''), "snippet": f"Research operation timed out after {timeout_seconds} seconds"}],
                    "last_updated": "timeout"
                }
            # Longer backoff for timeout errors
            await asyncio.sleep(min(10, 2 ** attempt))  # Cap at 10 seconds
            
        except (ReadTimeout, ConnectTimeout) as e:
            logger.warning(f"[VENDOR_RESEARCH_BATCH] Vendor {vendor_index} - Connection timeout executing research for {vendor_name} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"[VENDOR_RESEARCH_BATCH] Vendor {vendor_index} - Research for {vendor_name} connection failed after {max_retries} attempts: {e}")
                # Return partial result for connection timeouts
                return {
                    "vendor": {"canonical_name": vendor_name, "website": tool_input.get('url', ''), "disambiguation_notes": "Connection timeout"},
                    "business_profile": {"summary": "Analysis unavailable - connection timeout", "value_proposition": "Analysis unavailable - connection timeout", "features": [], "offerings": [], "pricing": "Not available", "faqs": [], "target_customers": []},
                    "pain_point_alignment": {"given_pain_point": tool_input.get('pain_point', ''), "how_addressed": "Analysis unavailable - connection timeout", "notable_gaps": "Analysis unavailable - connection timeout"},
                    "reviews_and_complaints": {"sources": [], "overall_sentiment": "Not available"},
                    "evidence": [{"type": "connection_error", "title": f"Connection timeout for {vendor_name}", "url": tool_input.get('url', ''), "snippet": f"Connection timeout: {str(e)}"}],
                    "last_updated": "connection_timeout"
                }
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["connection closed", "connection", "timeout", "network", "ssl", "certificate"]):
                logger.warning(f"[VENDOR_RESEARCH_BATCH] Vendor {vendor_index} - Connection error executing research for {vendor_name} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"[VENDOR_RESEARCH_BATCH] Vendor {vendor_index} - Research for {vendor_name} connection failed after {max_retries} attempts: {e}")
                    # Return partial result for connection errors
                    return {
                        "vendor": {"canonical_name": vendor_name, "website": tool_input.get('url', ''), "disambiguation_notes": "Connection error"},
                        "business_profile": {"summary": "Analysis unavailable - connection error", "value_proposition": "Analysis unavailable - connection error", "features": [], "offerings": [], "pricing": "Not available", "faqs": [], "target_customers": []},
                        "pain_point_alignment": {"given_pain_point": tool_input.get('pain_point', ''), "how_addressed": "Analysis unavailable - connection error", "notable_gaps": "Analysis unavailable - connection error"},
                        "reviews_and_complaints": {"sources": [], "overall_sentiment": "Not available"},
                        "evidence": [{"type": "connection_error", "title": f"Connection error for {vendor_name}", "url": tool_input.get('url', ''), "snippet": f"Connection error: {str(e)}"}],
                        "last_updated": "connection_error"
                    }
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                # Non-retryable error, log and return partial result
                logger.error(f"[VENDOR_RESEARCH_BATCH] Vendor {vendor_index} - Non-retryable error executing research for {vendor_name}: {e}")
                logger.error(f"[VENDOR_RESEARCH_BATCH] Vendor {vendor_index} - Exception details for {vendor_name}", exc_info=True)
                return {
                    "vendor": {"canonical_name": vendor_name, "website": tool_input.get('url', ''), "disambiguation_notes": "Processing error"},
                    "business_profile": {"summary": f"Analysis failed: {str(e)}", "value_proposition": "Analysis unavailable - processing error", "features": [], "offerings": [], "pricing": "Not available", "faqs": [], "target_customers": []},
                    "pain_point_alignment": {"given_pain_point": tool_input.get('pain_point', ''), "how_addressed": "Analysis unavailable - processing error", "notable_gaps": "Analysis unavailable - processing error"},
                    "reviews_and_complaints": {"sources": [], "overall_sentiment": "Not available"},
                    "evidence": [{"type": "processing_error", "title": f"Processing error for {vendor_name}", "url": tool_input.get('url', ''), "snippet": f"Processing error: {str(e)}"}],
                    "last_updated": "processing_error"
                }
    
    # Fallback - should not reach here
    return None


@mcp.tool()
async def aggregate_gap_analysis(
    research_outputs: List[Dict[str, Any]],
    pain_points: List[Dict[str, Any]],
    merge_strategy: str = "by_pain_point_id",
    output_format: str = "comprehensive_gap_analysis"
) -> Dict[str, Any]:
    """
    Simplified aggregate gap analysis - saves LLM response directly to manifest.
    No post-processing, no heuristics, no fallbacks.
    
    Args:
        research_outputs: List of vendor research results from different pain points
        pain_points: Original pain points data for context
        merge_strategy: Strategy for merging results (unused in simplified version)
        output_format: Format of the output (unused in simplified version)
    
    Returns:
        Raw LLM response as JSON (no transformation)
    """

    from scout_agent.llm.backends import DeepSeekBackend
    
    logger.info(f"[AGGREGATE_GAP_ANALYSIS] Starting simplified gap analysis aggregation")
    logger.info(f"[AGGREGATE_GAP_ANALYSIS] - Research outputs: {len(research_outputs) if isinstance(research_outputs, list) else 'not_list'}")
    logger.info(f"[AGGREGATE_GAP_ANALYSIS] - Pain points: {len(pain_points) if isinstance(pain_points, list) else 'not_list'}")
    
    try:
        # Initialize LLM manager
            llm_manager = LLMManager()
            config = get_config()
            
        # Initialize DeepSeek backend
        if not config.api.deepseek_api_key:
            raise Exception("No DeepSeek API key found in configuration")
        
                    deepseek_config = LLMConfig(
                        backend_type=LLMBackendType.DEEPSEEK,
                        model_name="deepseek-chat",
                        api_key=config.api.deepseek_api_key,
            temperature=0.2,
            max_tokens=8192
                    )
                    
                    deepseek_backend = DeepSeekBackend(deepseek_config)
                    await llm_manager.register_backend(deepseek_backend, is_default=True)
        logger.info("[AGGREGATE_GAP_ANALYSIS] DeepSeek backend initialized")
        
        # Prepare synthesis data
            synthesis_data = {
                "vendor_research_outputs": research_outputs,
                "pain_points": pain_points,
                "analysis_context": {
                "analysis_timestamp": datetime.now().isoformat(),
                "total_research_outputs": len(research_outputs),
                "total_pain_points": len(pain_points)
            }
        }
        
        # Load prompt template
            prompt_path = Path(__file__).parent.parent.parent / "prompts" / "gap_finder_agent" / "collect_aggregate.prompt"
            
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
        
                with open(prompt_path, 'r') as f:
                    prompt_template = f.read()
                
                # Prepare substitutions for the prompt
                substitutions = {
                    "research_outputs_count": str(len(research_outputs)),
                    "pain_points_count": str(len(pain_points)),
                    "market_context": "Software testing and quality assurance tools market",
                    "analysis_scope": "Competitive landscape analysis and gap identification"
                }
                
                # Replace template variables
                prompt_content = prompt_template
                for key, value in substitutions.items():
                    prompt_content = prompt_content.replace(f"{{{{{key}}}}}", value)
                
                # Append the actual research data
                prompt_content += f"\n\nResearch Data to Analyze:\n{json.dumps(synthesis_data, indent=2)}"
                
        # Create LLM request
                llm_request = LLMRequest(
                    messages=[{"role": "user", "content": prompt_content}],
            system_prompt="You are an expert market analyst. Return ONLY valid JSON following the specified format. Do not include any markdown formatting or code blocks.",
                    temperature=0.2
                )
                
                # Generate synthesis using LLM
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] Sending request to LLM")
                llm_response = await llm_manager.generate(llm_request)
                
        if not llm_response or not llm_response.success or not llm_response.content:
            raise Exception(f"LLM request failed: {llm_response.error if llm_response and hasattr(llm_response, 'error') else 'Unknown error'}")
        
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] LLM response received (length: {len(llm_response.content)})")
        
        # Try to parse as JSON directly
        try:
            result = json.loads(llm_response.content)
            logger.info(f"[AGGREGATE_GAP_ANALYSIS] Direct JSON parsing successful")
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
                        import re
                        code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', llm_response.content)
                        if code_block_match:
                logger.info(f"[AGGREGATE_GAP_ANALYSIS] Extracting JSON from code block")
                result = json.loads(code_block_match.group(1).strip())
                                else:
                # Try to find JSON object in the response
                            json_match = re.search(r'\{[\s\S]*\}', llm_response.content)
                            if json_match:
                    logger.info(f"[AGGREGATE_GAP_ANALYSIS] Extracting JSON from response")
                    result = json.loads(json_match.group(0))
                            else:
                    raise Exception("Could not extract valid JSON from LLM response")
        
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] Successfully parsed LLM response")
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] Result keys: {list(result.keys()) if isinstance(result, dict) else 'not_dict'}")
        
        # Return the LLM response directly - no post-processing
        return result
        
    except Exception as e:
        logger.error(f"[AGGREGATE_GAP_ANALYSIS] Error during gap analysis: {str(e)}")
        logger.error(f"[AGGREGATE_GAP_ANALYSIS] Exception details", exc_info=True)
        raise e


# Create global instances to avoid conflicts during rapid calls
global_triager = None
global_identifier = None

# Create ASGI app for uvicorn
app = server.asgi_app()

if __name__ == "__main__":
    # Run with SSE transport
    mcp.run(transport="sse")