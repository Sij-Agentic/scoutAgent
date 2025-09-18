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
from scout_agent.llm.manager import LLMManager
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
        input_text = f"URL: {url}\nTitle: {url.split('/')[-1] if url else 'Unknown'}\nContent: {content[:3000]}..."  # Truncate content to avoid token limits
        
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
        vendors_list = vendors_list[:2]
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
    Aggregate vendor research results for comprehensive gap analysis using LLM synthesis.
    
    Args:
        research_outputs: List of vendor research results from different pain points
        pain_points: Original pain points data for context
        merge_strategy: Strategy for merging results (by_pain_point_id, by_vendor, comprehensive)
        output_format: Format of the output (comprehensive_gap_analysis, summary, detailed)
    
    Returns:
        Aggregated analysis with market gaps, vendor landscape, and opportunities
    """
    
    # TEMPLATE RESOLUTION FIX: Handle template strings in research_outputs
    logger.info(f"[AGGREGATE_GAP_ANALYSIS] Raw research_outputs type: {type(research_outputs)}")
    logger.info(f"[AGGREGATE_GAP_ANALYSIS] Raw research_outputs content: {research_outputs[:2] if isinstance(research_outputs, list) else 'not list'}")
    
    # Check if research_outputs contains template strings that need resolution
    resolved_research_outputs = []
    for i, output in enumerate(research_outputs):
        if isinstance(output, str) and output.startswith('${'):
            logger.warning(f"[AGGREGATE_GAP_ANALYSIS] Found unresolved template string: {output}")
            logger.warning(f"[AGGREGATE_GAP_ANALYSIS] This indicates a template resolution issue in the workflow")
            # For now, skip this output as it's not resolved
            continue
        else:
            resolved_research_outputs.append(output)
    
    research_outputs = resolved_research_outputs
    logger.info(f"[AGGREGATE_GAP_ANALYSIS] After template resolution - research_outputs length: {len(research_outputs)}")

    logger.info("[AGGREGATE_GAP_ANALYSIS] Starting gap analysis aggregation")
    logger.info(f"[AGGREGATE_GAP_ANALYSIS] Input research_outputs length: {len(research_outputs)}")
    logger.info(f"[AGGREGATE_GAP_ANALYSIS] Input pain_points length: {len(pain_points)}")
    logger.info(f"[AGGREGATE_GAP_ANALYSIS] Merge strategy: {merge_strategy}")
    logger.info(f"[AGGREGATE_GAP_ANALYSIS] Output format: {output_format}")
    logger.debug(f"[AGGREGATE_GAP_ANALYSIS] Research outputs structure: {[{k: type(v).__name__ for k, v in item.items()} if isinstance(item, dict) else f'list_with_{len(item)}_items' for item in research_outputs[:3]]}")
    logger.debug(f"[AGGREGATE_GAP_ANALYSIS] Pain points structure: {[{k: type(v).__name__ for k, v in item.items()} for item in pain_points[:3]]}")
    
    try:
        # Initialize aggregation containers
        all_vendors = []
        vendor_by_pain_point = {}
        pain_point_coverage = {}
        
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] Initialized aggregation containers")
        
        # Process each research output
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] Processing {len(research_outputs)} research outputs")
        
        for i, research_output in enumerate(research_outputs):
            logger.info(f"[AGGREGATE_GAP_ANALYSIS] Processing research output {i+1}/{len(research_outputs)}")
            logger.debug(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} type: {type(research_output)}")
            logger.debug(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} keys: {list(research_output.keys()) if isinstance(research_output, dict) else 'not dict'}")
            
            if not research_output or not isinstance(research_output, (dict, list)):
                logger.warning(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} - Skipping invalid research output: {type(research_output)}")
                continue
                
            # Extract batch results - handle both direct array and wrapped object formats
            if isinstance(research_output, list):
                # Handle direct array from template (e.g., ${vendor_research_pp1_output.research_results})
                batch_results = research_output
                logger.info(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} - Processing direct array with {len(batch_results)} items")
            elif isinstance(research_output, dict):
                # Handle wrapped object format - try both field names for compatibility
                batch_results = research_output.get("batch_results", research_output.get("research_results", []))
                logger.info(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} - Processing wrapped object with {len(batch_results)} items")
                if "research_results" in research_output and "batch_results" not in research_output:
                    logger.info(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} - Using 'research_results' field (vendor_research_batch format)")
            else:
                batch_results = []
                logger.warning(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} - Unexpected research_output type: {type(research_output)}")
            
            logger.debug(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} - Batch results structure: {[{k: type(v).__name__ for k, v in result.items()} for result in batch_results[:3]] if batch_results else 'empty'}")
            
            successful_results = 0
            for j, result in enumerate(batch_results):
                logger.debug(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} - Processing batch result {j+1}/{len(batch_results)}")
                logger.debug(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} - Result {j+1} success: {result.get('success', False)}")
                
                if not result.get("success", False):
                    logger.warning(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} - Result {j+1} - Skipping unsuccessful result: {result.get('error', 'No error message')}")
                    continue
                    
                # Handle both vendor_data (expected format) and research_data (vendor_research_batch format)
                vendor_data = result.get("vendor_data", {})
                if not vendor_data and "research_data" in result:
                    # Convert vendor_research_batch format to expected format
                    research_data = result.get("research_data", {})
                    vendor_name_from_result = result.get("vendor_name", "Unknown")
                    
                    # Flatten the research_data structure
                    vendor_data = {
                        "name": vendor_name_from_result,
                        "business_profile": research_data.get("business_profile", {}),
                        "features": research_data.get("features", {}),
                        "pricing": research_data.get("pricing", {}),
                        "target_customers": research_data.get("target_customers", {}),
                        "pain_point_address": research_data.get("pain_point_address", {}),
                        "strengths": research_data.get("strengths", {}),
                        "limitations": research_data.get("limitations", {}),
                        "reviews": research_data.get("reviews", {}),
                        "evidence": research_data.get("evidence", {})
                    }
                    logger.info(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} - Result {j+1} - Converted research_data format for vendor: {vendor_name_from_result}")
                
                logger.debug(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} - Result {j+1} - Vendor data keys: {list(vendor_data.keys()) if vendor_data else 'No vendor data'}")
                
                if not vendor_data:
                    logger.warning(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} - Result {j+1} - Skipping result with no vendor data")
                    continue
                    
                # Add pain point context
                pain_point_id = f"pp_{i + 1}"
                vendor_data["associated_pain_point_id"] = pain_point_id
                vendor_data["pain_point_context"] = pain_points[i] if i < len(pain_points) else {}
                
                vendor_name = vendor_data.get('name', 'Unknown')
                logger.info(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} - Result {j+1} - Processing vendor: {vendor_name} for {pain_point_id}")
                
                all_vendors.append(vendor_data)
                
                # Group by pain point
                if pain_point_id not in vendor_by_pain_point:
                    vendor_by_pain_point[pain_point_id] = []
                    logger.debug(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} - Created new pain point group: {pain_point_id}")
                vendor_by_pain_point[pain_point_id].append(vendor_data)
                
                successful_results += 1
                logger.debug(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} - Result {j+1} - Added vendor {vendor_name} (total vendors: {len(all_vendors)})")
            
            logger.info(f"[AGGREGATE_GAP_ANALYSIS] Output {i+1} - Processed {successful_results}/{len(batch_results)} successful results")
        
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] Research output processing complete")
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] - Total vendors collected: {len(all_vendors)}")
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] - Pain points with vendors: {len(vendor_by_pain_point)}")
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] - Pain point IDs: {list(vendor_by_pain_point.keys())}")
        
        for pp_id, vendors in vendor_by_pain_point.items():
            vendor_names = [v.get('name', 'Unknown') for v in vendors[:3]]
            logger.info(f"[AGGREGATE_GAP_ANALYSIS] - {pp_id}: {len(vendors)} vendors (sample: {vendor_names})")
        
        # Create comprehensive vendor summary for gap finder stages
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] Creating comprehensive vendor summary with {len(all_vendors)} vendors")
        
        vendor_summary = _create_comprehensive_vendor_summary(all_vendors, vendor_by_pain_point, pain_points)
        
        # Use enhanced LLM synthesis with proper error handling
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] Starting enhanced LLM-based synthesis")
        
        try:
            # Initialize LLM manager with proper configuration
            llm_manager = LLMManager()
            
            # Initialize DeepSeek backend like other functions do
            from scout_agent.llm.backends import DeepSeekBackend
            logger.info("[AGGREGATE_GAP_ANALYSIS] Starting LLM backend initialization")
            config = get_config()
            
            # Only try to register DeepSeek backend
            if config.api.deepseek_api_key:
                try:
                    logger.info("[AGGREGATE_GAP_ANALYSIS] Initializing DeepSeek backend")
                    # Create LLMConfig as a dataclass instance
                    deepseek_config = LLMConfig(
                        backend_type=LLMBackendType.DEEPSEEK,
                        model_name="deepseek-chat",
                        api_key=config.api.deepseek_api_key,
                        temperature=0.7,
                        max_tokens=4096
                    )
                    
                    deepseek_backend = DeepSeekBackend(deepseek_config)
                    await llm_manager.register_backend(deepseek_backend, is_default=True)
                    logger.info("[AGGREGATE_GAP_ANALYSIS] DeepSeek backend registered successfully")
                except Exception as e:
                    logger.error(f"[AGGREGATE_GAP_ANALYSIS] Failed to initialize DeepSeek backend: {e}")
                    raise Exception(f"Failed to initialize DeepSeek backend: {e}")
            else:
                logger.error("[AGGREGATE_GAP_ANALYSIS] No DeepSeek API key found in configuration")
                raise Exception("No DeepSeek API key found in configuration")
            
            # Prepare comprehensive synthesis data
            synthesis_data = {
                "vendor_research_outputs": research_outputs,
                "pain_points": pain_points,
                "vendor_summary": vendor_summary,
                "analysis_context": {
                    "total_vendors": len(all_vendors),
                    "unique_vendors": len(set(v.get('name', '').lower() for v in all_vendors if v.get('name'))),
                    "pain_points_covered": len(vendor_by_pain_point),
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }
            
            # Load and prepare the synthesis prompt with proper substitutions
            prompt_path = Path(__file__).parent.parent.parent / "prompts" / "gap_finder_agent" / "collect_aggregate.prompt"
            
            if prompt_path.exists():
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
                
                # Create LLM request with proper configuration - no token limits for production
                llm_request = LLMRequest(
                    messages=[{"role": "user", "content": prompt_content}],
                    system_prompt="You are an expert market analyst specializing in competitive intelligence and gap analysis. Analyze the provided vendor research data and return a comprehensive JSON analysis following the specified format.",
                    temperature=0.2
                    # No max_tokens limit - let DeepSeek use full context window for comprehensive analysis
                )
                
                # Generate synthesis using LLM
                logger.info(f"[AGGREGATE_GAP_ANALYSIS] Sending enhanced synthesis request to LLM")
                llm_response = await llm_manager.generate(llm_request)
                
                # Debug logging for LLM response
                logger.info(f"[AGGREGATE_GAP_ANALYSIS] LLM response received: {llm_response is not None}")
                if llm_response:
                    logger.info(f"[AGGREGATE_GAP_ANALYSIS] LLM response success: {llm_response.success}")
                    logger.info(f"[AGGREGATE_GAP_ANALYSIS] LLM response content length: {len(llm_response.content) if llm_response.content else 0}")
                    logger.debug(f"[AGGREGATE_GAP_ANALYSIS] LLM response content preview: {llm_response.content[:200] if llm_response.content else 'None'}...")
                    if hasattr(llm_response, 'error') and llm_response.error:
                        logger.error(f"[AGGREGATE_GAP_ANALYSIS] LLM response error: {llm_response.error}")
                
                if llm_response and llm_response.success and llm_response.content:
                    logger.info(f"[AGGREGATE_GAP_ANALYSIS] LLM response content (first 500 chars): {llm_response.content[:500]}")
                    
                    # Parse LLM response as JSON
                    try:
                        synthesis_result = json.loads(llm_response.content)
                        logger.info(f"[AGGREGATE_GAP_ANALYSIS] LLM synthesis completed successfully")
                        
                        # Extract components from LLM synthesis
                        market_gaps = synthesis_result.get("market_gaps", [])
                        vendor_analysis = synthesis_result.get("competitive_landscape", {})
                        opportunities = synthesis_result.get("opportunities", [])
                        executive_summary = synthesis_result.get("executive_summary", {})
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"[AGGREGATE_GAP_ANALYSIS] Failed to parse LLM response as JSON: {e}")
                        logger.error(f"[AGGREGATE_GAP_ANALYSIS] Raw LLM response: '{llm_response.content}'")
                        logger.error(f"[AGGREGATE_GAP_ANALYSIS] Response length: {len(llm_response.content)}")
                        logger.error(f"[AGGREGATE_GAP_ANALYSIS] Response type: {type(llm_response.content)}")
                        
                        # Try to extract JSON from response, handling markdown code blocks
                        import re
                        
                        # First try to extract from ```json code blocks
                        code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', llm_response.content)
                        if code_block_match:
                            logger.info(f"[AGGREGATE_GAP_ANALYSIS] Found JSON in code block, attempting to parse")
                            try:
                                synthesis_result = json.loads(code_block_match.group(1).strip())
                                logger.info(f"[AGGREGATE_GAP_ANALYSIS] Code block JSON parsing successful")
                                
                                # Extract components from LLM synthesis
                                market_gaps = synthesis_result.get("market_gaps", [])
                                vendor_analysis = synthesis_result.get("competitive_landscape", {})
                                opportunities = synthesis_result.get("opportunities", [])
                                executive_summary = synthesis_result.get("executive_summary", {})
                            except json.JSONDecodeError as e2:
                                logger.error(f"[AGGREGATE_GAP_ANALYSIS] Code block JSON parsing failed: {e2}")
                                # Fall back to original regex pattern
                                json_match = re.search(r'\{[\s\S]*\}', llm_response.content)
                                if json_match:
                                    logger.info(f"[AGGREGATE_GAP_ANALYSIS] Trying fallback JSON pattern")
                                    try:
                                        synthesis_result = json.loads(json_match.group(0))
                                        logger.info(f"[AGGREGATE_GAP_ANALYSIS] Fallback JSON parsing successful")
                                        
                                        # Extract components from LLM synthesis
                                        market_gaps = synthesis_result.get("market_gaps", [])
                                        vendor_analysis = synthesis_result.get("competitive_landscape", {})
                                        opportunities = synthesis_result.get("opportunities", [])
                                        executive_summary = synthesis_result.get("executive_summary", {})
                                    except json.JSONDecodeError as e3:
                                        logger.error(f"[AGGREGATE_GAP_ANALYSIS] All JSON parsing attempts failed: {e3}")
                                        raise json.JSONDecodeError(f"Failed to parse LLM response as JSON: {e}", llm_response.content, 0)
                                else:
                                    logger.error(f"[AGGREGATE_GAP_ANALYSIS] No JSON pattern found in response")
                                    raise json.JSONDecodeError(f"Failed to parse LLM response as JSON: {e}", llm_response.content, 0)
                        else:
                            # Fall back to original regex pattern if no code blocks found
                            json_match = re.search(r'\{[\s\S]*\}', llm_response.content)
                            if json_match:
                                logger.info(f"[AGGREGATE_GAP_ANALYSIS] Found JSON match, attempting to parse")
                                try:
                                    synthesis_result = json.loads(json_match.group(0))
                                    logger.info(f"[AGGREGATE_GAP_ANALYSIS] Regex-extracted JSON parsing successful")
                                    
                                    # Extract components from LLM synthesis
                                    market_gaps = synthesis_result.get("market_gaps", [])
                                    vendor_analysis = synthesis_result.get("competitive_landscape", {})
                                    opportunities = synthesis_result.get("opportunities", [])
                                    executive_summary = synthesis_result.get("executive_summary", {})
                                except json.JSONDecodeError as e2:
                                    logger.error(f"[AGGREGATE_GAP_ANALYSIS] Regex-extracted JSON parsing also failed: {e2}")
                                    raise json.JSONDecodeError(f"Failed to parse LLM response as JSON: {e}", llm_response.content, 0)
                            else:
                                logger.error(f"[AGGREGATE_GAP_ANALYSIS] No JSON pattern found in response")
                                raise json.JSONDecodeError(f"Failed to parse LLM response as JSON: {e}", llm_response.content, 0)
                elif llm_response and not llm_response.success:
                    logger.error(f"[AGGREGATE_GAP_ANALYSIS] LLM request failed: {llm_response.error if hasattr(llm_response, 'error') else 'Unknown error'}")
                    raise ValueError(f"LLM request failed: {llm_response.error if hasattr(llm_response, 'error') else 'Unknown error'}")
                else:
                    logger.error(f"[AGGREGATE_GAP_ANALYSIS] Empty or invalid LLM response")
                    logger.error(f"[AGGREGATE_GAP_ANALYSIS] llm_response is None: {llm_response is None}")
                    if llm_response:
                        logger.error(f"[AGGREGATE_GAP_ANALYSIS] llm_response.content: '{llm_response.content}'")
                        logger.error(f"[AGGREGATE_GAP_ANALYSIS] llm_response.success: {llm_response.success}")
                    raise ValueError("Empty or invalid LLM response")
            else:
                logger.error(f"[AGGREGATE_GAP_ANALYSIS] Prompt template not found at {prompt_path}")
                raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
                
        except Exception as e:
            logger.error(f"[AGGREGATE_GAP_ANALYSIS] Error during LLM synthesis: {str(e)}")
            logger.error(f"[AGGREGATE_GAP_ANALYSIS] Exception details", exc_info=True)
            raise Exception(f"Error during LLM synthesis: {str(e)}")
        
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] Analysis complete - found {len(market_gaps)} gaps and {len(opportunities)} opportunities")
        
        # Create comprehensive analysis
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] Creating comprehensive analysis")
        
        coverage_percentage = (len(vendor_by_pain_point) / len(pain_points)) * 100 if pain_points else 0
        
        # Extract vendor count from LLM response (check multiple possible locations)
        llm_vendor_count = 0
        llm_vendors_list = []
        
        # Try to extract from vendor_analysis first
        if isinstance(vendor_analysis, dict):
            llm_vendor_count = vendor_analysis.get("vendor_count", 0)
            llm_vendors_list = vendor_analysis.get("key_players", [])
        
        # If not found, try to extract from market_gaps competitive_landscape
        if llm_vendor_count == 0 and isinstance(market_gaps, list) and len(market_gaps) > 0:
            for gap in market_gaps:
                if isinstance(gap, dict) and "competitive_landscape" in gap:
                    comp_landscape = gap["competitive_landscape"]
                    if isinstance(comp_landscape, dict):
                        gap_vendor_count = comp_landscape.get("vendor_count", 0)
                        gap_vendors_list = comp_landscape.get("key_players", [])
                        if gap_vendor_count > llm_vendor_count:
                            llm_vendor_count = gap_vendor_count
                            llm_vendors_list = gap_vendors_list
        
        # Calculate unique vendors from the extracted list
        if isinstance(llm_vendors_list, list) and len(llm_vendors_list) > 0:
            # Handle both string names and dict objects
            unique_vendors = len(set(
                (v.get("name", "").lower() if isinstance(v, dict) else str(v).lower())
                for v in llm_vendors_list if v
            ))
        else:
            unique_vendors = 0
        
        # If LLM didn't provide vendor count, fall back to all_vendors
        if llm_vendor_count == 0 and len(all_vendors) > 0:
            llm_vendor_count = len(all_vendors)
            unique_vendors = len(set(v.get("name", "").lower() for v in all_vendors if v.get("name")))
        
        high_opportunity_gaps = len([g for g in market_gaps if g.get("opportunity_score", 0) > 70])
        
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] Analysis metrics:")
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] - Coverage: {coverage_percentage:.1f}% ({len(vendor_by_pain_point)}/{len(pain_points)} pain points)")
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] - LLM vendor count: {llm_vendor_count}")
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] - Unique vendors: {unique_vendors}")
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] - High opportunity gaps: {high_opportunity_gaps}")
        
        analysis = {
            "market_gaps": market_gaps,
            "vendor_landscape": vendor_analysis,
            "opportunities": opportunities,
            "vendor_summary": vendor_summary,  # Comprehensive vendor information for gap finder stages
            "executive_summary": executive_summary,  # High-level insights and recommendations
            "pain_point_coverage": {
                "total_pain_points": len(pain_points),
                "covered_pain_points": len(vendor_by_pain_point),
                "coverage_percentage": coverage_percentage
            },
            "summary": {
                "total_vendors_found": llm_vendor_count,  # Use LLM vendor count instead of empty all_vendors
                "unique_vendors": unique_vendors,
                "high_opportunity_gaps": high_opportunity_gaps,
                "competitive_intensity": vendor_analysis.get("competitive_intensity", "unknown")
            },
            "metadata": {
                "merge_strategy": merge_strategy,
                "output_format": output_format,
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] Gap analysis aggregation completed successfully")
        logger.debug(f"[AGGREGATE_GAP_ANALYSIS] Final analysis keys: {list(analysis.keys())}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"[AGGREGATE_GAP_ANALYSIS] Exception during gap analysis aggregation: {str(e)}")
        logger.error(f"[AGGREGATE_GAP_ANALYSIS] Exception details", exc_info=True)
        
        error_result = {
            "error": f"Failed to aggregate gap analysis: {str(e)}",
            "success": False,
            "market_gaps": [],
            "vendor_landscape": {},
            "opportunities": []
        }
        
        logger.info(f"[AGGREGATE_GAP_ANALYSIS] Returning error result")
        return error_result


def _create_comprehensive_vendor_summary(all_vendors: List[Dict[str, Any]], vendor_by_pain_point: Dict[str, List], pain_points: List[Dict]) -> Dict[str, Any]:
    """Create comprehensive vendor summary for gap finder stages."""
    logger.info("[AGGREGATE_GAP_ANALYSIS] Creating comprehensive vendor summary")
    
    # Organize vendors by categories
    vendor_categories = {
        "enterprise": [],
        "smb": [],
        "niche": [],
        "emerging": []
    }
    
    # Analyze pricing models and features across all vendors
    pricing_models = set()
    feature_categories = set()
    target_segments = set()
    
    # Detailed vendor profiles
    vendor_profiles = []
    
    for vendor in all_vendors:
        vendor_name = vendor.get('name', 'Unknown')
        
        # Extract key information
        business_profile = vendor.get('business_profile', {})
        features = vendor.get('features', {})
        pricing = vendor.get('pricing', {})
        target_customers = vendor.get('target_customers', {})
        pain_point_address = vendor.get('pain_point_address', {})
        strengths = vendor.get('strengths', {})
        limitations = vendor.get('limitations', {})
        
        # Categorize vendor
        company_size = business_profile.get('company_size', '').lower()
        if 'enterprise' in company_size or 'large' in company_size:
            vendor_categories['enterprise'].append(vendor_name)
        elif 'small' in company_size or 'startup' in company_size:
            vendor_categories['emerging'].append(vendor_name)
        elif 'medium' in company_size or 'smb' in company_size:
            vendor_categories['smb'].append(vendor_name)
        else:
            vendor_categories['niche'].append(vendor_name)
        
        # Extract pricing models
        if pricing:
            pricing_model = pricing.get('model', '').lower()
            if pricing_model:
                pricing_models.add(pricing_model)
        
        # Extract feature categories
        if features:
            for feature_key in features.keys():
                feature_categories.add(feature_key)
        
        # Extract target segments
        if target_customers:
            segments = target_customers.get('segments', [])
            if isinstance(segments, list):
                target_segments.update(segments)
            elif isinstance(segments, str):
                target_segments.add(segments)
        
        # Create detailed vendor profile
        vendor_profile = {
            "name": vendor_name,
            "category": _categorize_vendor(vendor),
            "business_profile": business_profile,
            "key_features": _extract_key_features(features),
            "pricing_summary": _extract_pricing_summary(pricing),
            "target_segments": _extract_target_segments(target_customers),
            "pain_point_alignment": _extract_pain_point_alignment(pain_point_address),
            "competitive_strengths": _extract_strengths(strengths),
            "limitations": _extract_limitations(limitations),
            "associated_pain_point": vendor.get('associated_pain_point_id', 'unknown')
        }
        
        vendor_profiles.append(vendor_profile)
    
    # Create market landscape analysis
    market_landscape = {
        "total_vendors": len(all_vendors),
        "vendor_distribution": {
            category: len(vendors) for category, vendors in vendor_categories.items()
        },
        "pricing_models": list(pricing_models),
        "feature_categories": list(feature_categories),
        "target_segments": list(target_segments),
        "competitive_intensity": _assess_competitive_intensity(all_vendors, vendor_by_pain_point)
    }
    
    # Create pain point analysis
    pain_point_analysis = []
    for pp_id, vendors in vendor_by_pain_point.items():
        pp_index = int(pp_id.split('_')[1]) - 1 if '_' in pp_id else 0
        pain_point = pain_points[pp_index] if pp_index < len(pain_points) else {}
        
        analysis = {
            "pain_point_id": pp_id,
            "pain_point_title": pain_point.get('title', 'Unknown'),
            "vendor_count": len(vendors),
            "vendor_names": [v.get('name', 'Unknown') for v in vendors],
            "solution_diversity": _assess_solution_diversity(vendors),
            "market_maturity": _assess_market_maturity(vendors),
            "average_pricing": _calculate_average_pricing(vendors)
        }
        pain_point_analysis.append(analysis)
    
    return {
        "vendor_profiles": vendor_profiles,
        "market_landscape": market_landscape,
        "pain_point_analysis": pain_point_analysis,
        "synthesis_metadata": {
            "total_vendors_analyzed": len(all_vendors),
            "pain_points_covered": len(vendor_by_pain_point),
            "analysis_timestamp": datetime.now().isoformat(),
            "data_quality": "comprehensive"
        }
    }


def _create_enhanced_fallback_synthesis(all_vendors: List[Dict[str, Any]], vendor_by_pain_point: Dict[str, List], pain_points: List[Dict], vendor_summary: Dict[str, Any]) -> tuple:
    """Create enhanced fallback synthesis using comprehensive heuristic analysis when LLM synthesis fails."""
    logger.info("[AGGREGATE_GAP_ANALYSIS] Using enhanced fallback heuristic synthesis")
    
    # Create comprehensive market gaps analysis
    market_gaps = []
    for pp_id, vendors in vendor_by_pain_point.items():
        pp_index = int(pp_id.split('_')[1]) - 1 if '_' in pp_id else 0
        pain_point = pain_points[pp_index] if pp_index < len(pain_points) else {}
        
        # Analyze competitive landscape for this pain point
        competitive_landscape = {
            "market_maturity": _assess_market_maturity(vendors),
            "competitive_intensity": "high" if len(vendors) > 5 else "medium" if len(vendors) > 2 else "low",
            "vendor_count": len(vendors),
            "key_players": [v.get('name', 'Unknown') for v in vendors[:3]],
            "solution_categories": list(set(_categorize_vendor(v) for v in vendors)),
            "pricing_models": list(set(_extract_pricing_model(v) for v in vendors if _extract_pricing_model(v)))
        }
        
        # Identify gaps based on vendor analysis
        identified_gaps = _identify_market_gaps(vendors, pain_point)
        
        # Assess opportunity
        opportunity_analysis = {
            "market_size_estimate": "medium",  # Heuristic assessment
            "competition_level": competitive_landscape["competitive_intensity"],
            "technical_feasibility": "medium",
            "gtm_complexity": "medium",
            "differentiation_potential": "high" if len(identified_gaps) > 2 else "medium",
            "opportunity_score": _calculate_opportunity_score(competitive_landscape, identified_gaps)
        }
        
        # Strategic recommendations
        strategic_recommendations = {
            "priority_level": "high" if opportunity_analysis["opportunity_score"] > 70 else "medium",
            "market_entry_strategy": _suggest_market_entry_strategy(competitive_landscape),
            "competitive_positioning": _suggest_competitive_positioning(vendors),
            "partnership_opportunities": _identify_partnership_opportunities(vendors),
            "key_risks": _identify_key_risks(competitive_landscape, vendors)
        }
        
        gap_analysis = {
            "pain_point_id": pp_id,
            "pain_point": pain_point.get('title', 'Unknown'),
            "competitive_landscape": competitive_landscape,
            "identified_gaps": identified_gaps,
            "opportunity_analysis": opportunity_analysis,
            "strategic_recommendations": strategic_recommendations
        }
        
        market_gaps.append(gap_analysis)
    
    # Create vendor landscape analysis
    vendor_analysis = {
        "market_overview": {
            "total_vendors": len(all_vendors),
            "market_segments": vendor_summary["market_landscape"]["vendor_distribution"],
            "pricing_diversity": len(vendor_summary["market_landscape"]["pricing_models"]),
            "feature_diversity": len(vendor_summary["market_landscape"]["feature_categories"])
        },
        "competitive_intensity": vendor_summary["market_landscape"]["competitive_intensity"],
        "market_trends": _identify_market_trends(all_vendors),
        "vendor_positioning": _analyze_vendor_positioning(vendor_summary["vendor_profiles"])
    }
    
    # Create opportunities analysis
    opportunities = []
    for gap in market_gaps:
        if gap["opportunity_analysis"]["opportunity_score"] > 60:
            opportunity = {
                "title": f"Market opportunity in {gap['pain_point']}",
                "description": f"Gap identified in {gap['pain_point']} with {gap['opportunity_analysis']['opportunity_score']} opportunity score",
                "market_size": gap["opportunity_analysis"]["market_size_estimate"],
                "competition_level": gap["opportunity_analysis"]["competition_level"],
                "recommended_approach": gap["strategic_recommendations"]["market_entry_strategy"],
                "priority": gap["strategic_recommendations"]["priority_level"]
            }
            opportunities.append(opportunity)
    
    # Create executive summary
    executive_summary = {
        "key_insights": [
            f"Analyzed {len(all_vendors)} vendors across {len(pain_points)} pain points",
            f"Identified {len([g for g in market_gaps if g['opportunity_analysis']['opportunity_score'] > 70])} high-opportunity gaps",
            f"Market shows {vendor_analysis['competitive_intensity']} competitive intensity"
        ],
        "top_opportunities": [op["title"] for op in opportunities[:3]],
        "market_trends": vendor_analysis["market_trends"],
        "recommended_focus_areas": [
            gap["pain_point"] for gap in market_gaps 
            if gap["strategic_recommendations"]["priority_level"] == "high"
        ][:3]
    }
    
    return vendor_analysis, market_gaps, opportunities, executive_summary


# Helper functions for vendor analysis
def _categorize_vendor(vendor: Dict[str, Any]) -> str:
    """Categorize vendor based on business profile."""
    business_profile = vendor.get('business_profile', {})
    company_size = business_profile.get('company_size', '').lower()
    
    if 'enterprise' in company_size or 'large' in company_size:
        return 'enterprise'
    elif 'small' in company_size or 'startup' in company_size:
        return 'emerging'
    elif 'medium' in company_size or 'smb' in company_size:
        return 'smb'
    else:
        return 'niche'


def _extract_key_features(features: Dict[str, Any]) -> List[str]:
    """Extract key features from vendor feature data."""
    if not features:
        return []
    
    key_features = []
    for category, feature_data in features.items():
        if isinstance(feature_data, dict):
            key_features.extend(feature_data.keys())
        elif isinstance(feature_data, list):
            key_features.extend(feature_data)
        elif isinstance(feature_data, str):
            key_features.append(feature_data)
    
    return key_features[:10]  # Limit to top 10 features


def _extract_pricing_summary(pricing: Dict[str, Any]) -> Dict[str, Any]:
    """Extract pricing summary from vendor pricing data."""
    if not pricing:
        return {"model": "unknown", "range": "unknown"}
    
    return {
        "model": pricing.get('model', 'unknown'),
        "range": pricing.get('range', 'unknown'),
        "currency": pricing.get('currency', 'USD')
    }


def _extract_target_segments(target_customers: Dict[str, Any]) -> List[str]:
    """Extract target segments from vendor target customer data."""
    if not target_customers:
        return []
    
    segments = target_customers.get('segments', [])
    if isinstance(segments, list):
        return segments
    elif isinstance(segments, str):
        return [segments]
    else:
        return []


def _extract_pain_point_alignment(pain_point_address: Dict[str, Any]) -> Dict[str, Any]:
    """Extract pain point alignment information."""
    if not pain_point_address:
        return {"alignment_score": "unknown", "addressed_aspects": []}
    
    return {
        "alignment_score": pain_point_address.get('alignment_score', 'unknown'),
        "addressed_aspects": pain_point_address.get('addressed_aspects', []),
        "gaps": pain_point_address.get('gaps', [])
    }


def _extract_strengths(strengths: Dict[str, Any]) -> List[str]:
    """Extract competitive strengths."""
    if not strengths:
        return []
    
    if isinstance(strengths, dict):
        return list(strengths.keys())[:5]
    elif isinstance(strengths, list):
        return strengths[:5]
    else:
        return [str(strengths)]


def _extract_limitations(limitations: Dict[str, Any]) -> List[str]:
    """Extract vendor limitations."""
    if not limitations:
        return []
    
    if isinstance(limitations, dict):
        return list(limitations.keys())[:5]
    elif isinstance(limitations, list):
        return limitations[:5]
    else:
        return [str(limitations)]


def _assess_competitive_intensity(all_vendors: List[Dict[str, Any]], vendor_by_pain_point: Dict[str, List]) -> str:
    """Assess overall competitive intensity."""
    avg_vendors_per_pain_point = sum(len(vendors) for vendors in vendor_by_pain_point.values()) / len(vendor_by_pain_point) if vendor_by_pain_point else 0
    
    if avg_vendors_per_pain_point > 5:
        return "high"
    elif avg_vendors_per_pain_point > 2:
        return "medium"
    else:
        return "low"


def _assess_solution_diversity(vendors: List[Dict[str, Any]]) -> str:
    """Assess diversity of solutions for a pain point."""
    categories = set(_categorize_vendor(v) for v in vendors)
    
    if len(categories) > 2:
        return "high"
    elif len(categories) > 1:
        return "medium"
    else:
        return "low"


def _assess_market_maturity(vendors: List[Dict[str, Any]]) -> str:
    """Assess market maturity based on vendor characteristics."""
    enterprise_count = sum(1 for v in vendors if _categorize_vendor(v) == 'enterprise')
    total_vendors = len(vendors)
    
    if enterprise_count / total_vendors > 0.5:
        return "mature"
    elif total_vendors > 3:
        return "growing"
    else:
        return "emerging"


def _calculate_average_pricing(vendors: List[Dict[str, Any]]) -> str:
    """Calculate average pricing range."""
    pricing_data = [v.get('pricing', {}) for v in vendors if v.get('pricing')]
    
    if not pricing_data:
        return "unknown"
    
    # Simple heuristic based on pricing models
    subscription_count = sum(1 for p in pricing_data if 'subscription' in p.get('model', '').lower())
    
    if subscription_count > len(pricing_data) / 2:
        return "subscription-based"
    else:
        return "mixed-pricing"


def _extract_pricing_model(vendor: Dict[str, Any]) -> str:
    """Extract pricing model from vendor data."""
    pricing = vendor.get('pricing', {})
    return pricing.get('model', '') if pricing else ''


def _identify_market_gaps(vendors: List[Dict[str, Any]], pain_point: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify market gaps based on vendor analysis."""
    gaps = []
    
    # Feature gaps
    all_features = set()
    for vendor in vendors:
        features = vendor.get('features', {})
        if features:
            all_features.update(features.keys())
    
    if len(all_features) < 5:  # Heuristic: if few feature categories, there might be gaps
        gaps.append({
            "gap_type": "feature",
            "description": "Limited feature diversity in current solutions",
            "affected_segments": ["all"],
            "severity": "medium"
        })
    
    # Pricing gaps
    pricing_models = set(_extract_pricing_model(v) for v in vendors if _extract_pricing_model(v))
    if len(pricing_models) < 2:
        gaps.append({
            "gap_type": "pricing",
            "description": "Limited pricing model diversity",
            "affected_segments": ["price-sensitive"],
            "severity": "medium"
        })
    
    # Segment gaps
    categories = set(_categorize_vendor(v) for v in vendors)
    if 'smb' not in categories:
        gaps.append({
            "gap_type": "segment",
            "description": "Underserved SMB market segment",
            "affected_segments": ["smb"],
            "severity": "high"
        })
    
    return gaps


def _calculate_opportunity_score(competitive_landscape: Dict[str, Any], identified_gaps: List[Dict[str, Any]]) -> int:
    """Calculate opportunity score based on competitive landscape and gaps."""
    base_score = 50
    
    # Adjust based on competitive intensity
    intensity = competitive_landscape.get("competitive_intensity", "medium")
    if intensity == "low":
        base_score += 20
    elif intensity == "high":
        base_score -= 10
    
    # Adjust based on number of gaps
    gap_bonus = min(len(identified_gaps) * 10, 30)
    base_score += gap_bonus
    
    # Adjust based on gap severity
    high_severity_gaps = sum(1 for gap in identified_gaps if gap.get("severity") == "high")
    base_score += high_severity_gaps * 5
    
    return min(max(base_score, 0), 100)


def _suggest_market_entry_strategy(competitive_landscape: Dict[str, Any]) -> str:
    """Suggest market entry strategy based on competitive landscape."""
    intensity = competitive_landscape.get("competitive_intensity", "medium")
    vendor_count = competitive_landscape.get("vendor_count", 0)
    
    if intensity == "low" and vendor_count < 3:
        return "Direct competition with differentiated features"
    elif intensity == "high":
        return "Niche market focus with specialized solution"
    else:
        return "Partnership-based entry with existing players"


def _suggest_competitive_positioning(vendors: List[Dict[str, Any]]) -> str:
    """Suggest competitive positioning based on existing vendors."""
    categories = [_categorize_vendor(v) for v in vendors]
    
    if 'enterprise' in categories and 'smb' not in categories:
        return "SMB-focused solution with simplified features"
    elif 'smb' in categories and 'enterprise' not in categories:
        return "Enterprise-grade solution with advanced features"
    else:
        return "Mid-market solution bridging enterprise and SMB needs"


def _identify_partnership_opportunities(vendors: List[Dict[str, Any]]) -> List[str]:
    """Identify potential partnership opportunities."""
    partnerships = []
    
    vendor_names = [v.get('name', 'Unknown') for v in vendors]
    
    # Suggest partnerships with complementary vendors
    if len(vendor_names) > 0:
        partnerships.append(f"Integration partnership with {vendor_names[0]}")
    
    if len(vendor_names) > 1:
        partnerships.append(f"Channel partnership with {vendor_names[1]}")
    
    return partnerships[:3]


def _identify_key_risks(competitive_landscape: Dict[str, Any], vendors: List[Dict[str, Any]]) -> List[str]:
    """Identify key risks in the market."""
    risks = []
    
    intensity = competitive_landscape.get("competitive_intensity", "medium")
    if intensity == "high":
        risks.append("High competitive pressure from established players")
    
    vendor_count = competitive_landscape.get("vendor_count", 0)
    if vendor_count > 5:
        risks.append("Market saturation risk")
    
    enterprise_vendors = [v for v in vendors if _categorize_vendor(v) == 'enterprise']
    if len(enterprise_vendors) > 2:
        risks.append("Dominant enterprise players may expand to other segments")
    
    return risks[:3]


def _identify_market_trends(all_vendors: List[Dict[str, Any]]) -> List[str]:
    """Identify market trends based on vendor analysis."""
    trends = []
    
    # Analyze pricing trends
    subscription_count = sum(1 for v in all_vendors if 'subscription' in _extract_pricing_model(v).lower())
    if subscription_count > len(all_vendors) / 2:
        trends.append("Shift towards subscription-based pricing models")
    
    # Analyze vendor categories
    categories = [_categorize_vendor(v) for v in all_vendors]
    emerging_count = categories.count('emerging')
    if emerging_count > len(all_vendors) / 3:
        trends.append("Increasing number of emerging/startup solutions")
    
    # Feature trends
    all_features = set()
    for vendor in all_vendors:
        features = vendor.get('features', {})
        if features:
            all_features.update(features.keys())
    
    if 'ai' in ' '.join(all_features).lower() or 'automation' in ' '.join(all_features).lower():
        trends.append("Growing focus on AI and automation capabilities")
    
    return trends[:5]


def _analyze_vendor_positioning(vendor_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze vendor positioning in the market."""
    positioning = {
        "market_leaders": [],
        "challengers": [],
        "niche_players": [],
        "emerging_vendors": []
    }
    
    for profile in vendor_profiles:
        vendor_name = profile.get('name', 'Unknown')
        category = profile.get('category', 'niche')
        
        if category == 'enterprise':
            positioning['market_leaders'].append(vendor_name)
        elif category == 'smb':
            positioning['challengers'].append(vendor_name)
        elif category == 'emerging':
            positioning['emerging_vendors'].append(vendor_name)
        else:
            positioning['niche_players'].append(vendor_name)
    
    return positioning
    # Return individual components as expected by the calling function
    return vendor_analysis, market_gaps, opportunities, executive_summary


def _analyze_vendor_landscape_heuristic(vendors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the overall vendor landscape using heuristics (fallback method)."""
    if not vendors:
        return {"competitive_intensity": "low", "vendor_categories": [], "market_maturity": "emerging"}
    
    # Categorize vendors by business model/type
    categories = {}
    total_funding = 0
    funded_vendors = 0
    
    for vendor in vendors:
        # Extract business model or category
        business_profile = vendor.get("business_profile", {})
        category = business_profile.get("category", "unknown")
        
        if category not in categories:
            categories[category] = []
        categories[category].append(vendor)
        
        # Track funding information if available
        funding = business_profile.get("funding_info", {})
        if funding and funding.get("total_funding"):
            try:
                total_funding += float(funding["total_funding"])
                funded_vendors += 1
            except (ValueError, TypeError):
                pass
    
    # Determine competitive intensity
    vendor_count = len(vendors)
    if vendor_count > 20:
        intensity = "very_high"
    elif vendor_count > 10:
        intensity = "high"
    elif vendor_count > 5:
        intensity = "medium"
    else:
        intensity = "low"
    
    return {
        "vendor_count": vendor_count,
        "competitive_intensity": intensity,
        "vendor_categories": list(categories.keys()),
        "category_distribution": {k: len(v) for k, v in categories.items()},
        "market_maturity": "mature" if vendor_count > 15 else "growing" if vendor_count > 5 else "emerging",
        "funding_landscape": {
            "total_funding": total_funding,
            "funded_vendors": funded_vendors,
            "average_funding": total_funding / funded_vendors if funded_vendors > 0 else 0
        }
    }


def _identify_market_gaps_heuristic(vendor_by_pain_point: Dict[str, List], pain_points: List[Dict]) -> List[Dict[str, Any]]:
    """Identify market gaps based on vendor coverage and pain point analysis (heuristic fallback)."""
    gaps = []
    
    for i, pain_point in enumerate(pain_points):
        pain_point_id = f"pp_{i + 1}"
        vendors = vendor_by_pain_point.get(pain_point_id, [])
        
        # Analyze gap characteristics
        vendor_count = len(vendors)
        
        # Calculate opportunity score based on pain point severity and vendor coverage
        pain_severity = pain_point.get("severity_score", 50)
        market_size_indicator = pain_point.get("market_size_indicator", 50)
        
        # Lower vendor count + higher pain severity = higher opportunity
        coverage_score = max(0, 100 - (vendor_count * 10))  # Decreases as vendor count increases
        opportunity_score = (pain_severity * 0.4) + (coverage_score * 0.4) + (market_size_indicator * 0.2)
        
        gap = {
            "pain_point_id": pain_point_id,
            "pain_point": pain_point,
            "vendor_count": vendor_count,
            "opportunity_score": min(100, max(0, opportunity_score)),
            "gap_type": _classify_gap_type(vendor_count, pain_severity),
            "market_readiness": _assess_market_readiness(pain_point, vendors),
            "competitive_landscape": "crowded" if vendor_count > 10 else "moderate" if vendor_count > 3 else "sparse",
            "vendors": vendors[:5]  # Include top 5 vendors for context
        }
        
        gaps.append(gap)
    
    # Sort by opportunity score
    gaps.sort(key=lambda x: x["opportunity_score"], reverse=True)
    
    return gaps


def _classify_gap_type(vendor_count: int, pain_severity: float) -> str:
    """Classify the type of market gap."""
    if vendor_count == 0:
        return "blue_ocean" if pain_severity > 70 else "unvalidated_need"
    elif vendor_count <= 2:
        return "emerging_market" if pain_severity > 60 else "niche_opportunity"
    elif vendor_count <= 5:
        return "competitive_gap" if pain_severity > 70 else "incremental_opportunity"
    else:
        return "saturated_market"


def _assess_market_readiness(pain_point: Dict, vendors: List[Dict]) -> str:
    """Assess how ready the market is for solutions."""
    # Simple heuristic based on pain point characteristics and vendor presence
    urgency = pain_point.get("urgency_score", 50)
    frequency = pain_point.get("frequency_score", 50)
    vendor_count = len(vendors)
    
    readiness_score = (urgency + frequency) / 2
    
    if readiness_score > 80 and vendor_count > 0:
        return "ready"
    elif readiness_score > 60:
        return "developing"
    elif readiness_score > 40:
        return "early"
    else:
        return "nascent"


def _assess_opportunities_heuristic(gaps: List[Dict], vendor_analysis: Dict) -> List[Dict[str, Any]]:
    """Assess and prioritize opportunities based on gaps and market analysis (heuristic fallback)."""
    opportunities = []
    
    for gap in gaps:
        if gap["opportunity_score"] < 30:  # Skip low-opportunity gaps
            continue
            
        opportunity = {
            "gap_id": gap["pain_point_id"],
            "opportunity_type": gap["gap_type"],
            "priority": "high" if gap["opportunity_score"] > 70 else "medium" if gap["opportunity_score"] > 50 else "low",
            "market_entry_difficulty": _assess_entry_difficulty(gap, vendor_analysis),
            "recommended_approach": _recommend_approach(gap),
            "key_differentiators": _identify_differentiators(gap),
            "risk_factors": _identify_risk_factors(gap),
            "opportunity_score": gap["opportunity_score"]
        }
        
        opportunities.append(opportunity)
    
    return opportunities


def _assess_entry_difficulty(gap: Dict, vendor_analysis: Dict) -> str:
    """Assess difficulty of market entry."""
    vendor_count = gap["vendor_count"]
    competitive_intensity = vendor_analysis.get("competitive_intensity", "low")
    
    if competitive_intensity == "very_high" or vendor_count > 15:
        return "very_high"
    elif competitive_intensity == "high" or vendor_count > 8:
        return "high"
    elif competitive_intensity == "medium" or vendor_count > 3:
        return "medium"
    else:
        return "low"


def _recommend_approach(gap: Dict) -> str:
    """Recommend market approach based on gap characteristics."""
    gap_type = gap["gap_type"]
    vendor_count = gap["vendor_count"]
    
    if gap_type == "blue_ocean":
        return "pioneer_and_educate"
    elif gap_type == "emerging_market":
        return "fast_follower"
    elif gap_type == "competitive_gap":
        return "differentiate_and_specialize"
    elif vendor_count > 10:
        return "niche_focus_or_avoid"
    else:
        return "direct_competition"


def _identify_differentiators(gap: Dict) -> List[str]:
    """Identify potential differentiators based on gap analysis."""
    differentiators = []
    
    pain_point = gap["pain_point"]
    vendors = gap["vendors"]
    
    # Generic differentiators based on gap type
    if gap["gap_type"] == "blue_ocean":
        differentiators.extend(["first_mover_advantage", "market_education", "category_creation"])
    elif gap["vendor_count"] < 3:
        differentiators.extend(["superior_ux", "better_pricing", "specialized_features"])
    else:
        differentiators.extend(["niche_specialization", "integration_capabilities", "customer_service"])
    
    return differentiators


def _identify_risk_factors(gap: Dict) -> List[str]:
    """Identify risk factors for the opportunity."""
    risks = []
    
    if gap["gap_type"] == "blue_ocean":
        risks.extend(["market_validation_risk", "customer_education_cost"])
    elif gap["vendor_count"] > 10:
        risks.extend(["high_competition", "price_pressure", "customer_acquisition_cost"])
    
    if gap["market_readiness"] in ["early", "nascent"]:
        risks.append("market_timing_risk")
    
    return risks


# Create global instances to avoid conflicts during rapid calls
global_triager = None
global_identifier = None

# Create ASGI app for uvicorn
app = server.asgi_app()

if __name__ == "__main__":
    # Run with SSE transport
    mcp.run(transport="sse")