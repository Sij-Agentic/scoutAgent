import json
import os
import hashlib
import asyncio
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
        
        # Check cache
        if use_cache:
            cached = self.cache.load("vendors", key, None)
            if cached is not None:
                logger.info(f"Using cached vendor identification for {url}")
                return cached
        
        # Ensure LLM backends are initialized
        await self._initialize_llm_backends()
        
        # Prepare input for LLM - use content directly like triage_content
        input_text = content[:3000]
        
        # Call LLM
        try:
            llm_request = LLMRequest(
                messages=[{"role": "user", "content": input_text}],
                system_prompt=self.prompt_template,
                temperature=0.2,
                max_tokens=1000
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
                self.cache.save("vendors", key, result)
            
            return result
        except Exception as e:
            error_msg = f"Error identifying vendors: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}


@mcp.tool()
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
    try:
        logger.info(f"[IDENTIFY_VENDORS] Starting tool call with {len(contents)} content items, use_cache={use_cache}")
        
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
        
        # Identify vendors in each content item
        vendor_results = []
        logger.info(f"[IDENTIFY_VENDORS] Processing {len(contents)} content items")
        
        for i, item in enumerate(contents):
            try:
                url = item.get("url", "")
                content = item.get("content", "")
                logger.info(f"[IDENTIFY_VENDORS] Processing item {i+1}/{len(contents)}: {url[:100]}...")
                
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
                logger.info(f"[IDENTIFY_VENDORS] Successfully processed item {i+1}")
                
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
                
            except Exception as e:
                logger.error(f"[IDENTIFY_VENDORS] Error processing item {i+1} ({url}): {e}")
                # Continue processing other items even if one fails
                vendor_results.append({
                    "url": url,
                    "success": False,
                    "error": f"Processing error: {str(e)}"
                })
                continue
        
        logger.info(f"[IDENTIFY_VENDORS] Completed processing. Total results: {len(vendor_results)}")
        
        final_result = {
            "content": [
                TextContent(type="text", text=json.dumps({"vendor_results": vendor_results}))
            ]
        }
        logger.info(f"[IDENTIFY_VENDORS] Returning final result with {len(vendor_results)} vendor results")
        return final_result
        
    except Exception as e:
        logger.error(f"[IDENTIFY_VENDORS] Unexpected error in identify_vendors: {e}")
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
    logger.info(f"Starting batch vendor research for {len(vendors_list)} vendors")
    
    # Deduplicate vendors if requested
    if deduplicate:
        seen_names = set()
        unique_vendors = []
        for vendor in vendors_list:
            vendor_name = vendor.get('name', '').lower().strip()
            if vendor_name and vendor_name not in seen_names:
                seen_names.add(vendor_name)
                unique_vendors.append(vendor)
        
        logger.info(f"Deduplicated {len(vendors_list)} vendors to {len(unique_vendors)} unique vendors")
        vendors_list = unique_vendors
    
    # Initialize Vendor Research Tool
    research_tool = VendorResearchTool()
    
    # Research each vendor
    research_results = []
    successful_research = 0
    failed_research = 0
    
    for vendor in vendors_list:
        try:
            vendor_name = vendor.get('name', '')
            vendor_url = vendor.get('url')
            
            if not vendor_name:
                logger.warning("Skipping vendor with no name")
                continue
            
            # Prepare the input for the research tool
            tool_input = {
                'vendor_name': vendor_name,
                'pain_point': pain_point
            }
            
            if vendor_url:
                tool_input['url'] = vendor_url
            
            # Call the tool asynchronously
            logger.info(f"Researching vendor: {vendor_name}")
            result = await research_tool.forward(**tool_input)
            
            # Parse the result
            try:
                result_json = json.loads(result)
                research_results.append({
                    "vendor_name": vendor_name,
                    "research_data": result_json,
                    "success": True
                })
                successful_research += 1
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing research result for {vendor_name}: {str(e)}")
                research_results.append({
                    "vendor_name": vendor_name,
                    "error": f"JSON parsing error: {str(e)}",
                    "success": False
                })
                failed_research += 1
                
        except Exception as e:
            logger.error(f"Error researching vendor {vendor.get('name', 'unknown')}: {str(e)}")
            research_results.append({
                "vendor_name": vendor.get('name', 'unknown'),
                "error": str(e),
                "success": False
            })
            failed_research += 1
    
    return {
        "research_results": research_results,
        "summary": {
            "total_vendors": len(vendors_list),
            "successful_research": successful_research,
            "failed_research": failed_research,
            "deduplication_enabled": deduplicate
        },
        "pain_point": pain_point
    }


@mcp.tool()
async def aggregate_gap_analysis(
    research_outputs: List[Dict[str, Any]],
    pain_points: List[Dict[str, Any]],
    merge_strategy: str = "by_pain_point_id",
    output_format: str = "comprehensive_gap_analysis"
) -> Dict[str, Any]:
    """
    Aggregate vendor research results for comprehensive gap analysis.
    
    Args:
        research_outputs: List of vendor research results from different pain points
        pain_points: Original pain points data for context
        merge_strategy: Strategy for merging results (by_pain_point_id, by_vendor, comprehensive)
        output_format: Format of the output (comprehensive_gap_analysis, summary, detailed)
    
    Returns:
        Aggregated analysis with market gaps, vendor landscape, and opportunities
    """
    try:
        # Initialize aggregation containers
        all_vendors = []
        vendor_by_pain_point = {}
        pain_point_coverage = {}
        
        # Process each research output
        for i, research_output in enumerate(research_outputs):
            if not research_output or not isinstance(research_output, dict):
                continue
                
            # Extract batch results if available
            batch_results = research_output.get("batch_results", [])
            
            for result in batch_results:
                if not result.get("success", False):
                    continue
                    
                vendor_data = result.get("vendor_data", {})
                if not vendor_data:
                    continue
                    
                # Add pain point context
                pain_point_id = f"pp_{i + 1}"
                vendor_data["associated_pain_point_id"] = pain_point_id
                vendor_data["pain_point_context"] = pain_points[i] if i < len(pain_points) else {}
                
                all_vendors.append(vendor_data)
                
                # Group by pain point
                if pain_point_id not in vendor_by_pain_point:
                    vendor_by_pain_point[pain_point_id] = []
                vendor_by_pain_point[pain_point_id].append(vendor_data)
        
        # Analyze vendor landscape
        vendor_analysis = _analyze_vendor_landscape(all_vendors)
        
        # Identify market gaps
        market_gaps = _identify_market_gaps(vendor_by_pain_point, pain_points)
        
        # Generate opportunity assessment
        opportunities = _assess_opportunities(market_gaps, vendor_analysis)
        
        # Create comprehensive analysis
        analysis = {
            "market_gaps": market_gaps,
            "vendor_landscape": vendor_analysis,
            "opportunities": opportunities,
            "pain_point_coverage": {
                "total_pain_points": len(pain_points),
                "covered_pain_points": len(vendor_by_pain_point),
                "coverage_percentage": (len(vendor_by_pain_point) / len(pain_points)) * 100 if pain_points else 0
            },
            "summary": {
                "total_vendors_found": len(all_vendors),
                "unique_vendors": len(set(v.get("name", "").lower() for v in all_vendors if v.get("name"))),
                "high_opportunity_gaps": len([g for g in market_gaps if g.get("opportunity_score", 0) > 70]),
                "competitive_intensity": vendor_analysis.get("competitive_intensity", "unknown")
            },
            "metadata": {
                "merge_strategy": merge_strategy,
                "output_format": output_format,
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
        return analysis
        
    except Exception as e:
        return {
            "error": f"Failed to aggregate gap analysis: {str(e)}",
            "success": False,
            "market_gaps": [],
            "vendor_landscape": {},
            "opportunities": []
        }


def _analyze_vendor_landscape(vendors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the overall vendor landscape."""
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


def _identify_market_gaps(vendor_by_pain_point: Dict[str, List], pain_points: List[Dict]) -> List[Dict[str, Any]]:
    """Identify market gaps based on vendor coverage and pain point analysis."""
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


def _assess_opportunities(gaps: List[Dict], vendor_analysis: Dict) -> List[Dict[str, Any]]:
    """Assess and prioritize opportunities based on gaps and market analysis."""
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