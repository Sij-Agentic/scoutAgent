"""
DAG Node Templates for Scout Agent
Programmatic construction of tool nodes to avoid LLM JSON truncation issues.
"""

from typing import Dict, List, Any


def create_reddit_search_node(keywords: List[str], subreddits: List[str], 
                             per_query_limit: int = 50, comment_limit: int = 200) -> Dict[str, Any]:
    """Create a Reddit search and fetch tool node."""
    
    # Build the Python code for the node
    code = f'''# MCP tool call for reddit_search_and_fetch_threads
import json
from pathlib import Path

print("DEBUG: Starting reddit tool execution")

# Define parameters as Python dict
params = {{
    "keywords": {keywords},
    "subreddits": {subreddits},
    "per_query_limit": {per_query_limit},
    "include_comments": True,
    "comment_depth": 2,
    "comment_limit": {comment_limit},
    "use_cache": True
}}

print(f"DEBUG: Calling MCP tool with params: {{params}}")

# Call the MCP tool (synchronous wrapper)
result = mcp_call("reddit_search_and_fetch_threads", params)

print(f"DEBUG: MCP call completed, result type: {{type(result)}}")
print(f"DEBUG: Result keys: {{list(result.keys()) if isinstance(result, dict) else 'not dict'}}")

# Save result to manifest
print("DEBUG: About to call save_to_manifest")
save_to_manifest("stages.scout_collect.reddit", result)
print("DEBUG: save_to_manifest call completed")

if isinstance(result, dict) and 'threads' in result:
    threads_count = len(result.get('threads', []))
    comments_count = sum(len(t.get('comments', [])) for t in result.get('threads', []))
    print(f"DEBUG: Processed {{threads_count}} threads and {{comments_count}} comments")'''

    return {
        "id": "scout_collect_reddit",
        "type": "tool", 
        "tool": "reddit_search_and_fetch_threads",
        "params": {
            "keywords": keywords,
            "subreddits": subreddits,
            "per_query_limit": per_query_limit,
            "include_comments": True,
            "comment_depth": 2,
            "comment_limit": comment_limit,
            "use_cache": True
        },
        "code": code,
        "language": "python",
        "outputs": ["stages.scout_collect.reddit"]
    }


def create_twitter_search_node(keywords: List[str], max_results: int = 100) -> Dict[str, Any]:
    """Create a Twitter search and fetch tool node."""
    
    code = f'''# MCP tool call for twitter_search_and_fetch_tweets
import json
from pathlib import Path

print("DEBUG: Starting twitter tool execution")

# Define parameters as Python dict
params = {{
    "keywords": {keywords},
    "max_results": {max_results},
    "include_replies": False
}}

print(f"DEBUG: Calling MCP tool with params: {{params}}")

# Call the MCP tool (synchronous wrapper)
result = mcp_call("twitter_search_and_fetch_tweets", params)

print(f"DEBUG: MCP call completed, result type: {{type(result)}}")

# Save result to manifest
save_to_manifest("stages.scout_collect.twitter", result)

if isinstance(result, dict):
    tweet_count = len(result.get('tweets', []))
    print(f"DEBUG: Processed {{tweet_count}} tweets")'''

    return {
        "id": "scout_collect_twitter",
        "type": "tool",
        "tool": "twitter_search_and_fetch_tweets", 
        "params": {
            "keywords": keywords,
            "max_results": max_results,
            "include_replies": False
        },
        "code": code,
        "language": "python",
        "outputs": ["stages.scout_collect.twitter"]
    }


def create_dag_from_metadata(metadata: Dict[str, Any], available_sources: List[str]) -> Dict[str, Any]:
    """
    Construct a complete DAG from LLM-generated metadata.
    
    Args:
        metadata: LLM-generated metadata with keywords, subreddits, etc.
        available_sources: List of available data sources (e.g., ["reddit", "twitter"])
    
    Returns:
        Complete DAG structure with programmatically generated nodes
    """
    
    nodes = []
    
    # Extract parameters from metadata
    keywords = metadata.get("enhanced_keywords", metadata.get("keywords", []))
    subreddits = metadata.get("optimized_subreddits", metadata.get("subreddits", []))
    
    # Create nodes based on available sources
    if "reddit" in available_sources:
        reddit_node = create_reddit_search_node(
            keywords=keywords,
            subreddits=subreddits,
            per_query_limit=50,  # Default value
            comment_limit=200    # Default value
        )
        nodes.append(reddit_node)
    
    if "twitter" in available_sources:
        twitter_node = create_twitter_search_node(
            keywords=keywords,
            max_results=100     # Default value
        )
        nodes.append(twitter_node)
    
    # Return complete DAG structure
    return {
        "dag": {
            "nodes": nodes
        },
        "metadata": metadata.get("metadata", metadata)
    }


def get_default_research_parameters() -> Dict[str, Any]:
    """Get default research parameters for fallback scenarios."""
    return {
        "per_query_limit": 50,
        "include_comments": True,
        "comment_depth": 2,
        "comment_limit": 200,
        "use_cache": True,
        "max_results": 100
    }
