"""
Web search and fetching tools for MCP.

This module provides a server with tools for web search and data fetching.
"""

import asyncio
import json
import logging
import httpx
import re
from typing import Dict, List, Any, Optional
from urllib.parse import quote_plus

from ..server.base import Server, Context

# Configure logging
logger = logging.getLogger("mcp.servers.web_search")

# Create server
server = Server(
    name="Web Search Tools",
    description="Tools for searching the web and fetching content from external sources"
)

# Default HTTP client with appropriate headers
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


@server.tool(name="search_web", description="Search the web for information")
async def search_web(query: str, num_results: int = 5, ctx: Context = None) -> List[Dict[str, str]]:
    """
    Search the web for information on a given query.
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 5)
        ctx: Context object (optional)
    
    Returns:
        List of search results with title, url, and snippet
    """
    if ctx:
        ctx.log("INFO", f"Searching web for: {query}")
    
    try:
        # Use a simplified approach for demo purposes
        # In production, you would use a proper search API
        search_url = f"https://www.google.com/search?q={quote_plus(query)}"
        
        async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True) as client:
            response = await client.get(search_url)
            
            if response.status_code != 200:
                if ctx:
                    ctx.log("ERROR", f"Search request failed with status: {response.status_code}")
                return [{"error": f"Search failed with status code: {response.status_code}"}]
            
            # Very simple extraction of search results for demo
            # In production, use a proper HTML parser or API
            content = response.text
            results = []
            
            # Extract result blocks (this is a simplified approach)
            # In reality, you'd use a proper parser like BeautifulSoup
            result_blocks = re.findall(r'<div class="g">(.*?)</div>', content, re.DOTALL)
            
            for block in result_blocks[:num_results]:
                # Extract title, URL and snippet
                title_match = re.search(r'<h3[^>]*>(.*?)</h3>', block, re.DOTALL)
                url_match = re.search(r'<a href="([^"]*)"', block)
                snippet_match = re.search(r'<span class="st">(.*?)</span>', block, re.DOTALL)
                
                if title_match and url_match:
                    # Clean up the matches
                    title = re.sub(r'<.*?>', '', title_match.group(1))
                    url = url_match.group(1)
                    snippet = ""
                    if snippet_match:
                        snippet = re.sub(r'<.*?>', '', snippet_match.group(1))
                    
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet
                    })
            
            if ctx:
                ctx.log("INFO", f"Found {len(results)} search results")
            
            return results
            
    except Exception as e:
        if ctx:
            ctx.log("ERROR", f"Search error: {str(e)}")
        return [{"error": f"Search error: {str(e)}"}]


@server.tool(name="reddit_thread_fetcher", description="Fetch content from Reddit threads")
async def reddit_thread_fetcher(
    query: str = "", 
    thread_url: Optional[str] = None, 
    subreddit: Optional[str] = None,
    sort_by: str = "hot",
    limit: int = 10,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Fetch content from Reddit threads or search Reddit.
    
    Args:
        query: Search query for Reddit (optional)
        thread_url: Direct URL to a Reddit thread (optional)
        subreddit: Specific subreddit to search in (optional)
        sort_by: How to sort results (hot, new, top) (default: hot)
        limit: Maximum number of results (default: 10)
        ctx: Context object (optional)
    
    Returns:
        Dictionary with Reddit posts/comments data
    """
    if ctx:
        ctx.log("INFO", f"Fetching Reddit content: query={query}, thread_url={thread_url}, subreddit={subreddit}")
    
    try:
        # Convert thread URL to JSON API URL if provided
        if thread_url:
            # Add .json to the end of the URL to get the API response
            if not thread_url.endswith('.json'):
                if thread_url.endswith('/'):
                    api_url = f"{thread_url}.json"
                else:
                    api_url = f"{thread_url}/.json"
            else:
                api_url = thread_url
        
        # Search for posts based on query
        elif query:
            if subreddit:
                # Search within a subreddit
                api_url = f"https://www.reddit.com/r/{subreddit}/search.json?q={quote_plus(query)}&restrict_sr=1&sort={sort_by}&limit={limit}"
            else:
                # Search all of Reddit
                api_url = f"https://www.reddit.com/search.json?q={quote_plus(query)}&sort={sort_by}&limit={limit}"
        
        # Just get posts from a subreddit
        elif subreddit:
            api_url = f"https://www.reddit.com/r/{subreddit}/{sort_by}.json?limit={limit}"
        
        else:
            return {"error": "You must provide either a query, thread_url, or subreddit"}
        
        # Add user agent to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 MCP-Tool-RedditFetcher/1.0"
        }
        
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            response = await client.get(api_url)
            
            if response.status_code != 200:
                if ctx:
                    ctx.log("ERROR", f"Reddit request failed with status: {response.status_code}")
                return {"error": f"Reddit API request failed with status code: {response.status_code}"}
            
            # Parse the JSON response
            data = response.json()
            
            # Extract relevant information (simplified for demo)
            if thread_url:
                # For a thread, extract the post and comments
                post_data = data[0]["data"]["children"][0]["data"] if data and len(data) > 0 else {}
                comments_data = [c["data"] for c in data[1]["data"]["children"] if "data" in c] if len(data) > 1 else []
                
                # Clean up the data
                post = {
                    "title": post_data.get("title", ""),
                    "text": post_data.get("selftext", ""),
                    "author": post_data.get("author", ""),
                    "score": post_data.get("score", 0),
                    "url": post_data.get("url", ""),
                    "created_utc": post_data.get("created_utc", 0)
                }
                
                comments = []
                for comment in comments_data:
                    if "body" in comment:  # Only include actual comments
                        comments.append({
                            "text": comment.get("body", ""),
                            "author": comment.get("author", ""),
                            "score": comment.get("score", 0),
                            "created_utc": comment.get("created_utc", 0)
                        })
                
                result = {
                    "post": post,
                    "comments": comments,
                    "comment_count": len(comments)
                }
            
            else:
                # For search or subreddit listing, extract the posts
                posts_data = data["data"]["children"] if "data" in data and "children" in data["data"] else []
                
                posts = []
                for post_data in posts_data:
                    if "data" in post_data:
                        post = post_data["data"]
                        posts.append({
                            "title": post.get("title", ""),
                            "text": post.get("selftext", ""),
                            "author": post.get("author", ""),
                            "score": post.get("score", 0),
                            "url": post.get("url", ""),
                            "permalink": f"https://www.reddit.com{post.get('permalink', '')}",
                            "created_utc": post.get("created_utc", 0),
                            "num_comments": post.get("num_comments", 0)
                        })
                
                result = {
                    "query": query,
                    "subreddit": subreddit,
                    "sort_by": sort_by,
                    "posts": posts,
                    "post_count": len(posts)
                }
            
            if ctx:
                ctx.log("INFO", f"Successfully fetched Reddit content")
            
            return result
            
    except Exception as e:
        if ctx:
            ctx.log("ERROR", f"Reddit fetching error: {str(e)}")
        return {"error": f"Reddit fetching error: {str(e)}"}
