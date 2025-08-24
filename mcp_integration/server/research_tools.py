import json
from typing import Dict, Any, List, Optional

from mcp.types import TextContent

from scout_agent.mcp_integration.server.base import MCPServer
from scout_agent.sources.validation_data_manager import ValidationDataManager
from scout_agent.sources.hn_client import HNClient
from scout_agent.sources.serp_client import SerpApiClient
from scout_agent.sources.twitter_client import TwitterClient
from scout_agent.sources.review_client import ReviewSitesClient
from scout_agent.sources.reddit_client import RedditClient

# Create server instance
server = MCPServer(name="research-tools")

# Initialize data sources
validation_manager = ValidationDataManager()


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
async def search_reddit(
    query: str,
    subreddits: Optional[List[str]] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Search Reddit for posts matching a query.
    
    Args:
        query: Search query
        subreddits: Optional list of subreddits to search in
        limit: Maximum number of results to return
        
    Returns:
        List of Reddit posts
    """
    if validation_manager.reddit_client is None:
        validation_manager.reddit_client = RedditClient()
    
    posts = validation_manager.reddit_client.search_posts(
        query=query,
        subreddits=subreddits,
        limit=limit,
        use_cache=True,
    )
    
    payload = {"posts": posts}
    return {
        "content": [
            TextContent(type="text", text=json.dumps(payload))
        ]
    }


@server.tool()
async def get_reddit_thread(
    post_id: str,
    depth: int = 2,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Get a Reddit post with its comments.
    
    Args:
        post_id: Reddit post ID
        depth: Comment depth
        limit: Maximum number of comments to return
        
    Returns:
        Post with comments
    """
    if validation_manager.reddit_client is None:
        validation_manager.reddit_client = RedditClient()
    
    thread = validation_manager.reddit_client.get_post_with_comments(
        post_id=post_id,
        depth=depth,
        limit=limit,
        use_cache=True,
    )
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps(thread))
        ]
    }


@server.tool()
async def search_hn(
    query: str,
    tags: Optional[List[str]] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Search Hacker News for posts matching a query.
    
    Args:
        query: Search query
        tags: Optional list of tags to filter by (e.g., "story", "comment", "poll")
        limit: Maximum number of results to return
        
    Returns:
        List of HN posts
    """
    if validation_manager.hn_client is None:
        validation_manager.hn_client = HNClient()
    
    posts = validation_manager.hn_client.search_posts(
        query=query,
        tags=tags,
        limit=limit,
        use_cache=True,
    )
    
    payload = {"posts": posts}
    return {
        "content": [
            TextContent(type="text", text=json.dumps(payload))
        ]
    }


@server.tool()
async def get_hn_thread(
    item_id: str,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Get a Hacker News item with its comments.
    
    Args:
        item_id: HN item ID
        limit: Maximum number of comments to return
        
    Returns:
        Item with comments
    """
    if validation_manager.hn_client is None:
        validation_manager.hn_client = HNClient()
    
    thread = validation_manager.hn_client.get_item_with_comments(
        item_id=item_id,
        limit=limit,
        use_cache=True,
    )
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps(thread))
        ]
    }


@server.tool()
async def search_google(
    query: str,
    location: str = "United States",
    num_results: int = 10,
) -> Dict[str, Any]:
    """
    Search Google and return results.
    
    Args:
        query: Search query
        location: Location to search from
        num_results: Number of results to return
        
    Returns:
        Search results
    """
    if validation_manager.serp_client is None:
        validation_manager.serp_client = SerpApiClient()
    
    data = validation_manager.serp_client.search_google(
        query=query,
        location=location,
        num_results=num_results,
        use_cache=True,
    )
    
    normalized = validation_manager.serp_client.normalize_search_results(data)
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps({"results": normalized}))
        ]
    }


@server.tool()
async def get_google_trends(
    query: str = None,
    date: str = "now 7-d",
) -> Dict[str, Any]:
    """
    Get Google Trends data.
    
    Args:
        query: Optional search query to get related trends
        date: Time range (e.g., "now 1-d", "now 7-d", "today 12-m")
        
    Returns:
        Trends data
    """
    if validation_manager.serp_client is None:
        validation_manager.serp_client = SerpApiClient()
    
    data = validation_manager.serp_client.get_google_trends(
        query=query,
        date=date,
        use_cache=True,
    )
    
    normalized = validation_manager.serp_client.normalize_trends_results(data)
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps({"trends": normalized}))
        ]
    }


@server.tool()
async def search_news(
    query: str,
    time_period: str = "w",  # "h" (hour), "d" (day), "w" (week), "m" (month), "y" (year)
    num_results: int = 10,
) -> Dict[str, Any]:
    """
    Search Google News for articles.
    
    Args:
        query: Search query
        time_period: Time period to search in
        num_results: Number of results to return
        
    Returns:
        News articles
    """
    if validation_manager.serp_client is None:
        validation_manager.serp_client = SerpApiClient()
    
    data = validation_manager.serp_client.search_news(
        query=query,
        time_period=time_period,
        num_results=num_results,
        use_cache=True,
    )
    
    normalized = validation_manager.serp_client.normalize_news_results(data)
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps({"news": normalized}))
        ]
    }


@server.tool()
async def search_twitter(
    query: str,
    limit: int = 20,
    min_likes: int = 0,
) -> Dict[str, Any]:
    """
    Search Twitter for tweets matching a query.
    
    Args:
        query: Search query
        limit: Maximum number of results to return
        min_likes: Minimum number of likes
        
    Returns:
        List of tweets
    """
    if validation_manager.twitter_client is None:
        validation_manager.twitter_client = TwitterClient()
    
    tweets = validation_manager.twitter_client.search_by_topic(
        topic=query,
        limit=limit,
        min_likes=min_likes,
        use_cache=True,
        force_mock=True,  # Always use mock data for development
    )
    
    payload = {"tweets": tweets}
    return {
        "content": [
            TextContent(type="text", text=json.dumps(payload))
        ]
    }


@server.tool()
async def search_product_reviews(
    product: str,
    sites: Optional[List[str]] = None,
    num_results: int = 5,
) -> Dict[str, Any]:
    """
    Search for product reviews across review sites.
    
    Args:
        product: Product name to search for
        sites: List of review sites to search on (defaults to g2, capterra)
        num_results: Number of results per site
        
    Returns:
        Product reviews from various sites
    """
    if validation_manager.review_client is None:
        validation_manager.review_client = ReviewSitesClient()
    
    if sites is None:
        sites = ["g2", "capterra"]
    
    reviews = validation_manager.review_client.search_multiple_sites(
        product=product,
        sites=sites,
        num_results=num_results,
        use_cache=True,
    )
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps({"reviews": reviews}))
        ]
    }


@server.tool()
async def compare_products(
    product1: str,
    product2: str,
    site: str = "g2",
) -> Dict[str, Any]:
    """
    Compare two products using review sites.
    
    Args:
        product1: First product name
        product2: Second product name
        site: Review site to search on
        
    Returns:
        Comparison results
    """
    if validation_manager.review_client is None:
        validation_manager.review_client = ReviewSitesClient()
    
    comparison = validation_manager.review_client.compare_products(
        product1=product1,
        product2=product2,
        site=site,
        use_cache=True,
    )
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps({"comparison": comparison}))
        ]
    }


@server.tool()
async def validate_pain_point(
    pain_point: str,
    context: Optional[str] = None,
    keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate a pain point by collecting supporting evidence from all sources.
    
    Args:
        pain_point: The pain point to validate
        context: Additional context about the pain point
        keywords: Additional keywords to search for
        
    Returns:
        Validation results with evidence from multiple sources
    """
    validation_result = validation_manager.validate_pain_point(
        pain_point=pain_point,
        context=context,
        keywords=keywords,
        use_cache=True,
    )
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps(validation_result))
        ]
    }


@server.tool()
async def get_market_trends(
    topic: str,
    related_terms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Get market trends data for a topic.
    
    Args:
        topic: Main topic to research
        related_terms: Additional related terms to search for
        
    Returns:
        Market trends data
    """
    trends_data = validation_manager.get_market_trends(
        topic=topic,
        related_terms=related_terms,
        use_cache=True,
    )
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps(trends_data))
        ]
    }


# Expose ASGI app for uvicorn: `uvicorn scout_agent.mcp_integration.server.research_tools:app`
app = server.asgi_app()

# Allow running the module directly
if __name__ == "__main__":
    import uvicorn
    print("Starting research tools server on port 8002...")
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
