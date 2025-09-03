import json
from typing import Dict, Any, List, Optional
from datetime import datetime

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

# For running as a module
mcp = server._mcp

# Initialize data sources
validation_manager = ValidationDataManager()

# Main block to make the file runnable as a module
if __name__ == "__main__":
    mcp.run(transport="sse")


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
async def reddit_research(
    query: str,
    subreddits: Optional[List[str]] = None,
    post_limit: int = 10,
    include_comments: bool = True,
    comment_depth: int = 2,
    comment_limit: int = 20,
    min_score: int = 3,
    time_filter: str = "all",  # all, year, month, week, day
) -> Dict[str, Any]:
    """
    Comprehensive Reddit research tool that searches for posts and retrieves comments in a single call.
    
    Args:
        query: Search query
        subreddits: Optional list of subreddits to search in
        post_limit: Maximum number of posts to return
        include_comments: Whether to include comments for each post
        comment_depth: Depth of comment tree to retrieve (if include_comments is True)
        comment_limit: Maximum number of comments per post (if include_comments is True)
        min_score: Minimum score (upvotes) for posts to include
        time_filter: Time filter for posts (all, year, month, week, day)
        
    Returns:
        Dictionary with posts and their comments
    """
    if validation_manager.reddit_client is None:
        validation_manager.reddit_client = RedditClient()
    
    # First search for posts
    posts = validation_manager.reddit_client.search_posts(
        query=query,
        subreddits=subreddits,
        limit=post_limit * 2,  # Get more posts than needed for filtering
        use_cache=True,
    )
    
    # Filter posts by score
    filtered_posts = [post for post in posts if post.get("score", 0) >= min_score]
    filtered_posts = filtered_posts[:post_limit]  # Limit to requested number
    
    result = {
        "posts": filtered_posts,
        "posts_with_comments": []
    }
    
    # If requested, get comments for each post
    if include_comments and filtered_posts:
        posts_with_comments = []
        for post in filtered_posts:
            post_id = post.get("id")
            if post_id:
                try:
                    thread = validation_manager.reddit_client.get_post_with_comments(
                        post_id=post_id,
                        depth=comment_depth,
                        limit=comment_limit,
                        use_cache=True,
                    )
                    posts_with_comments.append(thread)
                except Exception as e:
                    # If we fail to get comments for a post, just include the post without comments
                    posts_with_comments.append({
                        "post": post,
                        "comments": []
                    })
        
        result["posts_with_comments"] = posts_with_comments
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps(result))
        ]
    }


@server.tool()
async def hackernews_research(
    query: str,
    tags: Optional[List[str]] = None,
    post_limit: int = 10,
    include_comments: bool = True,
    comment_limit: int = 30,
    min_points: int = 5,
    sort_by: str = "popularity",  # popularity, date, relevance
) -> Dict[str, Any]:
    """
    Comprehensive Hacker News research tool that searches for posts and retrieves comments in a single call.
    
    Args:
        query: Search query
        tags: Optional list of tags to filter by (e.g., "story", "comment", "poll")
        post_limit: Maximum number of posts to return
        include_comments: Whether to include comments for each post
        comment_limit: Maximum number of comments per post (if include_comments is True)
        min_points: Minimum points for posts to include
        sort_by: How to sort results (popularity, date, relevance)
        
    Returns:
        Dictionary with posts and their comments
    """
    if validation_manager.hn_client is None:
        validation_manager.hn_client = HNClient()
    
    # First search for posts
    posts = validation_manager.hn_client.search_posts(
        query=query,
        tags=tags,
        limit=post_limit * 2,  # Get more posts than needed for filtering
        use_cache=True,
    )
    
    # Filter posts by points
    filtered_posts = [post for post in posts if post.get("points", 0) >= min_points]
    filtered_posts = filtered_posts[:post_limit]  # Limit to requested number
    
    result = {
        "posts": filtered_posts,
        "posts_with_comments": []
    }
    
    # If requested, get comments for each post
    if include_comments and filtered_posts:
        posts_with_comments = []
        for post in filtered_posts:
            item_id = post.get("objectID") or post.get("id")
            if item_id:
                try:
                    thread = validation_manager.hn_client.get_item_with_comments(
                        item_id=item_id,
                        limit=comment_limit,
                        use_cache=True,
                    )
                    posts_with_comments.append(thread)
                except Exception as e:
                    # If we fail to get comments for a post, just include the post without comments
                    posts_with_comments.append({
                        "item": post,
                        "comments": []
                    })
        
        result["posts_with_comments"] = posts_with_comments
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps(result))
        ]
    }


@server.tool()
async def google_research(
    query: str,
    location: str = "United States",
    search_limit: int = 10,
    include_news: bool = True,
    news_time_period: str = "w",  # "h" (hour), "d" (day), "w" (week), "m" (month), "y" (year)
    news_limit: int = 5,
    include_trends: bool = True,
    trends_date_range: str = "now 7-d",
    include_autocomplete: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive Google research tool that searches web, news, trends, and autocomplete in a single call.
    
    Args:
        query: Search query
        location: Location to search from
        search_limit: Number of web search results to return
        include_news: Whether to include news results
        news_time_period: Time period for news search
        news_limit: Number of news results to return
        include_trends: Whether to include Google Trends data
        trends_date_range: Time range for trends data
        include_autocomplete: Whether to include autocomplete suggestions
        
    Returns:
        Dictionary with web search results, news, trends, and autocomplete suggestions
    """
    if validation_manager.serp_client is None:
        validation_manager.serp_client = SerpApiClient()
    
    result = {}
    
    # Get web search results
    search_data = validation_manager.serp_client.search_google(
        query=query,
        location=location,
        num_results=search_limit,
        use_cache=True,
    )
    result["web_results"] = validation_manager.serp_client.normalize_search_results(search_data)
    
    # Get news results if requested
    if include_news:
        news_data = validation_manager.serp_client.search_news(
            query=query,
            time_period=news_time_period,
            num_results=news_limit,
            use_cache=True,
        )
        result["news_results"] = validation_manager.serp_client.normalize_news_results(news_data)
    
    # Get trends data if requested
    if include_trends:
        trends_data = validation_manager.serp_client.get_google_trends(
            query=query,
            date=trends_date_range,
            use_cache=True,
        )
        result["trends_data"] = validation_manager.serp_client.normalize_trends_results(trends_data)
    
    # Get autocomplete suggestions if requested
    if include_autocomplete:
        try:
            autocomplete_data = validation_manager.serp_client.get_autocomplete(
                query=query,
                use_cache=True,
            )
            result["autocomplete_suggestions"] = autocomplete_data.get("suggestions", [])
        except Exception as e:
            result["autocomplete_suggestions"] = []
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps(result))
        ]
    }


@server.tool()
async def twitter_research(
    query: str,
    limit: int = 20,
    min_likes: int = 0,
    include_user_tweets: bool = False,
    user_handles: Optional[List[str]] = None,
    user_tweet_limit: int = 10,
    include_conversations: bool = True,
    conversation_limit: int = 5,
) -> Dict[str, Any]:
    """
    Comprehensive Twitter research tool that searches tweets, user timelines, and conversations in a single call.
    
    Args:
        query: Search query
        limit: Maximum number of search results to return
        min_likes: Minimum number of likes for tweets
        include_user_tweets: Whether to include tweets from specific users
        user_handles: List of Twitter user handles to get tweets from (if include_user_tweets is True)
        user_tweet_limit: Maximum number of tweets per user (if include_user_tweets is True)
        include_conversations: Whether to include conversation threads for top search results
        conversation_limit: Maximum number of conversations to include
        
    Returns:
        Dictionary with search results, user tweets, and conversations
    """
    if validation_manager.twitter_client is None:
        validation_manager.twitter_client = TwitterClient()
    
    result = {}
    
    # Search tweets by query
    search_tweets = validation_manager.twitter_client.search_by_topic(
        topic=query,
        limit=limit,
        min_likes=min_likes,
        use_cache=True,
        force_mock=True,  # Always use mock data for development
    )
    result["search_results"] = search_tweets
    
    # Get tweets from specific users if requested
    if include_user_tweets and user_handles:
        user_tweets = {}
        for handle in user_handles:
            try:
                tweets = validation_manager.twitter_client.get_user_tweets(
                    username=handle,
                    limit=user_tweet_limit,
                    use_cache=True,
                    force_mock=True,  # Always use mock data for development
                )
                user_tweets[handle] = tweets
            except Exception as e:
                user_tweets[handle] = []
        
        result["user_tweets"] = user_tweets
    
    # Get conversation threads for top search results if requested
    if include_conversations and search_tweets:
        conversations = []
        # Sort tweets by likes to get the most engaging ones
        top_tweets = sorted(search_tweets, key=lambda t: t.get("likes", 0), reverse=True)[:conversation_limit]
        
        for tweet in top_tweets:
            tweet_id = tweet.get("id")
            if tweet_id:
                try:
                    conversation = validation_manager.twitter_client.fetch_conversation(
                        tweet_id=tweet_id,
                        use_cache=True,
                        force_mock=True,  # Always use mock data for development
                    )
                    conversations.append(conversation)
                except Exception as e:
                    pass  # Skip if we can't get the conversation
        
        result["conversations"] = conversations
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps(result))
        ]
    }


@server.tool()
async def product_research(
    product: str,
    sites: Optional[List[str]] = None,
    review_limit: int = 5,
    include_alternatives: bool = True,
    max_alternatives: int = 3,
    compare_with: Optional[List[str]] = None,
    include_sentiment: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive product research tool that searches for reviews, alternatives, and comparisons in a single call.
    
    Args:
        product: Product name to search for
        sites: List of review sites to search on (defaults to g2, capterra)
        review_limit: Number of reviews per site
        include_alternatives: Whether to include alternative products
        max_alternatives: Maximum number of alternatives to include
        compare_with: List of product names to compare with the main product
        include_sentiment: Whether to include sentiment analysis of reviews
        
    Returns:
        Dictionary with product reviews, alternatives, and comparisons
    """
    if validation_manager.review_client is None:
        validation_manager.review_client = ReviewSitesClient()
    
    if sites is None:
        sites = ["g2", "capterra"]
    
    result = {}
    
    # Get product reviews
    reviews = validation_manager.review_client.search_multiple_sites(
        product=product,
        sites=sites,
        num_results=review_limit,
        use_cache=True,
    )
    result["reviews"] = reviews
    
    # Find alternative products if requested
    if include_alternatives:
        try:
            alternatives = validation_manager.review_client.find_alternatives(
                product=product,
                max_results=max_alternatives,
                use_cache=True,
            )
            result["alternatives"] = alternatives
        except Exception as e:
            result["alternatives"] = []
    
    # Compare with other products if requested
    if compare_with:
        comparisons = {}
        for other_product in compare_with:
            try:
                comparison = validation_manager.review_client.compare_products(
                    product1=product,
                    product2=other_product,
                    site="g2",  # Default to G2 for comparisons
                    use_cache=True,
                )
                comparisons[other_product] = comparison
            except Exception as e:
                comparisons[other_product] = {"error": str(e)}
        
        result["comparisons"] = comparisons
    
    # Add sentiment analysis if requested
    if include_sentiment and reviews:
        try:
            # Simple sentiment analysis based on review content
            # In a real implementation, this would use NLP
            sentiment = {
                "overall": "neutral",
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0
            }
            
            # For now, just count positive/negative keywords in reviews
            positive_keywords = ["great", "excellent", "good", "best", "love", "easy", "helpful", "recommend"]
            negative_keywords = ["bad", "poor", "difficult", "hard", "terrible", "worst", "hate", "avoid"]
            
            for site, site_reviews in reviews.items():
                for review in site_reviews:
                    text = review.get("text", "").lower()
                    pos_count = sum(1 for word in positive_keywords if word in text)
                    neg_count = sum(1 for word in negative_keywords if word in text)
                    
                    if pos_count > neg_count:
                        sentiment["positive_count"] += 1
                    elif neg_count > pos_count:
                        sentiment["negative_count"] += 1
                    else:
                        sentiment["neutral_count"] += 1
            
            # Determine overall sentiment
            if sentiment["positive_count"] > sentiment["negative_count"]:
                sentiment["overall"] = "positive"
            elif sentiment["negative_count"] > sentiment["positive_count"]:
                sentiment["overall"] = "negative"
            
            result["sentiment_analysis"] = sentiment
        except Exception as e:
            result["sentiment_analysis"] = {"overall": "unknown", "error": str(e)}
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps(result))
        ]
    }


@server.tool()
async def comprehensive_research(
    topic: str,
    context: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    depth: str = "medium",  # "light", "medium", "deep"
    max_results_per_source: int = 10,
) -> Dict[str, Any]:
    """
    Perform comprehensive research across all available sources with a single call.
    
    Args:
        topic: Main topic to research
        context: Additional context about the topic
        keywords: Additional keywords to search for
        sources: List of sources to use (defaults to all: "reddit", "hn", "google", "twitter", "reviews")
        depth: Research depth determining how much data to collect
        max_results_per_source: Maximum number of results to return per source
        
    Returns:
        Comprehensive research results from all specified sources
    """
    import logging
    logger = logging.getLogger("comprehensive_research")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    logger.debug(f"Starting comprehensive research for topic: {topic}")
    logger.debug(f"Context: {context}")
    logger.debug(f"Keywords: {keywords}")
    logger.debug(f"Sources: {sources}")
    logger.debug(f"Depth: {depth}")
    logger.debug(f"Max results per source: {max_results_per_source}")
    if sources is None:
        sources = ["reddit", "hn", "google", "twitter", "reviews"]
    
    # Adjust limits based on research depth
    if depth == "light":
        post_limit = 5
        comment_limit = 10
        search_limit = 5
    elif depth == "deep":
        post_limit = 20
        comment_limit = 30
        search_limit = 15
    else:  # medium
        post_limit = 10
        comment_limit = 20
        search_limit = 10
    
    # Prepare search queries
    queries = [topic]
    if keywords:
        queries.extend(keywords)
    if context:
        queries.append(f"{topic} {context}")
    
    # Limit to max_results_per_source
    post_limit = min(post_limit, max_results_per_source)
    search_limit = min(search_limit, max_results_per_source)
    
    result = {
        "topic": topic,
        "context": context,
        "keywords": keywords,
        "sources_used": sources,
        "depth": depth,
        "data": {}
    }
    
    def make_json_serializable(obj):
        """Convert objects to JSON serializable format"""
        if obj is None:
            return None
        
        if hasattr(obj, "__dict__"):
            # For objects with __dict__, convert to dict
            return {k: make_json_serializable(v) for k, v in obj.__dict__.items() 
                   if not k.startswith("_")}
        elif isinstance(obj, dict):
            # For dictionaries, recursively convert values
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # For lists, recursively convert items
            return [make_json_serializable(item) for item in obj]
        elif hasattr(obj, "type") and hasattr(obj, "text") and callable(getattr(obj, "__str__", None)):
            # Special case for TextContent-like objects
            return {"type": obj.type, "text": obj.text}
        else:
            # Return primitive types as is
            return obj
    
    def extract_data_from_response(response):
        """Helper function to extract data from MCP tool responses"""
        logger.debug(f"Extracting data from response type: {type(response)}")
        
        # If response is already a dict, use it directly
        if isinstance(response, dict):
            # If it's an MCP response with content structure
            if "content" in response and isinstance(response["content"], list) and len(response["content"]) > 0:
                content_item = response["content"][0]
                logger.debug(f"Content item type: {type(content_item)}")
                
                # Handle TextContent dict format
                if isinstance(content_item, dict):
                    if "text" in content_item:
                        try:
                            logger.debug(f"Parsing text field from content item")
                            data = json.loads(content_item["text"])
                            # Make sure the data is JSON serializable
                            return make_json_serializable(data)
                        except Exception as e:
                            logger.debug(f"Failed to parse text as JSON: {e}")
                            return None
            
            # If it doesn't have the expected structure but is a dict, return it
            return make_json_serializable(response)
        
        # Handle object-style response with attributes
        try:
            content_list = getattr(response, "content", [])
            if content_list and len(content_list) > 0:
                content = content_list[0]
                text = getattr(content, "text", "{}")
                data = json.loads(text)
                return make_json_serializable(data)
        except Exception as e:
            logger.debug(f"Failed to extract data from object attributes: {e}")
        
        return None
    
    # Collect data from each requested source
    if "reddit" in sources:
        try:
            logger.debug("Calling reddit_research...")
            reddit_data = await reddit_research(
                query=topic,
                post_limit=post_limit,
                include_comments=True,
                comment_depth=2,
                comment_limit=comment_limit,
                min_score=3
            )
            logger.debug(f"Reddit data type: {type(reddit_data)}")
            
            # Use our helper function to extract the data
            extracted_data = extract_data_from_response(reddit_data)
            if extracted_data:
                result["data"]["reddit"] = extracted_data
                logger.debug("Successfully added Reddit data")
            else:
                logger.debug("Failed to extract Reddit data")
        except Exception as e:
            logger.debug(f"Error processing Reddit data: {str(e)}")
            result["data"]["reddit"] = {"error": str(e)}
    
    if "hn" in sources:
        try:
            logger.debug("Calling hackernews_research...")
            hn_data = await hackernews_research(
                query=topic,
                post_limit=post_limit,
                include_comments=True,
                comment_limit=comment_limit,
                min_points=5
            )
            logger.debug(f"HN data type: {type(hn_data)}")
            
            # Use our helper function to extract the data
            extracted_data = extract_data_from_response(hn_data)
            if extracted_data:
                result["data"]["hackernews"] = extracted_data
                logger.debug("Successfully added HN data")
            else:
                logger.debug("Failed to extract HN data")
        except Exception as e:
            logger.debug(f"Error processing HN data: {str(e)}")
            result["data"]["hackernews"] = {"error": str(e)}
    
    if "google" in sources:
        try:
            logger.debug("Calling google_research...")
            google_data = await google_research(
                query=topic,
                search_limit=search_limit,
                include_news=True,
                news_limit=5,
                include_trends=True
            )
            logger.debug(f"Google data type: {type(google_data)}")
            
            # Use our helper function to extract the data
            extracted_data = extract_data_from_response(google_data)
            if extracted_data:
                result["data"]["google"] = extracted_data
                logger.debug("Successfully added Google data")
            else:
                logger.debug("Failed to extract Google data")
        except Exception as e:
            logger.debug(f"Error processing Google data: {str(e)}")
            result["data"]["google"] = {"error": str(e)}
    
    if "twitter" in sources:
        try:
            logger.debug("Calling twitter_research...")
            twitter_data = await twitter_research(
                query=topic,
                limit=post_limit,
                min_likes=5,
                include_conversations=True,
                conversation_limit=3
            )
            logger.debug(f"Twitter data type: {type(twitter_data)}")
            
            # Use our helper function to extract the data
            extracted_data = extract_data_from_response(twitter_data)
            if extracted_data:
                result["data"]["twitter"] = extracted_data
                logger.debug("Successfully added Twitter data")
            else:
                logger.debug("Failed to extract Twitter data")
        except Exception as e:
            logger.debug(f"Error processing Twitter data: {str(e)}")
            result["data"]["twitter"] = {"error": str(e)}
    
    if "reviews" in sources and context:
        # Try to extract product names from context or topic
        product_terms = []
        if "vs" in topic:
            # Handle comparison queries
            products = topic.split("vs")
            product_terms = [p.strip() for p in products]
        else:
            # Use the topic as product name
            product_terms = [topic]
        
        if product_terms:
            try:
                logger.debug("Calling product_research...")
                reviews_data = await product_research(
                    product=product_terms[0],
                    review_limit=5,
                    include_alternatives=True,
                    compare_with=product_terms[1:] if len(product_terms) > 1 else None
                )
                logger.debug(f"Reviews data type: {type(reviews_data)}")
                
                # Use our helper function to extract the data
                extracted_data = extract_data_from_response(reviews_data)
                if extracted_data:
                    result["data"]["reviews"] = extracted_data
                    logger.debug("Successfully added Reviews data")
                else:
                    logger.debug("Failed to extract Reviews data")
            except Exception as e:
                logger.debug(f"Error processing Reviews data: {str(e)}")
                result["data"]["reviews"] = {"error": str(e)}
    
    # Add summary metadata
    result["summary"] = {
        "sources_successful": [source for source in sources if source in result["data"] and result["data"][source] and not (isinstance(result["data"][source], dict) and "error" in result["data"][source])],
        "total_data_points": sum(1 for source in result["data"].values() if isinstance(source, dict) for _ in source.values()),
        "timestamp": datetime.now().isoformat()
    }
    
    # Log the successful sources for debugging
    logger.debug(f"Sources in data: {list(result['data'].keys())}")
    logger.debug(f"Successful sources: {result['summary']['sources_successful']}")
    
    # Make sure the entire result is JSON serializable
    serializable_result = make_json_serializable(result)
    
    return {
        "content": [
            TextContent(type="text", text=json.dumps(serializable_result))
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
