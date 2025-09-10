"""
Validation Data Manager for coordinating data collection from multiple sources.

This module provides a unified interface for collecting and validating data
from various sources including Reddit, Hacker News, SERP API, Twitter, and review sites.
"""

import time
from typing import Any, Dict, List, Optional, Union

from scout_agent.config import get_config
from scout_agent.custom_logging import get_logger
from scout_agent.sources.reddit_client import RedditClient
from scout_agent.sources.hn_client import HNClient
from scout_agent.sources.serper_client import SerperApiClient
from scout_agent.sources.twitter_client import TwitterClient
from scout_agent.sources.review_client import ReviewSitesClient

logger = get_logger("sources.validation_data_manager")


class ValidationDataManager:
    """Manager for collecting and validating data from multiple sources."""
    
    def __init__(self) -> None:
        """Initialize the validation data manager with all data source clients."""
        self.cfg = get_config()
        
        # Initialize clients
        try:
            self.reddit_client = RedditClient()
            logger.info("Reddit client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Reddit client: {e}")
            self.reddit_client = None
        
        try:
            self.hn_client = HNClient()
            logger.info("HN client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize HN client: {e}")
            self.hn_client = None
        
        try:
            self.serp_client = SerperApiClient()
            logger.info("SERP API client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize SERP API client: {e}")
            self.serp_client = None
        
        try:
            self.twitter_client = TwitterClient()
            logger.info("Twitter client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Twitter client: {e}")
            self.twitter_client = None
        
        try:
            self.review_client = ReviewSitesClient()
            logger.info("Review sites client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Review sites client: {e}")
            self.review_client = None
    
    def collect_data_for_topic(
        self,
        topic: str,
        keywords: Optional[List[str]] = None,
        subreddits: Optional[List[str]] = None,
        review_sites: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Collect data for a topic from all available sources.
        
        Args:
            topic: Main topic to research
            keywords: Additional keywords to search for (defaults to [topic])
            subreddits: Specific subreddits to search on Reddit
            review_sites: Specific review sites to search
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with collected data from all sources
        """
        if keywords is None:
            keywords = [topic]
        
        if review_sites is None:
            review_sites = ["g2", "capterra"]
        
        results = {
            "topic": topic,
            "keywords": keywords,
            "sources": {},
            "timestamp": int(time.time()),
        }
        
        # Collect Reddit data
        if self.reddit_client:
            try:
                reddit_data = self.reddit_client.fetch_conversations(
                    keywords=keywords,
                    subreddits=subreddits,
                    use_cache=use_cache,
                )
                results["sources"]["reddit"] = reddit_data
                logger.info(f"Collected {len(reddit_data)} Reddit items for topic '{topic}'")
            except Exception as e:
                logger.error(f"Failed to collect Reddit data: {e}")
                results["sources"]["reddit"] = {"error": str(e)}
        
        # Collect HN data
        if self.hn_client:
            try:
                hn_data = self.hn_client.fetch_conversations(
                    keywords=keywords,
                    use_cache=use_cache,
                )
                results["sources"]["hn"] = hn_data
                logger.info(f"Collected {len(hn_data)} HN items for topic '{topic}'")
            except Exception as e:
                logger.error(f"Failed to collect HN data: {e}")
                results["sources"]["hn"] = {"error": str(e)}
        
        # Collect Twitter data
        if self.twitter_client:
            try:
                twitter_data = self.twitter_client.fetch_conversations(
                    keywords=keywords,
                    use_cache=use_cache,
                )
                results["sources"]["twitter"] = twitter_data
                logger.info(f"Collected {len(twitter_data)} Twitter items for topic '{topic}'")
            except Exception as e:
                logger.error(f"Failed to collect Twitter data: {e}")
                results["sources"]["twitter"] = {"error": str(e)}
        
        # Collect Google Trends data
        if self.serp_client:
            try:
                trends_data = {}
                for kw in keywords:
                    trend = self.serp_client.get_google_trends(
                        query=kw,
                        use_cache=use_cache,
                    )
                    trends_data[kw] = self.serp_client.normalize_trends_results(trend)
                
                results["sources"]["trends"] = trends_data
                logger.info(f"Collected Google Trends data for {len(trends_data)} keywords")
            except Exception as e:
                logger.error(f"Failed to collect Google Trends data: {e}")
                results["sources"]["trends"] = {"error": str(e)}
        
        # Collect News data
        if self.serp_client:
            try:
                news_data = {}
                for kw in keywords:
                    news = self.serp_client.search_news(
                        query=kw,
                        use_cache=use_cache,
                    )
                    news_data[kw] = self.serp_client.normalize_news_results(news)
                
                results["sources"]["news"] = news_data
                logger.info(f"Collected news data for {len(news_data)} keywords")
            except Exception as e:
                logger.error(f"Failed to collect news data: {e}")
                results["sources"]["news"] = {"error": str(e)}
        
        # Collect Review data
        if self.review_client:
            try:
                review_data = self.review_client.fetch_product_reviews(
                    products=keywords,
                    sites=review_sites,
                    use_cache=use_cache,
                )
                results["sources"]["reviews"] = review_data
                logger.info(f"Collected {len(review_data)} review items for topic '{topic}'")
            except Exception as e:
                logger.error(f"Failed to collect review data: {e}")
                results["sources"]["reviews"] = {"error": str(e)}
        
        return results
    
    def validate_pain_point(
        self,
        pain_point: str,
        context: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Validate a pain point by collecting supporting evidence from all sources.
        
        Args:
            pain_point: The pain point to validate
            context: Additional context about the pain point
            keywords: Additional keywords to search for
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with validation results
        """
        # Generate search keywords from the pain point
        if keywords is None:
            keywords = [pain_point]
            # Add variations if context is provided
            if context:
                keywords.append(f"{pain_point} {context}")
        
        # Collect data for the pain point
        data = self.collect_data_for_topic(
            topic=pain_point,
            keywords=keywords,
            use_cache=use_cache,
        )
        
        # Add validation metadata
        validation_result = {
            "pain_point": pain_point,
            "context": context,
            "data": data,
            "validation_timestamp": int(time.time()),
        }
        
        return validation_result
    
    def compare_products(
        self,
        product1: str,
        product2: str,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Compare two products using review sites and search data.
        
        Args:
            product1: First product name
            product2: Second product name
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with comparison data
        """
        results = {
            "products": [product1, product2],
            "sources": {},
            "timestamp": int(time.time()),
        }
        
        # Get review comparisons
        if self.review_client:
            try:
                review_sites = ["g2", "capterra", "trustpilot"]
                comparisons = {}
                
                for site in review_sites:
                    comparison = self.review_client.compare_products(
                        product1=product1,
                        product2=product2,
                        site=site,
                        use_cache=use_cache,
                    )
                    comparisons[site] = comparison
                
                results["sources"]["reviews"] = comparisons
                logger.info(f"Collected comparison data for {product1} vs {product2}")
            except Exception as e:
                logger.error(f"Failed to collect review comparison data: {e}")
                results["sources"]["reviews"] = {"error": str(e)}
        
        # Get search comparisons
        if self.serp_client:
            try:
                search_data = self.serp_client.search_google(
                    query=f"{product1} vs {product2}",
                    use_cache=use_cache,
                )
                results["sources"]["search"] = self.serp_client.normalize_search_results(search_data)
                logger.info(f"Collected search data for {product1} vs {product2}")
            except Exception as e:
                logger.error(f"Failed to collect search comparison data: {e}")
                results["sources"]["search"] = {"error": str(e)}
        
        return results
    
    def get_market_trends(
        self,
        topic: str,
        related_terms: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Get market trends data for a topic.
        
        Args:
            topic: Main topic to research
            related_terms: Additional related terms to search for
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with market trends data
        """
        terms = [topic]
        if related_terms:
            terms.extend(related_terms)
        
        results = {
            "topic": topic,
            "terms": terms,
            "sources": {},
            "timestamp": int(time.time()),
        }
        
        # Get Google Trends data
        if self.serp_client:
            try:
                trends_data = {}
                for term in terms:
                    # Get trends for different time periods
                    for period, label in [
                        ("now 1-d", "last_day"),
                        ("now 7-d", "last_week"),
                        ("today 1-m", "last_month"),
                        ("today 12-m", "last_year"),
                    ]:
                        trend = self.serp_client.get_google_trends(
                            query=term,
                            date=period,
                            use_cache=use_cache,
                        )
                        if term not in trends_data:
                            trends_data[term] = {}
                        
                        trends_data[term][label] = self.serp_client.normalize_trends_results(trend)
                
                results["sources"]["trends"] = trends_data
                logger.info(f"Collected Google Trends data for {len(trends_data)} terms")
            except Exception as e:
                logger.error(f"Failed to collect Google Trends data: {e}")
                results["sources"]["trends"] = {"error": str(e)}
        
        # Get news data
        if self.serp_client:
            try:
                news_data = {}
                for term in terms:
                    news = self.serp_client.search_news(
                        query=term,
                        time_period="m",  # Last month
                        use_cache=use_cache,
                    )
                    news_data[term] = self.serp_client.normalize_news_results(news)
                
                results["sources"]["news"] = news_data
                logger.info(f"Collected news data for {len(news_data)} terms")
            except Exception as e:
                logger.error(f"Failed to collect news data: {e}")
                results["sources"]["news"] = {"error": str(e)}
        
        return results
