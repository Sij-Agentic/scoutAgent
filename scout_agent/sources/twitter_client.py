import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from scout_agent.config import get_config
from scout_agent.custom_logging import get_logger
from scout_agent.data_cache.file_cache import FileCache

logger = get_logger("sources.twitter_client")


@dataclass
class TwitterAuth:
    bearer_token: Optional[str] = None  # Optional for snscrape which doesn't require auth


class TwitterClient:
    """Twitter client using snscrape for scraping tweets without API authentication.
    
    This implementation uses snscrape via subprocess to avoid direct dependency.
    Make sure snscrape is installed: pip install snscrape
    """
    
    def __init__(self) -> None:
        cfg = get_config()
        bearer_token = (cfg.api.__dict__.get("twitter_bearer_token") or 
                        self._getenv_default("SCOUT_TWITTER_BEARER_TOKEN"))
        
        self.auth = TwitterAuth(bearer_token=bearer_token)
        self.cache = FileCache(base_subdir="twitter_cache")
        self.cfg = cfg
        
        # Check if snscrape is installed
        self.snscrape_available = False
        try:
            subprocess.run(["snscrape", "--version"], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE, 
                          check=True)
            self.snscrape_available = True
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("snscrape not found. Please install with: pip install git+https://github.com/JustAnotherArchivist/snscrape.git")
    
    def _getenv_default(self, k: str) -> Optional[str]:
        import os
        return os.getenv(k) or None
    
    @staticmethod
    def _hash_key(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()
    
    def _run_snscrape(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Run snscrape command and parse the output.
        
        Args:
            query: Twitter search query
            limit: Maximum number of tweets to return
            
        Returns:
            List of tweets as dictionaries
        """
        if not self.snscrape_available:
            logger.warning("snscrape not available. Returning empty results.")
            return []
            
        cmd = [
            "snscrape",
            "--jsonl",
            "--progress",
            "--max-results", str(limit),
            "twitter-search", query
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False  # Don't raise exception on non-zero exit
            )
            
            if result.returncode != 0:
                logger.warning(f"snscrape exited with code {result.returncode}: {result.stderr}")
                if "No items were returned" in result.stderr:
                    return []  # No results is not an error
                
                # Check if it's just a warning about max results
                if "Max results reached" in result.stderr:
                    pass  # This is expected behavior
                else:
                    # Real error
                    logger.error(f"snscrape error: {result.stderr}")
                    return []
            
            # Parse JSON lines
            tweets = []
            for line in result.stdout.splitlines():
                if line.strip():
                    try:
                        tweet = json.loads(line)
                        tweets.append(tweet)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tweet JSON: {line}")
            
            return tweets
            
        except Exception as e:
            logger.error(f"Error running snscrape: {e}")
            return []
    
    def search_tweets(
        self,
        query: str,
        limit: int = 100,
        use_cache: bool = True,
        force_mock: bool = True,  # Always use mock data in development
    ) -> List[Dict[str, Any]]:
        """Search tweets and return results.
        
        Args:
            query: Twitter search query
            limit: Maximum number of tweets to return
            use_cache: Whether to use the cache
            force_mock: Whether to force using mock data (for development)
            
        Returns:
            List of tweets
        """
        # Generate cache key
        cache_key = self._hash_key(json.dumps({
            "query": query,
            "limit": limit,
        }, sort_keys=True))
        
        # Try cache first
        ttl = self.cfg.agent.cache_ttl if (use_cache and self.cfg.agent.enable_caching) else None
        if use_cache:
            cached = self.cache.load("search", cache_key, ttl)
            if cached is not None:
                return cached
        
        if force_mock or not self.snscrape_available:
            # Generate mock data for development
            logger.info(f"Generating mock data for query: {query}")
            mock_tweets = self._generate_mock_tweets(query, limit)
            self.cache.save("search", cache_key, mock_tweets)
            return mock_tweets
        
        # Run snscrape
        tweets = self._run_snscrape(query, limit=min(limit, self.cfg.search.max_results))
        
        # Normalize tweets
        normalized_tweets = self._normalize_tweets(tweets)
        
        # Save to cache
        self.cache.save("search", cache_key, normalized_tweets)
        return normalized_tweets
    
    def get_user_tweets(
        self,
        username: str,
        limit: int = 100,
        use_cache: bool = True,
        force_mock: bool = True,  # Always use mock data in development
    ) -> List[Dict[str, Any]]:
        """Get tweets from a specific user.
        
        Args:
            username: Twitter username (without @)
            limit: Maximum number of tweets to return
            use_cache: Whether to use the cache
            force_mock: Whether to force using mock data (for development)
            
        Returns:
            List of tweets
        """
        query = f"from:{username}"
        return self.search_tweets(query, limit=limit, use_cache=use_cache, force_mock=force_mock)
    
    def search_by_topic(
        self,
        topic: str,
        limit: int = 100,
        min_likes: int = 0,
        min_retweets: int = 0,
        language: str = "en",
        use_cache: bool = True,
        force_mock: bool = True,  # Always use mock data in development
    ) -> List[Dict[str, Any]]:
        """Search tweets by topic with additional filters.
        
        Args:
            topic: Topic to search for
            limit: Maximum number of tweets to return
            min_likes: Minimum number of likes
            min_retweets: Minimum number of retweets
            language: Language code
            use_cache: Whether to use the cache
            force_mock: Whether to force using mock data (for development)
            
        Returns:
            List of tweets
        """
        query_parts = [topic]
        
        if min_likes > 0:
            query_parts.append(f"min_faves:{min_likes}")
        
        if min_retweets > 0:
            query_parts.append(f"min_retweets:{min_retweets}")
        
        if language:
            query_parts.append(f"lang:{language}")
        
        query = " ".join(query_parts)
        return self.search_tweets(query, limit=limit, use_cache=use_cache, force_mock=force_mock)
    
    def _generate_mock_tweets(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Generate mock tweets for development purposes.
        
        Args:
            query: Search query
            limit: Number of mock tweets to generate
            
        Returns:
            List of mock tweets
        """
        mock_tweets = []
        current_time = int(time.time())
        
        # Sample usernames and display names
        usernames = ["dev_user", "tech_enthusiast", "coder123", "programmer", "debugger"]
        display_names = ["Developer", "Tech Enthusiast", "Coder", "Programmer", "Debugger"]
        
        # Sample hashtags related to the query
        hashtags = [f"#{query.replace(' ', '')}", "#coding", "#development", "#tech", "#programming"]
        
        for i in range(min(limit, 20)):  # Generate up to 20 mock tweets
            # Create a mock tweet with the query in the content
            tweet_id = f"{current_time - i * 3600}"
            mock_tweet = {
                "id": tweet_id,
                "url": f"https://twitter.com/user/status/{tweet_id}",
                "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_time - i * 3600)),
                "content": f"This is a mock tweet about {query}. #coding #{query.replace(' ', '')}",
                "username": usernames[i % len(usernames)],
                "display_name": display_names[i % len(display_names)],
                "retweet_count": i * 3,
                "like_count": i * 10,
                "reply_count": i,
                "quote_count": i // 2,
                "is_retweet": False,
                "hashtags": [hashtags[i % len(hashtags)], hashtags[(i + 1) % len(hashtags)]],
                "mentions": [],
                "language": "en",
            }
            mock_tweets.append(mock_tweet)
        
        return mock_tweets
    
    def _normalize_tweets(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize tweets to a common format.
        
        Args:
            tweets: Raw tweets from snscrape
            
        Returns:
            List of normalized tweets
        """
        normalized = []
        
        for tweet in tweets:
            # Extract the essential information
            normalized_tweet = {
                "id": tweet.get("id"),
                "url": tweet.get("url"),
                "date": tweet.get("date"),
                "content": tweet.get("content"),
                "username": tweet.get("user", {}).get("username") if isinstance(tweet.get("user"), dict) else tweet.get("username"),
                "display_name": tweet.get("user", {}).get("displayname") if isinstance(tweet.get("user"), dict) else tweet.get("display_name"),
                "retweet_count": tweet.get("retweetCount", 0),
                "like_count": tweet.get("likeCount", 0),
                "reply_count": tweet.get("replyCount", 0),
                "quote_count": tweet.get("quoteCount", 0),
                "is_retweet": tweet.get("isRetweet", False),
                "hashtags": tweet.get("hashtags", []),
                "mentions": [user.get("username") for user in tweet.get("mentionedUsers", []) if user.get("username")] if tweet.get("mentionedUsers") else [],
                "language": tweet.get("lang"),
            }
            
            # Add media information if available
            if "media" in tweet:
                media_items = []
                for media in tweet.get("media", []):
                    media_type = media.get("type")
                    if media_type == "photo":
                        media_items.append({
                            "type": "photo",
                            "url": media.get("fullUrl"),
                        })
                    elif media_type == "video":
                        media_items.append({
                            "type": "video",
                            "thumbnail": media.get("thumbnailUrl"),
                            "duration": media.get("duration"),
                        })
                
                normalized_tweet["media"] = media_items
            
            normalized.append(normalized_tweet)
        
        return normalized
    
    def fetch_conversations(
        self,
        keywords: List[str],
        limit_per_keyword: int = 20,
        min_likes: int = 10,
        language: str = "en",
        use_cache: bool = True,
        force_mock: bool = True,  # Always use mock data in development
    ) -> List[Dict[str, Any]]:
        """High-level helper: search for multiple keywords and get popular tweets.
        
        Args:
            keywords: List of keywords to search for
            limit_per_keyword: Maximum number of tweets per keyword
            min_likes: Minimum number of likes for tweets
            language: Language code
            use_cache: Whether to use the cache
            force_mock: Whether to force using mock data (for development)
            
        Returns:
            List of tweets
        """
        results = []
        
        for keyword in keywords:
            tweets = self.search_by_topic(
                keyword,
                limit=limit_per_keyword,
                min_likes=min_likes,
                language=language,
                use_cache=use_cache,
                force_mock=force_mock,
            )
            
            for tweet in tweets:
                results.append({
                    "type": "twitter_tweet",
                    "tweet": tweet,
                    "keyword": keyword,
                    "source": tweet.get("url"),
                })
                
                # politeness delay
                time.sleep(self.cfg.search.rate_limit_delay)
        
        return results
