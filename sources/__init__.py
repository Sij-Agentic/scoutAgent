"""Sources package for ScoutAgent validation tools.

This package contains clients for various data sources used by the validation agent:
- Reddit API client
- Hacker News (HN) Algolia API client
- SERP API client for Google search, trends, and autocomplete
- Twitter client using snscrape
- Review sites client for G2, Capterra, etc. via SERP API
- DuckDuckGo client for web search
"""

from scout_agent.sources.reddit_client import RedditClient
from scout_agent.sources.hn_client import HNClient
from scout_agent.sources.serp_client import SerpApiClient
from scout_agent.sources.twitter_client import TwitterClient
from scout_agent.sources.review_client import ReviewSitesClient
from scout_agent.sources.duckduckgo_client import DuckDuckGoClient

__all__ = [
    'RedditClient',
    'HNClient',
    'SerpApiClient',
    'TwitterClient',
    'ReviewSitesClient',
    'DuckDuckGoClient',
]
