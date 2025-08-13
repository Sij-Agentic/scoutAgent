#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import List, Optional

# Ensure both package root (scout_agent) and project root are on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))  # .../scout_agent
PROJECT_ROOT = os.path.dirname(ROOT)  # .../
for p in (PROJECT_ROOT, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import get_config
from custom_logging import get_logger
from sources.reddit_client import RedditClient

logger = get_logger("scripts.fetch_reddit")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fetch and cache Reddit conversations.")
    ap.add_argument("--keywords", type=str, required=True, help="Comma-separated list of search keywords")
    ap.add_argument("--subreddits", type=str, default="", help="Comma-separated list of subreddits to restrict search")
    ap.add_argument("--per-query-limit", type=int, default=20, help="Max posts to fetch per keyword")
    ap.add_argument("--include-comments", action="store_true", help="Fetch comments for each post")
    ap.add_argument("--comment-depth", type=int, default=2, help="Comment tree depth")
    ap.add_argument("--comment-limit", type=int, default=50, help="Max comments to fetch")
    ap.add_argument("--no-cache", action="store_true", help="Bypass cache and refresh data")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = get_config()

    keywords: List[str] = [s.strip() for s in args.keywords.split(",") if s.strip()]
    subreddits: Optional[List[str]] = [s.strip() for s in args.subreddits.split(",") if s.strip()] if args.subreddits else None

    client = RedditClient()
    results = client.fetch_conversations(
        keywords=keywords,
        subreddits=subreddits,
        per_query_limit=args.per_query_limit,
        include_comments=args.include_comments,
        comment_depth=args.comment_depth,
        comment_limit=args.comment_limit,
        use_cache=not args.no_cache,
    )

    # Print brief summary
    threads = sum(1 for r in results if r.get("type") == "reddit_thread")
    posts = sum(1 for r in results if r.get("type") == "reddit_post")
    print(f"Fetched {len(results)} items: {threads} threads with comments, {posts} posts only.")
    if results:
        sample = results[0]
        print("Sample:")
        print(json.dumps({k: sample[k] for k in ("type", "source", "post") if k in sample}, indent=2)[:800])


if __name__ == "__main__":
    main()
