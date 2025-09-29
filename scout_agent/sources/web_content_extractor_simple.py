"""
Simplified web content extractor without trafilatura dependency
This is a fallback version for container builds
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class WebContentExtractor:
    """Simplified tool for extracting content from web pages without trafilatura."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def extract_content(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Extract content from a web page using requests and BeautifulSoup.
        
        Args:
            url: The URL to extract content from
            
        Returns:
            Dictionary with extracted content or None if extraction fails
        """
        try:
            logger.info(f"Extracting content from: {url}")
            
            # Fetch the page
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract title
            title = ""
            if soup.title:
                title = soup.title.get_text().strip()
            
            # Extract main content
            # Try to find main content area
            main_content = None
            
            # Look for common content selectors
            content_selectors = [
                'main', 'article', '[role="main"]', 
                '.content', '.post-content', '.entry-content',
                '.article-content', '.story-content'
            ]
            
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # If no main content found, use body
            if not main_content:
                main_content = soup.body
            
            if main_content:
                # Extract text content
                text_content = main_content.get_text(separator=' ', strip=True)
                
                # Clean up whitespace
                text_content = ' '.join(text_content.split())
                
                return {
                    'url': url,
                    'title': title,
                    'content': text_content,
                    'status': 'success',
                    'method': 'beautifulsoup'
                }
            else:
                logger.warning(f"No content found for URL: {url}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return None
    
    def extract_links(self, url: str) -> Optional[list]:
        """
        Extract links from a web page.
        
        Args:
            url: The URL to extract links from
            
        Returns:
            List of links or None if extraction fails
        """
        try:
            logger.info(f"Extracting links from: {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True)
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    from urllib.parse import urljoin
                    href = urljoin(url, href)
                
                links.append({
                    'url': href,
                    'text': text
                })
            
            return links
            
        except Exception as e:
            logger.error(f"Error extracting links from {url}: {str(e)}")
            return None
