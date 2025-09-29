"""
VendorResearchTool - A tool for deep research on vendors including features, offerings, and reviews.
"""
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Import config for API keys
try:
    from scout_agent.config import config
except ImportError:
    config = None

import requests
import trafilatura
from litellm import completion
from pydantic import BaseModel, Field
from smolagents.tools import Tool

# LLM prompt for processing scraped content
VENDOR_ANALYSIS_PROMPT = """Analyze the following information about {vendor_name} and extract the following details:

1. Company Overview:
   - Official Name: 
   - Website: 
   - Brief Description: 

2. Key Features (list top 5-10):
   - 

3. Pricing Information (if available):
   - 

4. Target Customers:
   - 

5. How they address "{pain_point}":
   - 

6. Notable Strengths:
   - 

7. Potential Limitations:
   - 

Source Content:
{content}"""

class VendorResearchResult(BaseModel):
    """Result of vendor research."""
    vendor: Dict[str, str] = Field(
        ...,
        description="Vendor identification information"
    )
    business_profile: Dict[str, Any] = Field(
        ...,
        description="Business information including features, offerings, and value proposition"
    )
    pain_point_alignment: Dict[str, str] = Field(
        ...,
        description="How the vendor addresses the given pain point"
    )
    reviews_and_complaints: Dict[str, Any] = Field(
        ...,
        description="Aggregated reviews and complaints from various sources"
    )
    evidence: List[Dict[str, str]] = Field(
        ...,
        description="Source evidence for the information gathered"
    )
    last_updated: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp of when the research was last updated"
    )

class VendorResearchTool(Tool):
    # Required Tool class attributes
    name = "vendor_research"
    description = "Conduct deep research on a vendor including their offerings, features, and reviews."
    inputs = {
        "vendor_name": {
            "type": "string",
            "description": "The name of the vendor to research"
        },
        "pain_point": {
            "type": "string",
            "description": "The specific pain point or use case to focus the research on"
        },
        "url": {
            "type": "string",
            "description": "Optional URL of the vendor's website",
            "required": False,
            "nullable": True
        }
    }
    output_type = "string"
    output_schema = {
        "type": "object",
        "properties": {
            "vendor": {
                "type": "object",
                "properties": {
                    "canonical_name": {"type": "string"},
                    "website": {"type": "string"},
                    "disambiguation_notes": {"type": "string"}
                }
            },
            "business_profile": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "value_proposition": {"type": "string"},
                    "features": {"type": "array", "items": {"type": "string"}},
                    "offerings": {"type": "array", "items": {"type": "string"}},
                    "pricing": {"type": "string"},
                    "faqs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "answer": {"type": "string"}
                            }
                        }
                    },
                    "target_customers": {"type": "array", "items": {"type": "string"}}
                }
            },
            "pain_point_alignment": {
                "type": "object",
                "properties": {
                    "given_pain_point": {"type": "string"},
                    "how_addressed": {"type": "string"},
                    "notable_gaps": {"type": "string"}
                }
            },
            "reviews_and_complaints": {
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "url": {"type": "string"},
                                "summary": {"type": "string"},
                                "top_pros": {"type": "array", "items": {"type": "string"}},
                                "top_cons": {"type": "array", "items": {"type": "string"}},
                                "notable_complaints": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                    "overall_sentiment": {"type": "string"}
                }
            },
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "snippet": {"type": "string"}
                    }
                }
            },
            "last_updated": {"type": "string", "format": "date-time"}
        }
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.serper_api_key = os.getenv("SCOUT_SERPER_API_KEY")
        if not self.serper_api_key:
            raise ValueError("SCOUT_SERPER_API_KEY environment variable is required")
        
        # Get DeepSeek API key from config or environment
        self.deepseek_api_key = None
        if config and hasattr(config, 'api') and config.api.deepseek_api_key:
            self.deepseek_api_key = config.api.deepseek_api_key
        else:
            self.deepseek_api_key = os.getenv("SCOUT_DEEPSEEK_API_KEY")
        
        if not self.deepseek_api_key:
            self.logger.warning("DeepSeek API key not found - LLM analysis will be disabled")
        
        # Setup logging
        import logging
        from pathlib import Path
        
        # Create logs directory if it doesn't exist
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Create file handler which logs even debug messages
        log_file = self.logs_dir / "vendor_research.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.is_initialized = False
        
    def _save_debug_info(self, prefix: str, data: Any, vendor_name: str):
        """Save debug information to a JSON file."""
        from pathlib import Path
        import json
        
        debug_dir = self.logs_dir / "debug" / vendor_name.lower().replace(" ", "_")
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a timestamp for the filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the data to a JSON file
        debug_file = debug_dir / f"{prefix}_{timestamp}.json"
        with open(debug_file, 'w', encoding='utf-8') as f:
            if isinstance(data, (dict, list)):
                json.dump(data, f, indent=2, ensure_ascii=False)
            elif hasattr(data, 'model_dump'):
                json.dump(data.model_dump(), f, indent=2, ensure_ascii=False)
            else:
                f.write(str(data))
                
        self.logger.debug(f"Saved {prefix} to {debug_file}")
        return str(debug_file)
        
    async def _scrape_website(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape and extract content from a website using trafilatura with fallback to requests."""
        try:
            self.logger.info(f"Attempting to scrape website: {url}")
            
            # First try with trafilatura
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    result = trafilatura.extract(
                        downloaded,
                        include_links=True,
                        include_tables=True,
                        include_images=False,
                        output_format='json',
                        include_comments=False
                    )
                    if result:
                        self.logger.info("Successfully scraped using trafilatura")
                        return json.loads(result)
            except Exception as e:
                self.logger.warning(f"Trafilatura failed: {str(e)}")
            
            # Fallback to requests with proper headers
            try:
                import requests
                from bs4 import BeautifulSoup
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text
                text = soup.get_text()
                
                # Break into lines and remove leading/trailing space
                lines = (line.strip() for line in text.splitlines())
                # Break multi-headlines into a line each
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                # Drop blank lines
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                if text:
                    self.logger.info("Successfully scraped using requests + BeautifulSoup")
                    return {
                        'url': url,
                        'title': soup.title.string if soup.title else '',
                        'text': text,
                        'status': 'success',
                        'method': 'requests+beautifulsoup4'
                    }
                
            except Exception as e:
                self.logger.warning(f"Requests + BeautifulSoup fallback failed: {str(e)}")
            
            # If all else fails, try with trafilatura using the raw response content
            if 'response' in locals():
                try:
                    result = trafilatura.extract(
                        response.content,
                        include_links=True,
                        include_tables=True,
                        include_images=False,
                        output_format='json',
                        include_comments=False
                    )
                    if result:
                        self.logger.info("Successfully scraped using trafilatura with raw content")
                        return json.loads(result)
                except Exception as e:
                    self.logger.warning(f"Trafilatura with raw content failed: {str(e)}")
            
            self.logger.error(f"All scraping methods failed for {url}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error in _scrape_website for {url}: {str(e)}", exc_info=True)
            return None
            
    def _analyze_with_llm(self, vendor_name: str, pain_point: str, content: str) -> Dict[str, Any]:
        """Analyze scraped content using DeepSeek LLM."""
        if not self.deepseek_api_key:
            self.logger.warning("DeepSeek API key not available - skipping LLM analysis")
            return {}
        
        try:
            prompt = VENDOR_ANALYSIS_PROMPT.format(
                vendor_name=vendor_name,
                pain_point=pain_point,
                content=content[:15000]  # Limit content length
            )
            
            # Use DeepSeek model with synchronous completion and API key
            response = completion(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
                api_base="https://api.deepseek.com/v1",  # DeepSeek API endpoint
                api_key=self.deepseek_api_key  # Add API key
            )
            
            # Parse the LLM response into structured format
            result = self._parse_llm_response(response.choices[0].message.content)
            self.logger.info(f"LLM analysis completed successfully for {vendor_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in LLM analysis for {vendor_name}: {str(e)}", exc_info=True)
            return {}
            
    def _parse_llm_response(self, llm_output: str) -> Dict[str, Any]:
        """Parse the LLM's text response into structured data.
        
        NOTE: This method should ONLY be called when LLM analysis succeeds.
        It does NOT extract fallback data from raw scraped content.
        """
        result = {
            'vendor': {
                'canonical_name': '',
                'website': '',
                'description': ''
            },
            'business_profile': {
                'features': [],
                'pricing': '',
                'target_customers': [],
                'value_proposition': ''  # Only populated from successful LLM analysis
            },
            'pain_point_alignment': {
                'how_addressed': ''
            }
        }
        
        # Extract company name
        name_match = re.search(r'Official Name:\s*(.+)', llm_output, re.IGNORECASE)
        if name_match:
            result['vendor']['canonical_name'] = name_match.group(1).strip()
            
        # Extract website
        website_match = re.search(r'Website:\s*(.+)', llm_output, re.IGNORECASE)
        if website_match:
            result['vendor']['website'] = website_match.group(1).strip()
            
        # Extract description
        desc_match = re.search(r'Brief Description:\s*(.+?)(?:\n\s*\d+\.|$)', 
                             llm_output, re.IGNORECASE | re.DOTALL)
        if desc_match:
            result['vendor']['description'] = desc_match.group(1).strip()
            
        # Extract value proposition from Notable Strengths section
        # ONLY from properly formatted LLM response, not raw scraped content
        strengths_match = re.search(r'Notable Strengths[^:]*:\s*([\s\S]+?)(?:\n\s*\d+\.|$)', 
                                  llm_output, re.IGNORECASE)
        if strengths_match:
            result['business_profile']['value_proposition'] = strengths_match.group(1).strip()
            
        # Extract features
        features_section = re.search(r'Key Features.*?:\s*([\s\S]+?)(?:\n\s*\d+\.|$)', 
                                   llm_output, re.IGNORECASE)
        if features_section:
            features = re.findall(r'\-\s*(.+)', features_section.group(1))
            result['business_profile']['features'] = features
            
        # Extract pricing
        pricing_match = re.search(r'Pricing Information[^:]*:\s*([\s\S]+?)(?:\n\s*\d+\.|$)', 
                                llm_output, re.IGNORECASE)
        if pricing_match:
            result['business_profile']['pricing'] = pricing_match.group(1).strip()
            
        # Extract target customers
        customers_section = re.search(r'Target Customers[^:]*:\s*([\s\S]+?)(?:\n\s*\d+\.|$)', 
                                    llm_output, re.IGNORECASE)
        if customers_section:
            customers = re.findall(r'\-\s*(.+)', customers_section.group(1))
            result['business_profile']['target_customers'] = customers
            
        # Extract pain point alignment
        pain_point_match = re.search(r'How they address[^:]*:\s*([\s\S]+?)(?:\n\s*\d+\.|$)', 
                                   llm_output, re.IGNORECASE)
        if pain_point_match:
            result['pain_point_alignment']['how_addressed'] = pain_point_match.group(1).strip()
            
        return result
        
    async def _research_vendor(self, vendor_name: str, pain_point: str, url: str = None) -> Dict[str, Any]:
        """
        Perform the actual vendor research with web scraping and LLM processing.
        
        Args:
            vendor_name: The name of the vendor to research
            pain_point: The specific pain point or use case to focus on
            url: Optional URL of the vendor's website
            
        Returns:
            Dictionary containing the research results
        """
        self.logger.info(f"Starting research for vendor: {vendor_name}")
        self.logger.info(f"Pain point: {pain_point}")
        self.logger.info(f"URL: {url if url else 'Not provided'}")
        
        # Initialize result structure
        result = {
            'vendor': {
                'canonical_name': vendor_name,
                'website': url or '',
                'disambiguation_notes': ''
            },
            'business_profile': {
                'summary': '',
                'value_proposition': '',
                'features': [],
                'offerings': [],
                'pricing': '',
                'faqs': [],
                'target_customers': []
            },
            'pain_point_alignment': {
                'given_pain_point': pain_point,
                'how_addressed': '',
                'notable_gaps': ''
            },
            'reviews_and_complaints': {
                'sources': [],
                'overall_sentiment': ''
            },
            'evidence': [],
            'last_updated': datetime.utcnow().isoformat()
        }
        
        # Save initial result structure
        self._save_debug_info("initial_result", result, vendor_name)
        
        # If URL is not provided, try to find it via search
        if not url:
            search_results = self._search_web(f"{vendor_name} official website")
            if search_results and 'link' in search_results[0]:
                url = search_results[0]['link']
                result['vendor']['website'] = url
                result['evidence'].append({
                    'type': 'website_discovery',
                    'title': f"Found official website for {vendor_name}",
                    'url': url,
                    'snippet': f"Discovered official website at {url}"
                })
        
        # Scrape and analyze the vendor's website if URL is available
        if url:
            try:
                self.logger.info(f"Scraping website: {url}")
                # Scrape the website
                scraped_data = await self._scrape_website(url)
                
                # Save scraped data for debugging
                if scraped_data:
                    self._save_debug_info("scraped_data", scraped_data, vendor_name)
                    
                    # Extract main content
                    content = scraped_data.get('text', '')
                    self.logger.info(f"Extracted {len(content)} characters of content")
                    
                    if content:
                        # Save content sample for debugging
                        content_sample = content[:1000] + "..." if len(content) > 1000 else content
                        self._save_debug_info("content_sample", {"length": len(content), "sample": content_sample}, vendor_name)
                        
                        # Analyze content with LLM
                        self.logger.info("Analyzing content with LLM...")
                        llm_analysis = self._analyze_with_llm(vendor_name, pain_point, content)
                        
                        # Save LLM analysis for debugging
                        if llm_analysis:
                            self._save_debug_info("llm_analysis", llm_analysis, vendor_name)
                            
                            # Update vendor info
                            if llm_analysis.get('vendor', {}).get('canonical_name'):
                                result['vendor']['canonical_name'] = llm_analysis['vendor']['canonical_name']
                            if llm_analysis.get('vendor', {}).get('description'):
                                result['business_profile']['summary'] = llm_analysis['vendor']['description']
                            
                            # Update business profile - ONLY use LLM analysis results
                            if llm_analysis.get('business_profile', {}).get('value_proposition'):
                                result['business_profile']['value_proposition'] = llm_analysis['business_profile']['value_proposition']
                            if llm_analysis.get('business_profile', {}).get('features'):
                                result['business_profile']['features'] = llm_analysis['business_profile']['features']
                            if llm_analysis.get('business_profile', {}).get('pricing'):
                                result['business_profile']['pricing'] = llm_analysis['business_profile']['pricing']
                            if llm_analysis.get('business_profile', {}).get('target_customers'):
                                result['business_profile']['target_customers'] = llm_analysis['business_profile']['target_customers']
                            
                            # Update pain point alignment
                            if llm_analysis.get('pain_point_alignment', {}).get('how_addressed'):
                                result['pain_point_alignment']['how_addressed'] = llm_analysis['pain_point_alignment']['how_addressed']
                            
                            result['evidence'].append({
                                'type': 'website_analysis',
                                'title': f"Analyzed {vendor_name} website",
                                'url': url,
                                'snippet': f"Extracted key information from {url} using LLM analysis"
                            })
                            
                            self.logger.info("Successfully processed website with LLM analysis")
                        else:
                            self.logger.warning(f"LLM analysis failed for {vendor_name} - no value proposition will be extracted")
                            # DO NOT extract value proposition from raw content when LLM fails
                            # This prevents stock/cached data from being used as fallback
                            result['business_profile']['value_proposition'] = 'Analysis unavailable - LLM service failed'
                            result['pain_point_alignment']['how_addressed'] = 'Analysis unavailable - LLM service failed'
                            result['pain_point_alignment']['notable_gaps'] = 'Analysis unavailable - LLM service failed'
                    else:
                        self.logger.warning("No content extracted from the website")
                else:
                    self.logger.warning("No data returned from website scraping")
                    
            except Exception as e:
                error_msg = f"Error during website analysis: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                result['evidence'].append({
                    'type': 'error',
                    'title': f"Error analyzing {vendor_name} website",
                    'url': url or '',
                    'snippet': f"Failed to analyze website: {str(e)}"
                })
        
        # Get reviews and complaints
        result['reviews_and_complaints'] = self._get_reviews(vendor_name)
        
        # Note: Removed fallback web search for features/value proposition
        # This ensures we can clearly identify when LLM analysis fails
        # rather than getting misleading generic data from web snippets
        
        return result
    
    async def forward(self, vendor_name: str, pain_point: str, url: str = None) -> str:
        """
        Conduct research on a vendor based on the provided information.
        
        Args:
            vendor_name: The name of the vendor to research
            pain_point: The specific pain point or use case to focus on
            url: Optional URL of the vendor's website
            
        Returns:
            JSON string containing the research results
        """
        # Convert the research result to JSON string
        result = await self._research_vendor(vendor_name, pain_point, url)
        return json.dumps(result, default=str)
    
    def _search_web(self, query: str) -> List[Dict[str, str]]:
        """Perform a web search using Serper API."""
        url = "https://google.serper.dev/search"
        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }
        payload = {
            'q': query,
            'gl': 'us',
            'hl': 'en',
            'num': 5
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            results = response.json()
            
            # Extract relevant information from the response
            search_results = []
            
            # Organic results
            for result in results.get('organic', []):
                search_results.append({
                    'title': result.get('title', ''),
                    'link': result.get('link', ''),
                    'snippet': result.get('snippet', '')
                })
                
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error performing web search: {str(e)}")
            return []
    
    def _get_reviews(self, vendor_name: str) -> Dict[str, Any]:
        """Search for reviews and complaints about the vendor."""
        self.logger.info(f"Searching for reviews of {vendor_name}")
        
        # Define review sites to search
        review_sites = [
            'trustpilot.com',
            'g2.com',
            'capterra.com',
            'crozdesk.com',
            'softwareadvice.com'
        ]
        
        # Initialize reviews dictionary
        reviews = {
            'sources': [],
            'overall_sentiment': 'Not available',
            'review_count': 0,
            'sources_searched': review_sites.copy()
        }
        
        try:
            # Search for reviews
            review_query = f"{vendor_name} reviews"
            review_results = self._search_web(review_query)
            
            # Save raw search results for debugging
            self._save_debug_info("review_search_results", review_results, vendor_name)
            
            if review_results:
                self.logger.info(f"Found {len(review_results)} potential review sources")
                
                # Filter and process review sources
                for result in review_results[:5]:  # Limit to top 5 review sources
                    source_url = result.get('link', '')
                    source_domain = source_url.split('/')[2] if len(source_url.split('/')) > 2 else ''
                    
                    # Only include known review sites
                    if any(site in source_domain for site in review_sites):
                        review_source = {
                            'source': source_domain,
                            'title': result.get('title', '').split('|')[0].strip(),
                            'url': source_url,
                            'summary': result.get('snippet', ''),
                            'type': 'review_site'
                        }
                        reviews['sources'].append(review_source)
                        self.logger.debug(f"Added review source: {source_domain}")
            
            # If no review sites found, try direct searches on known review platforms
            if not reviews['sources']:
                self.logger.info("No review sites found in initial search, trying direct searches...")
                
                for site in review_sites:
                    try:
                        site_query = f"{vendor_name} site:{site}"
                        site_results = self._search_web(site_query)
                        
                        if site_results:
                            result = site_results[0]  # Take top result
                            reviews['sources'].append({
                                'source': site,
                                'title': result.get('title', '').split('|')[0].strip(),
                                'url': result.get('link', ''),
                                'summary': result.get('snippet', ''),
                                'type': 'direct_search'
                            })
                            self.logger.debug(f"Found direct review source: {site}")
                    except Exception as e:
                        self.logger.warning(f"Error searching {site}: {str(e)}")
            
            # Update review count
            reviews['review_count'] = len(reviews['sources'])
            
            # Simple sentiment analysis based on review snippets
            if reviews['sources']:
                positive_terms = ['great', 'excellent', 'amazing', 'love', 'best', 'recommend', 'awesome', 'outstanding', 'satisfied', 'happy']
                negative_terms = ['bad', 'poor', 'terrible', 'worst', 'awful', 'disappoint', 'avoid', 'issue', 'problem', 'frustrat']
                
                positive_count = 0
                negative_count = 0
                
                for source in reviews['sources']:
                    summary = source.get('summary', '').lower()
                    if any(term in summary for term in positive_terms):
                        positive_count += 1
                    if any(term in summary for term in negative_terms):
                        negative_count += 1
                
                self.logger.info(f"Sentiment analysis - Positive: {positive_count}, Negative: {negative_count}")
                
                if positive_count > negative_count:
                    reviews['overall_sentiment'] = 'Mostly positive'
                elif negative_count > positive_count:
                    reviews['overall_sentiment'] = 'Mostly negative'
                else:
                    reviews['overall_sentiment'] = 'Mixed'
            
            self.logger.info(f"Found {reviews['review_count']} review sources with {reviews['overall_sentiment'].lower()} sentiment")
            
        except Exception as e:
            error_msg = f"Error while getting reviews: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            reviews['error'] = error_msg
        
        # Save final reviews data for debugging
        self._save_debug_info("reviews_final", reviews, vendor_name)
            
        return reviews

    async def _run(self, input_str: str) -> str:
        """Execute the vendor research tool."""
        try:
            # Parse input string as JSON
            input_data = json.loads(input_str)
            vendor_name = input_data.get('vendor_name')
            pain_point = input_data.get('pain_point')
            url = input_data.get('url')
            
            if not vendor_name or not pain_point:
                raise ValueError("Both 'vendor_name' and 'pain_point' are required")
                
            # Run the research
            result = await self.forward(vendor_name, pain_point, url)
            return json.dumps(result, indent=2)
            
        except json.JSONDecodeError:
            return "Error: Invalid JSON input"
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _arun(self, input_str: str) -> str:
        """Async version of run."""
        return await self._run(input_str)
