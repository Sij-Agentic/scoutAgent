"""
Text processing tools for MCP.

This module provides a server with tools for text extraction and processing.
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional
from html import unescape

from ..server.base import Server, Context

# Try to import optional dependencies
try:
    import bs4
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    # Ensure the necessary NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

# Configure logging
logger = logging.getLogger("mcp.servers.text_processing")

# Create server
server = Server(
    name="Text Processing Tools",
    description="Tools for extracting and processing text from various sources"
)


@server.tool(name="extract_text_from_html", description="Extract plain text from HTML content")
async def extract_text_from_html(html: str, extract_links: bool = False, ctx: Context = None) -> Dict[str, Any]:
    """
    Extract plain text from HTML content.
    
    Args:
        html: HTML content to extract text from
        extract_links: Whether to also extract links (default: False)
        ctx: Context object (optional)
    
    Returns:
        Dictionary with extracted text and optionally links
    """
    if ctx:
        ctx.log("INFO", f"Extracting text from HTML (extract_links={extract_links})")
    
    try:
        # Use BeautifulSoup if available for better extraction
        if HAS_BS4:
            if ctx:
                ctx.log("INFO", "Using BeautifulSoup for extraction")
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "noscript", "iframe", "head"]):
                script.extract()
            
            # Extract text
            text = soup.get_text(separator=" ")
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Extract links if requested
            links = []
            if extract_links:
                for link in soup.find_all('a'):
                    href = link.get('href', '')
                    text = link.get_text().strip()
                    if href and text:
                        links.append({
                            "url": href,
                            "text": text
                        })
        
        else:
            # Fallback to regex-based extraction if BeautifulSoup is not available
            if ctx:
                ctx.log("INFO", "Using regex fallback for extraction")
            
            # Remove script and style content
            html = re.sub(r'<script.*?</script>', ' ', html, flags=re.DOTALL)
            html = re.sub(r'<style.*?</style>', ' ', html, flags=re.DOTALL)
            
            # Extract text (remove all HTML tags)
            text = re.sub(r'<[^>]+>', ' ', html)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            text = unescape(text)  # Convert HTML entities
            
            # Extract links if requested
            links = []
            if extract_links:
                link_pattern = r'<a[^>]+href=["\'](.*?)["\'][^>]*>(.*?)</a>'
                for match in re.finditer(link_pattern, html):
                    url = match.group(1)
                    link_text = re.sub(r'<[^>]+>', '', match.group(2)).strip()
                    if url and link_text:
                        links.append({
                            "url": url,
                            "text": link_text
                        })
        
        result = {
            "text": text,
            "char_count": len(text),
            "word_count": len(text.split())
        }
        
        if extract_links:
            result["links"] = links
            result["link_count"] = len(links)
        
        if ctx:
            ctx.log("INFO", f"Extracted {result['char_count']} characters, {result['word_count']} words")
        
        return result
        
    except Exception as e:
        if ctx:
            ctx.log("ERROR", f"Text extraction error: {str(e)}")
        return {"error": f"Text extraction error: {str(e)}"}


@server.tool(name="summarize_text", description="Summarize a long text")
async def summarize_text(
    text: str, 
    max_sentences: int = 5,
    algorithm: str = "extractive",
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Summarize a long text by extracting the most important sentences.
    
    Args:
        text: The text to summarize
        max_sentences: Maximum number of sentences in the summary (default: 5)
        algorithm: Summarization algorithm to use (extractive or abstractive) (default: extractive)
        ctx: Context object (optional)
    
    Returns:
        Dictionary with the summarized text and metadata
    """
    if ctx:
        ctx.log("INFO", f"Summarizing text using {algorithm} algorithm (max_sentences={max_sentences})")
    
    try:
        # Validate parameters
        if algorithm not in ["extractive", "abstractive"]:
            algorithm = "extractive"  # Default to extractive if invalid
        
        # For now, we only implement extractive summarization
        # Abstractive would require more advanced NLP libraries
        if algorithm == "abstractive":
            if ctx:
                ctx.log("WARNING", "Abstractive summarization not implemented, falling back to extractive")
            algorithm = "extractive"
        
        # Extractive summarization (basic implementation)
        if algorithm == "extractive":
            # Split into sentences
            if HAS_NLTK:
                sentences = sent_tokenize(text)
            else:
                # Simple sentence splitting fallback
                sentences = re.split(r'(?<=[.!?])\s+', text)
            
            if not sentences:
                return {"summary": "", "original_length": len(text), "summary_length": 0}
            
            # Simple scoring based on position and sentence length
            # In a production system, you would use more sophisticated scoring
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                # Prefer sentences at the beginning and with reasonable length
                position_score = 1.0 if i < 3 else (1.0 / (i + 1))
                length_score = min(1.0, len(sentence.split()) / 20)
                score = position_score * 0.6 + length_score * 0.4
                scored_sentences.append((sentence, score, i))
            
            # Sort by score (descending) then by original position (ascending)
            sorted_sentences = sorted(scored_sentences, key=lambda x: (-x[1], x[2]))
            
            # Take top N sentences but preserve original order
            top_sentences = sorted(
                [s for s, _, _ in sorted_sentences[:max_sentences]], 
                key=lambda x: sentences.index(x)
            )
            
            # Join sentences
            summary = " ".join(top_sentences)
            
            result = {
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": round(len(summary) / len(text) * 100, 2) if len(text) > 0 else 0,
                "sentence_count": len(top_sentences)
            }
            
            if ctx:
                ctx.log("INFO", f"Summarized text from {len(text)} chars to {len(summary)} chars")
            
            return result
        
    except Exception as e:
        if ctx:
            ctx.log("ERROR", f"Summarization error: {str(e)}")
        return {"error": f"Summarization error: {str(e)}"}
