"""
Research Agent for ScoutAgent.

Specialized agent for web research, information gathering, and data collection.
"""

from typing import Dict, Any, List
import json
import re
from urllib.parse import urljoin, urlparse

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

from agents.base import BaseAgent, AgentInput, AgentOutput
from config import get_config


class ResearchAgent(BaseAgent):
    """
    Research agent for web scraping, search, and information gathering.
    
    Capabilities:
    - Web search and scraping
    - Content extraction and analysis
    - Source validation and credibility scoring
    - Structured data collection
    """
    
    def __init__(self, name="research", **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = get_config()
    
    def plan(self, agent_input: AgentInput) -> Dict[str, Any]:
        """
        Plan research strategy based on input.
        
        Args:
            agent_input: Contains research query and parameters
            
        Returns:
            Research plan with search strategy and sources
        """
        query = str(agent_input.data)
        
        plan = {
            'query': query,
            'search_strategy': 'comprehensive',
            'sources': [
                'web_search',
                'academic_papers',
                'news_articles',
                'official_docs'
            ],
            'extraction_methods': [
                'text_content',
                'structured_data',
                'metadata'
            ],
            'validation_criteria': [
                'source_credibility',
                'recency',
                'relevance_score'
            ],
            'max_results': agent_input.metadata.get('max_results', 10),
            'timeout': self.config.search.timeout_seconds
        }
        
        self.log(f"Research plan created for query: {query}")
        return plan
    
    def think(self, agent_input: AgentInput, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze research requirements and prioritize sources.
        
        Args:
            agent_input: Original input
            plan: Research plan
            
        Returns:
            Analysis and source prioritization
        """
        query = plan['query']
        
        # Analyze query type
        query_type = self._classify_query(query)
        
        # Prioritize sources based on query type
        source_priority = self._prioritize_sources(query_type)
        
        # Estimate search complexity
        complexity = self._estimate_complexity(query)
        
        thoughts = {
            'query_type': query_type,
            'source_priority': source_priority,
            'complexity': complexity,
            'estimated_time': self._estimate_time(complexity),
            'key_concepts': self._extract_key_concepts(query),
            'search_queries': self._generate_search_queries(query)
        }
        
        self.log(f"Research analysis complete - query type: {query_type}")
        return thoughts
    
    def act(self, agent_input: AgentInput, plan: Dict[str, Any], thoughts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute research based on plan and analysis.
        
        Args:
            agent_input: Original input
            plan: Research plan
            thoughts: Analysis from thinking phase
            
        Returns:
            Research results with structured data
        """
        query = plan['query']
        search_queries = thoughts['search_queries']
        
        results = {
            'query': query,
            'search_queries': search_queries,
            'sources': [],
            'summary': None,
            'key_findings': [],
            'metadata': {
                'total_sources': 0,
                'credible_sources': 0,
                'recency_score': 0.0,
                'relevance_score': 0.0
            }
        }
        
        # Execute searches
        for search_query in search_queries:
            self.log(f"Searching: {search_query}")
            search_results = self._perform_search(search_query)
            
            # Process and validate results
            for result in search_results:
                processed = self._process_search_result(result)
                if processed:
                    results['sources'].append(processed)
        
        # Generate summary
        results['summary'] = self._generate_summary(results['sources'])
        results['key_findings'] = self._extract_key_findings(results['sources'])
        
        # Calculate metadata
        results['metadata']['total_sources'] = len(results['sources'])
        results['metadata']['credible_sources'] = len([s for s in results['sources'] if s.get('credible', False)])
        results['metadata']['recency_score'] = self._calculate_recency_score(results['sources'])
        results['metadata']['relevance_score'] = self._calculate_relevance_score(results['sources'], query)
        
        self.log(f"Research completed - found {len(results['sources'])} sources")
        return results
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of research query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how to', 'tutorial', 'guide', 'steps']):
            return 'tutorial'
        elif any(word in query_lower for word in ['definition', 'what is', 'explain']):
            return 'definition'
        elif any(word in query_lower for word in ['comparison', 'vs', 'versus', 'compare']):
            return 'comparison'
        elif any(word in query_lower for word in ['news', 'latest', 'recent', 'update']):
            return 'news'
        elif any(word in query_lower for word in ['research', 'study', 'paper', 'academic']):
            return 'academic'
        else:
            return 'general'
    
    def _prioritize_sources(self, query_type: str) -> List[str]:
        """Prioritize sources based on query type."""
        priorities = {
            'tutorial': ['official_docs', 'tutorials', 'community_forums'],
            'definition': ['encyclopedias', 'official_docs', 'academic_sources'],
            'comparison': ['reviews', 'comparisons', 'official_docs'],
            'news': ['news_sites', 'official_announcements', 'social_media'],
            'academic': ['academic_papers', 'research_institutions', 'official_docs'],
            'general': ['web_search', 'official_docs', 'community_sources']
        }
        return priorities.get(query_type, priorities['general'])
    
    def _estimate_complexity(self, query: str) -> str:
        """Estimate research complexity based on query."""
        word_count = len(query.split())
        if word_count <= 3:
            return 'simple'
        elif word_count <= 10:
            return 'moderate'
        else:
            return 'complex'
    
    def _estimate_time(self, complexity: str) -> int:
        """Estimate research time in seconds."""
        estimates = {'simple': 30, 'moderate': 60, 'complex': 120}
        return estimates.get(complexity, 60)
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query."""
        # Simple keyword extraction
        words = query.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        concepts = [word for word in words if word not in stop_words and len(word) > 3]
        return concepts[:5]  # Limit to top 5 concepts
    
    def _generate_search_queries(self, query: str) -> List[str]:
        """Generate multiple search queries from original query."""
        queries = [query]
        
        # Add variations
        query_lower = query.lower()
        if 'how to' not in query_lower:
            queries.append(f"how to {query}")
        if 'what is' not in query_lower:
            queries.append(f"what is {query}")
        if 'best' not in query_lower:
            queries.append(f"best {query}")
        
        return queries[:3]  # Limit to 3 queries
    
    def _perform_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search (mock implementation)."""
        # Check if we have the required dependencies
        if not REQUESTS_AVAILABLE:
            self.log("requests library not available, using mock results", level='warning')
        
        # Mock implementation - in real implementation, integrate with search APIs
        mock_results = [
            {
                'title': f'Result for {query}',
                'url': f'https://example.com/search?q={query.replace(" ", "+")}',
                'snippet': f'Information about {query}...',
                'source': 'example.com',
                'date': '2024-01-01',
                'credibility': 0.8
            }
        ]
        return mock_results
    
    def _process_search_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate a search result."""
        processed = {
            'title': result.get('title', ''),
            'url': result.get('url', ''),
            'snippet': result.get('snippet', ''),
            'source': result.get('source', ''),
            'date': result.get('date', ''),
            'credibility': result.get('credibility', 0.5),
            'relevance_score': 0.7,  # Mock relevance
            'content_type': 'web_page'
        }
        return processed
    
    def _generate_summary(self, sources: List[Dict[str, Any]]) -> str:
        """Generate summary from sources."""
        if not sources:
            return "No relevant information found."
        
        # Simple summary generation
        snippets = [s.get('snippet', '') for s in sources[:3]]
        summary = " ".join(snippets)
        return summary[:500] + "..." if len(summary) > 500 else summary
    
    def _extract_key_findings(self, sources: List[Dict[str, Any]]) -> List[str]:
        """Extract key findings from sources."""
        findings = []
        for source in sources[:5]:
            if source.get('snippet'):
                findings.append(source['snippet'])
        return findings
    
    def _calculate_recency_score(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate recency score based on source dates."""
        # Mock implementation
        return 0.7
    
    def _calculate_relevance_score(self, sources: List[Dict[str, Any]], query: str) -> float:
        """Calculate relevance score based on query matching."""
        # Mock implementation
        return 0.8


if __name__ == "__main__":
    # Test the research agent
    agent = ResearchAgent()
    
    test_input = AgentInput(
        data={'query': 'Python async programming best practices'},
        metadata={'max_results': 5}
    )
    
    result = agent.execute(test_input)
    print(f"Research completed: {result.success}")
    print(f"Found {len(result.result.get('sources', []))} sources")
    print(f"Summary: {result.result.get('summary', 'No summary')}")
