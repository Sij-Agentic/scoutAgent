"""
Research Agent for ScoutAgent.

Specialized agent for web research, information gathering, and data collection.
"""

from typing import Dict, Any, List
import json
import re
import asyncio
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
from llm.utils import LLMAgentMixin, load_prompt_template


class ResearchAgent(LLMAgentMixin, BaseAgent):
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
        LLMAgentMixin.__init__(self, preferred_backend='ollama')
        self.config = get_config()

    def _extract_json(self, raw_content: str) -> Dict[str, Any]:
        """Extracts the first valid JSON object from a raw string."""
        # Find the start of the JSON object
        json_start_index = raw_content.find('{')
        if json_start_index == -1:
            self.log("No JSON object found in the response.", level='error')
            return {}

        # Find the end of the JSON object
        json_end_index = raw_content.rfind('}')
        if json_end_index == -1:
            self.log("JSON object end not found.", level='error')
            return {}

        json_string = raw_content[json_start_index:json_end_index + 1]

        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            self.log(f"Failed to parse JSON: {e}", level='error')
            self.log(f"Raw JSON string: {json_string}", level='debug')
            return {}

    async def plan(self, agent_input: AgentInput) -> Dict[str, Any]:
        """
        Plan research strategy based on input.
        """
        query = agent_input.data.get('query', '')
        self.log(f"Generating research plan for query: {query}")

        prompt = load_prompt_template(
            self.prompts_dir, 'plan.prompt', query=query
        )

        response = await self.llm.get_response_async(prompt)
        plan = self._extract_json(response)

        self.log(f"Research plan created for query: {query}")
        return plan

    async def think(self, agent_input: AgentInput, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze research requirements and prioritize sources.
        """
        self.log("Analyzing research plan and requirements...")
        prompt = load_prompt_template(
            self.prompts_dir, 'think.prompt', plan=json.dumps(plan, indent=2)
        )

        response = await self.llm.get_response_async(prompt)
        thoughts = self._extract_json(response)

        self.log(f"Research analysis complete - query type: {thoughts.get('query_type', 'N/A')}")
        return thoughts

    async def act(self, agent_input: AgentInput, plan: Dict[str, Any], thoughts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute research based on plan and analysis.
        """
        query = plan.get('query', '')
        search_queries = thoughts.get('search_queries', [query])

        self.log(f"Executing research for query: {query}")

        results = {
            'query': query,
            'search_queries': search_queries,
            'sources': [],
            'summary': None,
            'key_findings': [],
            'metadata': {
                'total_sources': 0,
                'credible_sources': 0,
                'recency_score': 0,
                'relevance_score': 0
            }
        }

        all_sources = []
        for q in search_queries:
            # In a real implementation, this would be an async call to a search tool
            search_results = self._perform_search(q)
            for res in search_results:
                processed = self._process_search_result(res)
                all_sources.append(processed)

        results['sources'] = all_sources[:plan.get('max_results', 10)]
        results['summary'] = self._generate_summary(results['sources'])
        results['key_findings'] = self._extract_key_findings(results['sources'])

        # Update metadata
        results['metadata']['total_sources'] = len(results['sources'])
        results['metadata']['recency_score'] = self._calculate_recency_score(results['sources'])
        results['metadata']['relevance_score'] = self._calculate_relevance_score(results['sources'], query)

        self.log(f"Research complete, found {len(results['sources'])} sources.")
        return results

    def _perform_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search (mock implementation)."""
        if not REQUESTS_AVAILABLE:
            self.log("requests library not available, using mock results", level='warning')

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
        return {
            'title': result.get('title', ''),
            'url': result.get('url', ''),
            'snippet': result.get('snippet', ''),
            'source': result.get('source', ''),
            'date': result.get('date', ''),
            'credibility': result.get('credibility', 0.5),
            'relevance_score': 0.7,  # Mock relevance
            'content_type': 'web_page'
        }

    def _generate_summary(self, sources: List[Dict[str, Any]]) -> str:
        """Generate summary from sources."""
        if not sources:
            return "No relevant information found."
        snippets = [s.get('snippet', '') for s in sources[:3]]
        summary = " ".join(snippets)
        return summary[:500] + "..." if len(summary) > 500 else summary

    def _extract_key_findings(self, sources: List[Dict[str, Any]]) -> List[str]:
        """Extract key findings from sources."""
        return [s['snippet'] for s in sources[:5] if s.get('snippet')]

    def _calculate_recency_score(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate recency score based on source dates."""
        return 0.7  # Mock implementation

    def _calculate_relevance_score(self, sources: List[Dict[str, Any]], query: str) -> float:
        """Calculate relevance score based on query matching."""
        return 0.8  # Mock implementation


async def main():
    """Main function to test the agent."""
    agent = ResearchAgent()

    test_input = AgentInput(
        data={'query': 'Python async programming best practices'},
        metadata={'max_results': 5}
    )

    result = await agent.execute(test_input)
    print(f"Research completed: {result.success}")
    if result.success:
        print(f"Found {len(result.result.get('sources', []))} sources")
        print(f"Summary: {result.result.get('summary', 'No summary')}")

if __name__ == "__main__":
    asyncio.run(main())
