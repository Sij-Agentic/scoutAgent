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

from .base import BaseAgent, AgentInput, AgentOutput
from config import get_config
from llm.utils import LLMAgentMixin


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
        BaseAgent.__init__(self, name=name, **kwargs)
        LLMAgentMixin.__init__(self, preferred_backend='ollama')


    def _extract_json(self, raw_content: str) -> Dict[str, Any]:
        """Extracts the first valid JSON object from a raw string."""
        # This regex is designed to find a JSON block enclosed in ```json ... ```
        match = re.search(r"```json\n(.*?)\n```", raw_content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Fallback to finding the first '{' and last '}'
            start = raw_content.find('{')
            end = raw_content.rfind('}')
            if start != -1 and end != -1:
                json_str = raw_content[start:end+1]
            else:
                self.log("No JSON object found in the response.", level='warning')
                return {}

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.log(f"Failed to decode JSON: {e}", level='error')
            self.log(f"Raw content for debugging: {raw_content}", level='debug')
            return {}

    async def plan(self, agent_input: AgentInput) -> Dict[str, Any]:
        """
        Determines the research plan based on the user query.
        This is the first step in the agent's lifecycle.
        """
        query = agent_input.data
        self.log(f"Generating research plan for query: {query}")

        from llm.utils import load_prompt_template
        prompt_template = load_prompt_template(
            'plan.prompt',
            agent_name=self.name,
            substitutions={'query': query}
        )

        response = await self.llm_generate(prompt_template)
        plan = self._extract_json(response)

        if not plan:
            self.log("Failed to generate a valid research plan.", level='error')
            return {"error": "Plan generation failed."}

        self.log("Research plan generated successfully.")
        return plan

    async def think(self, agent_input: AgentInput, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the plan and determines the best course of action.
        This is the second step, where the agent refines its approach.
        """
        query = agent_input.data
        plan_str = json.dumps(plan, indent=2)
        self.log(f"Thinking about research plan for query: {query}")

        from llm.utils import load_prompt_template
        prompt_template = load_prompt_template(
            'think.prompt',
            agent_name=self.name,
            substitutions={'query': query, 'plan': plan_str}
        )

        response = await self.llm_generate(prompt_template)
        thoughts = self._extract_json(response)

        if not thoughts:
            self.log("Failed to generate valid thoughts from the plan.", level='error')
            return {"error": "Thinking process failed."}

        self.log("Thinking process completed.")
        return thoughts

    async def act(self, agent_input: AgentInput, plan: Dict[str, Any], thoughts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the research plan based on the generated thoughts.
        This is the final step where the agent performs the actual research tasks.
        """
        self.log("Executing research plan...")

        # In a real implementation, this would involve calling search tools, APIs, etc.
        # For this refactoring, we'll simulate the action based on the plan.
        research_steps = plan.get('research_steps', [])
        search_queries = thoughts.get('search_queries', [])

        if not research_steps and not search_queries:
            self.log("No research steps or search queries found in the plan/thoughts.", level='warning')
            return {"summary": "No actionable research steps provided."}

        # Simulate fetching results
        results = []
        for i, query in enumerate(search_queries):
            self.log(f"Executing search: '{query}'")
            await asyncio.sleep(0.1)  # Simulate async I/O
            results.append({
                "query": query,
                "summary": f"This is a summary for the search query '{query}'. It contains relevant information.",
                "sources": [f"https://example.com/source{i+1}"]
            })

        final_summary = thoughts.get('summary', 'The research was conducted based on the plan.')

        output = {
            "summary": final_summary,
            "results": results,
            "steps_taken": research_steps
        }

        self.log("Research execution completed.")
        return output

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
