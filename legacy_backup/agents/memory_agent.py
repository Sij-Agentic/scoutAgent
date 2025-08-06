"""
MemoryAgent - Knowledge Management and Long-term Learning Agent

This agent specializes in managing long-term knowledge, learning from past
analyses, and providing intelligent insights for future workflows.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from agents.base import BaseAgent, AgentInput, AgentOutput, AgentState
from llm.utils import LLMAgentMixin
import json
import re
from memory.persistence import MemoryManager
from memory.context import ContextManager
from config import get_config


@dataclass
class KnowledgeEntry:
    """Represents a knowledge entry in the system."""
    id: str
    type: str  # pain_point, market_gap, solution, insight, pattern
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    confidence: float
    source: str
    tags: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryAgentInput:
    """Input for MemoryAgent."""
    workflow_data: Dict[str, Any]  # Complete workflow data
    operation: str = "store"  # store, retrieve, analyze, update
    query: Optional[str] = None  # For retrieval operations
    filters: Optional[Dict[str, Any]] = None  # For filtering
    context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.workflow_data and self.operation == "store":
            raise ValueError("Must provide workflow data for storage operation")
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MemoryAgentOutput:
    """Output from MemoryAgent."""
    knowledge_entries: List[KnowledgeEntry]
    insights: List[Dict[str, Any]]
    patterns: List[Dict[str, Any]]
    recommendations: List[str]
    storage_status: Dict[str, Any]
    retrieval_results: Optional[List[Dict[str, Any]]] = None
    result: Any = None
    metadata: Dict[str, Any] = None
    logs: List[str] = None
    execution_time: float = 0.0
    success: bool = True
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.logs is None:
            self.logs = []


class MemoryAgent(BaseAgent, LLMAgentMixin):
    """
    MemoryAgent for knowledge management and long-term learning.
    
    Manages persistent knowledge storage, pattern recognition, and 
    intelligent insights generation from historical workflow data.
    """
    
    def __init__(self, agent_id: str = None, config: Dict[str, Any] = None):
        # Explicitly call parent class initializers
        BaseAgent.__init__(self, 'memory', agent_id=agent_id, config=config)
        LLMAgentMixin.__init__(self, preferred_backend='ollama')
        
        # Additional initialization
        self.memory_manager = MemoryManager()
        self.context_manager = ContextManager()
        self.config = get_config()
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response text."""
        json_match = re.search(r'```(?:json)?\s*(.+?)\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to decode JSON: {e}")
                return {}
        else:
            self.logger.error("No JSON found in response")
            return {}
    
    async def plan(self, input_data: MemoryAgentInput) -> Dict[str, Any]:
        """Plan the memory management operation using LLM."""
        self.logger.info(f"Planning memory operation: {input_data.operation}")
        
        # Prepare data for prompt
        operation = input_data.operation
        data_sources = json.dumps(list(input_data.workflow_data.keys()) if input_data.workflow_data else [])
        
        # Load prompt template
        from llm.utils import load_prompt_template
        prompt_template = load_prompt_template(
            'plan.prompt',
            agent_name='memory_agent',
            substitutions={
                'operation': operation,
                'data_sources': data_sources
            }
        )
        
        # Generate plan using LLM
        response = await self.llm_generate(prompt_template)
        if not response:
            self.logger.error("Failed to generate memory plan")
            return {}
            
        # Extract plan from LLM response
        plan = self._extract_json(response)
        if not plan:
            self.logger.warning("Failed to parse memory plan from LLM response, using default")
            plan = {
                "operation": input_data.operation,
                "phases": self._get_operation_phases(input_data.operation),
                "expected_duration": 300,  # 5 minutes
                "data_sources": list(input_data.workflow_data.keys()) if input_data.workflow_data else []
            }
        
        self.logger.info(f"Memory plan created for operation: {operation}")
        self.state.plan = plan
        return plan
    
    async def think(self, input_data: MemoryAgentInput) -> Dict[str, Any]:
        """Analyze the memory operation requirements using LLM."""
        self.logger.info("Analyzing memory operation requirements...")
        
        # Prepare data for prompt
        operation = input_data.operation
        workflow_data_sample = json.dumps(input_data.workflow_data, indent=2)[:500] + "..."  # Truncate for prompt
        plan_str = json.dumps(self.state.plan, indent=2) if self.state.plan else "{}"
        
        # Load prompt template
        from llm.utils import load_prompt_template
        prompt_template = load_prompt_template(
            'think.prompt',
            agent_name='memory_agent',
            substitutions={
                'operation': operation,
                'workflow_data_sample': workflow_data_sample,
                'plan': plan_str
            }
        )
        
        # Generate analysis using LLM
        response = await self.llm_generate(prompt_template)
        if not response:
            self.logger.error("Failed to analyze memory operation requirements")
            return {}
            
        # Extract analysis from LLM response
        thoughts = self._extract_json(response)
        if not thoughts:
            self.logger.warning("Failed to parse memory analysis from LLM response, using default")
            thoughts = {
                "operation_type": input_data.operation,
                "data_complexity": self._assess_data_complexity(input_data.workflow_data),
                "storage_requirements": self._calculate_storage_requirements(input_data.workflow_data),
                "analysis_scope": self._determine_analysis_scope(input_data.operation),
                "expected_insights": 5 if input_data.operation in ["analyze", "retrieve"] else 0
            }
        
        self.logger.info(f"Memory operation requirements analyzed")
        self.state.thoughts = thoughts
        return thoughts
    
    async def act(self, input_data: MemoryAgentInput) -> MemoryAgentOutput:
        """Execute memory management operation using LLM."""
        self.logger.info(f"Executing memory operation: {input_data.operation}")
        start_time = datetime.now()
        
        # Prepare data for prompt
        operation = input_data.operation
        workflow_data_sample = json.dumps(input_data.workflow_data, indent=2)[:500] + "..."  # Truncate for prompt
        plan_str = json.dumps(self.state.plan, indent=2) if self.state.plan else "{}"
        thoughts_str = json.dumps(self.state.thoughts, indent=2) if self.state.thoughts else "{}"
        
        # Load prompt template
        from llm.utils import load_prompt_template
        prompt_template = load_prompt_template(
            'act.prompt',
            agent_name='memory_agent',
            substitutions={
                'operation': operation,
                'workflow_data_sample': workflow_data_sample,
                'plan': plan_str,
                'thoughts': thoughts_str
            }
        )
        
        # Generate execution results using LLM
        response = await self.llm_generate(prompt_template)
        if not response:
            self.logger.error(f"Failed to execute memory {operation} operation")
            raise ValueError(f"LLM execution failed for operation: {operation}")
        
        # Extract execution results from LLM response
        results = self._extract_json(response)
        if not results:
            self.logger.error(f"Failed to parse memory execution results from LLM response")
            
            # Fallback to traditional methods if LLM fails
            self.logger.info(f"Falling back to traditional execution methods")
            if input_data.operation == "store":
                return await self._store_knowledge(input_data.workflow_data)
            elif input_data.operation == "retrieve":
                return await self._retrieve_knowledge(input_data.query, input_data.filters)
            elif input_data.operation == "analyze":
                return await self._analyze_knowledge(input_data.workflow_data)
            elif input_data.operation == "update":
                return await self._update_knowledge(input_data.workflow_data)
            else:
                raise ValueError(f"Unsupported operation: {input_data.operation}")
        
        # Convert the LLM response into a proper MemoryAgentOutput
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Create stub knowledge entries if none provided
        if "knowledge_entries" not in results:
            results["knowledge_entries"] = []
            
        # Create a proper response object
        output = MemoryAgentOutput(
            knowledge_entries=[],  # We'll need to convert dict entries to KnowledgeEntry objects
            insights=results.get("insights", []),
            patterns=results.get("patterns", []),
            recommendations=results.get("recommendations", []),
            storage_status=results.get("storage_status", {"status": "success"}),
            retrieval_results=results.get("retrieval_results"),
            success=results.get("success", True),
            error=results.get("error"),
            execution_time=execution_time
        )
        
        # Convert knowledge entry dictionaries to objects
        for entry_dict in results.get("knowledge_entries", []):
            try:
                entry = KnowledgeEntry(
                    id=entry_dict.get("id", "unknown"),
                    type=entry_dict.get("type", "unknown"),
                    content=entry_dict.get("content", {}),
                    metadata=entry_dict.get("metadata", {}),
                    confidence=entry_dict.get("confidence", 0.5),
                    source=entry_dict.get("source", "memory_agent"),
                    tags=entry_dict.get("tags", []),
                    timestamp=datetime.now()
                )
                output.knowledge_entries.append(entry)
            except Exception as e:
                self.logger.error(f"Error creating knowledge entry: {e}")
        
        self.logger.info(f"Memory {operation} operation completed successfully")
        return output
    
    async def _store_knowledge(self, workflow_data: Dict[str, Any]) -> MemoryAgentOutput:
        """Store workflow data as knowledge entries."""
        knowledge_entries = []
        
        # Extract and store pain points
        pain_points = workflow_data.get("scout", {}).get("pain_points", [])
        for point in pain_points:
            entry = self._create_knowledge_entry(
                "pain_point", point, "scout_agent", point.get("tags", [])
            )
            knowledge_entries.append(entry)
        
        # Extract and store market gaps
        market_gaps = workflow_data.get("gap_finder", {}).get("market_gaps", [])
        for gap in market_gaps:
            entry = self._create_knowledge_entry(
                "market_gap", gap, "gap_finder_agent", ["market_analysis", "opportunity"]
            )
            knowledge_entries.append(entry)
        
        # Extract and store solutions
        solutions = workflow_data.get("builder", {}).get("solution_prototypes", [])
        for solution in solutions:
            entry = self._create_knowledge_entry(
                "solution", solution, "builder_agent", ["solution", "prototype"]
            )
            knowledge_entries.append(entry)
        
        # Extract insights and patterns
        insights = self._extract_insights(workflow_data)
        for insight in insights:
            entry = self._create_knowledge_entry(
                "insight", insight, "memory_agent", insight.get("tags", [])
            )
            knowledge_entries.append(entry)
        
        # Store entries
        storage_status = await self._persist_entries(knowledge_entries)
        
        # Analyze patterns
        patterns = await self._identify_patterns(knowledge_entries)
        
        # Generate recommendations
        recommendations = self._generate_memory_recommendations(knowledge_entries, patterns)
        
        return MemoryAgentOutput(
            knowledge_entries=knowledge_entries,
            insights=insights,
            patterns=patterns,
            recommendations=recommendations,
            storage_status=storage_status
        )
    
    async def _retrieve_knowledge(self, query: str, filters: Dict[str, Any]) -> MemoryAgentOutput:
        """Retrieve knowledge based on query and filters."""
        # Mock retrieval - in real implementation, use vector search
        retrieved_entries = []
        
        # Simulate search results
        if query:
            mock_entries = [
                {
                    "id": "mock_1",
                    "type": "pain_point",
                    "content": {"description": f"Related to {query}"},
                    "confidence": 0.9,
                    "source": "historical_analysis"
                }
            ]
            retrieved_entries.extend(mock_entries)
        
        return MemoryAgentOutput(
            knowledge_entries=[],
            insights=[],
            patterns=[],
            recommendations=["Refine search query", "Use specific filters"],
            storage_status={"status": "success", "entries_stored": 0},
            retrieval_results=retrieved_entries
        )
    
    async def _analyze_knowledge(self, workflow_data: Dict[str, Any]) -> MemoryAgentOutput:
        """Analyze knowledge for patterns and insights."""
        # Create knowledge entries from current workflow
        entries = []
        
        # Pain points analysis
        pain_points = workflow_data.get("scout", {}).get("pain_points", [])
        for point in pain_points:
            entry = self._create_knowledge_entry(
                "pain_point", point, "scout_agent", point.get("tags", [])
            )
            entries.append(entry)
        
        # Identify patterns
        patterns = await self._identify_patterns(entries)
        
        # Generate insights
        insights = await self._generate_analytical_insights(entries, patterns)
        
        # Create recommendations
        recommendations = await self._generate_analytical_recommendations(insights, patterns)
        
        return MemoryAgentOutput(
            knowledge_entries=entries,
            insights=insights,
            patterns=patterns,
            recommendations=recommendations,
            storage_status={"status": "analysis_complete", "entries_analyzed": len(entries)}
        )
    
    async def _update_knowledge(self, workflow_data: Dict[str, Any]) -> MemoryAgentOutput:
        """Update existing knowledge with new information."""
        # Similar to store but with update logic
        return await self._store_knowledge(workflow_data)
    
    def _create_knowledge_entry(self, entry_type: str, content: Dict[str, Any], 
                              source: str, tags: List[str]) -> KnowledgeEntry:
        """Create a knowledge entry from workflow data."""
        import uuid
        
        return KnowledgeEntry(
            id=str(uuid.uuid4()),
            type=entry_type,
            content=content,
            metadata={
                "source_agent": source,
                "workflow_id": getattr(self.state, 'workflow_id', f"memory_agent_{self.agent_id}"),
                "timestamp": datetime.now()
            },
            confidence=content.get("confidence", 0.8),
            source=source,
            tags=tags,
            timestamp=datetime.now()
        )
    
    def _extract_insights(self, workflow_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from workflow data."""
        insights = []
        
        # Pain point insights
        pain_points = workflow_data.get("scout", {}).get("pain_points", [])
        if pain_points:
            insights.append({
                "type": "pain_point_pattern",
                "description": f"Discovered {len(pain_points)} pain points across workflow",
                "severity_distribution": self._analyze_severity_distribution(pain_points),
                "tags": ["pain_points", "discovery"]
            })
        
        # Market gap insights
        market_gaps = workflow_data.get("gap_finder", {}).get("market_gaps", [])
        if market_gaps:
            avg_opportunity = sum(gap.get("opportunity_score", 0) for gap in market_gaps) / len(market_gaps)
            insights.append({
                "type": "market_opportunity",
                "description": f"Average opportunity score: {avg_opportunity:.1f}/100",
                "total_market_size": sum(gap.get("market_size", 0) for gap in market_gaps),
                "tags": ["market_analysis", "opportunities"]
            })
        
        # Solution insights
        solutions = workflow_data.get("builder", {}).get("solution_prototypes", [])
        if solutions:
            avg_cost = sum(s.get("estimated_cost", 0) for s in solutions) / len(solutions)
            insights.append({
                "type": "solution_viability",
                "description": f"Average solution development cost: ${avg_cost:,.0f}",
                "solution_count": len(solutions),
                "tags": ["solutions", "cost_analysis"]
            })
        
        return insights
    
    def _analyze_severity_distribution(self, pain_points: List[Dict]) -> Dict[str, int]:
        """Analyze severity distribution of pain points."""
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for point in pain_points:
            severity = point.get("severity", "medium").lower()
            if severity in distribution:
                distribution[severity] += 1
        
        return distribution
    
    async def _identify_patterns(self, entries: List[KnowledgeEntry]) -> List[Dict[str, Any]]:
        """Identify patterns in knowledge entries."""
        patterns = []
        
        # Group by type
        type_groups = {}
        for entry in entries:
            if entry.type not in type_groups:
                type_groups[entry.type] = []
            type_groups[entry.type].append(entry)
        
        # Identify patterns for each type
        for entry_type, group in type_groups.items():
            if entry_type == "pain_point":
                patterns.extend(self._analyze_pain_point_patterns(group))
            elif entry_type == "market_gap":
                patterns.extend(self._analyze_market_gap_patterns(group))
            elif entry_type == "solution":
                patterns.extend(self._analyze_solution_patterns(group))
        
        return patterns
    
    def _analyze_pain_point_patterns(self, entries: List[KnowledgeEntry]) -> List[Dict[str, Any]]:
        """Analyze pain point patterns."""
        patterns = []
        
        # Severity patterns
        severities = [e.content.get("severity", "medium") for e in entries]
        severity_counts = {"high": severities.count("high"), 
                          "medium": severities.count("medium"), 
                          "low": severities.count("low")}
        
        patterns.append({
            "type": "severity_distribution",
            "description": "Distribution of pain point severities",
            "data": severity_counts,
            "insight": f"High severity issues: {severity_counts['high']}/{len(entries)}"
        })
        
        return patterns
    
    def _analyze_market_gap_patterns(self, entries: List[KnowledgeEntry]) -> List[Dict[str, Any]]:
        """Analyze market gap patterns."""
        patterns = []
        
        # Opportunity score patterns
        scores = [e.content.get("opportunity_score", 0) for e in entries]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        patterns.append({
            "type": "opportunity_score_pattern",
            "description": "Distribution of opportunity scores",
            "data": {"average": avg_score, "range": [min(scores), max(scores)] if scores else [0, 0]},
            "insight": f"Average opportunity score: {avg_score:.1f}/100"
        })
        
        return patterns
    
    def _analyze_solution_patterns(self, entries: List[KnowledgeEntry]) -> List[Dict[str, Any]]:
        """Analyze solution patterns."""
        patterns = []
        
        # Cost patterns
        costs = [e.content.get("estimated_cost", 0) for e in entries]
        avg_cost = sum(costs) / len(costs) if costs else 0
        
        patterns.append({
            "type": "cost_pattern",
            "description": "Distribution of solution development costs",
            "data": {"average": avg_cost, "range": [min(costs), max(costs)] if costs else [0, 0]},
            "insight": f"Average development cost: ${avg_cost:,.0f}"
        })
        
        return patterns
    
    async def _generate_analytical_insights(self, entries: List[KnowledgeEntry], 
                                          patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate analytical insights from patterns."""
        insights = []
        
        for pattern in patterns:
            if pattern["type"] == "severity_distribution":
                data = pattern["data"]
                if data["high"] > data["medium"] + data["low"]:
                    insights.append({
                        "type": "critical_pattern",
                        "description": "High concentration of severe pain points indicates urgent market need",
                        "severity": "high",
                        "tags": ["critical", "urgent"]
                    })
        
        return insights
    
    async def _generate_analytical_recommendations(self, insights: List[Dict[str, Any]], 
                                                 patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on analytical insights."""
        recommendations = [
            "Store all workflow data for future pattern analysis",
            "Regularly update knowledge base with new findings",
            "Use historical patterns to improve future analyses",
            "Establish feedback loops for continuous learning",
            "Monitor trends in pain point severity and market opportunities"
        ]
        
        return recommendations
    
    async def _persist_entries(self, entries: List[KnowledgeEntry]) -> Dict[str, Any]:
        """Persist knowledge entries to storage."""
        # Mock persistence - in real implementation, use database
        return {
            "status": "success",
            "entries_stored": len(entries),
            "storage_location": "memory_system",
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_operation_phases(self, operation: str) -> List[str]:
        """Get operation-specific phases."""
        phases = {
            "store": ["data_extraction", "entry_creation", "persistence", "validation"],
            "retrieve": ["query_processing", "search", "filtering", "ranking"],
            "analyze": ["data_ingestion", "pattern_analysis", "insight_generation", "recommendation"],
            "update": ["data_comparison", "merge_strategy", "update_execution", "validation"]
        }
        
        return phases.get(operation, ["processing", "completion"])
    
    def _assess_data_complexity(self, workflow_data: Dict[str, Any]) -> str:
        """Assess the complexity of workflow data."""
        total_items = 0
        for agent_data in workflow_data.values():
            if isinstance(agent_data, dict):
                total_items += len(agent_data)
        
        if total_items < 10:
            return "low"
        elif total_items < 50:
            return "medium"
        else:
            return "high"
    
    def _calculate_storage_requirements(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate storage requirements for workflow data."""
        estimated_size = 0
        for agent_data in workflow_data.values():
            estimated_size += len(str(agent_data))
        
        return {
            "estimated_size_kb": estimated_size / 1024,
            "entries_count": len(workflow_data),
            "complexity": self._assess_data_complexity(workflow_data)
        }
    
    def _determine_analysis_scope(self, operation: str) -> str:
        """Determine the analysis scope based on operation."""
        scopes = {
            "store": "data_persistence",
            "retrieve": "targeted_search",
            "analyze": "comprehensive_analysis",
            "update": "incremental_update"
        }
        
        return scopes.get(operation, "standard_processing")
    
    def _generate_memory_recommendations(self, entries: List[KnowledgeEntry], 
                                       patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for memory management."""
        return [
            "Implement automated knowledge base updates",
            "Establish data retention policies",
            "Create knowledge validation workflows",
            "Set up pattern monitoring alerts",
            "Regular knowledge base optimization"
        ]


# Register the agent - moved to agent_registry.py
# from .base import register_agent
# register_agent("memory_agent", MemoryAgent)


# Test code - run this file directly to test the MemoryAgent
if __name__ == "__main__":
    import asyncio
    
    async def test_agent():
        print("Testing MemoryAgent with prompt-driven lifecycle...")
        
        # Create a sample workflow data for testing
        sample_data = {
            "analysis_agent": {
                "pain_points": [
                    {"description": "Difficult navigation", "severity": "high"},
                    {"description": "Slow loading times", "severity": "medium"}
                ],
                "market_gaps": [
                    {"description": "Mobile-friendly interface", "potential": "high"}
                ]
            },
            "research_agent": {
                "findings": ["Competitor A has better UX", "Users prefer simplicity"]
            }
        }
        
        # Create agent input
        agent_input = MemoryAgentInput(
            workflow_data=sample_data,
            operation="store"
        )
        
        # Initialize the agent
        agent = MemoryAgent()
        
        # Execute the agent's full lifecycle
        try:
            print("\n1. Planning memory operation...")
            plan = await agent.plan(agent_input)
            print(f"Plan: {json.dumps(plan, indent=2)}")
            
            print("\n2. Analyzing memory operation requirements...")
            thoughts = await agent.think(agent_input)
            print(f"Analysis: {json.dumps(thoughts, indent=2)}")
            
            print("\n3. Executing memory operation...")
            result = await agent.act(agent_input)
            print("\nMemory operation completed!")
            
            # Display the results
            print(f"Success: {result.success}")
            print(f"Knowledge entries: {len(result.knowledge_entries)}")
            print(f"Insights: {len(result.insights)}")
            print(f"Recommendations: {len(result.recommendations)}")
            if result.recommendations:
                print("\nRecommendations:")
                for rec in result.recommendations[:3]:  # Show first 3 recommendations
                    print(f"- {rec}")
                    
            return result
            
        except Exception as e:
            print(f"Error testing MemoryAgent: {e}")
            raise e
    
    # Run the test
    asyncio.run(test_agent())
