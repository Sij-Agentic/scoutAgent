"""
GapFinderAgent - Market Gap Analysis Agent

This agent specializes in analyzing market gaps and opportunities
based on validated pain points and market research.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from .base import BaseAgent, AgentInput, AgentOutput, AgentState
from scout_agent.llm.utils import LLMAgentMixin
from scout_agent.config import get_config
from scout_agent.services.execution.code_executor import AgentCodeExecutor
from scout_agent.memory.manifest_manager import ManifestManager


@dataclass
class MarketGap:
    """Represents a discovered market gap."""
    gap_description: str
    market_size: float  # in USD
    competition_level: str  # low, medium, high
    opportunity_score: float  # 0-100
    target_segments: List[str]
    solution_ideas: List[str]
    barriers_to_entry: List[str]
    estimated_tam: float  # Total Addressable Market
    estimated_sam: float  # Serviceable Available Market
    estimated_som: float  # Serviceable Obtainable Market
    risk_factors: List[str]
    timeline_to_market: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GapFinderInput:
    """Input for GapFinderAgent."""
    validated_pain_points: List[Dict[str, Any]]  # From ValidatorAgent
    market_context: str
    analysis_scope: str = "comprehensive"  # quick, focused, comprehensive
    include_competitive_analysis: bool = True
    include_market_sizing: bool = True
    context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.validated_pain_points:
            raise ValueError("Must provide validated pain points for gap analysis")
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GapFinderOutput:
    """Output from GapFinderAgent."""
    market_gaps: List[MarketGap]
    prioritized_opportunities: List[Dict[str, Any]]
    market_analysis: Dict[str, Any]
    competitive_landscape: Dict[str, Any]
    recommendations: List[str]
    risk_assessment: Dict[str, Any]
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


class GapFinderAgent(BaseAgent, LLMAgentMixin):
    """
    GapFinderAgent for analyzing market gaps and opportunities.
    
    Uses comprehensive market analysis, competitive research,
    and financial modeling to identify and prioritize market gaps.
    """
    
    def __init__(self, agent_id: str = None):
        BaseAgent.__init__(self, name="gap_finder", agent_id=agent_id)
        LLMAgentMixin.__init__(self, preferred_backend='ollama')
        # self.analysis_agent = AnalysisAgent()  # Commented out - not needed for plan stage
        # self.research_agent = ResearchAgent()  # Commented out - not needed for plan stage
        self.config = get_config()
        self.name = "gap_finder"
        self.preferred_backend = 'ollama'
    
    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Adapter: accept AgentInput, coerce to GapFinderInput, and run."""
        try:
            ctx = agent_input.context or {}
            data = agent_input.data or {}
            validated = []
            market_ctx = ""
            if isinstance(data, dict):
                validated = data.get("validated_pain_points") or data.get("pain_points") or []
                market_ctx = data.get("market_context") or ctx.get("market_context", "")
            gap_input = GapFinderInput(
                validated_pain_points=validated,
                market_context=market_ctx,
                analysis_scope=ctx.get("analysis_scope", "comprehensive"),
                include_competitive_analysis=bool(ctx.get("include_competitive_analysis", True)),
                include_market_sizing=bool(ctx.get("include_market_sizing", True)),
                context=agent_input.context,
                metadata=agent_input.metadata,
            )
            self._update_status('planning')
            plan = await self.plan(gap_input)
            self._update_status('thinking')
            thoughts = await self.think(gap_input)
            self._update_status('acting')
            result = await self.act(gap_input)
            self._update_status('completed')
            if isinstance(result, AgentOutput):
                # Already AgentOutput-like
                result.metadata = {**(result.metadata or {}), 'plan': plan, 'thoughts': thoughts, 'agent_name': self.name, 'agent_id': self.agent_id}
                return result
            # Wrap arbitrary result
            return AgentOutput(
                result=result,
                metadata={'agent_id': self.agent_id, 'agent_name': self.name, 'plan': plan, 'thoughts': thoughts},
                logs=self.execution_logs,
                execution_time=0.0,
                success=True,
            )
        except Exception as e:
            self._update_status('failed')
            return AgentOutput(
                result=None,
                metadata={'agent_id': self.agent_id, 'agent_name': self.name},
                logs=self.execution_logs,
                execution_time=0.0,
                success=False,
                error=str(e),
            )
    
    async def plan(self, input_data: GapFinderInput, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Plan the market gap analysis process."""
        self.logger.info(f"Planning market gap analysis for {len(input_data.validated_pain_points)} pain points")
        
        # Store run_id in state if provided
        if run_id:
            self.state.run_id = run_id
        
        # Step 1: Retrieve data from validator act stage if not provided directly
        validated_pain_points = input_data.validated_pain_points
        if not validated_pain_points:
            self.logger.info("No pain points provided directly, attempting to retrieve from manifest")
            try:
                from scout_agent.memory.manifest_manager import ManifestManager
                
                # Determine run directory
                from pathlib import Path
                project_root = Path(__file__).resolve().parents[2]
                current_run_id = run_id or getattr(self.state, "run_id", "latest")
                run_dir = project_root / "data" / "runs" / current_run_id
                manifest_path = run_dir / "run_manifest.json"
                
                # Try to load from manifest
                manifest_manager = ManifestManager(manifest_path, create_if_missing=False)
                
                # Try to get output from validator_think
                validator_output = manifest_manager.get_node_output("validator_think")
                
                # If not found, try fallbacks
                if not validator_output or "validated_pain_points" not in validator_output:
                    self.logger.info("Could not find validator_think output, trying fallbacks")
                    for node_id in ["validator_act", "validator"]:
                        validator_output = manifest_manager.get_node_output(node_id)
                        if validator_output and "validated_pain_points" in validator_output:
                            self.logger.info(f"Found validated pain points in {node_id}")
                            break
                
                if validator_output and "validated_pain_points" in validator_output:
                    validated_pain_points = validator_output["validated_pain_points"]
                    self.logger.info(f"Retrieved {len(validated_pain_points)} pain points from manifest")
                else:
                    self.logger.warning("Could not find validated pain points in manifest")
            except Exception as e:
                self.logger.error(f"Error retrieving pain points from manifest: {str(e)}")
        
        # Step 2: Generate a plan using the plan.prompt template
        try:
            from scout_agent.llm.utils import load_prompt_template
            
            # Prepare substitutions for the prompt template
            substitutions = {
                "pain_points_count": str(len(validated_pain_points)),
                "market_context": input_data.market_context,
                "analysis_scope": input_data.analysis_scope,
                "include_competitive_analysis": str(input_data.include_competitive_analysis),
                "include_market_sizing": str(input_data.include_market_sizing)
            }
            
            # Load the prompt template with substitutions
            prompt_content = load_prompt_template(
                "plan.prompt", 
                agent_name=self.name,
                substitutions=substitutions
            )
            
            # Generate plan using LLM
            llm_response = await self.llm_generate(
                prompt=prompt_content,
                task_type="planning"
            )
            
            # Extract JSON from LLM response
            plan = self._extract_json(llm_response)
            
            if not plan:
                self.logger.warning("Failed to extract valid JSON from LLM response, using fallback plan")
                plan = self._create_fallback_plan(input_data)
        except Exception as e:
            self.logger.error(f"Error generating plan: {str(e)}")
            plan = self._create_fallback_plan(input_data)
        
        # Step 3: Generate discovery queries for each pain point using plan_discovery.prompt
        discovery_queries = {}
        try:
            for i, pain_point in enumerate(validated_pain_points):
                self.logger.info(f"Generating discovery queries for pain point {i+1}/{len(validated_pain_points)}")
                
                # Extract pain point details - handle different possible structures
                pain_point_text = ""
                if isinstance(pain_point, dict):
                    # Direct dictionary format
                    pain_point_text = pain_point.get("pain_point", "") or pain_point.get("description", "")
                elif isinstance(pain_point, str):
                    # Simple string format
                    pain_point_text = pain_point
                
                if not pain_point_text:
                    self.logger.warning(f"Could not extract text from pain point: {pain_point}")
                    continue
                
                # Default number of queries per category
                n_queries = 3
                
                # Prepare substitutions for the discovery prompt
                discovery_substitutions = {
                    "pain_point_description": pain_point_text,
                    "n_queries": str(n_queries)
                }
                
                # Load the discovery prompt template with substitutions
                discovery_prompt = load_prompt_template(
                    "plan_discovery.prompt",
                    agent_name="gap_finder_agent",
                    substitutions=discovery_substitutions
                )
                
                # Generate discovery queries using LLM
                discovery_response = await self.llm_generate(
                    prompt=discovery_prompt,
                    task_type="discovery"
                )
                
                # Extract JSON from LLM response
                queries = self._extract_json(discovery_response)
                
                if queries:
                    discovery_queries[pain_point_text] = queries
                    self.logger.info(f"Generated discovery queries for pain point {i+1} in {len(queries)} categories")
                else:
                    self.logger.warning(f"Failed to generate discovery queries for pain point {i+1}")
        except Exception as e:
            self.logger.error(f"Error generating discovery queries: {str(e)}")
        
        # Add discovery queries to the plan
        if discovery_queries:
            plan["discovery_queries"] = discovery_queries
        
        # Step 4: Generate DAG metadata for gap finder stages
        dag_metadata = self._generate_dag_metadata(validated_pain_points, discovery_queries)
        plan["dag_metadata"] = dag_metadata
        
        # Step 5: Add execution strategy based on analysis scope and requirements
        execution_strategy = self._generate_execution_strategy(input_data, len(validated_pain_points))
        plan["execution_strategy"] = execution_strategy
        
        self.state.plan = plan
        return plan
        
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text with enhanced error handling and fallback strategies."""
        if not text or not text.strip():
            self.logger.warning("Empty or whitespace-only text provided to _extract_json")
            return {}
            
        import re
        import json
        
        # Log the input for debugging
        self.logger.debug(f"Extracting JSON from text (first 200 chars): {text[:200]}...")
        
        # Strategy 1: Direct JSON parsing
        try:
            result = json.loads(text.strip())
            self.logger.debug("Successfully parsed JSON directly")
            return result
        except json.JSONDecodeError as e:
            self.logger.debug(f"Direct JSON parsing failed: {e}")
        
        # Strategy 2: Extract from markdown code blocks
        code_block_patterns = [
            r'```(?:json)?\s*([\s\S]*?)\s*```',  # Standard markdown
            r'`([^`]+)`',  # Inline code
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    result = json.loads(match.strip())
                    self.logger.debug(f"Successfully parsed JSON from code block: {pattern}")
                    return result
                except json.JSONDecodeError:
                    continue
        
        # Strategy 3: Find JSON objects between curly braces
        json_patterns = [
            r'\{[\s\S]*\}',  # Greedy match for complete JSON
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested braces
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # Clean up common formatting issues
                    cleaned = self._clean_json_string(match)
                    result = json.loads(cleaned)
                    self.logger.debug(f"Successfully parsed JSON after cleaning: {pattern}")
                    return result
                except json.JSONDecodeError:
                    continue
        
        # Strategy 4: Try to extract key-value pairs and reconstruct JSON
        try:
            reconstructed = self._reconstruct_json_from_text(text)
            if reconstructed:
                self.logger.debug("Successfully reconstructed JSON from text patterns")
                return reconstructed
        except Exception as e:
            self.logger.debug(f"JSON reconstruction failed: {e}")
        
        self.logger.warning(f"Failed to extract valid JSON from text. Text preview: {text[:100]}...")
        return {}
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean common JSON formatting issues."""
        import re
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', json_str.strip())
        
        # Fix trailing commas
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        
        # Fix missing quotes around keys (common LLM mistake)
        cleaned = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
        
        # Fix single quotes to double quotes
        cleaned = re.sub(r"'([^']*)'\s*:", r'"\1":', cleaned)
        cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)
        
        return cleaned
    
    def _reconstruct_json_from_text(self, text: str) -> Dict[str, Any]:
        """Attempt to reconstruct JSON from text patterns."""
        import re
        
        result = {}
        
        # Look for key-value patterns
        patterns = [
            r'"([^"]+)"\s*:\s*"([^"]+)"',  # "key": "value"
            r'"([^"]+)"\s*:\s*([\d.]+)',    # "key": number
            r'"([^"]+)"\s*:\s*(true|false)', # "key": boolean
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*"([^"]+)"',  # key: "value"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for key, value in matches:
                # Try to convert value to appropriate type
                if value.lower() in ('true', 'false'):
                    result[key] = value.lower() == 'true'
                elif value.replace('.', '').replace('-', '').isdigit():
                    result[key] = float(value) if '.' in value else int(value)
                else:
                    result[key] = value
        
        return result if result else None
    
    def _generate_dag_metadata(self, validated_pain_points: List[Dict[str, Any]], discovery_queries: Dict[str, Any]) -> Dict[str, Any]:
        """Generate DAG metadata for gap finder stages with proper dependencies and parallelization."""
        from scout_agent.dag.node import NodeType, NodeConfig
        import uuid
        
        nodes = []
        edges = []
        
        # Generate unique node IDs to prevent overwrites
        def generate_node_id(base_name: str, suffix: str = "") -> str:
            unique_id = str(uuid.uuid4())[:8]
            return f"{base_name}_{unique_id}{suffix}" if suffix else f"{base_name}_{unique_id}"
        
        # Step 1: Create search_links nodes (parallel for each pain point)
        search_link_nodes = []
        for i, pain_point in enumerate(validated_pain_points):
            pain_point_text = ""
            if isinstance(pain_point, dict):
                pain_point_text = pain_point.get("pain_point", "") or pain_point.get("description", "")
            elif isinstance(pain_point, str):
                pain_point_text = pain_point
            
            if not pain_point_text:
                continue
                
            # Get queries for this pain point
            queries_for_pain_point = discovery_queries.get(pain_point_text, {})
            all_queries = []
            for category_queries in queries_for_pain_point.values():
                if isinstance(category_queries, list):
                    all_queries.extend(category_queries)
            
            # If no queries found, use default queries based on pain point
            if not all_queries:
                # Generate default search queries from pain point description
                pain_point_keywords = pain_point_text.lower().replace(',', ' ').split()
                key_terms = [word for word in pain_point_keywords if len(word) > 3 and word not in ['with', 'from', 'that', 'this', 'they', 'have', 'been', 'will', 'when', 'where', 'what', 'how']]
                if key_terms:
                    all_queries = [
                        f"{' '.join(key_terms[:3])} solutions",
                        f"{' '.join(key_terms[:2])} tools",
                        f"{' '.join(key_terms[:2])} problems"
                    ]
                else:
                    all_queries = [f"solutions for {pain_point_text[:30]}"]
                
            node_id = generate_node_id("search_links", f"_pp{i+1}")
            search_link_nodes.append({
                "node_id": node_id,
                "name": f"search_links_pp{i+1}",
                "description": f"Search for links related to pain point {i+1}: {pain_point_text[:50]}...",
                "node_type": "TOOL",
                "tool_name": "search_links",
                "dependencies": [],
                "inputs": {
                    "queries": all_queries,
                    "pain_point_id": f"pp{i+1}",
                    "pain_point_title": pain_point_text,
                    "num_results": 3,
                    "use_cache": True
                },
                "config": {
                    "timeout_seconds": 300,
                    "retry_count": 2,
                    "metadata": {
                        "stage": "search_links",
                        "pain_point_index": i,
                        "pain_point_id": f"pp{i+1}"
                    }
                },
                "output_manifest_key": f"search_links_pp{i+1}_output"
            })
        
        nodes.extend(search_link_nodes)
        
        # Step 2: Create extract_content nodes (parallel for each search_links output)
        extract_content_nodes = []
        for search_node in search_link_nodes:
            node_id = generate_node_id("extract_content", f"_pp{search_node['config']['metadata']['pain_point_index']+1}")
            extract_content_nodes.append({
                "node_id": node_id,
                "name": f"extract_content_pp{search_node['config']['metadata']['pain_point_index']+1}",
                "description": f"Extract content from URLs found for pain point {search_node['config']['metadata']['pain_point_index']+1}",
                "node_type": "TOOL",
                "tool_name": "extract_content",
                "dependencies": [search_node["node_id"]],
                "inputs": {
                    "urls": f"${{{search_node['output_manifest_key']}.query_results[*].results[*].link}}",  # Reference to search output URLs
                    "use_cache": True,
                    "include_comments": False,
                    "include_tables": True,
                    "include_links": True,
                    "include_images": False
                },
                "config": {
                    "timeout_seconds": 600,
                    "retry_count": 2,
                    "metadata": {
                        "stage": "extract_content",
                        "pain_point_index": search_node['config']['metadata']['pain_point_index'],
                        "pain_point_id": search_node['config']['metadata']['pain_point_id']
                    }
                },
                "output_manifest_key": f"extract_content_pp{search_node['config']['metadata']['pain_point_index']+1}_output"
            })
            
            # Add edge from search_links to extract_content
            edges.append({
                "from_node": search_node["node_id"],
                "to_node": node_id,
                "edge_type": "data_flow"
            })
        
        nodes.extend(extract_content_nodes)
        
        # Step 3: Create triage_content nodes (parallel for each extract_content output)
        triage_content_nodes = []
        for extract_node in extract_content_nodes:
            node_id = generate_node_id("triage_content", f"_pp{extract_node['config']['metadata']['pain_point_index']+1}")
            triage_content_nodes.append({
                "node_id": node_id,
                "name": f"triage_content_pp{extract_node['config']['metadata']['pain_point_index']+1}",
                "description": f"Triage extracted content for pain point {extract_node['config']['metadata']['pain_point_index']+1}",
                "node_type": "TOOL",
                "tool_name": "triage_content",
                "dependencies": [extract_node["node_id"]],
                "inputs": {
                    "contents": f"${{{extract_node['output_manifest_key']}.contents}}",  # Reference to extract output contents
                    "use_cache": True
                },
                "config": {
                    "timeout_seconds": 400,
                    "retry_count": 2,
                    "metadata": {
                        "stage": "triage_content",
                        "pain_point_index": extract_node['config']['metadata']['pain_point_index'],
                        "pain_point_id": extract_node['config']['metadata']['pain_point_id']
                    }
                },
                "output_manifest_key": f"triage_content_pp{extract_node['config']['metadata']['pain_point_index']+1}_output"
            })
            
            # Add edge from extract_content to triage_content
            edges.append({
                "from_node": extract_node["node_id"],
                "to_node": node_id,
                "edge_type": "data_flow"
            })
        
        nodes.extend(triage_content_nodes)
        
        # Step 4: Create identify_vendors nodes (parallel for each triage_content output)
        identify_vendors_nodes = []
        for triage_node in triage_content_nodes:
            node_id = generate_node_id("identify_vendors", f"_pp{triage_node['config']['metadata']['pain_point_index']+1}")
            identify_vendors_nodes.append({
                "node_id": node_id,
                "name": f"identify_vendors_pp{triage_node['config']['metadata']['pain_point_index']+1}",
                "description": f"Identify vendors from triaged content for pain point {triage_node['config']['metadata']['pain_point_index']+1}",
                "node_type": "TOOL",
                "tool_name": "identify_vendors",
                "dependencies": [triage_node["node_id"]],
                "inputs": {
                    "contents": f"${{{triage_node['output_manifest_key']}.triage_results}}",  # Reference to triage output results
                    "use_cache": True
                },
                "config": {
                    "timeout_seconds": 300,
                    "retry_count": 2,
                    "metadata": {
                        "stage": "identify_vendors",
                        "pain_point_index": triage_node['config']['metadata']['pain_point_index'],
                        "pain_point_id": triage_node['config']['metadata']['pain_point_id']
                    }
                },
                "output_manifest_key": f"identify_vendors_pp{triage_node['config']['metadata']['pain_point_index']+1}_output"
            })
            
            # Add edge from triage_content to identify_vendors
            edges.append({
                "from_node": triage_node["node_id"],
                "to_node": node_id,
                "edge_type": "data_flow"
            })
        
        nodes.extend(identify_vendors_nodes)
        
        # Step 5: Create vendor_research nodes (one per pain point, handling multiple vendors via deduplication)
        vendor_research_nodes = []
        for vendor_node in identify_vendors_nodes:
            node_id = generate_node_id("vendor_research", f"_pp{vendor_node['config']['metadata']['pain_point_index']+1}")
            
            # Get pain point text for the research context
            pain_point_index = vendor_node['config']['metadata']['pain_point_index']
            pain_point_text = ""
            if isinstance(validated_pain_points[pain_point_index], dict):
                pain_point_text = validated_pain_points[pain_point_index].get("pain_point", "") or validated_pain_points[pain_point_index].get("description", "")
            elif isinstance(validated_pain_points[pain_point_index], str):
                pain_point_text = validated_pain_points[pain_point_index]
            
            vendor_research_nodes.append({
                "node_id": node_id,
                "name": f"vendor_research_pp{vendor_node['config']['metadata']['pain_point_index']+1}",
                "description": f"Research identified vendors for pain point {vendor_node['config']['metadata']['pain_point_index']+1}",
                "node_type": "FUNCTION",  # Changed to FUNCTION to work with DAG engine
                "tool_name": "vendor_research_batch",  # Use batch processing tool
                "dependencies": [vendor_node["node_id"]],
                "inputs": {
                    "vendors_list": f"${{{vendor_node['output_manifest_key']}.vendor_results}}",  # Reference to all identified vendors
                    "pain_point": pain_point_text,
                    "deduplicate": True  # Enable vendor deduplication
                },
                "config": {
                    "timeout_seconds": 300,
                    "retry_count": 2,
                    "metadata": {
                        "pain_point_id": vendor_node['config']['metadata']['pain_point_id']
                    }
                },
                "output_manifest_key": f"vendor_research_pp{vendor_node['config']['metadata']['pain_point_index']+1}_output"
            })
            
            # Add edge from identify_vendors to vendor_research
            edges.append({
                "from_node": vendor_node["node_id"],
                "to_node": node_id,
                "edge_type": "data_flow"
            })
        
        nodes.extend(vendor_research_nodes)
        
        # Step 6: Create aggregation node to collect all vendor research outputs
        aggregation_node_id = generate_node_id("aggregate_research")
        aggregation_node = {
            "node_id": aggregation_node_id,
            "name": "aggregate_research_results",
            "description": "Aggregate all vendor research results for final analysis",
            "node_type": "FUNCTION",
            "tool_name": "aggregate_gap_analysis",
            "dependencies": [node["node_id"] for node in vendor_research_nodes],
            "inputs": {
                "research_outputs": [f"${{{node['output_manifest_key']}.research_data}}" for node in vendor_research_nodes],
                "pain_points": validated_pain_points,
                "merge_strategy": "by_pain_point_id",
                "output_format": "comprehensive_gap_analysis"
            },
            "config": {
                "timeout_seconds": 300,
                "retry_count": 1,
                "metadata": {
                    "stage": "aggregation",
                    "final_stage": True
                }
            },
            "output_manifest_key": "gap_finder_final_output"
        }
        
        nodes.append(aggregation_node)
        
        # Add edges from all vendor_research nodes to aggregation
        for vendor_research_node in vendor_research_nodes:
            edges.append({
                "from_node": vendor_research_node["node_id"],
                "to_node": aggregation_node_id,
                "edge_type": "data_flow"
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "execution_strategy": "parallel_by_pain_point",
            "total_nodes": len(nodes),
            "parallel_chains": len(validated_pain_points),
            "stages": ["search_links", "extract_content", "triage_content", "identify_vendors", "vendor_research", "aggregation"],
            "manifest_keys": {
                "search_links": [node["output_manifest_key"] for node in search_link_nodes],
                "extract_content": [node["output_manifest_key"] for node in extract_content_nodes],
                "triage_content": [node["output_manifest_key"] for node in triage_content_nodes],
                "identify_vendors": [node["output_manifest_key"] for node in identify_vendors_nodes],
                "vendor_research": [node["output_manifest_key"] for node in vendor_research_nodes],
                "final_output": aggregation_node["output_manifest_key"]
            },
            "manifest_handling": {
                "use_unique_keys": True,
                "merge_strategy": "by_output_key",
                "prevent_overwrites": True,
                "parallel_node_outputs": {
                    "search_links": "merge_by_pain_point",
                    "extract_content": "merge_by_pain_point",
                    "triage_content": "merge_by_pain_point",
                    "identify_vendors": "merge_by_pain_point",
                    "vendor_research": "merge_by_pain_point"
                }
            }
        }
    
    def _create_fallback_plan(self, input_data: GapFinderInput) -> Dict[str, Any]:
        """Create a fallback plan if LLM generation fails."""
        fallback_plan = {
            "operation": "market_gap_analysis",
            "phases": [
                "pain_point_clustering",
                "market_research",
                "competitive_analysis",
                "market_sizing",
                "opportunity_scoring",
                "risk_assessment",
                "prioritization"
            ],
            "data_sources": [
                "web_search",
                "market_reports",
                "competitor_websites",
                "review_platforms"
            ],
            "expected_duration": 900,  # 15 minutes
            "special_considerations": [
                f"Analysis scope: {input_data.analysis_scope}",
                f"Include competitive analysis: {input_data.include_competitive_analysis}",
                f"Include market sizing: {input_data.include_market_sizing}"
            ]
        }
        
        # Add execution strategy to fallback plan
        fallback_plan["execution_strategy"] = self._generate_execution_strategy(input_data, len(input_data.validated_pain_points))
        
        return fallback_plan
    
    def _generate_execution_strategy(self, input_data: GapFinderInput, pain_point_count: int) -> Dict[str, Any]:
        """Generate execution strategy based on input parameters and pain point count."""
        # Determine parallelization strategy based on pain point count
        if pain_point_count <= 2:
            parallelization = "sequential"
            max_concurrent = 1
        elif pain_point_count <= 5:
            parallelization = "limited_parallel"
            max_concurrent = 2
        else:
            parallelization = "full_parallel"
            max_concurrent = min(pain_point_count, 4)  # Cap at 4 for resource management
        
        # Determine timeout strategy based on analysis scope
        timeout_multipliers = {
            "quick": 0.5,
            "focused": 1.0,
            "comprehensive": 2.0
        }
        base_timeout = 300  # 5 minutes base
        timeout_multiplier = timeout_multipliers.get(input_data.analysis_scope, 1.0)
        node_timeout = int(base_timeout * timeout_multiplier)
        
        # Determine retry strategy
        retry_strategy = {
            "max_retries": 2 if input_data.analysis_scope == "comprehensive" else 1,
            "retry_delay": 30,  # seconds
            "exponential_backoff": True
        }
        
        # Resource allocation based on requirements
        resource_allocation = {
            "memory_limit": "2GB",
            "cpu_limit": "2 cores",
            "disk_space": "1GB"
        }
        
        # Adjust for competitive analysis and market sizing
        if input_data.include_competitive_analysis:
            resource_allocation["memory_limit"] = "3GB"
            node_timeout = int(node_timeout * 1.5)
        
        if input_data.include_market_sizing:
            resource_allocation["cpu_limit"] = "3 cores"
            node_timeout = int(node_timeout * 1.3)
        
        return {
            "execution_mode": "dag_based",
            "parallelization": parallelization,
            "max_concurrent_nodes": max_concurrent,
            "timeout_per_node": node_timeout,
            "total_timeout": node_timeout * pain_point_count * 2,  # Conservative estimate
            "retry_strategy": retry_strategy,
            "resource_allocation": resource_allocation,
            "error_handling": {
                "continue_on_error": True,
                "partial_results_acceptable": True,
                "fallback_to_simplified": input_data.analysis_scope != "quick"
            },
            "optimization": {
                "cache_intermediate_results": True,
                "reuse_similar_queries": True,
                "batch_similar_operations": parallelization != "sequential"
            },
            "monitoring": {
                "progress_reporting": True,
                "performance_metrics": True,
                "resource_usage_tracking": True
            }
        }
    
    async def think(self, input_data: GapFinderInput) -> Dict[str, Any]:
        """Analyze pain points to identify market gaps."""
        self.logger.info("Analyzing market gaps and opportunities...")
        
        # Cluster similar pain points
        clusters = await self._cluster_pain_points(input_data.validated_pain_points)
        
        # Analyze market context
        market_context = await self._analyze_market_context(
            input_data.market_context,
            clusters
        )
        
        # Prepare gap analysis strategy
        analysis_strategy = {
            "cluster_count": len(clusters),
            "market_segments": list(set(p.get("market", "") for p in input_data.validated_pain_points)),
            "analysis_approach": self._determine_analysis_approach(input_data.analysis_scope),
            "competitive_intensity": "medium",  # Will be updated
            "market_maturity": "growing",  # Will be analyzed
            "expected_gaps": min(len(clusters), 5)
        }
        
        return analysis_strategy
    
    async def act(self, input_data: GapFinderInput) -> GapFinderOutput:
        """Execute market gap analysis and return results."""
        self.logger.info("Executing market gap analysis...")
        
        start_time = datetime.now()
        
        # Cluster pain points into market opportunities
        clusters = await self._cluster_pain_points(input_data.validated_pain_points)
        
        # Analyze each cluster for market gaps
        market_gaps = []
        for cluster in clusters:
            gap = await self._analyze_market_gap(
                cluster,
                input_data.market_context,
                input_data.include_competitive_analysis,
                input_data.include_market_sizing
            )
            market_gaps.append(gap)
        
        # Analyze competitive landscape
        competitive_landscape = await self._analyze_competitive_landscape(
            market_gaps,
            input_data.market_context
        )
        
        # Prioritize opportunities
        prioritized_opportunities = self._prioritize_opportunities(market_gaps)
        
        # Generate market insights
        market_analysis = await self._generate_market_analysis(
            market_gaps,
            competitive_landscape
        )
        
        # Risk assessment
        risk_assessment = self._assess_risks(market_gaps, competitive_landscape)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            prioritized_opportunities,
            risk_assessment
        )
        
        return GapFinderOutput(
            market_gaps=market_gaps,
            prioritized_opportunities=prioritized_opportunities,
            market_analysis=market_analysis,
            competitive_landscape=competitive_landscape,
            recommendations=recommendations,
            risk_assessment=risk_assessment
        )

    async def collect(self, plan: Dict[str, Any], run_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute the collect stage - build and execute DAG nodes from plan stage."""
        self.logger.info("Starting gap_finder collect stage...")
        
        try:
            # Execute DAG tool nodes using the plan from the plan stage
            await self._execute_plan_non_agent_nodes(plan, run_id)
            
            # Aggregate and process final results
            final_output = await self._aggregate_collect_results(plan, run_id)
            
            # Store final output to manifest
            await self._write_stage_output("gapfinder_collect", final_output, run_id)
            
            self.logger.info("Gap finder collect stage completed successfully")
            return final_output
            
        except Exception as e:
            self.logger.error(f"Error in gap_finder collect stage: {str(e)}")
            raise
    
    async def _execute_plan_non_agent_nodes(self, plan: Dict[str, Any], run_id: Optional[str] = None):
        """Execute tool nodes from the DAG plan by directly calling MCP tools."""
        self.logger.info("Executing DAG tool nodes via direct MCP calls...")
        
        # Import MCP client for direct tool calls
        from scout_agent.mcp_integration.client.multi import MultiMCPClient
        from scout_agent.mcp_integration.config import load_server_configs
        
        # Get DAG nodes from plan
        self.logger.info(f"Plan structure keys: {list(plan.keys())}")
        
        # Try different possible locations for DAG nodes
        dag_nodes = []
        if "stages" in plan and "scout_plan" in plan["stages"]:
            dag_nodes = plan.get("stages", {}).get("scout_plan", {}).get("data", {}).get("dag", {}).get("nodes", [])
            self.logger.info(f"Found {len(dag_nodes)} nodes in stages.scout_plan.data.dag.nodes")
        elif "dag_metadata" in plan:
            dag_nodes = plan.get("dag_metadata", {}).get("nodes", [])
            self.logger.info(f"Found {len(dag_nodes)} nodes in dag_metadata.nodes")
        elif "dag" in plan:
            dag_nodes = plan.get("dag", {}).get("nodes", [])
            self.logger.info(f"Found {len(dag_nodes)} nodes in dag.nodes")
        
        # Filter tool nodes
        tool_nodes = [node for node in dag_nodes if "tool_name" in node]
        self.logger.info(f"Found {len(tool_nodes)} tool nodes to execute")
        
        if not tool_nodes:
            self.logger.info("No tool nodes found to execute")
            return
        
        # Initialize MCP client
        try:
            configs = load_server_configs()
            multi_client = MultiMCPClient(configs)
            await multi_client.initialize()
            self.logger.info("MCP client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP client: {e}")
            return
        
        # Execute each tool node in dependency order
        manifest_manager = ManifestManager(run_id or "default")
        
        # Sort nodes by dependency order
        def get_dependency_order(nodes):
            """Sort nodes by dependency order using topological sort."""
            # Create dependency graph
            node_map = {node.get("node_id"): node for node in nodes}
            in_degree = {node.get("node_id"): 0 for node in nodes}
            
            # Calculate in-degrees
            for node in nodes:
                node_id = node.get("node_id")
                dependencies = node.get("dependencies", [])
                for dep in dependencies:
                    if dep in in_degree:
                        in_degree[node_id] += 1
            
            # Topological sort
            queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
            result = []
            
            while queue:
                current = queue.pop(0)
                result.append(node_map[current])
                
                # Update in-degrees of dependent nodes
                for node in nodes:
                    if current in node.get("dependencies", []):
                        in_degree[node.get("node_id")] -= 1
                        if in_degree[node.get("node_id")] == 0:
                            queue.append(node.get("node_id"))
            
            return result
        
        # Sort tool nodes by dependency order
        ordered_nodes = get_dependency_order(tool_nodes)
        self.logger.info(f"Executing {len(ordered_nodes)} tool nodes in dependency order")
        
        for i, node in enumerate(ordered_nodes):
            try:
                node_id = node.get("node_id") or f"tool_node_{i}_{node.get('tool_name', 'unknown')}"
                tool_name = node.get("tool_name")
                inputs = node.get("inputs", {})
                
                self.logger.info(f"Executing tool node: {node_id} with tool: {tool_name}")
                
                # Resolve template variables in inputs before calling MCP tool
                resolved_inputs = self._resolve_template_variables(inputs, manifest_manager)
                
                # Make direct MCP tool call
                result = await multi_client.call_tool(tool_name, resolved_inputs)
                
                # Process and store result
                if result:
                    # Extract content from MCP response
                    if hasattr(result, 'content') and result.content:
                        content_item = result.content[0] if isinstance(result.content, list) else result.content
                        if hasattr(content_item, 'text'):
                            try:
                                # Try to parse as JSON
                                result_data = json.loads(content_item.text)
                            except json.JSONDecodeError:
                                # If not JSON, store as text
                                result_data = {"output": content_item.text}
                        else:
                            result_data = {"raw_result": str(content_item)}
                    else:
                        result_data = {"raw_result": str(result)}
                    
                    # Store to manifest
                    output_key = node.get("output_manifest_key", f"{node_id}_output")
                    manifest_manager.store_node_output(output_key, result_data)
                    
                    self.logger.info(f"Successfully executed and stored results for node: {node_id}")
                else:
                    self.logger.error(f"Tool node {node_id} returned no result")
                    
            except Exception as e:
                self.logger.error(f"Error executing tool node {node.get('node_id', 'unknown')}: {str(e)}")
                # Continue with other nodes even if one fails
                continue
        
        # Cleanup MCP client
        try:
            await multi_client.shutdown()
            self.logger.info("MCP client shutdown successfully")
        except Exception as e:
            self.logger.error(f"Error shutting down MCP client: {e}")
    
    async def _aggregate_collect_results(self, plan: Dict[str, Any], run_id: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate results from executed tool nodes and generate final collect output."""
        self.logger.info("Aggregating collect stage results...")
        
        manifest_manager = ManifestManager(run_id or "default")
        
        # Get all tool node results from manifest
        dag_nodes = plan.get("dag_metadata", {}).get("nodes", [])
        tool_nodes = [node for node in dag_nodes if node.get("node_type") == "TOOL"]
        
        aggregated_data = {
            "market_research_data": {},
            "competitive_analysis": {},
            "market_sizing_data": {},
            "vendor_analysis": {},
            "execution_metadata": {
                "total_nodes_executed": len(tool_nodes),
                "execution_timestamp": datetime.now().isoformat(),
                "run_id": run_id or "default"
            }
        }
        
        # Collect results from each tool node
        for node in tool_nodes:
            try:
                node_id = node.get("node_id")
                # Use the output_manifest_key to get the actual stored result
                output_key = node.get("output_manifest_key", f"{node_id}_output")
                node_result = manifest_manager.get_node_output(output_key)
                
                if node_result:
                    # Categorize results based on node type/purpose
                    if "market_research" in node_id.lower():
                        aggregated_data["market_research_data"][node_id] = node_result
                    elif "competitive" in node_id.lower():
                        aggregated_data["competitive_analysis"][node_id] = node_result
                    elif "sizing" in node_id.lower() or "market_size" in node_id.lower():
                        aggregated_data["market_sizing_data"][node_id] = node_result
                    elif "vendor" in node_id.lower():
                        aggregated_data["vendor_analysis"][node_id] = node_result
                    else:
                        # Generic tool results
                        if "tool_results" not in aggregated_data:
                            aggregated_data["tool_results"] = {}
                        aggregated_data["tool_results"][node_id] = node_result
                    
                    self.logger.info(f"Retrieved results for node: {node_id} using key: {output_key}")
                else:
                    self.logger.warning(f"No results found for node: {node_id} using key: {output_key}")
                        
            except Exception as e:
                self.logger.error(f"Error retrieving results for node {node.get('node_id', 'unknown')}: {str(e)}")
                continue
        
        # Generate summary insights
        aggregated_data["summary"] = self._generate_collect_summary(aggregated_data)
        
        return aggregated_data
    
    def _generate_collect_summary(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary insights from collected data."""
        return {
            "data_sources_collected": len([k for k in aggregated_data.keys() if k != "execution_metadata" and k != "summary"]),
            "market_research_sources": len(aggregated_data.get("market_research_data", {})),
            "competitive_analysis_sources": len(aggregated_data.get("competitive_analysis", {})),
            "market_sizing_sources": len(aggregated_data.get("market_sizing_data", {})),
            "vendor_analysis_sources": len(aggregated_data.get("vendor_analysis", {})),
            "collection_status": "completed",
            "next_stage_ready": True
        }
    
    def _resolve_template_variables(self, inputs: Dict[str, Any], manifest_manager: ManifestManager) -> Dict[str, Any]:
        """Resolve ${...} template variables in tool inputs using manifest data."""
        import re
        
        resolved_inputs = {}
        
        for key, value in inputs.items():
            if isinstance(value, str) and "${" in value:
                # Find all template variables in the format ${variable_name}
                template_pattern = r'\$\{([^}]+)\}'
                matches = re.findall(template_pattern, value)
                
                # Check if the entire value is a single template variable
                if len(matches) == 1 and value.strip() == f"${{{matches[0]}}}":
                    # Direct replacement - preserve original data type
                    match = matches[0]
                    try:
                        # Parse the template variable (e.g., "triage_content_pp1_output.contents")
                        if '.' in match:
                            node_key, field_path = match.split('.', 1)
                            # Backward compatibility: convert .contents to .triage_results
                            if field_path == 'contents':
                                field_path = 'triage_results'
                                self.logger.info(f"Converting deprecated .contents to .triage_results for {node_key}")
                        else:
                            node_key, field_path = match, None
                        
                        # First try to get the node output using the key directly
                        node_output = manifest_manager.get_node_output(node_key)
                        
                        # If not found, try to find the actual node_id that maps to this output key
                        if not node_output:
                            # Look through all stages to find a node that has this output_manifest_key
                            manifest = manifest_manager.get_manifest()
                            for stage_key, stage_data in manifest.get("stages", {}).items():
                                # Check if this stage was stored with the output key we're looking for
                                if stage_key == node_key and "data" in stage_data:
                                    node_output = stage_data["data"]
                                    break
                        
                        # Handle MCP tool output format: {"content": [{"type": "text", "text": "{...}"}]}
                        if node_output and isinstance(node_output, dict) and "content" in node_output:
                            content_list = node_output.get("content", [])
                            if content_list and isinstance(content_list, list) and len(content_list) > 0:
                                first_content = content_list[0]
                                if isinstance(first_content, dict) and first_content.get("type") == "text":
                                    text_content = first_content.get("text", "")
                                    if text_content:
                                        try:
                                            # Parse the JSON string to get the actual data
                                            node_output = json.loads(text_content)
                                        except json.JSONDecodeError:
                                            # If it's not valid JSON, keep the original text
                                            pass
                        
                        if node_output:
                            # Navigate to the specific field if specified
                            if field_path:
                                field_value = self._get_nested_field(node_output, field_path)
                            else:
                                field_value = node_output
                            
                            if field_value is not None:
                                # Direct assignment preserves data type (list, dict, etc.)
                                resolved_inputs[key] = field_value
                                self.logger.info(f"Resolved template variable ${{{match}}} to {type(field_value).__name__} with {len(field_value) if isinstance(field_value, (list, dict)) else 'N/A'} items")
                            else:
                                self.logger.warning(f"Template variable ${{{match}}} resolved to None")
                                resolved_inputs[key] = value  # Keep original if resolution fails
                        else:
                            self.logger.warning(f"No data found for template variable ${{{match}}} - using empty default")
                            # For missing triage_content data, provide appropriate defaults
                            if 'triage_content' in match:
                                if isinstance(value, list):
                                    resolved_inputs[key] = []  # Empty list for list inputs
                                elif isinstance(value, dict):
                                    resolved_inputs[key] = {}  # Empty dict for dict inputs
                                else:
                                    resolved_inputs[key] = ""  # Empty string for string inputs
                            else:
                                resolved_inputs[key] = value  # Keep original for other missing data
                            
                    except Exception as e:
                        self.logger.error(f"Error resolving template variable ${{{match}}}: {str(e)}")
                        resolved_inputs[key] = value  # Keep original on error
                else:
                    # String interpolation - convert to string
                    resolved_value = value
                    for match in matches:
                        try:
                            # Parse the template variable
                            if '.' in match:
                                node_key, field_path = match.split('.', 1)
                            else:
                                node_key, field_path = match, None
                            
                            # Get the node output from manifest
                            node_output = manifest_manager.get_node_output(node_key)
                            
                            if node_output:
                                # Navigate to the specific field if specified
                                if field_path:
                                    field_value = self._get_nested_field(node_output, field_path)
                                else:
                                    field_value = node_output
                                
                                # Replace the template variable with string representation
                                template_var = f"${{{match}}}"
                                if field_value is not None:
                                    resolved_value = resolved_value.replace(template_var, str(field_value))
                                    self.logger.info(f"Interpolated template variable {template_var} as string")
                                else:
                                    self.logger.warning(f"Template variable {template_var} resolved to None")
                            else:
                                self.logger.warning(f"No data found for template variable {template_var} - removing from string")
                                # For missing triage_content data in string interpolation, remove the template variable
                                if 'triage_content' in match:
                                    resolved_value = resolved_value.replace(template_var, "")
                                
                        except Exception as e:
                            self.logger.error(f"Error resolving template variable ${{{match}}}: {str(e)}")
                            continue
                    
                    resolved_inputs[key] = resolved_value
            else:
                # Non-template values pass through unchanged
                resolved_inputs[key] = value
        
        return resolved_inputs
    
    def _get_nested_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get a nested field from data using dot notation with support for multiple [*] expressions."""
        try:
            # Handle multiple array indexing like 'query_results[*].results[*].link'
            if '[*]' in field_path:
                # Split by [*] to get all parts
                parts = field_path.split('[*]')
                current = data
                
                # Process each part
                for i, part in enumerate(parts):
                    # Clean up the part (remove leading dots)
                    clean_part = part.lstrip('.')
                    
                    if i == 0:
                        # First part: navigate to the first array
                        if clean_part:
                            for subpart in clean_part.split('.'):
                                if subpart:
                                    current = current[subpart]
                    else:
                        # Subsequent parts: we're dealing with arrays
                        if isinstance(current, list):
                            new_current = []
                            for item in current:
                                if clean_part:
                                    # Navigate through the remaining path in each item
                                    try:
                                        item_value = item
                                        for subpart in clean_part.split('.'):
                                            if subpart:
                                                item_value = item_value[subpart]
                                        
                                        # If this results in a list, extend; otherwise append
                                        if isinstance(item_value, list):
                                            new_current.extend(item_value)
                                        else:
                                            new_current.append(item_value)
                                    except (KeyError, TypeError):
                                        continue
                                else:
                                    # No remaining path, just add the item
                                    new_current.append(item)
                            current = new_current
                        else:
                            # Not a list, can't process [*]
                            return None
                
                return current
            else:
                # Simple dot notation navigation
                current = data
                for part in field_path.split('.'):
                    if part:
                        current = current[part]
                return current
                
        except (KeyError, TypeError, IndexError) as e:
            self.logger.warning(f"Could not resolve field path '{field_path}': {str(e)}")
            return None
    
    async def _write_stage_output(self, stage_name: str, output_data: Dict[str, Any], run_id: Optional[str] = None):
        """Write stage output to manifest storage."""
        try:
            manifest_manager = ManifestManager(run_id or "default")
            manifest_manager.store_node_output(stage_name, output_data)
            self.logger.info(f"Successfully stored {stage_name} output to manifest")
        except Exception as e:
            self.logger.error(f"Error storing {stage_name} output: {str(e)}")
            raise

    async def _analyze_market_context(self, market_context: str, *args, **kwargs) -> Dict[str, Any]:
        """Analyze the overall market context and trends."""
        # Mock implementation for validation
        return {
            "market_size": "large",
            "growth_rate": "high",
            "maturity": "growing",
            "key_trends": ["digital_transformation", "automation", "ai_adoption"],
            "barriers": ["technical_complexity", "regulatory_compliance", "market_competition"]
        }
    
    async def _cluster_pain_points(self, pain_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster similar pain points into market opportunities."""
        clusters = []
        
        # Mock clustering - in real implementation, use NLP
        cluster_themes = [
            {"theme": "onboarding_complexity", "pain_points": []},
            {"theme": "integration_challenges", "pain_points": []},
            {"theme": "pricing_barriers", "pain_points": []},
            {"theme": "performance_issues", "pain_points": []},
            {"theme": "usability_problems", "pain_points": []}
        ]
        
        for point in pain_points:
            description = point.get("description", "").lower()
            tags = point.get("tags", [])
            
            # Simple clustering based on keywords
            if any(word in description for word in ["setup", "onboarding", "complex"]):
                cluster_themes[0]["pain_points"].append(point)
            elif any(word in description for word in ["integration", "connect", "sync"]):
                cluster_themes[1]["pain_points"].append(point)
            elif any(word in description for word in ["price", "cost", "expensive"]):
                cluster_themes[2]["pain_points"].append(point)
            elif any(word in description for word in ["slow", "performance", "lag"]):
                cluster_themes[3]["pain_points"].append(point)
            else:
                cluster_themes[4]["pain_points"].append(point)
        
        # Filter clusters with actual pain points
        return [cluster for cluster in cluster_themes if cluster["pain_points"]]
    
    async def _analyze_market_gap(self, cluster: Dict[str, Any], market_context: str,
                                include_competitive: bool, include_sizing: bool) -> MarketGap:
        """Analyze a specific market gap."""
        theme = cluster["theme"]
        pain_points = cluster["pain_points"]
        
        # Market research
        market_data = await self._research_market_for_theme(theme, market_context)
        
        # Competitive analysis
        competitive_data = {}
        if include_competitive:
            competitive_data = await self._analyze_competition_for_theme(theme, market_context)
        
        # Market sizing
        market_sizing = {}
        if include_sizing:
            market_sizing = await self._size_market_for_theme(theme, market_context, pain_points)
        
        # Calculate opportunity score
        opportunity_score = self._calculate_opportunity_score(
            market_data,
            competitive_data,
            len(pain_points)
        )
        
        # Generate solution ideas
        solution_ideas = self._generate_solution_ideas(theme, pain_points)
        
        # Assess barriers
        barriers = self._assess_barriers_to_entry(theme, market_context)
        
        # Risk factors
        risk_factors = self._identify_risk_factors(theme, market_context)
        
        return MarketGap(
            gap_description=f"Market gap in {theme} for {market_context}",
            market_size=market_sizing.get("total_market", 25000000),
            competition_level=competitive_data.get("competition_level", "medium"),
            opportunity_score=opportunity_score,
            target_segments=self._identify_target_segments(theme, pain_points),
            solution_ideas=solution_ideas,
            barriers_to_entry=barriers,
            estimated_tam=market_sizing.get("tam", 50000000),
            estimated_sam=market_sizing.get("sam", 25000000),
            estimated_som=market_sizing.get("som", 5000000),
            risk_factors=risk_factors,
            timeline_to_market="6-12 months"
        )
    
    async def _research_market_for_theme(self, theme: str, market_context: str) -> Dict[str, Any]:
        """Research market data for a specific theme."""
        # Mock research
        theme_markets = {
            "onboarding_complexity": {"size": 30000000, "growth": 0.2},
            "integration_challenges": {"size": 45000000, "growth": 0.15},
            "pricing_barriers": {"size": 25000000, "growth": 0.18},
            "performance_issues": {"size": 35000000, "growth": 0.12},
            "usability_problems": {"size": 40000000, "growth": 0.22}
        }
        
        return theme_markets.get(theme, {"size": 20000000, "growth": 0.15})
    
    async def _analyze_competition_for_theme(self, theme: str, market_context: str) -> Dict[str, Any]:
        """Analyze competition for a specific theme."""
        # Mock competitive analysis
        theme_competition = {
            "onboarding_complexity": {"level": "medium", "players": 8},
            "integration_challenges": {"level": "high", "players": 12},
            "pricing_barriers": {"level": "low", "players": 5},
            "performance_issues": {"level": "medium", "players": 7},
            "usability_problems": {"level": "high", "players": 15}
        }
        
        data = theme_competition.get(theme, {"level": "medium", "players": 10})
        return {
            "competition_level": data["level"],
            "direct_competitors": data["players"],
            "market_saturation": data["players"] * 0.05,
            "competitive_intensity": "high" if data["players"] > 10 else "medium"
        }
    
    async def _size_market_for_theme(self, theme: str, market_context: str, 
                                   pain_points: List[Dict]) -> Dict[str, float]:
        """Size the market for a specific theme."""
        # Mock market sizing
        base_size = 50000000  # $50M base
        
        # Adjust based on pain point count and severity
        pain_point_factor = min(len(pain_points) * 0.1, 1.5)
        
        tam = base_size * pain_point_factor
        sam = tam * 0.5  # Serviceable Available Market
        som = sam * 0.2  # Serviceable Obtainable Market
        
        return {
            "tam": tam,
            "sam": sam,
            "som": som,
            "total_market": tam
        }
    
    def _calculate_opportunity_score(self, market_data: Dict, competitive_data: Dict, 
                                   pain_point_count: int) -> float:
        """Calculate opportunity score for a market gap."""
        # Market size score (0-40 points)
        market_size = market_data.get("size", 0)
        size_score = min(market_size / 1000000, 40)  # $1M = 40 points
        
        # Competition score (0-30 points, reverse scoring)
        competition_level = competitive_data.get("level", "medium")
        competition_scores = {"low": 30, "medium": 20, "high": 10}
        competition_score = competition_scores.get(competition_level, 20)
        
        # Pain point intensity score (0-20 points)
        intensity_score = min(pain_point_count * 4, 20)
        
        # Growth potential score (0-10 points)
        growth_rate = market_data.get("growth", 0.15)
        growth_score = min(growth_rate * 50, 10)
        
        total_score = size_score + competition_score + intensity_score + growth_score
        return min(total_score, 100)
    
    def _generate_solution_ideas(self, theme: str, pain_points: List[Dict]) -> List[str]:
        """Generate solution ideas for a market gap."""
        solutions = {
            "onboarding_complexity": [
                "Interactive guided tours",
                "Simplified setup wizard",
                "Template-based configuration",
                "AI-powered onboarding assistant"
            ],
            "integration_challenges": [
                "Universal API connector",
                "Pre-built integrations",
                "Integration marketplace",
                "No-code integration builder"
            ],
            "pricing_barriers": [
                "Freemium model",
                "Usage-based pricing",
                "Small business tier",
                "Open-source alternative"
            ],
            "performance_issues": [
                "Performance optimization tools",
                "Caching solutions",
                "CDN integration",
                "Monitoring dashboard"
            ],
            "usability_problems": [
                "UI/UX redesign",
                "User testing platform",
                "Accessibility improvements",
                "Mobile-first design"
            ]
        }
        
        return solutions.get(theme, ["Custom solution development"])
    
    def _assess_barriers_to_entry(self, theme: str, market_context: str) -> List[str]:
        """Assess barriers to entry for a market gap."""
        common_barriers = [
            "Technical complexity",
            "Customer acquisition costs",
            "Integration requirements",
            "Regulatory compliance",
            "Brand recognition"
        ]
        
        theme_barriers = {
            "onboarding_complexity": ["User behavior change", "Integration complexity"],
            "integration_challenges": ["API limitations", "Legacy system compatibility"],
            "pricing_barriers": ["Competitive pricing pressure", "Cost structure optimization"],
            "performance_issues": ["Technical expertise required", "Infrastructure costs"],
            "usability_problems": ["Design expertise", "User research requirements"]
        }
        
        return common_barriers + theme_barriers.get(theme, [])
    
    def _identify_risk_factors(self, theme: str, market_context: str) -> List[str]:
        """Identify risk factors for a market gap."""
        common_risks = [
            "Market saturation",
            "Competitive response",
            "Technology changes",
            "Customer acquisition",
            "Scaling challenges"
        ]
        
        return common_risks
    
    def _identify_target_segments(self, theme: str, pain_points: List[Dict]) -> List[str]:
        """Identify target market segments."""
        segments = {
            "onboarding_complexity": ["SMBs", "Startups", "Non-technical users"],
            "integration_challenges": ["Mid-market companies", "Enterprises", "Tech teams"],
            "pricing_barriers": ["SMBs", "Startups", "Individual users"],
            "performance_issues": ["Enterprises", "High-volume users", "Global companies"],
            "usability_problems": ["All business sizes", "Non-technical users", "Mobile users"]
        }
        
        return segments.get(theme, ["General business users"])
    
    async def _analyze_competitive_landscape(self, market_gaps: List[MarketGap], 
                                           market_context: str) -> Dict[str, Any]:
        """Analyze the competitive landscape."""
        return {
            "total_gaps": len(market_gaps),
            "competitive_intensity": "medium",
            "market_maturity": "growing",
            "key_players": ["Established players", "Startups", "Tech giants"],
            "market_dynamics": ["Consolidation", "Innovation", "Disruption"],
            "opportunity_areas": [g.gap_description for g in market_gaps]
        }
    
    def _prioritize_opportunities(self, market_gaps: List[MarketGap]) -> List[Dict[str, Any]]:
        """Prioritize market opportunities by score."""
        opportunities = []
        
        for gap in market_gaps:
            opportunities.append({
                "gap_description": gap.gap_description,
                "opportunity_score": gap.opportunity_score,
                "market_size": gap.market_size,
                "competition_level": gap.competition_level,
                "priority": "high" if gap.opportunity_score > 70 else "medium" if gap.opportunity_score > 50 else "low"
            })
        
        # Sort by opportunity score
        return sorted(opportunities, key=lambda x: x["opportunity_score"], reverse=True)
    
    async def _generate_market_analysis(self, market_gaps: List[MarketGap], 
                                      competitive_landscape: Dict) -> Dict[str, Any]:
        """Generate comprehensive market analysis."""
        total_market_size = sum(gap.market_size for gap in market_gaps)
        avg_opportunity_score = sum(gap.opportunity_score for gap in market_gaps) / len(market_gaps)
        
        return {
            "total_addressable_market": total_market_size,
            "average_opportunity_score": avg_opportunity_score,
            "market_gaps_count": len(market_gaps),
            "market_segments": list(set(segment for gap in market_gaps for segment in gap.target_segments)),
            "key_trends": ["Digital transformation", "User experience focus", "Cost optimization"],
            "growth_drivers": ["Remote work", "SaaS adoption", "SMB digitization"]
        }
    
    def _assess_risks(self, market_gaps: List[MarketGap], 
                     competitive_landscape: Dict) -> Dict[str, Any]:
        """Assess risks for market opportunities."""
        return {
            "overall_risk_level": "medium",
            "market_risks": ["Competition", "Market saturation", "Economic downturn"],
            "technical_risks": ["Technology changes", "Scalability challenges"],
            "business_risks": ["Customer acquisition", "Pricing pressure", "Regulatory changes"],
            "mitigation_strategies": ["Focus on niche", "Build moat", "Customer validation"]
        }
    
    def _generate_recommendations(self, opportunities: List[Dict], 
                                risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if opportunities:
            top_opportunity = opportunities[0]
            recommendations.append(f"Focus on top opportunity: {top_opportunity['gap_description']}")
            recommendations.append(f"Target market size: ${top_opportunity['market_size']:,.0f}")
            
            if len(opportunities) > 3:
                recommendations.append("Consider building a platform addressing multiple gaps")
            
            recommendations.append("Validate with potential customers before full development")
            recommendations.append("Start with MVP targeting highest-scoring opportunity")
        
        return recommendations
    
    def _determine_analysis_approach(self, scope: str) -> str:
        """Determine analysis approach based on scope."""
        approaches = {
            "quick": "High-level analysis with key metrics",
            "focused": "Detailed analysis of top opportunities",
            "comprehensive": "Full market analysis with detailed modeling"
        }
        return approaches.get(scope, "focused")


# Register the agent - moved to agent_registry.py
# from .base import register_agent
# register_agent("gap_finder", GapFinderAgent)
