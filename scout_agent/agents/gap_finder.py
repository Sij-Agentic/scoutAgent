"""
GapFinderAgent - Market Gap Analysis Agent

This agent specializes in analyzing market gaps and opportunities
based on validated pain points and market research.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

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
    
    async def plan(self, input_data: Union[GapFinderInput, AgentInput], run_id: Optional[str] = None) -> Dict[str, Any]:
        """Plan the market gap analysis process."""
        
        # Store run_id in state if provided
        if run_id:
            self.state.run_id = run_id
        
        # Convert AgentInput to GapFinderInput if needed
        if isinstance(input_data, AgentInput):
            # Extract all needed fields from AgentInput
            ctx = input_data.context or {}
            data = input_data.data or {}
            validated_pain_points = []
            
            if isinstance(data, dict):
                validated_pain_points = data.get("validated_pain_points") or data.get("pain_points") or []
            elif isinstance(data, list):
                # Assume the list itself contains pain points
                validated_pain_points = data
            
            # Convert to proper GapFinderInput object for the rest of the method
            input_data = GapFinderInput(
                validated_pain_points=validated_pain_points,
                market_context=data.get("market_context", "") if isinstance(data, dict) else ctx.get("market_context", ""),
                analysis_scope=ctx.get("analysis_scope", "comprehensive"),
                include_competitive_analysis=bool(ctx.get("include_competitive_analysis", True)),
                include_market_sizing=bool(ctx.get("include_market_sizing", True)),
                context=ctx,
                metadata=input_data.metadata or {}
            )
            
            self.logger.info(f"Converted AgentInput to GapFinderInput with {len(validated_pain_points)} pain points")
            validated_pain_points = input_data.validated_pain_points
        else:
            # Already GapFinderInput
            validated_pain_points = input_data.validated_pain_points
            self.logger.info(f"Planning market gap analysis for {len(validated_pain_points)} pain points")
        
        # Step 1: Retrieve data from validator act stage if not provided directly
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
            # TEMPORARY: Limit pain points for discovery query generation during testing
            max_pain_points_for_discovery = 5  # MODERATE LIMITING: Increase from 2 to 5 for better data quality
            if len(validated_pain_points) > max_pain_points_for_discovery:
                self.logger.info(f"COST LIMITING: Reducing {len(validated_pain_points)} pain points to {max_pain_points_for_discovery} for discovery query generation")
                validated_pain_points_limited = validated_pain_points[:max_pain_points_for_discovery]
            else:
                validated_pain_points_limited = validated_pain_points

            for i, pain_point in enumerate(validated_pain_points_limited):
                self.logger.info(f"Generating discovery queries for pain point {i+1}/{len(validated_pain_points_limited)}")
                
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
                
                # Default number of queries per category (align with template example)
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
                
                # Generate discovery queries using LLM with JSON-optimized parameters
                discovery_response = await self.llm_generate(
                    prompt=discovery_prompt,
                    task_type="discovery",
                    temperature=0.1,  # Low temperature for structured JSON output
                    max_tokens=1000   # Sufficient tokens for 5-category JSON structure
                )
                
                # Extract JSON from LLM response
                self.logger.info(f"Discovery response for pain point {i+1} (length: {len(discovery_response)})")
                self.logger.debug(f"Raw discovery response: {discovery_response[:500]}...")
                
                queries = self._extract_json(discovery_response)
                
                if queries:
                    discovery_queries[pain_point_text] = queries
                    self.logger.info(f"Generated discovery queries for pain point {i+1} in {len(queries)} categories")
                else:
                    self.logger.warning(f"Failed to generate discovery queries for pain point {i+1}")
                    # Try to generate fallback queries based on pain point keywords
                    fallback_queries = self._generate_fallback_discovery_queries(pain_point_text)
                    if fallback_queries:
                        discovery_queries[pain_point_text] = fallback_queries
                        self.logger.info(f"Generated fallback discovery queries for pain point {i+1}")
                    else:
                        # Store the error information for debugging
                        discovery_queries[pain_point_text] = {
                            "error": f"Failed to parse JSON from LLM response",
                            "response_preview": discovery_response[:200] if discovery_response else "No response",
                            "response_length": len(discovery_response) if discovery_response else 0
                        }
        except Exception as e:
            self.logger.error(f"Error generating discovery queries: {str(e)}")
        
        # Add discovery queries to the plan
        if discovery_queries:
            plan["discovery_queries"] = discovery_queries
        
        # Step 4: Generate DAG metadata for gap finder stages (use limited set if applied)
        pain_points_for_dag = validated_pain_points_limited if 'validated_pain_points_limited' in locals() else validated_pain_points
        dag_metadata = self._generate_dag_metadata(pain_points_for_dag, discovery_queries)
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
        
        # Clean text first (remove common LLM response prefixes/suffixes)
        cleaned_text = text.strip()
        
        # Remove common LLM response prefixes
        prefixes_to_remove = [
            "Here's the JSON:", "Here is the JSON:", "JSON output:", "The JSON is:",
            "Based on the pain point, here are the discovery queries:",
            "```json", "```", "Here are the queries:", "The queries are:",
            "The discovery queries are:", "Discovery queries:"
        ]
        for prefix in prefixes_to_remove:
            if cleaned_text.lower().startswith(prefix.lower()):
                cleaned_text = cleaned_text[len(prefix):].strip()
        
        # Remove common suffixes
        suffixes_to_remove = ["```", "```json"]
        for suffix in suffixes_to_remove:
            if cleaned_text.lower().endswith(suffix.lower()):
                cleaned_text = cleaned_text[:-len(suffix)].strip()
        
        # Strategy 1: Direct JSON parsing (on cleaned text)
        try:
            result = json.loads(cleaned_text)
            self.logger.debug("Successfully parsed cleaned JSON directly")
            return result
        except json.JSONDecodeError as e:
            self.logger.debug(f"Direct JSON parsing failed on cleaned text: {e}")
            # Try original text as fallback
            try:
                result = json.loads(text.strip())
                self.logger.debug("Successfully parsed original JSON directly")
                return result
            except json.JSONDecodeError as e2:
                self.logger.debug(f"Direct JSON parsing failed on original text: {e2}")
        
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
                except json.JSONDecodeError as e:
                    self.logger.debug(f"JSON cleaning failed for pattern {pattern}: {e}")
                    self.logger.debug(f"Cleaned text that failed: {cleaned[:200]}...")
                    continue
        
        # Strategy 4: Try to extract key-value pairs and reconstruct JSON
        try:
            reconstructed = self._reconstruct_json_from_text(text)
            if reconstructed:
                self.logger.debug("Successfully reconstructed JSON from text patterns")
                return reconstructed
        except Exception as e:
            self.logger.debug(f"JSON reconstruction failed: {e}")
        
        self.logger.warning(f"Failed to extract valid JSON from text. Text preview: {text[:200]}...")
        self.logger.debug(f"Full text that failed JSON extraction: {text}")
        return {}
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean common JSON formatting issues."""
        import re
        
        # Remove extra whitespace but preserve structure
        cleaned = json_str.strip()
        
        # Fix trailing commas before } or ]
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        
        # Fix missing quotes around keys (common LLM mistake)
        cleaned = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
        
        # Fix single quotes to double quotes for keys
        cleaned = re.sub(r"'([^']*)'\s*:", r'"\1":', cleaned)
        
        # Fix single quotes to double quotes for values
        cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)
        
        # Fix array formatting (ensure proper spacing)
        cleaned = re.sub(r'\[\s*([^\[\]]*?)\s*\]', lambda m: '[' + ', '.join(f'"{item.strip()}"' if not item.strip().startswith('"') else item.strip() for item in m.group(1).split(',') if item.strip()) + ']', cleaned)
        
        # Ensure proper spacing around colons and commas
        cleaned = re.sub(r'\s*:\s*', ': ', cleaned)
        cleaned = re.sub(r'\s*,\s*', ', ', cleaned)
        
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
    
    def _generate_fallback_discovery_queries(self, pain_point_text: str) -> Dict[str, List[str]]:
        """Generate fallback discovery queries when LLM fails to produce valid JSON."""
        import re
        
        # Extract key terms from pain point
        words = re.findall(r'\b[a-zA-Z]{3,}\b', pain_point_text.lower())
        key_terms = [word for word in words if word not in {'that', 'this', 'with', 'from', 'have', 'they', 'their', 'them', 'than', 'when', 'where', 'what', 'which', 'would', 'could', 'should'}]
        
        if len(key_terms) < 2:
            return {}
        
        # Generate simple queries based on the pain point structure
        primary_terms = key_terms[:3]  # Take first 3 relevant terms
        
        return {
            "category_level": [
                f"{primary_terms[0]} {primary_terms[1]} tools",
                f"best {primary_terms[0]} solutions 2025", 
                f"AI {primary_terms[0]} SaaS"
            ],
            "problem_focused": [
                f"solutions for {primary_terms[0]} {primary_terms[1]}",
                f"tools to fix {primary_terms[0]}",
                f"how to solve {primary_terms[0]} problems"
            ],
            "alternatives_comparisons": [
                f"alternatives to {primary_terms[0]} tools",
                f"best {primary_terms[0]} comparison",
                f"top {primary_terms[0]} competitors"
            ],
            "review_sites": [
                f"site:g2.com {primary_terms[0]}",
                f"site:capterra.com {primary_terms[0]} {primary_terms[1]}",
                f"site:producthunt.com {primary_terms[0]}"
            ],
            "trend_recency": [
                f"new {primary_terms[0]} startups 2025",
                f"{primary_terms[0]} startup site:techcrunch.com",
                f"innovative {primary_terms[0]} solutions"
            ]
        }
    
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
                # Try multiple possible field names for pain point text
                pain_point_text = (
                    pain_point.get("pain_point", "") or 
                    pain_point.get("description", "") or
                    pain_point.get("text", "") or
                    pain_point.get("content", "")
                )
            elif isinstance(pain_point, str):
                pain_point_text = pain_point
            
            if not pain_point_text:
                continue
                
            # Get queries for this pain point
            queries_for_pain_point = discovery_queries.get(pain_point_text, {})
            all_queries = []
            
            # Check if we have valid query lists (not error objects)
            for category_name, category_queries in queries_for_pain_point.items():
                if isinstance(category_queries, list):
                    all_queries.extend(category_queries)
                elif isinstance(category_queries, dict) and "error" in category_queries:
                    # This category failed - we'll use fallback below
                    continue
            
            # If no valid queries found, use our sophisticated fallback
            if not all_queries:
                self.logger.info(f"Using fallback discovery queries for pain point: {pain_point_text[:50]}...")
                fallback_queries = self._generate_fallback_discovery_queries(pain_point_text)
                if fallback_queries:
                    # Use all queries from all categories
                    for category_queries in fallback_queries.values():
                        if isinstance(category_queries, list):
                            all_queries.extend(category_queries)
                    self.logger.info(f"Generated {len(all_queries)} fallback queries for pain point {i+1}")
                else:
                    # Final fallback if even our sophisticated fallback fails
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
                    self.logger.warning(f"Using basic keyword fallback for pain point {i+1}")
            else:
                self.logger.info(f"Using {len(all_queries)} LLM-generated discovery queries for pain point {i+1}")
                
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
                    "contents": f"${{{extract_node['output_manifest_key']}.content}}",  # Reference to extract output content
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
        for i, triage_node in enumerate(triage_content_nodes):
            extract_node = extract_content_nodes[i]  # Get corresponding extract_content node
            node_id = generate_node_id("identify_vendors", f"_pp{triage_node['config']['metadata']['pain_point_index']+1}")
            identify_vendors_nodes.append({
                "node_id": node_id,
                "name": f"identify_vendors_pp{triage_node['config']['metadata']['pain_point_index']+1}",
                "description": f"Identify vendors from triaged content for pain point {triage_node['config']['metadata']['pain_point_index']+1}",
                "node_type": "TOOL",
                "tool_name": "identify_vendors",
                "dependencies": [triage_node["node_id"], extract_node["node_id"]],  # Depend on both triage and extract
                "inputs": {
                    "contents": f"${{{extract_node['output_manifest_key']}.content}}",  # Reference to extract_content output with actual content
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
            
            # Add edges from both triage_content and extract_content to identify_vendors
            edges.append({
                "from_node": triage_node["node_id"],
                "to_node": node_id,
                "edge_type": "data_flow"
            })
            edges.append({
                "from_node": extract_node["node_id"],
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
                # Try multiple possible field names for pain point text
                pain_point_text = (
                    validated_pain_points[pain_point_index].get("pain_point", "") or 
                    validated_pain_points[pain_point_index].get("description", "") or
                    validated_pain_points[pain_point_index].get("text", "") or
                    validated_pain_points[pain_point_index].get("content", "")
                )
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
                    "vendors_list": f"${{{vendor_node['output_manifest_key']}.vendors}}",  # Reference to extracted vendors from successful results
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
                "research_outputs": [f"${{{node['output_manifest_key']}.research_results}}" for node in vendor_research_nodes],
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
    
    async def think(self, input_data: Union[GapFinderInput, AgentInput], run_id: str = None) -> Dict[str, Any]:
        """Analyze aggregate gap analysis data to prioritize opportunities and assess competitive positioning."""
        self.logger.info("Starting gap finder think stage...")
        
        try:
            # Convert AgentInput to GapFinderInput if needed
            if hasattr(input_data, 'data') and hasattr(input_data, 'context'):
                # This is an AgentInput, convert to GapFinderInput
                ctx = input_data.context or {}
                data = input_data.data or {}
                validated = []
                market_ctx = ""
                if isinstance(data, dict):
                    validated = data.get("validated_pain_points") or data.get("pain_points") or []
                    market_ctx = data.get("market_context") or ctx.get("market_context", "")
                
                # Debug logging
                self.logger.info(f"AgentInput data keys: {list(data.keys()) if isinstance(data, dict) else 'not_dict'}")
                self.logger.info(f"AgentInput context keys: {list(ctx.keys())}")
                self.logger.info(f"Found validated pain points: {len(validated)}")
                
                # If no validated pain points found, try to get from agent state
                if not validated:
                    validated = getattr(self.state, 'validated_pain_points', [])
                    self.logger.info(f"Got {len(validated)} pain points from agent state")
                
                # If still no pain points, create a fallback
                if not validated:
                    validated = [{"pain_point": "Market analysis for software testing tools", "description": "General market analysis"}]
                    self.logger.warning("No pain points found, using fallback")
                
                gap_input = GapFinderInput(
                    validated_pain_points=validated,
                    market_context=market_ctx or "Software testing and quality assurance tools market",
                    analysis_scope=ctx.get("analysis_scope", "comprehensive"),
                    include_competitive_analysis=bool(ctx.get("include_competitive_analysis", True)),
                    include_market_sizing=bool(ctx.get("include_market_sizing", True)),
                    context=input_data.context,
                    metadata=input_data.metadata,
                )
            else:
                # This is already a GapFinderInput
                gap_input = input_data
            
            # Get run_id from parameter or agent state
            current_run_id = run_id or getattr(self.state, 'run_id', None)
            if not current_run_id:
                raise Exception("No run_id provided for think stage")
            
            # Get manifest manager from agent state or create one
            manifest_manager = getattr(self.state, 'manifest_manager', None)
            if not manifest_manager:
                manifest_path = Path("data/runs") / current_run_id / "run_manifest.json"
                manifest_manager = ManifestManager(manifest_path=manifest_path)
            
            manifest = manifest_manager.get_manifest()
            
            # Load aggregate gap analysis from collect stage
            aggregate_data = self._load_aggregate_data(manifest)
            if not aggregate_data:
                raise Exception("No aggregate gap analysis data found from collect stage")
            
            # Load essential data for think stage (minimal, focused)
            essential_data = self._load_essential_think_data(manifest)
            
            # Prepare synthesis data
            synthesis_data = {
                "aggregate_gap_analysis": aggregate_data,
                "essential_data": essential_data,
                "analysis_context": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "market_context": gap_input.market_context,
                    "analysis_scope": gap_input.analysis_scope
                }
            }
            
            # Load and process the think prompt
            prompt_content = self._load_think_prompt()
            
            # Append the synthesis data to the prompt
            prompt_content += f"\n\nResearch Data to Analyze:\n{json.dumps(synthesis_data, indent=2)}"
            
            # Generate analysis using LLM mixin
            analysis_result = await self.llm_generate(prompt=prompt_content, task_type="think_analysis")
            
            # Parse the response as JSON
            if isinstance(analysis_result, str):
                self.logger.info(f"LLM response type: {type(analysis_result)}, length: {len(analysis_result)}")
                self.logger.info(f"LLM response preview: {analysis_result[:200]}...")
                # Parse structured text response instead of JSON
                analysis_result = self._parse_structured_text(analysis_result)
                self.logger.info(f"Successfully parsed LLM response, type: {type(analysis_result)}")
                
                # Check if we got an error from _parse_structured_text
                if isinstance(analysis_result, dict) and "error" in analysis_result:
                    self.logger.error(f"Structured text parsing failed: {analysis_result['error']}")
                    # Create a fallback result structure
                    analysis_result = self._create_fallback_think_result(gap_input)
                    self.logger.info(f"Created fallback result, type: {type(analysis_result)}")
            else:
                self.logger.info(f"LLM response is not a string, type: {type(analysis_result)}")
            
            # Store the result to manifest
            self._store_think_output_to_manifest(manifest_manager, analysis_result)
            
            self.logger.info("Think stage completed successfully")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in think stage: {e}")
            raise e
    
    async def act(self, input_data: Union[GapFinderInput, AgentInput], plan: Optional[Dict[str, Any]] = None, thoughts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the act stage - generate SaaS business recommendations."""
        try:
            self.logger.info("Starting gap finder act stage...")
            self.logger.info(f"Input data type: {type(input_data)}")
            self.logger.info(f"Input data: {input_data}")
            
            # Convert AgentInput to GapFinderInput if needed
            if not isinstance(input_data, GapFinderInput):
                # Handle case where input_data.data might be a list or dict
                if isinstance(input_data.data, list):
                    # If data is a list, treat it as validated_pain_points
                    validated_pain_points = input_data.data
                    market_context = ""
                    analysis_scope = "focused"
                elif isinstance(input_data.data, dict):
                    # If data is a dict, extract fields normally
                    validated_pain_points = input_data.data.get("validated_pain_points", [])
                    market_context = input_data.data.get("market_context", "")
                    analysis_scope = input_data.data.get("analysis_scope", "focused")
                else:
                    # Fallback
                    validated_pain_points = []
                    market_context = ""
                    analysis_scope = "focused"
                
                gap_input = GapFinderInput(
                    validated_pain_points=validated_pain_points,
                    market_context=market_context,
                    analysis_scope=analysis_scope
                )
            else:
                gap_input = input_data
            
            # Ensure we have pain points
            if not gap_input.validated_pain_points:
                # Try to get from agent state
                pain_points = getattr(self.state, 'validated_pain_points', [])
                if not pain_points:
                    self.logger.warning("No pain points found, using fallback")
                    pain_points = [{"id": "pp1", "title": "Software Testing Tools", "description": "Testing tools market analysis"}]
                # Create a new GapFinderInput with the pain points
                gap_input = GapFinderInput(
                    validated_pain_points=pain_points,
                    market_context=gap_input.market_context,
                    analysis_scope=gap_input.analysis_scope
                )
            
            # Initialize manifest manager
            run_id = getattr(self.state, 'run_id', None)
            if not run_id:
                self.logger.error("No run_id found in agent state")
                return {"error": "No run_id available"}
            
            manifest_path = Path("data/runs") / run_id / "run_manifest.json"
            manifest_manager = ManifestManager(manifest_path=manifest_path)
            
            # Load think stage data
            think_data = self._load_think_stage_data(manifest_manager)
            if not think_data:
                self.logger.error("No think stage data found")
                return {"error": "No think stage data available"}
            
            # Load essential data for context
            essential_data = self._load_essential_act_data(manifest_manager)
            
            # Prepare synthesis data
            synthesis_data = {
                "think_stage_output": think_data,
                "essential_data": essential_data,
                "market_context": gap_input.market_context,
                "analysis_scope": gap_input.analysis_scope
            }
            
            # Load act stage prompt
            prompt_content = self._load_act_prompt()
            if not prompt_content:
                self.logger.error("Failed to load act stage prompt")
                return {"error": "Failed to load act stage prompt"}
            
            # Append the synthesis data to the prompt
            prompt_content += f"\n\nThink Stage Data to Analyze:\n{json.dumps(synthesis_data, indent=2)}"
            
            # Generate SaaS recommendations using LLM mixin
            recommendations_result = await self.llm_generate(prompt=prompt_content, task_type="act_analysis")
            
            # Parse the response as structured text
            if isinstance(recommendations_result, str):
                self.logger.info(f"LLM response type: {type(recommendations_result)}, length: {len(recommendations_result)}")
                self.logger.info(f"LLM response preview: {recommendations_result[:200]}...")
                # Parse structured text response instead of JSON
                recommendations_result = self._parse_structured_act_text(recommendations_result)
                self.logger.info(f"Successfully parsed LLM response, type: {type(recommendations_result)}")
                
                # Check if we got an error from _parse_structured_act_text
                if isinstance(recommendations_result, dict) and "error" in recommendations_result:
                    self.logger.error(f"Structured text parsing failed: {recommendations_result['error']}")
                    # Create a fallback result structure
                    recommendations_result = self._create_fallback_act_result(gap_input)
                    self.logger.info(f"Created fallback result, type: {type(recommendations_result)}")
            else:
                self.logger.info(f"LLM response is not a string, type: {type(recommendations_result)}")
            
            # Store the result to manifest
            self._store_act_output_to_manifest(manifest_manager, recommendations_result)
            
            self.logger.info("Act stage completed successfully")
            return recommendations_result
            
        except Exception as e:
            self.logger.error(f"Error in act stage: {e}")
            raise e
    
    def _store_think_output_to_manifest(self, manifest_manager: ManifestManager, analysis_result: Dict[str, Any]) -> None:
        """Store think stage output to manifest."""
        try:
            # Get the manifest
            manifest = manifest_manager.get_manifest()
            
            # Ensure stages section exists
            if "stages" not in manifest:
                manifest["stages"] = {}
            
            # Store the think stage output
            manifest["stages"]["gap_finder_think"] = {
                "status": "completed",
                "updated_at": datetime.now().isoformat(),
                "data": analysis_result
            }
            
            # Save the manifest
            manifest_manager._save()
            self.logger.info("Stored think stage output to manifest")
            
        except Exception as e:
            self.logger.error(f"Error storing think output to manifest: {e}")
            raise e
    
    def _load_aggregate_data(self, manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load aggregate gap analysis data from collect stage output."""
        try:
            # Try to get from gap_finder_final_output in tool_results
            collect_stage = manifest.get("stages", {}).get("gap_finder_collect", {})
            tool_results = collect_stage.get("data", {}).get("tool_results", {})
            aggregate_output = tool_results.get("gap_finder_final_output")
            
            if aggregate_output:
                self.logger.info("Found aggregate data in gap_finder_collect tool_results")
                return aggregate_output
            
            # Fallback: try outputs section
            outputs = manifest.get("outputs", {})
            aggregate_output = outputs.get("gap_finder_final_output")
            
            if aggregate_output:
                self.logger.info("Found aggregate data in outputs section")
                return aggregate_output.get("data") if isinstance(aggregate_output, dict) else aggregate_output
            
            self.logger.warning("No aggregate gap analysis data found")
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading aggregate data: {e}")
            return None
    
    def _load_essential_think_data(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Load only essential data for think stage - minimal and focused."""
        essential_data = {}
        
        try:
            collect_stage = manifest.get("stages", {}).get("gap_finder_collect", {})
            tool_results = collect_stage.get("data", {}).get("tool_results", {})
            
            # Only extract URLs from search_links (not full content)
            for key, value in tool_results.items():
                if key.startswith("search_links_pp") and key.endswith("_output"):
                    urls = self._extract_urls_from_search_links(value)
                    if urls:
                        essential_data["search_urls"] = urls
                        self.logger.info(f"Extracted {len(urls)} URLs from {key}")
            
            self.logger.info(f"Loaded essential data: {list(essential_data.keys())}")
            return essential_data
            
        except Exception as e:
            self.logger.error(f"Error loading essential think data: {e}")
            return {}
    
    def _extract_urls_from_search_links(self, search_links_data: Any) -> List[Dict[str, str]]:
        """Extract just URLs and titles from search_links data."""
        urls = []
        
        try:
            # Parse MCP response format
            if isinstance(search_links_data, dict) and "content" in search_links_data:
                for content_item in search_links_data["content"]:
                    if isinstance(content_item, dict) and "text" in content_item:
                        try:
                            search_data = json.loads(content_item["text"])
                            if "query_results" in search_data:
                                for query_result in search_data["query_results"]:
                                    if "results" in query_result:
                                        for result in query_result["results"]:
                                            if "link" in result and "title" in result:
                                                urls.append({
                                                    "url": result["link"],
                                                    "title": result["title"],
                                                    "query": query_result.get("query", "")
                                                })
                        except json.JSONDecodeError:
                            continue
            
            return urls
            
        except Exception as e:
            self.logger.error(f"Error extracting URLs from search_links: {e}")
            return []
    
    def _load_think_prompt(self) -> str:
        """Load the think stage prompt template."""
        try:
            prompt_path = Path(__file__).parent.parent / "prompts" / "gap_finder_agent" / "think.prompt"
            
            if not prompt_path.exists():
                raise FileNotFoundError(f"Think prompt not found at {prompt_path}")
            
            with open(prompt_path, 'r') as f:
                prompt_content = f.read()
            
            self.logger.info("Loaded think stage prompt template")
            return prompt_content
            
        except Exception as e:
            self.logger.error(f"Error loading think prompt: {e}")
            raise e
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text with robust error handling."""
        import re
        import json
        
        # Default fallback result - we'll create it when needed
        default_result = None
        
        if not text or not isinstance(text, str):
            self.logger.error("Empty or non-string response from LLM")
            return {"error": "Empty or non-string response from LLM"}
        
        # Try to find JSON in markdown code blocks
        code_block_pattern = r'```(?:json)?\s*(.+?)\s*```'
        matches = re.search(code_block_pattern, text, re.DOTALL)
        
        if matches:
            self.logger.info("Found JSON in code block")
            json_str = matches.group(1)
        else:
            # Try to find JSON without code blocks
            self.logger.info("No code block found, trying to parse entire response as JSON")
            json_str = text.strip()
            
            # Try to find JSON object within the text using regex
            json_pattern = r'\{[\s\S]*\}'
            json_matches = re.search(json_pattern, text, re.DOTALL)
            if json_matches:
                self.logger.info("Found JSON-like structure in response")
                json_str = json_matches.group(0)
            else:
                # As a final attempt, slice from first { to last }
                first_idx = text.find('{')
                last_idx = text.rfind('}')
                if first_idx != -1 and last_idx != -1 and last_idx > first_idx:
                    json_str = text[first_idx:last_idx+1]
        
        # Clean the JSON string with comprehensive comma fixing
        json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)  # Remove trailing commas
        
        # Fix missing commas between object properties (comprehensive)
        json_str = re.sub(r'"\s*\n\s*"([^"]*)"\s*:', '",\n"\1":', json_str)
        json_str = re.sub(r'"\s*"([^"]*)"\s*:', '", "\1":', json_str)
        json_str = re.sub(r'}\s*\n\s*"', '},\n"', json_str)
        json_str = re.sub(r']\s*\n\s*"', '],\n"', json_str)
        json_str = re.sub(r'}\s*"', '}, "', json_str)
        json_str = re.sub(r']\s*"', '], "', json_str)
        
        # Fix missing commas between array elements
        json_str = re.sub(r'"\s*\n\s*"([^"]*)"\s*"', '",\n"\1", "', json_str)
        json_str = re.sub(r'"\s*"([^"]*)"\s*"', '", "\1", "', json_str)
        
        # Fix missing commas after values before closing braces/brackets
        json_str = re.sub(r'([^,}\]])\s*(\n\s*[}\]])', r'\1,\2', json_str)
        json_str = re.sub(r'([^,}\]])\s*([}\]])', r'\1,\2', json_str)
        
        # Fix missing commas after strings before other properties
        json_str = re.sub(r'"\s*\n\s*([a-zA-Z_][a-zA-Z0-9_]*\s*:)', '",\n\1', json_str)
        json_str = re.sub(r'"\s*([a-zA-Z_][a-zA-Z0-9_]*\s*:)', '", \1', json_str)
        
        try:
            parsed_json = json.loads(json_str)
            self.logger.info("Successfully parsed JSON")
            return parsed_json
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            try:
                # Try again with more aggressive cleaning
                self.logger.info("Attempting more aggressive JSON cleaning")
                json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)  # Add quotes to keys
                
                parsed_json = json.loads(json_str)
                self.logger.info("Successfully parsed JSON after aggressive cleaning")
                return parsed_json
            except Exception as e:
                self.logger.error(f"Failed to parse JSON after cleaning: {e}")
                return {"error": f"Failed to parse JSON after cleaning: {e}"}
    
    def _fix_json_syntax(self, json_str: str) -> str:
        """Attempt to fix common JSON syntax issues."""
        try:
            import re
            
            # Remove any trailing commas before closing braces/brackets
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            # Fix missing commas between array/object elements
            # Look for patterns like "value" "key" or } "key" or ] "key"
            json_str = re.sub(r'(["\}\]])"([^"]*)"\s*:', r'\1, "\2":', json_str)
            json_str = re.sub(r'(["\}\]])"([^"]*)"\s*"', r'\1, "\2", "', json_str)
            
            # Fix missing commas between object properties (enhanced)
            json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
            json_str = re.sub(r'}\s*\n\s*"', '},\n"', json_str)
            json_str = re.sub(r']\s*\n\s*"', '],\n"', json_str)
            
            # Additional comma fixes for common patterns
            json_str = re.sub(r'"\s*"([^"]*)"\s*:', '", "\1":', json_str)
            json_str = re.sub(r'}\s*"', '}, "', json_str)
            json_str = re.sub(r']\s*"', '], "', json_str)
            
            # Fix missing commas after values before closing braces
            json_str = re.sub(r'([^,}\]])\s*(\n\s*[}\]])', r'\1,\2', json_str)
            
            # Fix unterminated strings by adding quotes at the end
            json_str = re.sub(r'([^"])\s*$', r'\1"', json_str.strip())
            
            # Fix missing quotes around unquoted keys (but be careful not to quote values)
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            
            # Fix single quotes to double quotes
            json_str = json_str.replace("'", '"')
            
            # Fix common issues with numbers and booleans
            json_str = re.sub(r':\s*true\s*([^,}\]])(?=\s*[}\]])', r': true, \1', json_str)
            json_str = re.sub(r':\s*false\s*([^,}\]])(?=\s*[}\]])', r': false, \1', json_str)
            json_str = re.sub(r':\s*null\s*([^,}\]])(?=\s*[}\]])', r': null, \1', json_str)
            
            self.logger.info("Applied JSON syntax fixes")
            return json_str
            
        except Exception as e:
            self.logger.error(f"Failed to fix JSON syntax: {e}")
            raise Exception(f"Could not fix malformed JSON: {e}")
    
    def _create_fallback_think_result(self, gap_input: GapFinderInput) -> Dict[str, Any]:
        """Create a fallback think result when LLM parsing fails."""
        return {
            "summary": {
                "analysis_objective": "Strategic market analysis",
                "methodology": "LLM analysis with fallback structure",
                "total_vendors_analyzed": 0,
                "total_gaps_identified": 0,
                "analysis_confidence": "low",
                "key_findings": ["Analysis completed with fallback structure due to JSON parsing issues"],
                "analysis_timestamp": datetime.now().isoformat()
            },
            "prioritized_gaps": [],
            "vendor_landscape": [],
            "whitespace_opportunities": [],
            "strategic_insights": {
                "market_trends": [],
                "emerging_patterns": [],
                "competitive_dynamics": "Analysis incomplete due to technical issues",
                "barriers_to_entry": [],
                "success_factors": []
            },
            "risks_and_unknowns": [
                {
                    "risk_id": "json_parsing_failure",
                    "description": "LLM response could not be parsed as valid JSON",
                    "impact": "high",
                    "likelihood": "low",
                    "mitigation_strategy": "Review and fix JSON formatting in LLM response",
                    "data_needed": ["Valid JSON response from LLM"]
                }
            ],
            "next_actions_recommendation": [
                {
                    "action_id": "retry_analysis",
                    "action": "Retry think stage analysis with improved JSON formatting",
                    "priority": "high",
                    "owner": "system",
                    "expected_outcome": "Valid JSON analysis result",
                    "blocking_inputs": ["Fixed LLM response formatting"],
                    "timeline": "immediate"
                }
            ]
        }
    

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
            
            # NOTE: Removed outputs section save - gap_finder_final_output is now stored
            # directly in gap_finder_collect.data.tool_results via nuclear approach
            
            self.logger.info("Gap finder collect stage completed successfully")
            return final_output
            
        except Exception as e:
            self.logger.error(f"Error in gap_finder collect stage: {str(e)}")
            raise
    
    def _store_tool_output_in_collect_stage(self, manifest_manager, output_key: str, result_data: Dict[str, Any]):
        """Store tool output within the gap_finder_collect stage data structure."""
        try:
            # Get current manifest
            manifest = manifest_manager.get_manifest()
            
            # Ensure gap_finder_collect stage exists
            if "stages" not in manifest:
                manifest["stages"] = {}
            if "gap_finder_collect" not in manifest["stages"]:
                manifest["stages"]["gap_finder_collect"] = {
                    "data": {},
                    "updated_at": datetime.now().isoformat(),
                    "status": "in_progress"
                }
            
            # Store the tool output within gap_finder_collect data
            collect_data = manifest["stages"]["gap_finder_collect"]["data"]
            if "tool_results" not in collect_data:
                collect_data["tool_results"] = {}
            
            collect_data["tool_results"][output_key] = result_data
            manifest["stages"]["gap_finder_collect"]["updated_at"] = datetime.now().isoformat()
            
            # Save the updated manifest
            manifest_manager._save()
            self.logger.info(f"Stored tool output {output_key} within gap_finder_collect stage")
            
        except Exception as e:
            self.logger.error(f"Error storing tool output in collect stage: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
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
        # FIXED: Check gap_finder dag_metadata location first, then fallback to scout locations
        dag_nodes = []
        
        # Location 1: Check gap_finder dag_metadata first (where gap_finder nodes are stored)
        if "stages" in plan and "gap_finder_plan" in plan["stages"]:
            gap_finder_data = plan.get("stages", {}).get("gap_finder_plan", {}).get("data", {})
            if "dag_metadata" in gap_finder_data:
                dag_nodes = gap_finder_data.get("dag_metadata", {}).get("nodes", [])
                self.logger.info(f"Found {len(dag_nodes)} nodes in stages.gap_finder_plan.data.dag_metadata.nodes")
        
        # Location 1b: Also check gap_finder_plan.data.dag.nodes (alternative structure)
        if not dag_nodes and "stages" in plan and "gap_finder_plan" in plan["stages"]:
            gap_finder_data = plan.get("stages", {}).get("gap_finder_plan", {}).get("data", {})
            if "dag" in gap_finder_data:
                dag_nodes = gap_finder_data.get("dag", {}).get("nodes", [])
                self.logger.info(f"Found {len(dag_nodes)} nodes in stages.gap_finder_plan.data.dag.nodes")
        
        # Location 2: Check top-level dag_metadata (fallback)
        if not dag_nodes and "dag_metadata" in plan:
            dag_nodes = plan.get("dag_metadata", {}).get("nodes", [])
            self.logger.info(f"Found {len(dag_nodes)} nodes in dag_metadata.nodes")
        
        # Location 3: Fallback to scout_plan location (for backward compatibility)
        if not dag_nodes and "stages" in plan and "scout_plan" in plan["stages"]:
            dag_nodes = plan.get("stages", {}).get("scout_plan", {}).get("data", {}).get("dag", {}).get("nodes", [])
            self.logger.info(f"Found {len(dag_nodes)} nodes in stages.scout_plan.data.dag.nodes")
        
        # Location 4: Check top-level dag
        if not dag_nodes and "dag" in plan:
            dag_nodes = plan.get("dag", {}).get("nodes", [])
            self.logger.info(f"Found {len(dag_nodes)} nodes in dag.nodes")
        
        # Filter tool nodes (both TOOL and FUNCTION node types with tool_name)
        tool_nodes = [node for node in dag_nodes if "tool_name" in node]
        self.logger.info(f"Found {len(tool_nodes)} tool nodes to execute (including FUNCTION nodes)")
        
        # Log node types for debugging
        for node in tool_nodes:
            self.logger.info(f"  Node {node.get('node_id')}: type={node.get('node_type')}, tool={node.get('tool_name')}")
        
        if not tool_nodes:
            self.logger.info("No tool nodes found to execute")
            return
        
        # Initialize MCP client with enhanced connection parameters
        try:
            configs = load_server_configs()
            multi_client = MultiMCPClient(
                configs, 
                max_retries=5,  # Increased retries for better resilience
                connection_timeout=600,  # 300 second timeout for connections
                sse_read_timeout=3600  # 1 hour timeout for vendor research operations
            )
            await multi_client.initialize()
            self.logger.info("MCP client initialized successfully with enhanced connection parameters")
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP client: {e}")
            return
        
        # Execute each tool node in dependency order
        # Use the same manifest path construction as in other methods
        project_root = Path(__file__).resolve().parents[2]
        current_run_id = run_id or getattr(self.state, "run_id", "latest")
        run_dir = project_root / "data" / "runs" / current_run_id
        manifest_path = run_dir / "run_manifest.json"
        
        manifest_manager = ManifestManager(manifest_path, create_if_missing=False)
        
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
                
                # Make direct MCP tool call with timeout and retry logic
                result = await self._execute_tool_with_retry(multi_client, tool_name, resolved_inputs, node_id)
                
                # Process and store result
                if result:
                    # Extract content from MCP response
                    if hasattr(result, 'content') and result.content:
                        content_item = result.content[0] if isinstance(result.content, list) else result.content
                        if hasattr(content_item, 'text'):
                            # Check if this is an error response from the tool
                            if content_item.text.startswith("Error executing tool"):
                                self.logger.error(f"Tool {tool_name} returned error: {content_item.text}")
                                # Don't store error responses as successful results
                                continue
                            
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
                    if tool_name == "aggregate_gap_analysis" or output_key == "gap_finder_final_output":
                        # Store the aggregate output directly with no post-processing
                        self._store_tool_output_in_collect_stage(manifest_manager, output_key, result_data)
                        self.logger.info(f"Stored aggregate output for node: {node_id} (key={output_key})")
                    else:
                        # Store other tool outputs within gap_finder_collect stage context
                        self._store_tool_output_in_collect_stage(manifest_manager, output_key, result_data)
                        self.logger.info(f"Stored results for node: {node_id} within collect stage (key={output_key})")
                    
                else:
                    self.logger.error(f"Tool node {node_id} returned no result")
                    
            except Exception as e:
                self.logger.error(f"Error executing tool node {node.get('node_id', 'unknown')}: {str(e)}")
                # Store error information in manifest for debugging
                try:
                    error_data = {
                        "node_id": node.get('node_id', 'unknown'),
                        "tool_name": tool_name,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "timestamp": datetime.now().isoformat(),
                        "inputs": resolved_inputs if 'resolved_inputs' in locals() else inputs
                    }
                    manifest_manager.store_node_result(node.get('node_id', 'unknown'), {"error": error_data})
                    self.logger.info(f"Stored error data for node {node.get('node_id', 'unknown')} in manifest")
                except Exception as store_error:
                    self.logger.error(f"Failed to store error data for {node.get('node_id', 'unknown')}: {store_error}")
                # Continue with other nodes even if one fails
                continue
        
        # Cleanup MCP client
        try:
            await multi_client.shutdown()
            self.logger.info("MCP client shutdown successfully")
        except Exception as e:
            self.logger.error(f"Error shutting down MCP client: {e}")
    
    async def _execute_tool_with_retry(self, multi_client, tool_name: str, inputs: Dict[str, Any], node_id: str, max_retries: int = 5, timeout_seconds: int = 900):
        """Execute tool with timeout and retry logic."""
        import asyncio
        from httpx import ReadTimeout, ConnectTimeout
        
        # Use longer timeout for vendor research operations
        if "vendor_research" in tool_name.lower():
            timeout_seconds = 7200  # 2 hours for vendor research (increased for production)
            self.logger.info(f"Using extended timeout of {timeout_seconds}s for vendor research tool: {tool_name}")
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Executing {tool_name} for node {node_id} (attempt {attempt + 1}/{max_retries})")
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    multi_client.call_tool(tool_name, inputs),
                    timeout=timeout_seconds
                )
                
                self.logger.info(f"Successfully executed {tool_name} for node {node_id} on attempt {attempt + 1}")
                return result
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout executing {tool_name} for node {node_id} (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    raise Exception(f"Tool {tool_name} timed out after {max_retries} attempts")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except (ReadTimeout, ConnectTimeout) as e:
                self.logger.warning(f"Connection timeout executing {tool_name} for node {node_id} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Tool {tool_name} connection failed after {max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                if "Connection closed" in str(e) or "connection" in str(e).lower():
                    self.logger.warning(f"Connection error executing {tool_name} for node {node_id} (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        raise Exception(f"Tool {tool_name} connection failed after {max_retries} attempts: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Non-retryable error, re-raise immediately
                    self.logger.error(f"Non-retryable error executing {tool_name} for node {node_id}: {e}")
                    raise
    
    async def _aggregate_collect_results(self, plan: Dict[str, Any], run_id: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate results from executed tool nodes and generate final collect output."""
        self.logger.info("Aggregating collect stage results...")
        
        # Use the same manifest path construction as in other methods
        project_root = Path(__file__).resolve().parents[2]
        current_run_id = run_id or getattr(self.state, "run_id", "latest")
        run_dir = project_root / "data" / "runs" / current_run_id
        manifest_path = run_dir / "run_manifest.json"
        
        manifest_manager = ManifestManager(manifest_path, create_if_missing=False)
        
        # Get all tool node results from manifest
        # Try different possible locations for DAG nodes
        dag_nodes = []
        
        # Location 1: Check gap_finder dag_metadata first (where gap_finder nodes are stored)
        if "stages" in plan and "gap_finder_plan" in plan["stages"]:
            gap_finder_data = plan.get("stages", {}).get("gap_finder_plan", {}).get("data", {})
            if "dag_metadata" in gap_finder_data:
                dag_nodes = gap_finder_data.get("dag_metadata", {}).get("nodes", [])
                self.logger.info(f"Found {len(dag_nodes)} nodes in stages.gap_finder_plan.data.dag_metadata.nodes")
        
        # Location 2: Fallback to direct dag_metadata (legacy support)
        if not dag_nodes:
            dag_nodes = plan.get("dag_metadata", {}).get("nodes", [])
            if dag_nodes:
                self.logger.info(f"Found {len(dag_nodes)} nodes in direct dag_metadata.nodes")
        
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
        
        # ENHANCEMENT: Direct manifest scanning for vendor research outputs
        # This handles cases where vendor research nodes aren't in DAG metadata
        try:
            manifest = manifest_manager.get_manifest()
            stages = manifest.get("stages", {})
            
            # Look for vendor research outputs using known patterns
            vendor_research_patterns = [
                "vendor_research_pp1_output",
                "vendor_research_pp2_output", 
                "vendor_research_pp3_output",
                "vendor_research_pp4_output",
                "vendor_research_pp5_output"
            ]
            
            for pattern in vendor_research_patterns:
                if pattern in stages:
                    stage_data = stages[pattern]
                    if isinstance(stage_data, dict) and "data" in stage_data:
                        vendor_result = stage_data["data"]
                        
                        # Create a synthetic node ID for this result
                        synthetic_node_id = pattern.replace("_output", "")
                        
                        # Only add if not already present from DAG processing
                        if synthetic_node_id not in aggregated_data["vendor_analysis"]:
                            aggregated_data["vendor_analysis"][synthetic_node_id] = vendor_result
                            self.logger.info(f"Retrieved vendor research via direct manifest scan: {pattern}")
                            
                            # Update execution metadata
                            aggregated_data["execution_metadata"]["total_nodes_executed"] += 1
            
            # Also check for any other vendor-related stages
            for stage_name, stage_data in stages.items():
                if ("vendor" in stage_name.lower() and 
                    stage_name not in vendor_research_patterns and
                    isinstance(stage_data, dict) and "data" in stage_data):
                    
                    vendor_result = stage_data["data"]
                    synthetic_node_id = stage_name.replace("_output", "")
                    
                    if synthetic_node_id not in aggregated_data["vendor_analysis"]:
                        aggregated_data["vendor_analysis"][synthetic_node_id] = vendor_result
                        self.logger.info(f"Retrieved additional vendor data via manifest scan: {stage_name}")
                        aggregated_data["execution_metadata"]["total_nodes_executed"] += 1
                        
        except Exception as e:
            self.logger.error(f"Error during direct manifest scanning: {str(e)}")
        
        # NOTE: Removed reading from outputs['gap_finder_final_output'] to avoid stale data influence
        
        # Generate summary insights
        self.logger.info(f"DEBUG: Before _generate_collect_summary, aggregated_data keys: {list(aggregated_data.keys())}")
        if "summary" in aggregated_data:
            self.logger.info(f"DEBUG: Existing summary in aggregated_data: {aggregated_data['summary']}")
        aggregated_data["summary"] = self._generate_collect_summary(aggregated_data)
        self.logger.info(f"DEBUG: After _generate_collect_summary: {aggregated_data['summary']}")
        
        return aggregated_data
    
    def _generate_collect_summary(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary insights from collected data."""
        # Start with basic collection metrics
        summary = {
            "data_sources_collected": len([k for k in aggregated_data.keys() if k != "execution_metadata" and k != "summary"]),
            "market_research_sources": len(aggregated_data.get("market_research_data", {})),
            "competitive_analysis_sources": len(aggregated_data.get("competitive_analysis", {})),
            "market_sizing_sources": len(aggregated_data.get("market_sizing_data", {})),
            "vendor_analysis_sources": len(aggregated_data.get("vendor_analysis", {})),
            "collection_status": "completed",
            "next_stage_ready": True
        }
        
        # Preserve vendor data from aggregate results if available
        if "summary" in aggregated_data:
            existing_summary = aggregated_data["summary"]
            if isinstance(existing_summary, dict):
                # Preserve vendor-related fields from aggregate function
                vendor_fields = ["total_vendors_found", "unique_vendors", "high_opportunity_gaps", 
                               "competitive_intensity", "total_vendors_analyzed"]
                for field in vendor_fields:
                    if field in existing_summary:
                        summary[field] = existing_summary[field]
                        self.logger.info(f"Preserved {field}: {existing_summary[field]} from aggregate results")
        
        # Include vendor data from aggregate summary if available
        if "aggregate_summary" in aggregated_data:
            aggregate_summary = aggregated_data["aggregate_summary"]
            if isinstance(aggregate_summary, dict):
                # Include vendor-related fields from aggregate summary - OVERRIDE calculated values
                vendor_fields = ["total_vendors_found", "unique_vendors", "high_opportunity_gaps", 
                               "competitive_intensity", "total_vendors_analyzed"]
                for field in vendor_fields:
                    if field in aggregate_summary:
                        summary[field] = aggregate_summary[field]
                        self.logger.info(f"OVERRIDING {field}: {aggregate_summary[field]} from aggregate summary")
                        
                # Special priority for total_vendors_found from aggregate results
                if "total_vendors_found" in aggregate_summary and aggregate_summary["total_vendors_found"] > 0:
                    summary["total_vendors_found"] = aggregate_summary["total_vendors_found"]
                    self.logger.info(f"CRITICAL: Using aggregate total_vendors_found: {aggregate_summary['total_vendors_found']} (overriding calculated value)")
        
        return summary
    
    def _resolve_template_variables(self, inputs: Dict[str, Any], manifest_manager: ManifestManager) -> Dict[str, Any]:
        """Resolve ${...} template variables in tool inputs using manifest data."""
        import re
        
        resolved_inputs = {}
        
        for key, value in inputs.items():
            if isinstance(value, list):
                # Handle lists that may contain template strings
                resolved_list = []
                for item in value:
                    if isinstance(item, str) and "${" in item:
                        # Process the template string
                        template_pattern = r'\$\{([^}]+)\}'
                        matches = re.findall(template_pattern, item)
                        
                        if len(matches) == 1 and item.strip() == f"${{{matches[0]}}}":
                            match = matches[0]
                            try:
                                # Parse the template variable
                                if '.' in match:
                                    node_key, field_path = match.split('.', 1)
                                else:
                                    node_key, field_path = match, None
                                
                                # Get node output using the same logic as string templates
                                manifest = manifest_manager.get_manifest()
                                node_output = None
                                
                                # Special handling for vendor_research_pp1_output template
                                if node_key == "vendor_research_pp1_output":
                                    # Look for vendor_research_pp1 in vendor_analysis
                                    # Try both gap_finder_collect and gapfinder_collect naming conventions
                                    collect_stage_data = None
                                    for collect_key in ["gap_finder_collect", "gapfinder_collect"]:
                                        if collect_key in manifest.get("stages", {}):
                                            collect_stage_data = manifest["stages"][collect_key].get("data", {})
                                            break
                                    
                                    if collect_stage_data:
                                        vendor_analysis = collect_stage_data.get("vendor_analysis", {})
                                        if "vendor_research_pp1" in vendor_analysis:
                                            node_output = vendor_analysis["vendor_research_pp1"]
                                            self.logger.info(f"Found vendor_research_pp1_output in vendor_analysis")
                                        elif "vendor_research_batch" in vendor_analysis:
                                            node_output = vendor_analysis["vendor_research_batch"]
                                            self.logger.info(f"Found vendor_research_pp1_output via vendor_research_batch")

                                # Check gap_finder_collect stage first
                                collect_stage = manifest.get("stages", {}).get("gap_finder_collect", {})
                                if "data" in collect_stage and "tool_results" in collect_stage["data"]:
                                    tool_results = collect_stage["data"]["tool_results"]
                                    if node_key in tool_results:
                                        node_output = tool_results[node_key]
                                        self.logger.info(f"Found tool output {node_key} in gap_finder_collect stage")
                                
                                if node_output and field_path:
                                    # Extract the field
                                    if isinstance(node_output, dict) and field_path in node_output:
                                        field_value = node_output[field_path]
                                        # If the resolved value is a list, extend instead of append to flatten
                                        if isinstance(field_value, list):
                                            resolved_list.extend(field_value)
                                            self.logger.info(f"Flattened list template variable ${{{match}}} - extended {len(field_value)} items")
                                        else:
                                            resolved_list.append(field_value)
                                            self.logger.info(f"Resolved list template variable ${{{match}}} to {type(field_value).__name__}")
                                    else:
                                        resolved_list.append(item)  # Keep original
                                elif node_output:
                                    # If the resolved value is a list, extend instead of append to flatten
                                    if isinstance(node_output, list):
                                        resolved_list.extend(node_output)
                                        self.logger.info(f"Flattened list template variable ${{{match}}} - extended {len(node_output)} items")
                                    else:
                                        resolved_list.append(node_output)
                                else:
                                    resolved_list.append(item)  # Keep original
                            except Exception as e:
                                self.logger.error(f"Error resolving template variable {match}: {e}")
                                resolved_list.append(item)  # Keep original on error
                        else:
                            resolved_list.append(item)  # Keep non-template items
                    else:
                        resolved_list.append(item)  # Keep non-string items
                resolved_inputs[key] = resolved_list
            elif isinstance(value, str) and "${" in value:
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
                            # Backward compatibility: convert .contents based on node type
                            if field_path == 'contents':
                                # For extract_content nodes, use 'content' field
                                if 'extract_content' in node_key:
                                    field_path = 'content'
                                    self.logger.info(f"Converting .contents to .content for extract_content node {node_key}")
                                # For triage_content nodes, use 'content' field
                                elif 'triage_content' in node_key:
                                    field_path = 'content'
                                    self.logger.info(f"Converting deprecated .contents to .content for {node_key}")
                                else:
                                    # Default fallback - use 'content' field
                                    field_path = 'content'
                                    self.logger.warning(f"Unknown node type for {node_key}, defaulting to 'content' field")
                            elif field_path == 'triage_results':
                                # Convert triage_results to content for all node types
                                field_path = 'content'
                                self.logger.info(f"Converting .triage_results to .content for {node_key}")
                        else:
                            node_key, field_path = match, None
                        
                        # First try to get the node output from gap_finder_collect stage tool_results
                        manifest = manifest_manager.get_manifest()
                        node_output = None
                        
                        # Check gap_finder_collect stage first
                        collect_stage = manifest.get("stages", {}).get("gap_finder_collect", {})
                        if "data" in collect_stage and "tool_results" in collect_stage["data"]:
                            tool_results = collect_stage["data"]["tool_results"]
                            if node_key in tool_results:
                                node_output = tool_results[node_key]
                                self.logger.info(f"Found tool output {node_key} in gap_finder_collect stage")
                        
                        # If not found in collect stage, try the original logic
                        if not node_output:
                            node_output = manifest_manager.get_node_output(node_key)
                        
                        # If still not found, try to find the actual node_id that maps to this output key
                        if not node_output:
                            # Look through all stages to find a node that has this output_manifest_key
                            for stage_key, stage_data in manifest.get("stages", {}).items():
                                # Check if this stage was stored with the output key we're looking for
                                if stage_key == node_key and "data" in stage_data:
                                    node_output = stage_data["data"]
                                    break
                            
                            # If still not found, search through DAG nodes to find the one with matching output_manifest_key
                            if not node_output:
                                # First try gap_finder dag_metadata nodes (where gap_finder nodes are stored)
                                dag_nodes = []
                                gap_finder_data = manifest.get("stages", {}).get("gap_finder_plan", {}).get("data", {})
                                if "dag_metadata" in gap_finder_data:
                                    dag_nodes = gap_finder_data.get("dag_metadata", {}).get("nodes", [])
                                
                                # Fallback to top-level dag nodes if not found
                                if not dag_nodes:
                                    dag = manifest.get("dag", {})
                                    dag_nodes = dag.get("nodes", [])
                                
                                for node in dag_nodes:
                                    if node.get("output_manifest_key") == node_key:
                                        # Found the node, now get its data from the manifest
                                        actual_node_id = node.get("node_id")
                                        if actual_node_id:
                                            # Try to get the node output using the actual node_id
                                            node_output = manifest_manager.get_node_output(actual_node_id)
                                            if not node_output:
                                                # Look in stages for the actual node_id
                                                for stage_key, stage_data in manifest.get("stages", {}).items():
                                                    if stage_key == actual_node_id and "data" in stage_data:
                                                        node_output = stage_data["data"]
                                                        break
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
                                # Special handling for vendors field - extract from vendor_results structure FIRST
                                if field_path == 'vendors':
                                    vendor_results = self._get_nested_field(node_output, 'vendor_results')
                                    if vendor_results and isinstance(vendor_results, list):
                                        # Extract vendors from each vendor_results item
                                        all_vendors = []
                                        for result in vendor_results:
                                            if isinstance(result, dict) and result.get('success') and 'vendors' in result:
                                                vendors_data = result['vendors']
                                                # Handle nested vendors structure: vendors.vendors
                                                if isinstance(vendors_data, dict) and 'vendors' in vendors_data:
                                                    vendors = vendors_data['vendors']
                                                    if isinstance(vendors, list):
                                                        all_vendors.extend(vendors)
                                                # Handle direct vendors list (fallback)
                                                elif isinstance(vendors_data, list):
                                                    all_vendors.extend(vendors_data)
                                        if all_vendors:
                                            field_value = all_vendors
                                            self.logger.info(f"Extracted {len(all_vendors)} vendors from vendor_results structure for node {node_key}")
                                        else:
                                            field_value = None
                                    else:
                                        field_value = None
                                else:
                                    field_value = self._get_nested_field(node_output, field_path)
                                
                                # Enhanced fallback logic for field resolution failures
                                if field_value is None:
                                    # Define field mapping based on node type and requested field
                                    fallback_fields = []
                                    
                                    if field_path == 'contents':
                                        if 'extract_content' in node_key:
                                            fallback_fields = ['content']
                                        elif 'triage_content' in node_key:
                                            fallback_fields = ['triage_results']
                                        else:
                                            # Unknown node type, try common fields
                                            fallback_fields = ['content', 'triage_results', 'results', 'data']
                                    elif field_path == 'content':
                                        fallback_fields = ['contents', 'data', 'results']
                                    elif field_path == 'triage_results':
                                        fallback_fields = ['results', 'content', 'contents', 'data']
                                    elif field_path == 'vendor_results':
                                        fallback_fields = ['results', 'vendors', 'data', 'content']
                                    elif field_path == 'vendors':
                                        fallback_fields = ['vendor_results', 'results', 'data', 'content']
                                    elif field_path == 'results':
                                        fallback_fields = ['content', 'data', 'triage_results']
                                    
                                    # Try fallback fields
                                    for fallback_field in fallback_fields:
                                        field_value = self._get_nested_field(node_output, fallback_field)
                                        if field_value is not None:
                                            self.logger.info(f"Resolved field '{field_path}' using fallback '{fallback_field}' for node {node_key}")
                                            break
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
    
    def _parse_structured_text(self, text: str) -> Dict[str, Any]:
        """Parse structured text response into JSON format."""
        import re
        from datetime import datetime
        
        try:
            result = {
                "summary": {
                    "analysis_objective": "",
                    "methodology": "",
                    "total_vendors_analyzed": 0,
                    "total_gaps_identified": 0,
                    "analysis_confidence": "medium",
                    "key_findings": [],
                    "analysis_timestamp": datetime.now().isoformat()
                },
                "prioritized_gaps": [],
                "vendor_landscape": [],
                "whitespace_opportunities": [],
                "strategic_insights": {
                    "market_trends": [],
                    "emerging_patterns": [],
                    "competitive_dynamics": "",
                    "barriers_to_entry": [],
                    "success_factors": []
                },
                "risks_and_unknowns": [],
                "next_actions_recommendation": []
            }
            
            # Parse Summary section
            summary_match = re.search(r'## Summary(.*?)(?=## |$)', text, re.DOTALL)
            if summary_match:
                summary_text = summary_match.group(1)
                result["summary"]["analysis_objective"] = self._extract_field(summary_text, "Analysis Objective")
                result["summary"]["methodology"] = self._extract_field(summary_text, "Methodology")
                result["summary"]["total_vendors_analyzed"] = self._extract_number(summary_text, "Total Vendors Analyzed")
                result["summary"]["total_gaps_identified"] = self._extract_number(summary_text, "Total Gaps Identified")
                result["summary"]["analysis_confidence"] = self._extract_field(summary_text, "Analysis Confidence", "medium")
                result["summary"]["key_findings"] = self._extract_list(summary_text, "Key Findings")
            
            # Parse Prioritized Gaps section
            gaps_match = re.search(r'## Prioritized Gaps(.*?)(?=## |$)', text, re.DOTALL)
            if gaps_match:
                gaps_text = gaps_match.group(1)
                result["prioritized_gaps"] = self._parse_gaps(gaps_text)
            
            # Parse Vendor Landscape section
            vendors_match = re.search(r'## Vendor Landscape(.*?)(?=## |$)', text, re.DOTALL)
            if vendors_match:
                vendors_text = vendors_match.group(1)
                result["vendor_landscape"] = self._parse_vendors(vendors_text)
            
            # Parse Whitespace Opportunities section
            opportunities_match = re.search(r'## Whitespace Opportunities(.*?)(?=## |$)', text, re.DOTALL)
            if opportunities_match:
                opportunities_text = opportunities_match.group(1)
                result["whitespace_opportunities"] = self._parse_opportunities(opportunities_text)
            
            # Parse Strategic Insights section
            insights_match = re.search(r'## Strategic Insights(.*?)(?=## |$)', text, re.DOTALL)
            if insights_match:
                insights_text = insights_match.group(1)
                result["strategic_insights"]["market_trends"] = self._extract_list(insights_text, "Market Trends")
                result["strategic_insights"]["emerging_patterns"] = self._extract_list(insights_text, "Emerging Patterns")
                result["strategic_insights"]["competitive_dynamics"] = self._extract_field(insights_text, "Competitive Dynamics")
                result["strategic_insights"]["barriers_to_entry"] = self._extract_list(insights_text, "Barriers to Entry")
                result["strategic_insights"]["success_factors"] = self._extract_list(insights_text, "Success Factors")
            
            # Parse Risks and Unknowns section
            risks_match = re.search(r'## Risks and Unknowns(.*?)(?=## |$)', text, re.DOTALL)
            if risks_match:
                risks_text = risks_match.group(1)
                result["risks_and_unknowns"] = self._parse_risks(risks_text)
            
            # Parse Next Actions section
            actions_match = re.search(r'## Next Actions(.*?)(?=## |$)', text, re.DOTALL)
            if actions_match:
                actions_text = actions_match.group(1)
                result["next_actions_recommendation"] = self._parse_actions(actions_text)
            
            self.logger.info("Successfully parsed structured text response")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse structured text: {e}")
            return {"error": f"Failed to parse structured text: {e}"}
    
    def _extract_field(self, text: str, field_name: str, default: str = "") -> str:
        """Extract a field value from structured text."""
        import re
        pattern = rf'- {re.escape(field_name)}:\s*(.+?)(?:\n|$)'
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else default
    
    def _extract_number(self, text: str, field_name: str, default: int = 0) -> int:
        """Extract a number field from structured text."""
        import re
        pattern = rf'- {re.escape(field_name)}:\s*(\d+)'
        match = re.search(pattern, text)
        return int(match.group(1)) if match else default
    
    def _extract_list(self, text: str, field_name: str) -> list:
        """Extract a list field from structured text."""
        import re
        pattern = rf'- {re.escape(field_name)}:\s*(.+?)(?:\n- |\n## |$)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            list_text = match.group(1).strip()
            # Split by newlines and clean up
            items = [item.strip('- ').strip() for item in list_text.split('\n') if item.strip()]
            return [item for item in items if item]
        return []
    
    def _parse_gaps(self, gaps_text: str) -> list:
        """Parse gaps from structured text."""
        import re
        gaps = []
        gap_pattern = r'- Gap \d+:\s*(.+?)\s*-\s*Urgency:\s*(\d+),\s*Feasibility:\s*(\d+),\s*Impact:\s*(\d+)(.*?)(?=- Gap \d+:|$)'
        matches = re.finditer(gap_pattern, gaps_text, re.DOTALL)
        
        for i, match in enumerate(matches, 1):
            title = match.group(1).strip()
            urgency = int(match.group(2))
            feasibility = int(match.group(3))
            impact = int(match.group(4))
            details = match.group(5)
            
            rationale = self._extract_field(details, "Rationale")
            validation_path = self._extract_field(details, "Validation Path")
            
            gaps.append({
                "gap_id": f"gap_{i}",
                "title": title,
                "urgency_score": urgency,
                "feasibility_score": feasibility,
                "impact_score": impact,
                "overall_priority": (urgency + feasibility + impact) // 3,
                "rationale": rationale,
                "validation_path": validation_path
            })
        
        return gaps
    
    def _parse_vendors(self, vendors_text: str) -> list:
        """Parse vendors from structured text."""
        import re
        vendors = []
        vendor_pattern = r'- (.+?)\s*\((.+?)\)(.*?)(?=- .+?\(|$)'
        matches = re.finditer(vendor_pattern, vendors_text, re.DOTALL)
        
        for match in matches:
            vendor_name = match.group(1).strip()
            category = match.group(2).strip()
            details = match.group(3)
            
            strengths = self._extract_list(details, "Strengths")
            limitations = self._extract_list(details, "Limitations")
            market_position = self._extract_field(details, "Market Position")
            addressed_gaps = self._extract_list(details, "Addressed Gaps")
            
            vendors.append({
                "vendor_name": vendor_name,
                "category": category,
                "market_position": market_position,
                "strengths": strengths,
                "limitations": limitations,
                "addressed_gaps": addressed_gaps
            })
        
        return vendors
    
    def _parse_opportunities(self, opportunities_text: str) -> list:
        """Parse opportunities from structured text."""
        import re
        opportunities = []
        opp_pattern = r'- (.+?)(.*?)(?=- .+?|$)'
        matches = re.finditer(opp_pattern, opportunities_text, re.DOTALL)
        
        for i, match in enumerate(matches, 1):
            title = match.group(1).strip()
            details = match.group(2)
            
            description = self._extract_field(details, "Description")
            validation_path = self._extract_field(details, "Validation Path")
            potential_impact = self._extract_field(details, "Potential Impact")
            feasibility = self._extract_field(details, "Feasibility")
            
            opportunities.append({
                "opportunity_id": f"opp_{i}",
                "title": title,
                "description": description,
                "validation_path": validation_path,
                "potential_impact": potential_impact,
                "feasibility_assessment": feasibility
            })
        
        return opportunities
    
    def _parse_risks(self, risks_text: str) -> list:
        """Parse risks from structured text."""
        import re
        risks = []
        risk_pattern = r'- (.+?)\s*-\s*Impact:\s*(.+?),\s*Likelihood:\s*(.+?)(.*?)(?=- .+?|$)'
        matches = re.finditer(risk_pattern, risks_text, re.DOTALL)
        
        for i, match in enumerate(matches, 1):
            description = match.group(1).strip()
            impact = match.group(2).strip()
            likelihood = match.group(3).strip()
            details = match.group(4)
            
            mitigation_strategy = self._extract_field(details, "Mitigation Strategy")
            data_needed = self._extract_list(details, "Data Needed")
            
            risks.append({
                "risk_id": f"risk_{i}",
                "description": description,
                "impact": impact.lower(),
                "likelihood": likelihood.lower(),
                "mitigation_strategy": mitigation_strategy,
                "data_needed": data_needed
            })
        
        return risks
    
    def _parse_actions(self, actions_text: str) -> list:
        """Parse actions from structured text."""
        import re
        actions = []
        action_pattern = r'- (.+?)\s*-\s*Priority:\s*(.+?)(.*?)(?=- .+?|$)'
        matches = re.finditer(action_pattern, actions_text, re.DOTALL)
        
        for i, match in enumerate(matches, 1):
            action = match.group(1).strip()
            priority = match.group(2).strip()
            details = match.group(3)
            
            owner = self._extract_field(details, "Owner")
            expected_outcome = self._extract_field(details, "Expected Outcome")
            timeline = self._extract_field(details, "Timeline")
            
            actions.append({
                "action_id": f"action_{i}",
                "action": action,
                "priority": priority.lower(),
                "owner": owner,
                "expected_outcome": expected_outcome,
                "timeline": timeline
            })
        
        return actions
    
    def _load_think_stage_data(self, manifest_manager: ManifestManager) -> Dict[str, Any]:
        """Load think stage data from manifest."""
        try:
            manifest = manifest_manager.get_manifest()
            think_stage = manifest.get("stages", {}).get("gap_finder_think", {})
            return think_stage.get("data", {})
        except Exception as e:
            self.logger.error(f"Error loading think stage data: {e}")
            return {}
    
    def _load_essential_act_data(self, manifest_manager: ManifestManager) -> Dict[str, Any]:
        """Load essential data for act stage - minimal and focused."""
        essential_data = {}
        
        try:
            manifest = manifest_manager.get_manifest()
            collect_stage = manifest.get("stages", {}).get("gap_finder_collect", {})
            tool_results = collect_stage.get("data", {}).get("tool_results", {})
            
            # Only extract URLs from search_links (not full content)
            for key, value in tool_results.items():
                if key.startswith("search_links_pp") and key.endswith("_output"):
                    urls = self._extract_urls_from_search_links(value)
                    if urls:
                        essential_data["search_urls"] = urls
                        self.logger.info(f"Extracted {len(urls)} URLs from {key}")
            
            self.logger.info(f"Loaded essential act data: {list(essential_data.keys())}")
            return essential_data
            
        except Exception as e:
            self.logger.error(f"Error loading essential act data: {e}")
            return {}
    
    def _load_act_prompt(self) -> str:
        """Load act stage prompt template."""
        try:
            prompt_path = Path("scout_agent/prompts/gap_finder_agent/act.prompt")
            if prompt_path.exists():
                with open(prompt_path, 'r') as f:
                    return f.read()
            else:
                self.logger.error(f"Act prompt file not found: {prompt_path}")
                return ""
        except Exception as e:
            self.logger.error(f"Error loading act prompt: {e}")
            return ""
    
    def _store_act_output_to_manifest(self, manifest_manager: ManifestManager, recommendations_result: Dict[str, Any]) -> None:
        """Store act stage output to manifest."""
        try:
            # Get the manifest
            manifest = manifest_manager.get_manifest()
            
            # Ensure stages section exists
            if "stages" not in manifest:
                manifest["stages"] = {}
            
            # Store the act stage output
            manifest["stages"]["gap_finder_act"] = {
                "status": "completed",
                "updated_at": datetime.now().isoformat(),
                "data": recommendations_result
            }
            
            # Save the manifest
            manifest_manager._save()
            self.logger.info("Stored act stage output to manifest")
            
        except Exception as e:
            self.logger.error(f"Error storing act output to manifest: {e}")
            raise e
    
    def _parse_structured_act_text(self, text: str) -> Dict[str, Any]:
        """Parse structured text response into JSON format for act stage."""
        import re
        from datetime import datetime
        
        try:
            result = {
                "market_gaps_summary": {
                    "primary_gap": "",
                    "gap_count": "",
                    "market_impact": "",
                    "opportunity_size": ""
                },
                "identified_market_gaps": [],
                "strategic_recommendations": [],
                "market_intelligence": {
                    "key_insights": [],
                    "emerging_trends": [],
                    "competitive_dynamics": "",
                    "market_barriers": [],
                    "enabling_factors": []
                },
                "implementation_guidance": {
                    "priority_order": [],
                    "resource_requirements": [],
                    "timeline_considerations": [],
                    "success_metrics": []
                }
            }
            
            # Parse Market Gaps Summary section
            summary_match = re.search(r'## Market Gaps Summary(.*?)(?=## |$)', text, re.DOTALL)
            if summary_match:
                summary_text = summary_match.group(1)
                result["market_gaps_summary"]["primary_gap"] = self._extract_field(summary_text, "Primary Gap")
                result["market_gaps_summary"]["gap_count"] = self._extract_field(summary_text, "Gap Count")
                result["market_gaps_summary"]["market_impact"] = self._extract_field(summary_text, "Market Impact")
                result["market_gaps_summary"]["opportunity_size"] = self._extract_field(summary_text, "Opportunity Size")
            
            # Parse Identified Market Gaps section
            gaps_match = re.search(r'## Identified Market Gaps(.*?)(?=## |$)', text, re.DOTALL)
            if gaps_match:
                gaps_text = gaps_match.group(1)
                result["identified_market_gaps"] = self._parse_market_gaps(gaps_text)
            
            # Parse Strategic Recommendations section
            recommendations_match = re.search(r'## Strategic Recommendations(.*?)(?=## |$)', text, re.DOTALL)
            if recommendations_match:
                recommendations_text = recommendations_match.group(1)
                result["strategic_recommendations"] = self._parse_strategic_recommendations(recommendations_text)
            
            # Parse Market Intelligence section
            intelligence_match = re.search(r'## Market Intelligence(.*?)(?=## |$)', text, re.DOTALL)
            if intelligence_match:
                intelligence_text = intelligence_match.group(1)
                result["market_intelligence"]["key_insights"] = self._extract_list(intelligence_text, "Key Insights")
                result["market_intelligence"]["emerging_trends"] = self._extract_list(intelligence_text, "Emerging Trends")
                result["market_intelligence"]["competitive_dynamics"] = self._extract_field(intelligence_text, "Competitive Dynamics")
                result["market_intelligence"]["market_barriers"] = self._extract_list(intelligence_text, "Market Barriers")
                result["market_intelligence"]["enabling_factors"] = self._extract_list(intelligence_text, "Enabling Factors")
            
            # Parse Implementation Guidance section
            guidance_match = re.search(r'## Implementation Guidance(.*?)(?=## |$)', text, re.DOTALL)
            if guidance_match:
                guidance_text = guidance_match.group(1)
                result["implementation_guidance"]["priority_order"] = self._extract_list(guidance_text, "Priority Order")
                result["implementation_guidance"]["resource_requirements"] = self._extract_list(guidance_text, "Resource Requirements")
                result["implementation_guidance"]["timeline_considerations"] = self._extract_list(guidance_text, "Timeline Considerations")
                result["implementation_guidance"]["success_metrics"] = self._extract_list(guidance_text, "Success Metrics")
            
            self.logger.info("Successfully parsed structured act text response")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse structured act text: {e}")
            return {"error": f"Failed to parse structured act text: {e}"}
    
    def _parse_market_gaps(self, gaps_text: str) -> list:
        """Parse market gaps from structured text."""
        import re
        gaps = []
        
        # Look for gap entries with more specific patterns
        lines = gaps_text.split('\n')
        current_gap = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('- ') and not line.startswith('-   '):
                # Check if this is a gap name (doesn't contain a colon) or a field (contains a colon)
                if ':' in line:
                    # This is a field line like "- Description: value"
                    if current_gap:
                        field_line = line[2:].strip()  # Remove the "- "
                        field_name, field_value = field_line.split(':', 1)
                        field_name = field_name.strip().lower()
                        field_value = field_value.strip()
                        
                        if field_name == "description":
                            current_gap["description"] = field_value
                        elif field_name == "severity":
                            current_gap["severity"] = field_value
                        elif field_name == "affected segments":
                            current_gap["affected_segments"] = field_value
                        elif field_name == "evidence":
                            current_gap["evidence"] = field_value
                        elif field_name == "opportunity score":
                            current_gap["opportunity_score"] = field_value
                        elif field_name == "competitive landscape":
                            current_gap["competitive_landscape"] = field_value
                        elif field_name == "market timing":
                            current_gap["market_timing"] = field_value
                else:
                    # This is a new gap
                    if current_gap:
                        gaps.append(current_gap)
                    
                    gap_name = line[2:].strip()
                    current_gap = {
                        "gap_name": gap_name,
                        "description": "",
                        "severity": "",
                        "affected_segments": "",
                        "evidence": "",
                        "opportunity_score": "",
                        "competitive_landscape": "",
                        "market_timing": ""
                    }
        
        # Add the last gap if it exists
        if current_gap:
            gaps.append(current_gap)
        
        return gaps

    def _parse_strategic_recommendations(self, recommendations_text: str) -> list:
        """Parse strategic recommendations from structured text."""
        import re
        recommendations = []
        
        # Look for recommendation entries with more specific patterns
        lines = recommendations_text.split('\n')
        current_recommendation = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('- ') and not line.startswith('-   '):
                # Check if this is a recommendation name (doesn't contain a colon) or a field (contains a colon)
                if ':' in line:
                    # This is a field line like "- Target Gap: value"
                    if current_recommendation:
                        field_line = line[2:].strip()  # Remove the "- "
                        field_name, field_value = field_line.split(':', 1)
                        field_name = field_name.strip().lower()
                        field_value = field_value.strip()
                        
                        if field_name == "target gap":
                            current_recommendation["target_gap"] = field_value
                        elif field_name == "approach":
                            current_recommendation["approach"] = field_value
                        elif field_name == "target market":
                            current_recommendation["target_market"] = field_value
                        elif field_name == "value proposition":
                            current_recommendation["value_proposition"] = field_value
                        elif field_name == "implementation strategy":
                            current_recommendation["implementation_strategy"] = field_value
                        elif field_name == "success criteria":
                            current_recommendation["success_criteria"] = field_value
                        elif field_name == "risk factors":
                            current_recommendation["risk_factors"] = field_value
                else:
                    # This is a new recommendation
                    if current_recommendation:
                        recommendations.append(current_recommendation)
                    
                    recommendation_name = line[2:].strip()
                    current_recommendation = {
                        "recommendation_name": recommendation_name,
                        "target_gap": "",
                        "approach": "",
                        "target_market": "",
                        "value_proposition": "",
                        "implementation_strategy": "",
                        "success_criteria": "",
                        "risk_factors": ""
                    }
        
        # Add the last recommendation if it exists
        if current_recommendation:
            recommendations.append(current_recommendation)
        
        return recommendations

    def _parse_saas_recommendations(self, recommendations_text: str) -> list:
        """Parse SaaS recommendations from structured text."""
        import re
        recommendations = []
        
        # Look for business recommendations with more specific patterns
        # Pattern: - [Business Name] followed by details
        lines = recommendations_text.split('\n')
        current_recommendation = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('- ') and not line.startswith('-   '):
                # Check if this is a business name (doesn't contain a colon) or a field (contains a colon)
                if ':' in line:
                    # This is a field line like "- Description: value"
                    if current_recommendation:
                        field_line = line[2:].strip()  # Remove the "- "
                        field_name, field_value = field_line.split(':', 1)
                        field_name = field_name.strip().lower()
                        field_value = field_value.strip()
                        
                        if field_name == "description":
                            current_recommendation["description"] = field_value
                        elif field_name == "target gap":
                            current_recommendation["target_gap"] = field_value
                        elif field_name == "value proposition":
                            current_recommendation["value_proposition"] = field_value
                        elif field_name == "revenue model":
                            current_recommendation["revenue_model"] = field_value
                        elif field_name == "competitive advantage":
                            current_recommendation["competitive_advantage"] = field_value
                        elif field_name == "go to market":
                            current_recommendation["go_to_market"] = field_value
                else:
                    # This is a new business recommendation
                    if current_recommendation:
                        recommendations.append(current_recommendation)
                    
                    business_name = line[2:].strip()
                    current_recommendation = {
                        "business_name": business_name,
                        "description": "",
                        "target_gap": "",
                        "value_proposition": "",
                        "mvp_features": [],
                        "long_term_features": [],
                        "target_customers": [],
                        "revenue_model": "",
                        "competitive_advantage": "",
                        "go_to_market": ""
                    }
        
        # Add the last recommendation if it exists
        if current_recommendation:
            recommendations.append(current_recommendation)
        
        return recommendations
    
    def _create_fallback_act_result(self, gap_input: GapFinderInput) -> Dict[str, Any]:
        """Create a fallback act result when LLM parsing fails."""
        return {
            "market_gaps_summary": {
                "primary_gap": "Market gaps identified through analysis",
                "gap_count": "Multiple gaps discovered",
                "market_impact": "Significant market opportunity",
                "opportunity_size": "To be determined through further analysis"
            },
            "identified_market_gaps": [
                {
                    "gap_name": "Primary Market Gap",
                    "description": "Key market gap identified through research",
                    "severity": "High",
                    "affected_segments": "Multiple customer segments",
                    "evidence": "Research findings support this gap",
                    "opportunity_score": "8",
                    "competitive_landscape": "Limited existing solutions",
                    "market_timing": "Favorable market conditions"
                }
            ],
            "strategic_recommendations": [
                {
                    "recommendation_name": "Gap-Filling Solution",
                    "target_gap": "Primary Market Gap",
                    "approach": "Comprehensive solution development",
                    "target_market": "Underserved customer segments",
                    "value_proposition": "Addresses identified market needs",
                    "implementation_strategy": "Phased development approach",
                    "success_criteria": "Market validation and customer adoption",
                    "risk_factors": "Market timing and competitive response"
                }
            ],
            "market_intelligence": {
                "key_insights": ["Market gaps present significant opportunities"],
                "emerging_trends": ["Growing demand for gap-filling solutions"],
                "competitive_dynamics": "Limited competition in identified gaps",
                "market_barriers": ["Development complexity", "Market education needs"],
                "enabling_factors": ["Technology availability", "Market readiness"]
            },
            "implementation_guidance": {
                "priority_order": ["Address highest-impact gaps first"],
                "resource_requirements": ["Development team", "Market research"],
                "timeline_considerations": ["3-6 months for initial solution"],
                "success_metrics": ["Market validation", "Customer adoption"]
            }
        }


# Register the agent - moved to agent_registry.py
# from .base import register_agent
# register_agent("gap_finder", GapFinderAgent)
