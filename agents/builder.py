"""
Builder Agent - Business Solution Development

This agent transforms market gaps from the gap finder into comprehensive business solutions,
focusing on building viable SaaS businesses rather than just technical implementations.
"""

import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from .base import BaseAgent, AgentInput, AgentOutput
from ..llm.utils import LLMAgentMixin
from ..memory.manifest_manager import ManifestManager


# Input class for the builder agent
class BuilderInput:
    """Input for BuilderAgent."""
    def __init__(self, gap_finder_output: Dict[str, Any], market_context: str = "", analysis_scope: str = "focused"):
        self.gap_finder_output = gap_finder_output
        self.market_context = market_context
        self.analysis_scope = analysis_scope


class BuilderAgent(BaseAgent, LLMAgentMixin):
    """
    Builder Agent for creating comprehensive business solutions.
    
    Transforms market gaps into viable business concepts with complete go-to-market strategies,
    focusing on building successful SaaS businesses.
    """
    
    def __init__(self, agent_id: str = None):
        BaseAgent.__init__(self, name="builder", agent_id=agent_id)
        LLMAgentMixin.__init__(self, preferred_backend='deepseek')
        
        # Initialize backend preferences
        self.task_backend_preferences = {
            'think_analysis': 'deepseek',
            'act_development': 'deepseek',
            'default': 'deepseek'
        }
    
    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Main execution method - runs think and act stages."""
        try:
            self.logger.info("Starting builder agent execution...")
            self.logger.info(f"Input data type: {type(agent_input)}")
            self.logger.info(f"Input data: {agent_input}")
            
            # Convert AgentInput to BuilderInput if needed
            if not isinstance(agent_input, BuilderInput):
                # Handle case where input_data.data might be a dict
                if isinstance(agent_input.data, dict):
                    gap_finder_output = agent_input.data
                else:
                    self.logger.error("Invalid input data format")
            return AgentOutput(
                result=None,
                metadata={'agent_id': self.agent_id, 'agent_name': self.name},
                logs=self.execution_logs,
                execution_time=0.0,
                success=False,
                        error="Invalid input data format"
                    )
                
                builder_input = BuilderInput(
                    gap_finder_output=gap_finder_output,
                    market_context=agent_input.context.get("market_context", "") if agent_input.context else "",
                    analysis_scope=agent_input.context.get("analysis_scope", "focused") if agent_input.context else "focused"
                )
            else:
                builder_input = agent_input
            
            # Ensure we have gap finder output
            if not builder_input.gap_finder_output:
                self.logger.error("No gap finder output found")
                return AgentOutput(
                    result=None,
                    metadata={'agent_id': self.agent_id, 'agent_name': self.name},
                    logs=self.execution_logs,
                    execution_time=0.0,
                    success=False,
                    error="No gap finder output available"
                )
            
            # Initialize manifest manager
            run_id = getattr(self.state, 'run_id', None)
            if not run_id:
                self.logger.error("No run_id found in agent state")
                return AgentOutput(
                    result=None,
                    metadata={'agent_id': self.agent_id, 'agent_name': self.name},
                    logs=self.execution_logs,
                    execution_time=0.0,
                    success=False,
                    error="No run_id available"
                )
            
            manifest_path = Path("data/runs") / run_id / "run_manifest.json"
            manifest_manager = ManifestManager(manifest_path=manifest_path)
            
            # Run think stage
            self._update_status('thinking')
            think_result = await self.think(builder_input, manifest_manager)
            
            # Run act stage
            self._update_status('acting')
            act_result = await self.act(builder_input, manifest_manager, think_result)
            
            self._update_status('completed')
            
            return AgentOutput(
                result=act_result,
                metadata={
                    'agent_id': self.agent_id,
                    'agent_name': self.name,
                    'think_result': think_result,
                    'act_result': act_result
                },
                logs=self.execution_logs,
                execution_time=0.0,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in builder agent execution: {e}")
            self._update_status('failed')
            return AgentOutput(
                result=None,
                metadata={'agent_id': self.agent_id, 'agent_name': self.name},
                logs=self.execution_logs,
                execution_time=0.0,
                success=False,
                error=str(e)
            )
    
    async def plan(self, input_data: BuilderInput) -> Dict[str, Any]:
        """Plan phase is not used in BuilderAgent."""
        return {"status": "skipped", "reason": "BuilderAgent uses only think and act phases"}
    
    async def think(self, input_data: BuilderInput) -> Dict[str, Any]:
        """Analyze market gaps to design business solutions."""
        try:
            self.logger.info("Starting builder think stage...")
            
            # Initialize manifest manager
            run_id = getattr(self.state, 'run_id', None)
            if not run_id:
                self.logger.error("No run_id found in agent state")
                return {"error": "No run_id available"}
            
            manifest_path = Path("data/runs") / run_id / "run_manifest.json"
            manifest_manager = ManifestManager(manifest_path=manifest_path)
            
            # Load gap finder output
            gap_finder_data = self._load_gap_finder_output(manifest_manager)
            if not gap_finder_data:
                self.logger.error("No gap finder output found")
                return {"error": "No gap finder output available"}
            
            # Prepare synthesis data
            synthesis_data = {
                "gap_finder_output": gap_finder_data,
                "market_context": input_data.market_context,
                "analysis_scope": input_data.analysis_scope
            }
            
            # Load think stage prompt
            prompt_content = self._load_think_prompt()
            if not prompt_content:
                self.logger.error("Failed to load think stage prompt")
                return {"error": "Failed to load think stage prompt"}
            
            # Append the synthesis data to the prompt
            prompt_content += f"\n\nGap Finder Data to Analyze:\n{json.dumps(synthesis_data, indent=2)}"
            
            # Generate business analysis using LLM mixin
            analysis_result = await self.llm_generate(prompt=prompt_content, task_type="think_analysis")
            
            # Parse the response as structured text
            if isinstance(analysis_result, str):
                self.logger.info(f"LLM response type: {type(analysis_result)}, length: {len(analysis_result)}")
                self.logger.info(f"LLM response preview: {analysis_result[:200]}...")
                # Parse structured text response instead of JSON
                analysis_result = self._parse_structured_think_text(analysis_result)
                self.logger.info(f"Successfully parsed LLM response, type: {type(analysis_result)}")
                
                # Check if we got an error from _parse_structured_think_text
                if isinstance(analysis_result, dict) and "error" in analysis_result:
                    self.logger.error(f"Structured text parsing failed: {analysis_result['error']}")
                    # Create a fallback result structure
                    analysis_result = self._create_fallback_think_result(input_data)
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
    
    async def act(self, input_data: BuilderInput, think_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create comprehensive business solutions and go-to-market strategies."""
        try:
            self.logger.info("Starting builder act stage...")
            
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
            
            # Prepare synthesis data
            synthesis_data = {
                "think_stage_output": think_data,
                "market_context": input_data.market_context,
                "analysis_scope": input_data.analysis_scope
            }
            
            # Load act stage prompt
            prompt_content = self._load_act_prompt()
            if not prompt_content:
                self.logger.error("Failed to load act stage prompt")
                return {"error": "Failed to load act stage prompt"}
            
            # Append the synthesis data to the prompt
            prompt_content += f"\n\nThink Stage Data to Analyze:\n{json.dumps(synthesis_data, indent=2)}"
            
            # Generate business solutions using LLM mixin
            solutions_result = await self.llm_generate(prompt=prompt_content, task_type="act_development")
            
            # Parse the response as structured text
            if isinstance(solutions_result, str):
                self.logger.info(f"LLM response type: {type(solutions_result)}, length: {len(solutions_result)}")
                self.logger.info(f"LLM response preview: {solutions_result[:200]}...")
                # Parse structured text response instead of JSON
                solutions_result = self._parse_structured_act_text(solutions_result)
                self.logger.info(f"Successfully parsed LLM response, type: {type(solutions_result)}")
                
                # Check if we got an error from _parse_structured_act_text
                if isinstance(solutions_result, dict) and "error" in solutions_result:
                    self.logger.error(f"Structured text parsing failed: {solutions_result['error']}")
                    # Create a fallback result structure
                    solutions_result = self._create_fallback_act_result(input_data)
                    self.logger.info(f"Created fallback result, type: {type(solutions_result)}")
            else:
                self.logger.info(f"LLM response is not a string, type: {type(solutions_result)}")
            
            # Store the result to manifest
            self._store_act_output_to_manifest(manifest_manager, solutions_result)
            
            self.logger.info("Act stage completed successfully")
            return solutions_result
            
        except Exception as e:
            self.logger.error(f"Error in act stage: {e}")
            raise e
    
    def _load_gap_finder_output(self, manifest_manager: ManifestManager) -> Dict[str, Any]:
        """Load gap finder act stage output from manifest."""
        try:
            manifest = manifest_manager.get_manifest()
            gap_finder_act = manifest.get("stages", {}).get("gap_finder_act", {})
            return gap_finder_act.get("data", {})
            except Exception as e:
            self.logger.error(f"Error loading gap finder output: {e}")
            return {}
        
    def _load_think_stage_data(self, manifest_manager: ManifestManager) -> Dict[str, Any]:
        """Load think stage data from manifest."""
            try:
            manifest = manifest_manager.get_manifest()
            think_stage = manifest.get("stages", {}).get("builder_think", {})
            return think_stage.get("data", {})
            except Exception as e:
            self.logger.error(f"Error loading think stage data: {e}")
            return {}
    
    def _load_think_prompt(self) -> str:
        """Load think stage prompt template."""
        try:
            prompt_path = Path("scout_agent/prompts/builder_agent/think.prompt")
            if prompt_path.exists():
                with open(prompt_path, 'r') as f:
                    return f.read()
            else:
                self.logger.error(f"Think prompt file not found: {prompt_path}")
                return ""
        except Exception as e:
            self.logger.error(f"Error loading think prompt: {e}")
            return ""
    
    def _load_act_prompt(self) -> str:
        """Load act stage prompt template."""
        try:
            prompt_path = Path("scout_agent/prompts/builder_agent/act.prompt")
            if prompt_path.exists():
                with open(prompt_path, 'r') as f:
                    return f.read()
        else:
                self.logger.error(f"Act prompt file not found: {prompt_path}")
                return ""
        except Exception as e:
            self.logger.error(f"Error loading act prompt: {e}")
            return ""
    
    def _store_think_output_to_manifest(self, manifest_manager: ManifestManager, analysis_result: Dict[str, Any]) -> None:
        """Store think stage output to manifest."""
        try:
            # Get the manifest
            manifest = manifest_manager.get_manifest()
            
            # Ensure stages section exists
            if "stages" not in manifest:
                manifest["stages"] = {}
            
            # Store the think stage output
            manifest["stages"]["builder_think"] = {
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
    
    def _store_act_output_to_manifest(self, manifest_manager: ManifestManager, solutions_result: Dict[str, Any]) -> None:
        """Store act stage output to manifest."""
        try:
            # Get the manifest
            manifest = manifest_manager.get_manifest()
            
            # Ensure stages section exists
            if "stages" not in manifest:
                manifest["stages"] = {}
            
            # Store the act stage output
            manifest["stages"]["builder_act"] = {
                "status": "completed",
                "updated_at": datetime.now().isoformat(),
                "data": solutions_result
            }
            
            # Save the manifest
            manifest_manager._save()
            self.logger.info("Stored act stage output to manifest")
            
        except Exception as e:
            self.logger.error(f"Error storing act output to manifest: {e}")
            raise e
    
    def _parse_structured_think_text(self, text: str) -> Dict[str, Any]:
        """Parse structured text response into JSON format for think stage."""
        import re
        from datetime import datetime
        
        try:
            result = {
                "business_feasibility_analysis": {
                    "primary_opportunity": "",
                    "market_viability": "",
                    "business_feasibility": "",
                    "competitive_landscape": ""
                },
                "solution_concept": {
                    "solution_name": "",
                    "alternative_names": [],
                    "core_value_proposition": "",
                    "target_problem": "",
                    "target_customers": ""
                },
                "business_model_design": {
                    "revenue_model": "",
                    "pricing_strategy": "",
                    "cost_structure": "",
                    "key_partnerships": "",
                    "customer_acquisition": ""
                },
                "competitive_positioning": {
                    "differentiation": "",
                    "competitive_advantages": "",
                    "market_positioning": "",
                    "barriers_to_entry": ""
                },
                "implementation_considerations": {
                    "development_approach": "",
                    "technology_considerations": "",
                    "resource_requirements": "",
                    "timeline_considerations": "",
                    "risk_factors": ""
                }
            }
            
            # Parse Business Feasibility Analysis section
            feasibility_match = re.search(r'## Business Feasibility Analysis(.*?)(?=## |$)', text, re.DOTALL)
            if feasibility_match:
                feasibility_text = feasibility_match.group(1)
                result["business_feasibility_analysis"]["primary_opportunity"] = self._extract_field(feasibility_text, "Primary Opportunity")
                result["business_feasibility_analysis"]["market_viability"] = self._extract_field(feasibility_text, "Market Viability")
                result["business_feasibility_analysis"]["business_feasibility"] = self._extract_field(feasibility_text, "Business Feasibility")
                result["business_feasibility_analysis"]["competitive_landscape"] = self._extract_field(feasibility_text, "Competitive Landscape")
            
            # Parse Solution Concept section
            concept_match = re.search(r'## Solution Concept(.*?)(?=## |$)', text, re.DOTALL)
            if concept_match:
                concept_text = concept_match.group(1)
                result["solution_concept"]["solution_name"] = self._extract_field(concept_text, "Solution Name")
                result["solution_concept"]["alternative_names"] = self._extract_list(concept_text, "Alternative Names")
                result["solution_concept"]["core_value_proposition"] = self._extract_field(concept_text, "Core Value Proposition")
                result["solution_concept"]["target_problem"] = self._extract_field(concept_text, "Target Problem")
                result["solution_concept"]["target_customers"] = self._extract_field(concept_text, "Target Customers")
            
            # Parse Business Model Design section
            model_match = re.search(r'## Business Model Design(.*?)(?=## |$)', text, re.DOTALL)
            if model_match:
                model_text = model_match.group(1)
                result["business_model_design"]["revenue_model"] = self._extract_field(model_text, "Revenue Model")
                result["business_model_design"]["pricing_strategy"] = self._extract_field(model_text, "Pricing Strategy")
                result["business_model_design"]["cost_structure"] = self._extract_field(model_text, "Cost Structure")
                result["business_model_design"]["key_partnerships"] = self._extract_field(model_text, "Key Partnerships")
                result["business_model_design"]["customer_acquisition"] = self._extract_field(model_text, "Customer Acquisition")
            
            # Parse Competitive Positioning section
            positioning_match = re.search(r'## Competitive Positioning(.*?)(?=## |$)', text, re.DOTALL)
            if positioning_match:
                positioning_text = positioning_match.group(1)
                result["competitive_positioning"]["differentiation"] = self._extract_field(positioning_text, "Differentiation")
                result["competitive_positioning"]["competitive_advantages"] = self._extract_field(positioning_text, "Competitive Advantages")
                result["competitive_positioning"]["market_positioning"] = self._extract_field(positioning_text, "Market Positioning")
                result["competitive_positioning"]["barriers_to_entry"] = self._extract_field(positioning_text, "Barriers to Entry")
            
            # Parse Implementation Considerations section
            implementation_match = re.search(r'## Implementation Considerations(.*?)(?=## |$)', text, re.DOTALL)
            if implementation_match:
                implementation_text = implementation_match.group(1)
                result["implementation_considerations"]["development_approach"] = self._extract_field(implementation_text, "Development Approach")
                result["implementation_considerations"]["technology_considerations"] = self._extract_field(implementation_text, "Technology Considerations")
                result["implementation_considerations"]["resource_requirements"] = self._extract_field(implementation_text, "Resource Requirements")
                result["implementation_considerations"]["timeline_considerations"] = self._extract_field(implementation_text, "Timeline Considerations")
                result["implementation_considerations"]["risk_factors"] = self._extract_field(implementation_text, "Risk Factors")
            
            self.logger.info("Successfully parsed structured think text response")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse structured think text: {e}")
            return {"error": f"Failed to parse structured think text: {e}"}
    
    def _parse_structured_act_text(self, text: str) -> Dict[str, Any]:
        """Parse structured text response into JSON format for act stage."""
        import re
        from datetime import datetime
        
        try:
            result = {
                "business_solution_summary": {
                    "solution_name": "",
                    "business_concept": "",
                    "target_market": "",
                    "market_opportunity": "",
                    "business_model": ""
                },
                "product_strategy": {
                    "core_features": [],
                    "advanced_features": [],
                    "user_experience": "",
                    "design_philosophy": "",
                    "product_differentiation": ""
                },
                "business_model_pricing": {
                    "revenue_streams": "",
                    "pricing_strategy": "",
                    "cost_structure": "",
                    "unit_economics": "",
                    "profitability_timeline": ""
                },
                "go_to_market_strategy": {
                    "customer_acquisition": "",
                    "marketing_strategy": "",
                    "sales_strategy": "",
                    "partnership_strategy": "",
                    "launch_strategy": ""
                },
                "growth_scale_strategy": {
                    "growth_phases": "",
                    "expansion_strategy": "",
                    "team_building": "",
                    "funding_strategy": "",
                    "exit_strategy": ""
                },
                "competitive_strategy": {
                    "competitive_advantages": "",
                    "market_positioning": "",
                    "defensibility": "",
                    "competitive_response": "",
                    "market_leadership": ""
                },
                "success_metrics_milestones": {
                    "key_performance_indicators": [],
                    "milestone_timeline": "",
                    "success_criteria": "",
                    "risk_mitigation": "",
                    "long_term_vision": ""
                }
            }
            
            # Parse Business Solution Summary section
            summary_match = re.search(r'## Business Solution Summary(.*?)(?=## |$)', text, re.DOTALL)
            if summary_match:
                summary_text = summary_match.group(1)
                result["business_solution_summary"]["solution_name"] = self._extract_field(summary_text, "Solution Name")
                result["business_solution_summary"]["business_concept"] = self._extract_field(summary_text, "Business Concept")
                result["business_solution_summary"]["target_market"] = self._extract_field(summary_text, "Target Market")
                result["business_solution_summary"]["market_opportunity"] = self._extract_field(summary_text, "Market Opportunity")
                result["business_solution_summary"]["business_model"] = self._extract_field(summary_text, "Business Model")
            
            # Parse Product Strategy section
            product_match = re.search(r'## Product Strategy(.*?)(?=## |$)', text, re.DOTALL)
            if product_match:
                product_text = product_match.group(1)
                result["product_strategy"]["core_features"] = self._extract_list(product_text, "Core Features")
                result["product_strategy"]["advanced_features"] = self._extract_list(product_text, "Advanced Features")
                result["product_strategy"]["user_experience"] = self._extract_field(product_text, "User Experience")
                result["product_strategy"]["design_philosophy"] = self._extract_field(product_text, "Design Philosophy")
                result["product_strategy"]["product_differentiation"] = self._extract_field(product_text, "Product Differentiation")
            
            # Parse Business Model & Pricing section
            pricing_match = re.search(r'## Business Model & Pricing(.*?)(?=## |$)', text, re.DOTALL)
            if pricing_match:
                pricing_text = pricing_match.group(1)
                result["business_model_pricing"]["revenue_streams"] = self._extract_field(pricing_text, "Revenue Streams")
                result["business_model_pricing"]["pricing_strategy"] = self._extract_field(pricing_text, "Pricing Strategy")
                result["business_model_pricing"]["cost_structure"] = self._extract_field(pricing_text, "Cost Structure")
                result["business_model_pricing"]["unit_economics"] = self._extract_field(pricing_text, "Unit Economics")
                result["business_model_pricing"]["profitability_timeline"] = self._extract_field(pricing_text, "Profitability Timeline")
            
            # Parse Go-to-Market Strategy section
            gtm_match = re.search(r'## Go-to-Market Strategy(.*?)(?=## |$)', text, re.DOTALL)
            if gtm_match:
                gtm_text = gtm_match.group(1)
                result["go_to_market_strategy"]["customer_acquisition"] = self._extract_field(gtm_text, "Customer Acquisition")
                result["go_to_market_strategy"]["marketing_strategy"] = self._extract_field(gtm_text, "Marketing Strategy")
                result["go_to_market_strategy"]["sales_strategy"] = self._extract_field(gtm_text, "Sales Strategy")
                result["go_to_market_strategy"]["partnership_strategy"] = self._extract_field(gtm_text, "Partnership Strategy")
                result["go_to_market_strategy"]["launch_strategy"] = self._extract_field(gtm_text, "Launch Strategy")
            
            # Parse Growth & Scale Strategy section
            growth_match = re.search(r'## Growth & Scale Strategy(.*?)(?=## |$)', text, re.DOTALL)
            if growth_match:
                growth_text = growth_match.group(1)
                result["growth_scale_strategy"]["growth_phases"] = self._extract_field(growth_text, "Growth Phases")
                result["growth_scale_strategy"]["expansion_strategy"] = self._extract_field(growth_text, "Expansion Strategy")
                result["growth_scale_strategy"]["team_building"] = self._extract_field(growth_text, "Team Building")
                result["growth_scale_strategy"]["funding_strategy"] = self._extract_field(growth_text, "Funding Strategy")
                result["growth_scale_strategy"]["exit_strategy"] = self._extract_field(growth_text, "Exit Strategy")
            
            # Parse Competitive Strategy section
            competitive_match = re.search(r'## Competitive Strategy(.*?)(?=## |$)', text, re.DOTALL)
            if competitive_match:
                competitive_text = competitive_match.group(1)
                result["competitive_strategy"]["competitive_advantages"] = self._extract_field(competitive_text, "Competitive Advantages")
                result["competitive_strategy"]["market_positioning"] = self._extract_field(competitive_text, "Market Positioning")
                result["competitive_strategy"]["defensibility"] = self._extract_field(competitive_text, "Defensibility")
                result["competitive_strategy"]["competitive_response"] = self._extract_field(competitive_text, "Competitive Response")
                result["competitive_strategy"]["market_leadership"] = self._extract_field(competitive_text, "Market Leadership")
            
            # Parse Success Metrics & Milestones section
            metrics_match = re.search(r'## Success Metrics & Milestones(.*?)(?=## |$)', text, re.DOTALL)
            if metrics_match:
                metrics_text = metrics_match.group(1)
                result["success_metrics_milestones"]["key_performance_indicators"] = self._extract_list(metrics_text, "Key Performance Indicators")
                result["success_metrics_milestones"]["milestone_timeline"] = self._extract_field(metrics_text, "Milestone Timeline")
                result["success_metrics_milestones"]["success_criteria"] = self._extract_field(metrics_text, "Success Criteria")
                result["success_metrics_milestones"]["risk_mitigation"] = self._extract_field(metrics_text, "Risk Mitigation")
                result["success_metrics_milestones"]["long_term_vision"] = self._extract_field(metrics_text, "Long-term Vision")
            
            self.logger.info("Successfully parsed structured act text response")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse structured act text: {e}")
            return {"error": f"Failed to parse structured act text: {e}"}
    
    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract a field value from structured text."""
        import re
        
        # Look for the field name followed by a colon and value
        pattern = rf'- {re.escape(field_name)}:\s*(.+?)(?=\n- |$)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def _extract_list(self, text: str, field_name: str) -> List[str]:
        """Extract a list field from structured text."""
        import re
        
        # Look for the field name followed by a colon and list items
        pattern = rf'- {re.escape(field_name)}:\s*(.+?)(?=\n- |$)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            list_text = match.group(1).strip()
            # Split by newlines and clean up
            items = [item.strip('- ').strip() for item in list_text.split('\n') if item.strip()]
            return [item for item in items if item]
        return []
    
    def _create_fallback_think_result(self, input_data: BuilderInput) -> Dict[str, Any]:
        """Create a fallback think result when LLM parsing fails."""
        return {
            "business_feasibility_analysis": {
                "primary_opportunity": "SaaS solution for identified market gaps",
                "market_viability": "High market demand for gap-filling solutions",
                "business_feasibility": "Viable business opportunity with clear path to profitability",
                "competitive_landscape": "Moderate competition with opportunities for differentiation"
            },
            "solution_concept": {
                "solution_name": "GapSolver Pro",
                "alternative_names": ["MarketGap Solutions", "GapBridge Platform"],
                "core_value_proposition": "Comprehensive solution addressing identified market gaps",
                "target_problem": "Market gaps identified through analysis",
                "target_customers": "SMBs and startups seeking gap-filling solutions"
            },
            "business_model_design": {
                "revenue_model": "Subscription-based SaaS",
                "pricing_strategy": "Freemium with premium tiers",
                "cost_structure": "Development, infrastructure, and marketing costs",
                "key_partnerships": "Technology partners and distribution channels",
                "customer_acquisition": "Digital marketing and content strategy"
            },
            "competitive_positioning": {
                "differentiation": "First-mover advantage in identified gaps",
                "competitive_advantages": "Comprehensive gap analysis and solution design",
                "market_positioning": "Premium solution for gap-filling needs",
                "barriers_to_entry": "Domain expertise and market knowledge"
            },
            "implementation_considerations": {
                "development_approach": "Agile development with iterative releases",
                "technology_considerations": "Modern web technologies and cloud infrastructure",
                "resource_requirements": "Development team, design resources, and marketing",
                "timeline_considerations": "6-12 months for MVP development",
                "risk_factors": "Market timing and competitive response"
            }
        }
    
    def _create_fallback_act_result(self, input_data: BuilderInput) -> Dict[str, Any]:
        """Create a fallback act result when LLM parsing fails."""
        return {
            "business_solution_summary": {
                "solution_name": "GapSolver Pro",
                "business_concept": "Comprehensive SaaS platform addressing market gaps",
                "target_market": "SMBs and startups",
                "market_opportunity": "Significant market opportunity in gap-filling solutions",
                "business_model": "Subscription-based SaaS with freemium model"
            },
            "product_strategy": {
                "core_features": ["Gap analysis tools", "Solution recommendations", "Market insights"],
                "advanced_features": ["AI-powered insights", "Custom solutions", "Enterprise features"],
                "user_experience": "Intuitive, user-friendly interface",
                "design_philosophy": "Clean, modern design focused on usability",
                "product_differentiation": "Comprehensive gap analysis and solution design"
            },
            "business_model_pricing": {
                "revenue_streams": "Subscription fees, premium features, enterprise licenses",
                "pricing_strategy": "Freemium with $29/month starter, $99/month professional",
                "cost_structure": "Development, infrastructure, marketing, and operations",
                "unit_economics": "Positive unit economics with scalable pricing",
                "profitability_timeline": "18-24 months to profitability"
            },
            "go_to_market_strategy": {
                "customer_acquisition": "Content marketing, SEO, and partnerships",
                "marketing_strategy": "Digital marketing focused on gap-filling solutions",
                "sales_strategy": "Self-service with enterprise sales team",
                "partnership_strategy": "Technology and distribution partnerships",
                "launch_strategy": "Soft launch with beta users, then public launch"
            },
            "growth_scale_strategy": {
                "growth_phases": "MVP launch, feature expansion, market expansion",
                "expansion_strategy": "New markets and additional gap categories",
                "team_building": "Development, marketing, and sales teams",
                "funding_strategy": "Bootstrap initially, then seek Series A funding",
                "exit_strategy": "Strategic acquisition or IPO in 5-7 years"
            },
            "competitive_strategy": {
                "competitive_advantages": "First-mover advantage and comprehensive solution",
                "market_positioning": "Premium solution for gap-filling needs",
                "defensibility": "Network effects and data moats",
                "competitive_response": "Continuous innovation and customer focus",
                "market_leadership": "Become the leading gap-filling solution platform"
            },
            "success_metrics_milestones": {
                "key_performance_indicators": ["Monthly recurring revenue", "Customer acquisition cost", "Customer lifetime value"],
                "milestone_timeline": "6 months MVP, 12 months growth, 24 months scale",
                "success_criteria": "Profitable growth with strong customer satisfaction",
                "risk_mitigation": "Diversified revenue streams and strong customer relationships",
                "long_term_vision": "Leading platform for market gap analysis and solutions"
            }
        }
