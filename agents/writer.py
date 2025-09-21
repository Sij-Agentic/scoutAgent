"""
Writer Agent - Generates comprehensive HTML reports from business analysis data.

This agent transforms outputs from all previous agents into professional,
animated HTML reports suitable for executive presentation.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..llm.utils import LLMAgentMixin
from ..memory.manifest_manager import ManifestManager
from .base import BaseAgent


class WriterInput:
    """Input data structure for Writer Agent."""
    
    def __init__(self, 
                 builder_output: Dict[str, Any],
                 gap_finder_output: Optional[Dict[str, Any]] = None,
                 scout_output: Optional[Dict[str, Any]] = None,
                 screener_output: Optional[Dict[str, Any]] = None,
                 validator_output: Optional[Dict[str, Any]] = None,
                 report_style: str = "professional",
                 include_animations: bool = True):
        self.builder_output = builder_output
        self.gap_finder_output = gap_finder_output
        self.scout_output = scout_output
        self.screener_output = screener_output
        self.validator_output = validator_output
        self.report_style = report_style
        self.include_animations = include_animations


class WriterAgent(BaseAgent, LLMAgentMixin):
    """Writer Agent for generating comprehensive HTML reports."""
    
    def __init__(self, agent_id: str = None):
        BaseAgent.__init__(self, name="writer", agent_id=agent_id)
        LLMAgentMixin.__init__(self, preferred_backend='deepseek')
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM with DeepSeek backend
        self.llm_manager = None
        self.preferred_backend = 'deepseek'
        self.task_backend_preferences = {
            'think_development': 'deepseek',
            'act_development': 'deepseek'
        }
    
    async def plan(self, input_data: WriterInput) -> Dict[str, Any]:
        """Plan phase is not used in WriterAgent."""
        return {"status": "skipped", "reason": "WriterAgent uses only think and act phases"}
    
    async def think(self, input_data: WriterInput) -> Dict[str, Any]:
        """Analyze all agent outputs and plan comprehensive HTML report structure."""
        try:
            self.logger.info("Starting writer think stage...")
            
            # Get run_id and initialize manifest manager
            run_id = getattr(self.state, 'run_id', None)
            if not run_id:
                self.logger.error("No run_id found in agent state")
                return {"error": "No run_id available"}
            
            manifest_path = Path("data/runs") / run_id / "run_manifest.json"
            manifest_manager = ManifestManager(manifest_path=manifest_path)
            
            # Load all agent outputs for comprehensive analysis
            all_agent_data = self._load_all_agent_outputs(manifest_manager)
            
            # Prepare synthesis data
            synthesis_data = {
                "builder_output": input_data.builder_output,
                "gap_finder_output": input_data.gap_finder_output,
                "scout_output": input_data.scout_output,
                "screener_output": input_data.screener_output,
                "validator_output": input_data.validator_output,
                "all_agent_data": all_agent_data,
                "report_style": input_data.report_style,
                "include_animations": input_data.include_animations
            }
            
            # Load think stage prompt
            prompt_content = self._load_think_prompt()
            if not prompt_content:
                self.logger.error("Failed to load think stage prompt")
                return {"error": "Failed to load think stage prompt"}
            
            # Append the synthesis data to the prompt
            prompt_content += f"\n\nData to Analyze:\n{json.dumps(synthesis_data, indent=2)}"
            
            # Generate report analysis using LLM mixin
            analysis_result = await self.llm_generate(prompt=prompt_content, task_type="think_development")
            
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
            return {"error": f"Think stage failed: {str(e)}"}
    
    async def act(self, input_data: WriterInput, think_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive HTML report with animations and interactive elements."""
        try:
            self.logger.info("Starting writer act stage...")
            
            # Get run_id and initialize manifest manager
            run_id = getattr(self.state, 'run_id', None)
            if not run_id:
                self.logger.error("No run_id found in agent state")
                return {"error": "No run_id available"}
            
            manifest_path = Path("data/runs") / run_id / "run_manifest.json"
            manifest_manager = ManifestManager(manifest_path=manifest_path)
            
            # Load think stage data if not provided
            if think_result is None:
                think_result = self._load_think_stage_data(manifest_manager)
            
            # Load all agent outputs for comprehensive report generation
            all_agent_data = self._load_all_agent_outputs(manifest_manager)
            
            # Prepare synthesis data
            synthesis_data = {
                "think_analysis": think_result,
                "builder_output": input_data.builder_output,
                "gap_finder_output": input_data.gap_finder_output,
                "scout_output": input_data.scout_output,
                "screener_output": input_data.screener_output,
                "validator_output": input_data.validator_output,
                "all_agent_data": all_agent_data,
                "report_style": input_data.report_style,
                "include_animations": input_data.include_animations
            }
            
            # Load act stage prompt
            prompt_content = self._load_act_prompt()
            if not prompt_content:
                self.logger.error("Failed to load act stage prompt")
                return {"error": "Failed to load act stage prompt"}
            
            # Append the synthesis data to the prompt
            prompt_content += f"\n\nData to Generate Report From:\n{json.dumps(synthesis_data, indent=2)}"
            
            # Generate HTML report using LLM mixin
            html_result = await self.llm_generate(prompt=prompt_content, task_type="act_development")
            
            # Process the HTML response
            if isinstance(html_result, str):
                self.logger.info(f"LLM response type: {type(html_result)}, length: {len(html_result)}")
                self.logger.info(f"LLM response preview: {html_result[:200]}...")
                
                # Save HTML file and create metadata
                report_metadata = self._save_html_report(html_result, run_id)
                
                # Create final result structure
                final_result = {
                    "html_report": report_metadata,
                    "report_analysis": think_result,
                    "generation_timestamp": datetime.now().isoformat(),
                    "success": True
                }
                
                self.logger.info(f"Successfully generated HTML report: {report_metadata.get('file_path', 'unknown')}")
            else:
                self.logger.info(f"LLM response is not a string, type: {type(html_result)}")
                final_result = {"error": "Failed to generate HTML report"}
            
            # Store the result to manifest
            self._store_act_output_to_manifest(manifest_manager, final_result)
            
            self.logger.info("Act stage completed successfully")
            return final_result
        
        except Exception as e:
            self.logger.error(f"Error in act stage: {e}")
            return {"error": f"Act stage failed: {str(e)}"}
    
    def _load_all_agent_outputs(self, manifest_manager: ManifestManager) -> Dict[str, Any]:
        """Load outputs from all previous agents for comprehensive analysis."""
        try:
            manifest = manifest_manager.get_manifest()
            all_data = {}
            
            # Load builder output
            builder_stage = manifest.get("stages", {}).get("builder_act", {})
            if builder_stage.get("data"):
                all_data["builder_act"] = builder_stage["data"]
            
            # Load gap finder outputs
            gap_finder_think = manifest.get("stages", {}).get("gap_finder_think", {})
            if gap_finder_think.get("data"):
                all_data["gap_finder_think"] = gap_finder_think["data"]
            
            gap_finder_act = manifest.get("stages", {}).get("gap_finder_act", {})
            if gap_finder_act.get("data"):
                all_data["gap_finder_act"] = gap_finder_act["data"]
            
            # Load scout outputs
            scout_act = manifest.get("stages", {}).get("scout_act", {})
            if scout_act.get("data"):
                all_data["scout_act"] = scout_act["data"]
            
            # Load screener outputs
            screener_act = manifest.get("stages", {}).get("screener_act", {})
            if screener_act.get("data"):
                all_data["screener_act"] = screener_act["data"]
            
            # Load validator outputs
            validator_act = manifest.get("stages", {}).get("validator_act", {})
            if validator_act.get("data"):
                all_data["validator_act"] = validator_act["data"]
            
            self.logger.info(f"Loaded data from {len(all_data)} agent stages")
            return all_data
            
        except Exception as e:
            self.logger.error(f"Error loading agent outputs: {e}")
            return {}
    
    def _load_think_stage_data(self, manifest_manager: ManifestManager) -> Dict[str, Any]:
        """Load think stage data from manifest."""
        try:
            manifest = manifest_manager.get_manifest()
            think_stage = manifest.get("stages", {}).get("writer_think", {})
            return think_stage.get("data", {})
        except Exception as e:
            self.logger.error(f"Error loading think stage data: {e}")
            return {}
    
    def _load_think_prompt(self) -> str:
        """Load think stage prompt template."""
        try:
            prompt_path = Path("scout_agent/prompts/writer_agent/think.prompt")
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
            prompt_path = Path("scout_agent/prompts/writer_agent/act.prompt")
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
            manifest["stages"]["writer_think"] = {
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
    
    def _store_act_output_to_manifest(self, manifest_manager: ManifestManager, report_result: Dict[str, Any]) -> None:
        """Store act stage output to manifest."""
        try:
            # Get the manifest
            manifest = manifest_manager.get_manifest()
            
            # Ensure stages section exists
            if "stages" not in manifest:
                manifest["stages"] = {}
            
            # Store the act stage output
            manifest["stages"]["writer_act"] = {
                "status": "completed",
                "updated_at": datetime.now().isoformat(),
                "data": report_result
            }
            
            # Save the manifest
            manifest_manager._save()
            self.logger.info("Stored act stage output to manifest")
            
        except Exception as e:
            self.logger.error(f"Error storing act output to manifest: {e}")
            raise e
    
    def _save_html_report(self, html_content: str, run_id: str) -> Dict[str, Any]:
        """Save HTML report to file and return metadata."""
        try:
            # Create reports directory
            reports_dir = Path("data/runs") / run_id / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"business_analysis_report_{timestamp}.html"
            file_path = reports_dir / filename
            
            # Save HTML content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Calculate file size
            file_size = file_path.stat().st_size
            
            # Create metadata
            metadata = {
                "file_path": str(file_path),
                "filename": filename,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "generation_timestamp": datetime.now().isoformat(),
                "content_length": len(html_content)
            }
            
            self.logger.info(f"Saved HTML report: {file_path} ({metadata['file_size_mb']} MB)")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error saving HTML report: {e}")
            return {"error": f"Failed to save HTML report: {str(e)}"}
    
    def _parse_structured_think_text(self, text: str) -> Dict[str, Any]:
        """Parse structured text response from LLM for think stage."""
        try:
            result = {
                "report_analysis": {},
                "content_planning": {},
                "design_strategy": {},
                "animation_plan": {},
                "technical_requirements": {},
                "content_priorities": {}
            }
            
            # Parse Report Analysis section
            analysis_match = re.search(r'## Report Analysis(.*?)(?=## |$)', text, re.DOTALL)
            if analysis_match:
                analysis_text = analysis_match.group(1)
                result["report_analysis"]["executive_summary"] = self._extract_field(analysis_text, "Executive Summary")
                result["report_analysis"]["primary_business_opportunity"] = self._extract_field(analysis_text, "Primary Business Opportunity")
                result["report_analysis"]["market_landscape"] = self._extract_field(analysis_text, "Market Landscape")
                result["report_analysis"]["strategic_recommendations"] = self._extract_list(analysis_text, "Strategic Recommendations")
            
            # Parse Content Planning section
            planning_match = re.search(r'## Content Planning(.*?)(?=## |$)', text, re.DOTALL)
            if planning_match:
                planning_text = planning_match.group(1)
                result["content_planning"]["report_sections"] = self._extract_list(planning_text, "Report Sections")
                result["content_planning"]["narrative_flow"] = self._extract_field(planning_text, "Narrative Flow")
                result["content_planning"]["key_metrics_to_highlight"] = self._extract_list(planning_text, "Key Metrics to Highlight")
                result["content_planning"]["data_visualizations_needed"] = self._extract_list(planning_text, "Data Visualizations Needed")
            
            # Parse Design Strategy section
            design_match = re.search(r'## Design Strategy(.*?)(?=## |$)', text, re.DOTALL)
            if design_match:
                design_text = design_match.group(1)
                result["design_strategy"]["visual_theme"] = self._extract_field(design_text, "Visual Theme")
                result["design_strategy"]["typography"] = self._extract_field(design_text, "Typography")
                result["design_strategy"]["layout_approach"] = self._extract_field(design_text, "Layout Approach")
                result["design_strategy"]["interactive_elements"] = self._extract_list(design_text, "Interactive Elements")
            
            # Parse Animation Plan section
            animation_match = re.search(r'## Animation Plan(.*?)(?=## |$)', text, re.DOTALL)
            if animation_match:
                animation_text = animation_match.group(1)
                result["animation_plan"]["hero_section"] = self._extract_field(animation_text, "Hero Section")
                result["animation_plan"]["scroll_animations"] = self._extract_field(animation_text, "Scroll Animations")
                result["animation_plan"]["data_visualizations"] = self._extract_field(animation_text, "Data Visualizations")
                result["animation_plan"]["interactive_features"] = self._extract_field(animation_text, "Interactive Features")
                result["animation_plan"]["loading_transitions"] = self._extract_field(animation_text, "Loading Transitions")
            
            # Parse Technical Requirements section
            tech_match = re.search(r'## Technical Requirements(.*?)(?=## |$)', text, re.DOTALL)
            if tech_match:
                tech_text = tech_match.group(1)
                result["technical_requirements"]["html_structure"] = self._extract_field(tech_text, "HTML Structure")
                result["technical_requirements"]["css_framework"] = self._extract_field(tech_text, "CSS Framework")
                result["technical_requirements"]["animation_performance"] = self._extract_field(tech_text, "Animation Performance")
                result["technical_requirements"]["browser_compatibility"] = self._extract_field(tech_text, "Browser Compatibility")
            
            # Parse Content Priorities section
            priorities_match = re.search(r'## Content Priorities(.*?)(?=## |$)', text, re.DOTALL)
            if priorities_match:
                priorities_text = priorities_match.group(1)
                result["content_priorities"]["most_important_sections"] = self._extract_list(priorities_text, "Most Important Sections")
                result["content_priorities"]["supporting_data"] = self._extract_list(priorities_text, "Supporting Data")
                result["content_priorities"]["call_to_actions"] = self._extract_list(priorities_text, "Call-to-Actions")
                result["content_priorities"]["professional_presentation"] = self._extract_field(priorities_text, "Professional Presentation")
            
            self.logger.info("Successfully parsed structured think text response")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse structured think text: {e}")
            return {"error": f"Failed to parse structured think text: {e}"}
    
    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract a field value from structured text."""
        # Look for the field name followed by a colon and value (multiple formats)
        patterns = [
            rf'- \*?{re.escape(field_name)}\*?:\s*(.+?)(?=\n- |$)',
            rf'\*\*{re.escape(field_name)}\*\*:\s*(.+?)(?=\n\*\*|\n##|$)',
            rf'{re.escape(field_name)}:\s*(.+?)(?=\n\*\*|\n##|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""
    
    def _extract_list(self, text: str, field_name: str) -> List[str]:
        """Extract a list field from structured text."""
        # Look for the field name followed by a colon and list items (multiple formats)
        patterns = [
            rf'- \*?{re.escape(field_name)}\*?:\s*(.+?)(?=\n- |$)',
            rf'\*\*{re.escape(field_name)}\*\*:\s*(.+?)(?=\n\*\*|\n##|$)',
            rf'{re.escape(field_name)}:\s*(.+?)(?=\n\*\*|\n##|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                list_text = match.group(1).strip()
                # Split by newlines and clean up
                items = [item.strip('- ').strip() for item in list_text.split('\n') if item.strip()]
                return [item for item in items if item]
        return []
    
    def _create_fallback_think_result(self, input_data: WriterInput) -> Dict[str, Any]:
        """Create a fallback think result when LLM parsing fails."""
        return {
            "report_analysis": {
                "executive_summary": "Comprehensive business analysis report based on market research and gap analysis",
                "primary_business_opportunity": "SaaS solution identified through market gap analysis",
                "market_landscape": "Market analysis reveals significant opportunities for new solutions",
                "strategic_recommendations": ["Develop identified SaaS solution", "Implement go-to-market strategy", "Focus on target market segments"]
            },
            "content_planning": {
                "report_sections": ["Executive Summary", "Market Analysis", "Business Solution", "Implementation Plan", "Financial Projections"],
                "narrative_flow": "Logical progression from market analysis to business solution and implementation",
                "key_metrics_to_highlight": ["Market size", "Revenue projections", "Customer acquisition costs", "Growth metrics"],
                "data_visualizations_needed": ["Market size charts", "Revenue projections", "Timeline visualizations", "Competitive analysis"]
            },
            "design_strategy": {
                "visual_theme": "Professional and modern design with clean typography",
                "typography": "Clear, readable fonts with proper hierarchy",
                "layout_approach": "Responsive grid layout with clear sections",
                "interactive_elements": ["Hover effects", "Smooth scrolling", "Interactive charts"]
            },
            "animation_plan": {
                "hero_section": "Fade-in animations for key metrics",
                "scroll_animations": "Elements animate as they enter viewport",
                "data_visualizations": "Charts animate on load with smooth transitions",
                "interactive_features": "Hover effects and click interactions",
                "loading_transitions": "Smooth transitions between sections"
            },
            "technical_requirements": {
                "html_structure": "Semantic HTML5 with proper accessibility",
                "css_framework": "Custom CSS with responsive design",
                "animation_performance": "Optimized animations for smooth performance",
                "browser_compatibility": "Cross-browser support for modern browsers"
            },
            "content_priorities": {
                "most_important_sections": ["Executive Summary", "Business Solution", "Financial Projections"],
                "supporting_data": ["Market analysis", "Competitive landscape", "Implementation details"],
                "call_to_actions": ["Next steps", "Implementation recommendations", "Contact information"],
                "professional_presentation": "Executive-level quality suitable for decision-making"
            }
        }
