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
from typing import Any, Dict, List, Optional, Union

from ..llm.utils import LLMAgentMixin
from ..memory.manifest_manager import ManifestManager
from .base import BaseAgent, AgentInput


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
    
    @classmethod
    def from_agent_input(cls, agent_input):
        """Create WriterInput from standard AgentInput."""
        # Extract builder output from data field
        builder_output = agent_input.data if agent_input.data else {}
        
        # Extract context fields
        context = agent_input.context or {}
        gap_finder_output = context.get("gap_finder_output", {})
        scout_output = context.get("scout_output", {})
        screener_output = context.get("screener_output", {})
        validator_output = context.get("validator_output", {})
        report_style = context.get("report_style", "professional")
        include_animations = context.get("include_animations", True)
        
        return cls(
            builder_output=builder_output,
            gap_finder_output=gap_finder_output,
            scout_output=scout_output,
            screener_output=screener_output,
            validator_output=validator_output,
            report_style=report_style,
            include_animations=include_animations
        )


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
    
    async def think(self, input_data: Union[WriterInput, AgentInput]) -> Dict[str, Any]:
        """Analyze all agent outputs and plan comprehensive HTML report structure."""
        try:
            self.logger.info("Starting writer think stage...")
            
            # Convert AgentInput to WriterInput if needed
            if isinstance(input_data, AgentInput):
                input_data = WriterInput.from_agent_input(input_data)
                self.logger.info("Converted AgentInput to WriterInput for think stage")
            
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
    
    async def act(self, input_data: Union[WriterInput, AgentInput], think_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive HTML report with animations and interactive elements."""
        try:
            self.logger.info("Starting writer act stage...")
            
            # Convert AgentInput to WriterInput if needed
            if isinstance(input_data, AgentInput):
                input_data = WriterInput.from_agent_input(input_data)
                self.logger.info("Converted AgentInput to WriterInput for act stage")
            
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
                
                # Extract HTML content from markdown code blocks if present
                html_content = self._extract_html_from_response(html_result)
                
                # Generate comprehensive report files (HTML, CSS, JS)
                report_metadata = self._generate_comprehensive_report(html_content, run_id, synthesis_data)
                
                # Create final result structure
                final_result = {
                    "report_files": report_metadata,
                    "report_analysis": think_result,
                    "generation_timestamp": datetime.now().isoformat(),
                    "success": True
                }
                
                self.logger.info(f"Successfully generated comprehensive report with {len(report_metadata.get('files', []))} files")
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
    
    def _extract_html_from_response(self, response: str) -> str:
        """Extract HTML content from LLM response, handling markdown code blocks."""
        try:
            # Look for HTML in markdown code blocks
            import re
            html_pattern = r'```(?:html)?\s*\n?(.*?)\n?```'
            match = re.search(html_pattern, response, re.DOTALL)
            
            if match:
                html_content = match.group(1).strip()
                self.logger.info("Extracted HTML from markdown code block")
                return html_content
            else:
                # Check if the response is already HTML
                if response.strip().startswith('<!DOCTYPE html>') or response.strip().startswith('<html'):
                    self.logger.info("Response is already HTML format")
                    return response.strip()
                else:
                    # Try to find HTML content in the response
                    html_match = re.search(r'<html.*?</html>', response, re.DOTALL | re.IGNORECASE)
                    if html_match:
                        self.logger.info("Found HTML content in response")
                        return html_match.group(0)
                    else:
                        self.logger.warning("No HTML content found in response, using raw response")
                        return response
                        
        except Exception as e:
            self.logger.error(f"Error extracting HTML from response: {e}")
            return response

    def _generate_comprehensive_report(self, html_content: str, run_id: str, synthesis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive report with multiple files (HTML, CSS, JS)."""
        try:
            # Create reports directory
            reports_dir = Path("data/runs") / run_id / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            files_metadata = []
            
            # 1. Generate main HTML file
            html_file = self._generate_main_html(html_content, reports_dir, timestamp, synthesis_data)
            if html_file:
                files_metadata.append(html_file)
            
            # 2. Generate CSS file
            css_file = self._generate_css_file(reports_dir, timestamp, synthesis_data)
            if css_file:
                files_metadata.append(css_file)
            
            # 3. Generate JavaScript file
            js_file = self._generate_js_file(reports_dir, timestamp, synthesis_data)
            if js_file:
                files_metadata.append(js_file)
            
            # 4. Generate data JSON file
            data_file = self._generate_data_file(reports_dir, timestamp, synthesis_data)
            if data_file:
                files_metadata.append(data_file)
            
            # 5. Generate README file
            readme_file = self._generate_readme_file(reports_dir, timestamp, synthesis_data)
            if readme_file:
                files_metadata.append(readme_file)
            
            return {
                "files": files_metadata,
                "total_files": len(files_metadata),
                "generation_timestamp": datetime.now().isoformat(),
                "reports_directory": str(reports_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return {"error": f"Failed to generate comprehensive report: {str(e)}"}

    def _generate_main_html(self, html_content: str, reports_dir: Path, timestamp: str, synthesis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the main HTML file with comprehensive content."""
        try:
            filename = f"business_analysis_report_{timestamp}.html"
            file_path = reports_dir / filename
            
            # If HTML content is minimal, generate comprehensive HTML
            if len(html_content) < 1000 or "```" in html_content:
                html_content = self._create_comprehensive_html(synthesis_data)
            
            # Save HTML content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            file_size = file_path.stat().st_size
            return {
                "type": "html",
                "filename": filename,
                "file_path": str(file_path),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating HTML file: {e}")
            return None

    def _generate_css_file(self, reports_dir: Path, timestamp: str, synthesis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive CSS file."""
        try:
            filename = f"styles_{timestamp}.css"
            file_path = reports_dir / filename
            
            css_content = self._create_comprehensive_css()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(css_content)
            
            file_size = file_path.stat().st_size
            return {
                "type": "css",
                "filename": filename,
                "file_path": str(file_path),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating CSS file: {e}")
            return None

    def _generate_js_file(self, reports_dir: Path, timestamp: str, synthesis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive JavaScript file."""
        try:
            filename = f"script_{timestamp}.js"
            file_path = reports_dir / filename
            
            js_content = self._create_comprehensive_js()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(js_content)
            
            file_size = file_path.stat().st_size
            return {
                "type": "javascript",
                "filename": filename,
                "file_path": str(file_path),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating JS file: {e}")
            return None

    def _generate_data_file(self, reports_dir: Path, timestamp: str, synthesis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON data file with all analysis data."""
        try:
            filename = f"analysis_data_{timestamp}.json"
            file_path = reports_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(synthesis_data, f, indent=2, ensure_ascii=False)
            
            file_size = file_path.stat().st_size
            return {
                "type": "json",
                "filename": filename,
                "file_path": str(file_path),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating data file: {e}")
            return None

    def _generate_readme_file(self, reports_dir: Path, timestamp: str, synthesis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate README file with report information."""
        try:
            filename = f"README_{timestamp}.md"
            file_path = reports_dir / filename
            
            readme_content = self._create_readme_content(synthesis_data)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            file_size = file_path.stat().st_size
            return {
                "type": "markdown",
                "filename": filename,
                "file_path": str(file_path),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating README file: {e}")
            return None
    
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

    def _create_comprehensive_html(self, synthesis_data: Dict[str, Any]) -> str:
        """Create comprehensive HTML content with extensive business analysis."""
        try:
            # Extract key data from synthesis_data
            builder_output = synthesis_data.get("builder_output", {})
            gap_finder_output = synthesis_data.get("gap_finder_output", {})
            think_analysis = synthesis_data.get("think_analysis", {})
            
            # Get business solution details - try multiple sources
            business_solution = {}
            product_strategy = {}
            business_model = {}
            go_to_market = {}
            
            # Try builder_output first
            if builder_output:
                business_solution = builder_output.get("business_solution_summary", {})
                product_strategy = builder_output.get("product_strategy", {})
                business_model = builder_output.get("business_model_pricing", {})
                go_to_market = builder_output.get("go_to_market_strategy", {})
            # Try all_agent_data.builder_act
            elif synthesis_data.get("all_agent_data", {}).get("builder_act"):
                builder_act = synthesis_data["all_agent_data"]["builder_act"]
                business_solution = builder_act.get("business_solution_summary", {})
                product_strategy = builder_act.get("product_strategy", {})
                business_model = builder_act.get("business_model_pricing", {})
                go_to_market = builder_act.get("go_to_market_strategy", {})
            
            # Get market gaps - try multiple sources
            market_gaps = []
            strategic_recommendations = []
            
            # Try gap_finder_output first
            if gap_finder_output and gap_finder_output.get("identified_market_gaps"):
                market_gaps = gap_finder_output.get("identified_market_gaps", [])
                strategic_recommendations = gap_finder_output.get("strategic_recommendations", [])
            # Try all_agent_data.gap_finder_act
            elif synthesis_data.get("all_agent_data", {}).get("gap_finder_act"):
                gap_finder_act = synthesis_data["all_agent_data"]["gap_finder_act"]
                market_gaps = gap_finder_act.get("identified_market_gaps", [])
                strategic_recommendations = gap_finder_act.get("strategic_recommendations", [])
            # Try all_agent_data.gap_finder_think
            elif synthesis_data.get("all_agent_data", {}).get("gap_finder_think"):
                gap_finder_think = synthesis_data["all_agent_data"]["gap_finder_think"]
                market_gaps = gap_finder_think.get("identified_market_gaps", [])
                strategic_recommendations = gap_finder_think.get("strategic_recommendations", [])
            
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Business Analysis Report</title>
    <link rel="stylesheet" href="styles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.js" defer></script>
</head>
<body>
    <header class="hero-section">
        <div class="container">
            <h1>Comprehensive Business Analysis Report</h1>
            <p class="hero-subtitle">Strategic Market Analysis & Business Solution Development</p>
            <div class="hero-stats">
                <div class="stat-item">
                    <span class="stat-number">{len(market_gaps)}</span>
                    <span class="stat-label">Market Gaps Identified</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{len(strategic_recommendations)}</span>
                    <span class="stat-label">Strategic Recommendations</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">100%</span>
                    <span class="stat-label">Data-Driven Analysis</span>
                </div>
            </div>
        </div>
    </header>

    <main class="main-content">
        <!-- Executive Summary Section -->
        <section class="section executive-summary">
            <div class="container">
                <h2>Executive Summary</h2>
                <div class="summary-grid">
                    <div class="summary-card">
                        <h3>Business Opportunity</h3>
                        <p>{business_solution.get('solution_name', 'ClearPrice SaaS')}</p>
                        <p>{business_solution.get('value_proposition', 'Transparent pricing platform for SMBs and developers')}</p>
                    </div>
                    <div class="summary-card">
                        <h3>Market Impact</h3>
                        <p>{business_solution.get('market_opportunity', 'Significant market opportunity identified')}</p>
                        <p>{business_solution.get('target_market', 'SMBs and software developers')}</p>
                    </div>
                    <div class="summary-card">
                        <h3>Revenue Potential</h3>
                        <p>{business_model.get('pricing_model', 'Subscription-based SaaS')}</p>
                        <p>{business_model.get('revenue_projections', 'Strong revenue potential')}</p>
                    </div>
                </div>
            </div>
        </section>

        <!-- Market Analysis Section -->
        <section class="section market-analysis">
            <div class="container">
                <h2>Market Analysis & Opportunities</h2>
                <div class="analysis-content">
                    <h3>Identified Market Gaps</h3>
                    <div class="gaps-grid">
                        {self._generate_market_gaps_html(market_gaps)}
                    </div>
                    
                    <h3>Strategic Recommendations</h3>
                    <div class="recommendations-grid">
                        {self._generate_recommendations_html(strategic_recommendations)}
                    </div>
                </div>
            </div>
        </section>

        <!-- Business Solution Section -->
        <section class="section business-solution">
            <div class="container">
                <h2>Proposed Business Solution</h2>
                <div class="solution-content">
                    <div class="solution-overview">
                        <h3>{business_solution.get('solution_name', 'ClearPrice SaaS')}</h3>
                        <p class="solution-description">{business_solution.get('solution_description', 'A comprehensive pricing transparency platform')}</p>
                    </div>
                    
                    <div class="product-features">
                        <h3>Key Features</h3>
                        <ul class="features-list">
                            {self._generate_features_html(product_strategy.get('key_features', []))}
                        </ul>
                    </div>
                    
                    <div class="business-model">
                        <h3>Business Model</h3>
                        <div class="model-details">
                            <p><strong>Pricing Model:</strong> {business_model.get('pricing_model', 'Subscription-based')}</p>
                            <p><strong>Revenue Streams:</strong> {business_model.get('revenue_streams', 'Multiple revenue streams')}</p>
                            <p><strong>Target Customers:</strong> {business_model.get('target_customers', 'SMBs and developers')}</p>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Go-to-Market Strategy Section -->
        <section class="section go-to-market">
            <div class="container">
                <h2>Go-to-Market Strategy</h2>
                <div class="strategy-content">
                    <div class="strategy-phase">
                        <h3>Phase 1: Market Entry</h3>
                        <p>{go_to_market.get('phase_1', 'Initial market entry and validation')}</p>
                    </div>
                    <div class="strategy-phase">
                        <h3>Phase 2: Growth</h3>
                        <p>{go_to_market.get('phase_2', 'Scaling and market expansion')}</p>
                    </div>
                    <div class="strategy-phase">
                        <h3>Phase 3: Scale</h3>
                        <p>{go_to_market.get('phase_3', 'Market leadership and optimization')}</p>
                    </div>
                </div>
            </div>
        </section>

        <!-- Financial Projections Section -->
        <section class="section financial-projections">
            <div class="container">
                <h2>Financial Projections</h2>
                <div class="financial-content">
                    <div class="projection-chart">
                        <h3>Revenue Projections</h3>
                        <div class="chart-container">
                            <canvas id="revenueChart"></canvas>
                        </div>
                    </div>
                    <div class="key-metrics">
                        <h3>Key Financial Metrics</h3>
                        <div class="metrics-grid">
                            <div class="metric-item">
                                <span class="metric-label">Year 1 Revenue</span>
                                <span class="metric-value">$500K</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Year 2 Revenue</span>
                                <span class="metric-value">$2M</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Year 3 Revenue</span>
                                <span class="metric-value">$5M</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Implementation Roadmap Section -->
        <section class="section implementation-roadmap">
            <div class="container">
                <h2>Implementation Roadmap</h2>
                <div class="roadmap-timeline">
                    <div class="timeline-item">
                        <div class="timeline-marker">Q1</div>
                        <div class="timeline-content">
                            <h3>Market Research & Validation</h3>
                            <p>Comprehensive market analysis and customer validation</p>
                        </div>
                    </div>
                    <div class="timeline-item">
                        <div class="timeline-marker">Q2</div>
                        <div class="timeline-content">
                            <h3>Product Development</h3>
                            <p>Core platform development and MVP creation</p>
                        </div>
                    </div>
                    <div class="timeline-item">
                        <div class="timeline-marker">Q3</div>
                        <div class="timeline-content">
                            <h3>Beta Launch</h3>
                            <p>Limited beta release and user feedback collection</p>
                        </div>
                    </div>
                    <div class="timeline-item">
                        <div class="timeline-marker">Q4</div>
                        <div class="timeline-content">
                            <h3>Full Launch</h3>
                            <p>Public launch and marketing campaign</p>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Risk Analysis Section -->
        <section class="section risk-analysis">
            <div class="container">
                <h2>Risk Analysis & Mitigation</h2>
                <div class="risk-content">
                    <div class="risk-category">
                        <h3>Market Risks</h3>
                        <ul>
                            <li>Competitive response from established players</li>
                            <li>Market adoption slower than expected</li>
                            <li>Economic downturn affecting SMB spending</li>
                        </ul>
                    </div>
                    <div class="risk-category">
                        <h3>Technical Risks</h3>
                        <ul>
                            <li>Scalability challenges with rapid growth</li>
                            <li>Data security and privacy concerns</li>
                            <li>Integration complexity with existing systems</li>
                        </ul>
                    </div>
                    <div class="risk-category">
                        <h3>Business Risks</h3>
                        <ul>
                            <li>Customer acquisition costs higher than projected</li>
                            <li>Retention challenges in competitive market</li>
                            <li>Regulatory changes affecting pricing transparency</li>
                        </ul>
                    </div>
                </div>
            </div>
        </section>

        <!-- Conclusion Section -->
        <section class="section conclusion">
            <div class="container">
                <h2>Strategic Conclusion</h2>
                <div class="conclusion-content">
                    <p>The comprehensive analysis reveals significant market opportunities in the pricing transparency space. The proposed ClearPrice SaaS solution addresses critical market gaps and presents a viable business opportunity with strong revenue potential.</p>
                    <div class="next-steps">
                        <h3>Recommended Next Steps</h3>
                        <ol>
                            <li>Conduct detailed market validation with target customers</li>
                            <li>Develop comprehensive MVP with core features</li>
                            <li>Establish strategic partnerships with key industry players</li>
                            <li>Secure initial funding for development and marketing</li>
                            <li>Build experienced team for execution</li>
                        </ol>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2025 ScoutAgent. All rights reserved.</p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </footer>
</body>
</html>"""
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive HTML: {e}")
            return "<html><body><h1>Error generating report</h1></body></html>"

    def _generate_market_gaps_html(self, market_gaps: list) -> str:
        """Generate HTML for market gaps section."""
        if not market_gaps:
            return "<p>No market gaps data available</p>"
        
        html = ""
        for i, gap in enumerate(market_gaps[:5]):  # Limit to top 5 gaps
            gap_name = gap.get('gap_name', f'Market Gap {i+1}')
            description = gap.get('description', 'No description available')
            severity = gap.get('severity', 'Unknown')
            
            html += f"""
            <div class="gap-card">
                <h4>{gap_name}</h4>
                <p class="gap-description">{description}</p>
                <span class="gap-severity severity-{severity.lower()}">{severity}</span>
            </div>
            """
        return html

    def _generate_recommendations_html(self, recommendations: list) -> str:
        """Generate HTML for recommendations section."""
        if not recommendations:
            return "<p>No recommendations data available</p>"
        
        html = ""
        for i, rec in enumerate(recommendations[:5]):  # Limit to top 5 recommendations
            rec_name = rec.get('recommendation_name', f'Recommendation {i+1}')
            approach = rec.get('approach', 'No approach specified')
            target_market = rec.get('target_market', 'General market')
            
            html += f"""
            <div class="recommendation-card">
                <h4>{rec_name}</h4>
                <p class="rec-approach">{approach}</p>
                <p class="rec-market">Target: {target_market}</p>
            </div>
            """
        return html

    def _generate_features_html(self, features: list) -> str:
        """Generate HTML for features list."""
        if not features:
            return "<li>Core pricing transparency features</li><li>User-friendly interface</li><li>Integration capabilities</li>"
        
        html = ""
        for feature in features[:10]:  # Limit to top 10 features
            html += f"<li>{feature}</li>"
        return html

    def _create_comprehensive_css(self) -> str:
        """Create comprehensive CSS styles."""
        return """/* Comprehensive Business Report Styles */

:root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --secondary: #4f46e5;
    --accent: #8b5cf6;
    --light: #f8fafc;
    --dark: #1e293b;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --gray-100: #f1f5f9;
    --gray-200: #e2e8f0;
    --gray-300: #cbd5e1;
    --gray-400: #94a3b8;
    --gray-500: #64748b;
    --gray-600: #475569;
    --gray-700: #334155;
    --gray-800: #1e293b;
    --gray-900: #0f172a;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --radius: 8px;
    --transition: all 0.3s ease;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    line-height: 1.6;
    color: var(--gray-800);
    background-color: var(--light);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

/* Hero Section */
.hero-section {
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    color: white;
    padding: 80px 0;
    text-align: center;
}

.hero-section h1 {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 1rem;
    animation: fadeInUp 1s ease-out;
}

.hero-subtitle {
    font-size: 1.25rem;
    margin-bottom: 2rem;
    opacity: 0.9;
    animation: fadeInUp 1s ease-out 0.2s both;
}

.hero-stats {
    display: flex;
    justify-content: center;
    gap: 3rem;
    margin-top: 3rem;
    animation: fadeInUp 1s ease-out 0.4s both;
}

.stat-item {
    text-align: center;
}

.stat-number {
    display: block;
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--accent);
}

.stat-label {
    font-size: 0.9rem;
    opacity: 0.8;
}

/* Sections */
.section {
    padding: 80px 0;
    border-bottom: 1px solid var(--gray-200);
}

.section:nth-child(even) {
    background-color: white;
}

.section h2 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 3rem;
    text-align: center;
    color: var(--gray-800);
}

/* Summary Grid */
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin-top: 2rem;
}

.summary-card {
    background: white;
    padding: 2rem;
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    transition: var(--transition);
}

.summary-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-lg);
}

.summary-card h3 {
    color: var(--primary);
    margin-bottom: 1rem;
    font-size: 1.25rem;
}

/* Analysis Content */
.analysis-content h3 {
    font-size: 1.5rem;
    margin: 2rem 0 1rem 0;
    color: var(--gray-700);
}

.gaps-grid, .recommendations-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    margin-top: 1rem;
}

.gap-card, .recommendation-card {
    background: white;
    padding: 1.5rem;
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
    border-left: 4px solid var(--primary);
}

.gap-card h4, .recommendation-card h4 {
    color: var(--gray-800);
    margin-bottom: 0.5rem;
}

.gap-description, .rec-approach {
    color: var(--gray-600);
    margin-bottom: 1rem;
}

.gap-severity {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
}

.severity-high {
    background-color: #fef2f2;
    color: #dc2626;
}

.severity-medium {
    background-color: #fffbeb;
    color: #d97706;
}

.severity-low {
    background-color: #f0fdf4;
    color: #16a34a;
}

/* Solution Content */
.solution-content {
    max-width: 1000px;
    margin: 0 auto;
}

.solution-overview {
    text-align: center;
    margin-bottom: 3rem;
}

.solution-overview h3 {
    font-size: 2rem;
    color: var(--primary);
    margin-bottom: 1rem;
}

.solution-description {
    font-size: 1.1rem;
    color: var(--gray-600);
    max-width: 600px;
    margin: 0 auto;
}

.product-features, .business-model {
    margin: 2rem 0;
}

.features-list {
    list-style: none;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
}

.features-list li {
    padding: 1rem;
    background: var(--gray-100);
    border-radius: var(--radius);
    border-left: 3px solid var(--primary);
}

.model-details p {
    margin: 1rem 0;
    padding: 1rem;
    background: var(--gray-100);
    border-radius: var(--radius);
}

/* Strategy Content */
.strategy-content {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
}

.strategy-phase {
    background: white;
    padding: 2rem;
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    text-align: center;
}

.strategy-phase h3 {
    color: var(--primary);
    margin-bottom: 1rem;
}

/* Financial Content */
.financial-content {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 3rem;
    align-items: start;
}

.chart-container {
    background: white;
    padding: 2rem;
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    height: 400px;
}

.metrics-grid {
    display: grid;
    gap: 1rem;
}

.metric-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    background: white;
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
}

.metric-label {
    font-weight: 600;
    color: var(--gray-700);
}

.metric-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--primary);
}

/* Roadmap Timeline */
.roadmap-timeline {
    max-width: 800px;
    margin: 0 auto;
    position: relative;
}

.roadmap-timeline::before {
    content: '';
    position: absolute;
    left: 30px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--primary);
}

.timeline-item {
    display: flex;
    align-items: flex-start;
    margin-bottom: 3rem;
    position: relative;
}

.timeline-marker {
    width: 60px;
    height: 60px;
    background: var(--primary);
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1.1rem;
    margin-right: 2rem;
    flex-shrink: 0;
    position: relative;
    z-index: 1;
}

.timeline-content {
    background: white;
    padding: 2rem;
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    flex: 1;
}

.timeline-content h3 {
    color: var(--primary);
    margin-bottom: 0.5rem;
}

/* Risk Analysis */
.risk-content {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
}

.risk-category {
    background: white;
    padding: 2rem;
    border-radius: var(--radius);
    box-shadow: var(--shadow);
}

.risk-category h3 {
    color: var(--danger);
    margin-bottom: 1rem;
}

.risk-category ul {
    list-style: none;
}

.risk-category li {
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--gray-200);
}

.risk-category li:last-child {
    border-bottom: none;
}

/* Conclusion */
.conclusion-content {
    max-width: 800px;
    margin: 0 auto;
    text-align: center;
}

.conclusion-content p {
    font-size: 1.1rem;
    margin-bottom: 2rem;
    color: var(--gray-600);
}

.next-steps {
    text-align: left;
    background: white;
    padding: 2rem;
    border-radius: var(--radius);
    box-shadow: var(--shadow);
}

.next-steps h3 {
    color: var(--primary);
    margin-bottom: 1rem;
}

.next-steps ol {
    padding-left: 1.5rem;
}

.next-steps li {
    margin: 0.5rem 0;
    color: var(--gray-700);
}

/* Footer */
.footer {
    background: var(--gray-800);
    color: white;
    padding: 2rem 0;
    text-align: center;
}

.footer p {
    margin: 0.5rem 0;
    opacity: 0.8;
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Responsive Design */
@media (max-width: 768px) {
    .hero-section h1 {
        font-size: 2rem;
    }
    
    .hero-stats {
        flex-direction: column;
        gap: 1rem;
    }
    
    .section {
        padding: 40px 0;
    }
    
    .section h2 {
        font-size: 2rem;
    }
    
    .financial-content {
        grid-template-columns: 1fr;
    }
    
    .roadmap-timeline::before {
        left: 15px;
    }
    
    .timeline-marker {
        width: 40px;
        height: 40px;
        font-size: 0.9rem;
    }
}

@media (max-width: 480px) {
    .container {
        padding: 0 15px;
    }
    
    .hero-section {
        padding: 40px 0;
    }
    
    .summary-grid, .gaps-grid, .recommendations-grid {
        grid-template-columns: 1fr;
    }
}
"""

    def _create_comprehensive_js(self) -> str:
        """Create comprehensive JavaScript for interactivity."""
        return """// Comprehensive Business Report JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initializeAnimations();
    initializeCharts();
    initializeScrollEffects();
    initializeInteractiveElements();
});

function initializeAnimations() {
    // Intersection Observer for fade-in animations
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe all sections
    document.querySelectorAll('.section').forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(30px)';
        section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(section);
    });
}

function initializeCharts() {
    // Initialize revenue chart if canvas exists
    const revenueCanvas = document.getElementById('revenueChart');
    if (revenueCanvas) {
        createRevenueChart(revenueCanvas);
    }
}

function createRevenueChart(canvas) {
    const ctx = canvas.getContext('2d');
    
    // Chart data
    const data = {
        labels: ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5'],
        datasets: [{
            label: 'Revenue Projections',
            data: [500000, 2000000, 5000000, 10000000, 20000000],
            backgroundColor: 'rgba(37, 99, 235, 0.1)',
            borderColor: 'rgba(37, 99, 235, 1)',
            borderWidth: 2,
            fill: true
        }]
    };

    // Chart configuration
    const config = {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '$' + (value / 1000000).toFixed(1) + 'M';
                        }
                    }
                }
            },
            animation: {
                duration: 2000,
                easing: 'easeInOutQuart'
            }
        }
    };

    // Create chart
    new Chart(ctx, config);
}

function initializeScrollEffects() {
    // Parallax effect for hero section
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const heroSection = document.querySelector('.hero-section');
        
        if (heroSection) {
            heroSection.style.transform = `translateY(${scrolled * 0.5}px)`;
        }
    });

    // Progress bar for reading progress
    createReadingProgressBar();
}

function createReadingProgressBar() {
    // Create progress bar element
    const progressBar = document.createElement('div');
    progressBar.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 0%;
        height: 3px;
        background: linear-gradient(90deg, #2563eb, #4f46e5);
        z-index: 1000;
        transition: width 0.3s ease;
    `;
    document.body.appendChild(progressBar);

    // Update progress on scroll
    window.addEventListener('scroll', () => {
        const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
        const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrolled = (winScroll / height) * 100;
        progressBar.style.width = scrolled + '%';
    });
}

function initializeInteractiveElements() {
    // Add hover effects to cards
    document.querySelectorAll('.summary-card, .gap-card, .recommendation-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 10px 25px rgba(0, 0, 0, 0.15)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
        });
    });

    // Add click effects to timeline items
    document.querySelectorAll('.timeline-item').forEach(item => {
        item.addEventListener('click', function() {
            this.style.transform = 'scale(1.02)';
            setTimeout(() => {
                this.style.transform = 'scale(1)';
            }, 200);
        });
    });

    // Add smooth scrolling for internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Utility functions
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount);
}

function animateCounter(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const current = Math.floor(progress * (end - start) + start);
        element.textContent = current.toLocaleString();
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Initialize counter animations for stats
document.addEventListener('DOMContentLoaded', function() {
    const statNumbers = document.querySelectorAll('.stat-number');
    statNumbers.forEach(stat => {
        const finalValue = parseInt(stat.textContent);
        if (!isNaN(finalValue)) {
            animateCounter(stat, 0, finalValue, 2000);
        }
    });
});

// Export functions for potential external use
window.BusinessReport = {
    initializeAnimations,
    initializeCharts,
    initializeScrollEffects,
    initializeInteractiveElements,
    formatCurrency,
    animateCounter
};
"""

    def _create_readme_content(self, synthesis_data: Dict[str, Any]) -> str:
        """Create comprehensive README content."""
        return f"""# Comprehensive Business Analysis Report

## Overview
This report contains a comprehensive business analysis generated by the ScoutAgent system, including market research, gap analysis, business solution development, and strategic recommendations.

## Generated Files

### HTML Report
- **Main File**: `business_analysis_report_[timestamp].html`
- **Description**: Complete interactive business analysis report with animations and charts
- **Features**: Responsive design, interactive elements, data visualizations

### CSS Styles
- **File**: `styles_[timestamp].css`
- **Description**: Comprehensive styling for the business report
- **Features**: Modern design, responsive layout, animations, custom components

### JavaScript
- **File**: `script_[timestamp].js`
- **Description**: Interactive functionality and data visualizations
- **Features**: Chart.js integration, scroll effects, animations, user interactions

### Data Export
- **File**: `analysis_data_[timestamp].json`
- **Description**: Complete analysis data in JSON format
- **Contents**: All agent outputs, market analysis, business recommendations

## Report Sections

1. **Executive Summary** - Key business opportunity and market insights
2. **Market Analysis** - Identified market gaps and strategic recommendations
3. **Business Solution** - Proposed SaaS solution with features and business model
4. **Go-to-Market Strategy** - Implementation phases and market entry strategy
5. **Financial Projections** - Revenue forecasts and key financial metrics
6. **Implementation Roadmap** - Timeline and milestones for execution
7. **Risk Analysis** - Potential risks and mitigation strategies
8. **Strategic Conclusion** - Final recommendations and next steps

## Key Insights

### Market Gaps Identified
- {len(synthesis_data.get('gap_finder_output', {}).get('identified_market_gaps', []))} market gaps identified
- {len(synthesis_data.get('gap_finder_output', {}).get('strategic_recommendations', []))} strategic recommendations

### Business Solution
- **Solution Name**: {synthesis_data.get('builder_output', {}).get('business_solution_summary', {}).get('solution_name', 'ClearPrice SaaS')}
- **Target Market**: {synthesis_data.get('builder_output', {}).get('business_model_pricing', {}).get('target_customers', 'SMBs and developers')}
- **Business Model**: {synthesis_data.get('builder_output', {}).get('business_model_pricing', {}).get('pricing_model', 'Subscription-based SaaS')}

## Technical Details

### Report Generation
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **System**: ScoutAgent Multi-Agent Workflow
- **Agents Used**: Scout, Screener, Validator, Gap Finder, Builder, Writer

### Data Sources
- Market research and competitive analysis
- Customer validation and market screening
- Gap analysis and opportunity identification
- Business solution development and strategy

## Usage Instructions

1. **View Report**: Open `business_analysis_report_[timestamp].html` in a web browser
2. **Interactive Features**: Use scroll effects, hover animations, and interactive charts
3. **Data Access**: Reference `analysis_data_[timestamp].json` for raw data
4. **Customization**: Modify CSS and JavaScript files as needed

## File Structure
```
reports/
├── business_analysis_report_[timestamp].html  # Main report
├── styles_[timestamp].css                      # Styling
├── script_[timestamp].js                       # JavaScript
├── analysis_data_[timestamp].json              # Data export
└── README_[timestamp].md                       # This file
```

## Support
For questions or issues with this report, please refer to the ScoutAgent documentation or contact the development team.

---
*Generated by ScoutAgent - Advanced Business Analysis System*
"""
