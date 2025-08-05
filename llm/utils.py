"""
Utility functions for LLM integration with ScoutAgent.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from .base import LLMRequest, LLMResponse, LLMBackendType
from .manager import get_llm_manager, initialize_llm_backends
from custom_logging import get_logger


@dataclass
class AgentPrompt:
    """Structured prompt for agent operations."""
    system_prompt: str
    user_prompt: str
    context: Optional[Dict[str, Any]] = None
    examples: Optional[List[Dict[str, str]]] = None
    
    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to message format for LLM."""
        messages = []
        
        # Add system prompt
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add examples if provided
        if self.examples:
            for example in self.examples:
                if "user" in example:
                    messages.append({"role": "user", "content": example["user"]})
                if "assistant" in example:
                    messages.append({"role": "assistant", "content": example["assistant"]})
        
        # Add main user prompt
        messages.append({"role": "user", "content": self.user_prompt})
        
        return messages


class LLMAgentMixin:
    """
    Mixin class to add LLM capabilities to agents.
    
    This can be mixed into any BaseAgent subclass to provide LLM functionality.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize LLM capabilities."""
        super().__init__(*args, **kwargs)
        self.llm_logger = get_logger(f"llm.agent.{getattr(self, 'name', 'unknown')}")
        self._llm_manager = None
    
    def initialize_llm(self):
        """Initialize LLM capabilities synchronously."""
        # This method is called during agent construction
        # Actual LLM manager initialization happens lazily in _ensure_llm_initialized
        pass
    
    async def _ensure_llm_initialized(self):
        """Ensure LLM manager is initialized."""
        if self._llm_manager is None:
            self._llm_manager = get_llm_manager()
            
            # Initialize backends if none are available
            if not self._llm_manager.get_available_backends():
                await initialize_llm_backends()
    
    async def llm_generate(self, 
                          prompt: Union[str, AgentPrompt],
                          backend_type: Optional[str] = None,
                          temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None,
                          **kwargs) -> str:
        """
        Generate text using LLM.
        
        Args:
            prompt: Text prompt or structured AgentPrompt
            backend_type: Specific backend to use (optional)
            temperature: Temperature override
            max_tokens: Max tokens override
            **kwargs: Additional parameters
            
        Returns:
            Generated text content
        """
        await self._ensure_llm_initialized()
        
        # Convert prompt to messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, AgentPrompt):
            messages = prompt.to_messages()
        else:
            raise ValueError("Prompt must be string or AgentPrompt")
        
        # Create request
        request = LLMRequest(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=kwargs
        )
        
        try:
            response = await self._llm_manager.generate(request, backend_type)
            
            if response.success:
                self.llm_logger.debug(f"LLM generation successful: {len(response.content)} chars in {response.response_time:.2f}s")
                return response.content
            else:
                self.llm_logger.error(f"LLM generation failed: {response.error}")
                raise Exception(f"LLM generation failed: {response.error}")
                
        except Exception as e:
            self.llm_logger.error(f"Error in LLM generation: {e}")
            raise
    
    async def llm_stream_generate(self,
                                 prompt: Union[str, AgentPrompt],
                                 backend_type: Optional[str] = None,
                                 temperature: Optional[float] = None,
                                 max_tokens: Optional[int] = None,
                                 **kwargs):
        """
        Generate streaming text using LLM.
        
        Args:
            prompt: Text prompt or structured AgentPrompt
            backend_type: Specific backend to use (optional)
            temperature: Temperature override
            max_tokens: Max tokens override
            **kwargs: Additional parameters
            
        Yields:
            Text chunks as they are generated
        """
        await self._ensure_llm_initialized()
        
        # Convert prompt to messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, AgentPrompt):
            messages = prompt.to_messages()
        else:
            raise ValueError("Prompt must be string or AgentPrompt")
        
        # Create request
        request = LLMRequest(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_params=kwargs
        )
        
        try:
            async for chunk in self._llm_manager.stream_generate(request, backend_type):
                yield chunk
                
        except Exception as e:
            self.llm_logger.error(f"Error in LLM streaming: {e}")
            raise
    
    async def llm_analyze_text(self,
                              text: str,
                              analysis_type: str = "general",
                              context: Optional[str] = None,
                              backend_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze text using LLM with structured output.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis (sentiment, themes, summary, etc.)
            context: Additional context for analysis
            backend_type: Specific backend to use
            
        Returns:
            Analysis results as structured data
        """
        system_prompt = f"""You are an expert text analyst. Analyze the provided text for {analysis_type}.
        
        Provide your analysis in the following JSON format:
        {{
            "analysis_type": "{analysis_type}",
            "key_findings": ["finding1", "finding2", ...],
            "summary": "brief summary",
            "confidence": 0.0-1.0,
            "details": {{
                // specific analysis details based on type
            }}
        }}
        
        Be thorough but concise. Focus on actionable insights."""
        
        user_prompt = f"Text to analyze:\n\n{text}"
        if context:
            user_prompt += f"\n\nAdditional context: {context}"
        
        prompt = AgentPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        response = await self.llm_generate(prompt, backend_type=backend_type)
        
        try:
            import json
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback to text response if JSON parsing fails
            return {
                "analysis_type": analysis_type,
                "raw_response": response,
                "confidence": 0.5
            }
    
    async def llm_extract_insights(self,
                                  data: Union[str, Dict, List],
                                  insight_type: str = "general",
                                  backend_type: Optional[str] = None) -> List[str]:
        """
        Extract insights from data using LLM.
        
        Args:
            data: Data to extract insights from
            insight_type: Type of insights to extract
            backend_type: Specific backend to use
            
        Returns:
            List of insights
        """
        system_prompt = f"""You are an expert data analyst. Extract key {insight_type} insights from the provided data.
        
        Focus on:
        - Patterns and trends
        - Actionable findings
        - Important relationships
        - Potential opportunities or risks
        
        Provide insights as a JSON array of strings:
        ["insight1", "insight2", "insight3", ...]
        
        Each insight should be clear, specific, and actionable."""
        
        # Convert data to string if needed
        if isinstance(data, (dict, list)):
            import json
            data_str = json.dumps(data, indent=2)
        else:
            data_str = str(data)
        
        user_prompt = f"Data to analyze:\n\n{data_str}"
        
        prompt = AgentPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        response = await self.llm_generate(prompt, backend_type=backend_type)
        
        try:
            import json
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback to splitting by lines if JSON parsing fails
            return [line.strip() for line in response.split('\n') if line.strip()]
    
    def get_llm_status(self) -> Dict[str, Any]:
        """Get status of LLM backends."""
        if self._llm_manager is None:
            return {"status": "not_initialized", "backends": []}
        
        return {
            "status": "initialized",
            "backends": self._llm_manager.get_available_backends(),
            "healthy_backends": self._llm_manager.get_healthy_backends(),
            "default_backend": self._llm_manager.get_default_backend(),
            "health_status": self._llm_manager.get_all_health_status()
        }


# Utility functions for common LLM operations
async def quick_llm_generate(prompt: str, 
                           backend_type: Optional[str] = None,
                           temperature: float = 0.7,
                           max_tokens: int = 1000) -> str:
    """Quick utility function for simple LLM generation."""
    manager = get_llm_manager()
    
    # Initialize backends if needed
    if not manager.get_available_backends():
        await initialize_llm_backends()
    
    request = LLMRequest(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    response = await manager.generate(request, backend_type)
    return response.content if response.success else ""


async def llm_summarize(text: str, 
                       max_length: int = 200,
                       backend_type: Optional[str] = None) -> str:
    """Summarize text using LLM."""
    prompt = f"""Please provide a concise summary of the following text in approximately {max_length} words or less:

{text}

Summary:"""
    
    return await quick_llm_generate(
        prompt, 
        backend_type=backend_type,
        temperature=0.3,
        max_tokens=max_length * 2  # Rough token estimate
    )


async def llm_classify(text: str,
                      categories: List[str],
                      backend_type: Optional[str] = None) -> str:
    """Classify text into one of the provided categories."""
    categories_str = ", ".join(categories)
    
    prompt = f"""Classify the following text into one of these categories: {categories_str}

Text: {text}

Category:"""
    
    response = await quick_llm_generate(
        prompt,
        backend_type=backend_type,
        temperature=0.1,
        max_tokens=50
    )
    
    # Extract the category from the response
    response = response.strip().lower()
    for category in categories:
        if category.lower() in response:
            return category
    
    return categories[0] if categories else "unknown"


# Prompt templates for common agent operations
class AgentPromptTemplates:
    """Collection of prompt templates for common agent operations."""
    
    @staticmethod
    def market_analysis(pain_points: List[str], market_context: str) -> AgentPrompt:
        """Template for market gap analysis."""
        system_prompt = """You are an expert market analyst specializing in identifying market gaps and opportunities.
        
        Your task is to analyze pain points and market context to identify potential market gaps.
        Focus on:
        - Unmet needs in the market
        - Underserved customer segments
        - Competitive gaps
        - Emerging opportunities
        
        Provide structured analysis with specific, actionable insights."""
        
        pain_points_str = "\n".join([f"- {pp}" for pp in pain_points])
        user_prompt = f"""Market Context: {market_context}

Pain Points to Analyze:
{pain_points_str}

Please analyze these pain points in the given market context and identify potential market gaps and opportunities."""
        
        return AgentPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
    
    @staticmethod
    def solution_generation(gap_description: str, target_segments: List[str]) -> AgentPrompt:
        """Template for solution idea generation."""
        system_prompt = """You are a creative solution architect and product strategist.
        
        Your task is to generate innovative solution ideas for identified market gaps.
        Focus on:
        - Feasible and scalable solutions
        - User-centered design
        - Technology opportunities
        - Business model considerations
        
        Provide creative but practical solution ideas."""
        
        segments_str = ", ".join(target_segments)
        user_prompt = f"""Market Gap: {gap_description}

Target Segments: {segments_str}

Please generate innovative solution ideas that could address this market gap for the specified target segments."""
        
        return AgentPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
    
    @staticmethod
    def report_writing(data: Dict[str, Any], report_type: str, audience: str) -> AgentPrompt:
        """Template for report generation."""
        system_prompt = f"""You are an expert business writer specializing in {report_type} reports for {audience}.
        
        Your task is to create a comprehensive, well-structured report based on the provided data.
        Focus on:
        - Clear, professional writing
        - Logical structure and flow
        - Data-driven insights
        - Actionable recommendations
        
        Tailor the content and tone for the {audience} audience."""
        
        import json
        data_str = json.dumps(data, indent=2)
        user_prompt = f"""Please create a {report_type} report based on the following data:

{data_str}

The report should be professional, comprehensive, and tailored for {audience}."""
        
        return AgentPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
