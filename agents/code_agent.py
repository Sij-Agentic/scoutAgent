"""
Code Agent for ScoutAgent.

Specialized agent for code analysis, generation, and debugging.
"""

from typing import Dict, Any, List, Optional
import ast
import subprocess
import tempfile
import os
import re
from pathlib import Path

import json
import re

from agents.base import BaseAgent, AgentInput, AgentOutput
from config import get_config
from llm.utils import LLMAgentMixin


class CodeAgent(BaseAgent, LLMAgentMixin):
    """
    Code agent for analysis, generation, and debugging of code.
    
    Capabilities:
    - Code analysis and review
    - Code generation and refactoring
    - Debugging and error detection
    - Language detection and optimization
    - Security analysis
    """
    
    def __init__(self, name="code", **kwargs):
        BaseAgent.__init__(self, name=name, **kwargs)
        LLMAgentMixin.__init__(self, preferred_backend='ollama')
        self.supported_languages = {
            'python': {
                'extensions': ['.py', '.pyw'],
                'interpreter': 'python3',
                'linter': 'pylint',
                'formatter': 'black'
            },
            'javascript': {
                'extensions': ['.js', '.mjs'],
                'interpreter': 'node',
                'linter': 'eslint',
                'formatter': 'prettier'
            },
            'typescript': {
                'extensions': ['.ts', '.tsx'],
                'interpreter': 'ts-node',
                'linter': 'eslint',
                'formatter': 'prettier'
            },
            'java': {
                'extensions': ['.java'],
                'interpreter': 'java',
                'linter': 'checkstyle',
                'formatter': 'google-java-format'
            },
            'cpp': {
                'extensions': ['.cpp', '.cc', '.cxx'],
                'interpreter': 'g++',
                'linter': 'cppcheck',
                'formatter': 'clang-format'
            },
            'go': {
                'extensions': ['.go'],
                'interpreter': 'go run',
                'linter': 'golint',
                'formatter': 'gofmt'
            }
        }
    
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
        Plan code analysis or generation strategy using LLM.
        
        Args:
            agent_input: Contains code, task type, and parameters
            
        Returns:
            Code processing plan
        """
        code = agent_input.data.get('code', str(agent_input.data))
        task_type = agent_input.metadata.get('task_type', 'analyze')
        language = self._detect_language(code)
        
        self.log(f"Generating code plan for task: {task_type}, language: {language}")
        
        from llm.utils import load_prompt_template
        prompt_template = load_prompt_template(
            'plan.prompt',
            agent_name='code_agent',  # Use directory name instead of agent name
            substitutions={
                'code': code[:1000],  # Limit code size for prompt
                'task_type': task_type,
                'language': language
            }
        )
        
        response = await self.llm_generate(prompt_template)
        plan = self._extract_json(response)
        
        if not plan:
            self.log("Failed to generate a valid code plan.", level='error')
            return {"error": "Plan generation failed.", "task_type": task_type, "language": language}
        
        # Ensure these fields are present
        plan['task_type'] = task_type
        plan['language'] = language
        
        self.log(f"Code plan created for task: {task_type}")
        return plan
    
    async def think(self, agent_input: AgentInput, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze code and determine approach using LLM.
        
        Args:
            agent_input: Original input
            plan: Code processing plan
            
        Returns:
            Analysis and approach recommendations
        """
        code = agent_input.data.get('code', str(agent_input.data))
        task_type = plan['task_type']
        language = plan['language']
        plan_str = json.dumps(plan, indent=2)
        
        self.log(f"Analyzing code for {task_type} task in {language}")
        
        from llm.utils import load_prompt_template
        prompt_template = load_prompt_template(
            'think.prompt',
            agent_name='code_agent',  # Use directory name instead of agent name
            substitutions={
                'code': code[:2000],  # Limit code size for prompt
                'plan': plan_str,
                'task_type': task_type,
                'language': language
            }
        )
        
        response = await self.llm_generate(prompt_template)
        analysis = self._extract_json(response)
        
        if not analysis:
            self.log("Failed to generate valid code analysis.", level='error')
            return {"error": "Analysis generation failed.", "language": language}
        
        # Ensure language is included
        analysis['language'] = language
        
        self.log(f"Code analysis completed for {task_type} task")
        return analysis
    
    async def act(self, agent_input: AgentInput, plan: Dict[str, Any], thoughts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute code processing based on plan and analysis using LLM.
        
        Args:
            agent_input: Original input
            plan: Code processing plan
            thoughts: Analysis from thinking phase
            
        Returns:
            Processing results
        """
        code = agent_input.data.get('code', str(agent_input.data))
        task_type = plan['task_type']
        language = thoughts['language']
        plan_str = json.dumps(plan, indent=2)
        thoughts_str = json.dumps(thoughts, indent=2)
        
        self.log(f"Executing {task_type} task for {language} code")
        
        from llm.utils import load_prompt_template
        prompt_template = load_prompt_template(
            'act.prompt',
            agent_name='code_agent',  # Use directory name instead of agent name
            substitutions={
                'code': code[:3000],  # Limit code size for prompt
                'plan': plan_str,
                'thoughts': thoughts_str,
                'task_type': task_type,
                'language': language
            }
        )
        
        response = await self.llm_generate(prompt_template)
        result = self._extract_json(response)
        
        if not result:
            self.log("Failed to generate valid code processing result.", level='error')
            return {"error": "Code processing failed.", "language": language, "task_type": task_type}
        
        # Add original code and metadata to result
        result['original_code'] = code
        result['language'] = language
        result['task_type'] = task_type
        
        self.log(f"Code {task_type} task completed successfully")
        return result
    
    def _plan_analysis(self, agent_input: AgentInput) -> Dict[str, Any]:
        """Plan code analysis task."""
        return {
            'task_type': 'analyze',
            'steps': [
                'detect_language',
                'parse_code',
                'analyze_structure',
                'find_issues',
                'generate_report'
            ],
            'focus_areas': agent_input.metadata.get('focus_areas', ['syntax', 'style', 'complexity'])
        }
    
    def _plan_generation(self, agent_input: AgentInput) -> Dict[str, Any]:
        """Plan code generation task."""
        return {
            'task_type': 'generate',
            'steps': [
                'understand_requirements',
                'design_structure',
                'generate_code',
                'validate_syntax',
                'optimize_code'
            ],
            'target_language': agent_input.metadata.get('language', 'python'),
            'complexity': agent_input.metadata.get('complexity', 'moderate')
        }
    
    def _plan_debugging(self, agent_input: AgentInput) -> Dict[str, Any]:
        """Plan debugging task."""
        return {
            'task_type': 'debug',
            'steps': [
                'identify_errors',
                'analyze_stack_trace',
                'suggest_fixes',
                'validate_fixes',
                'test_solutions'
            ],
            'error_message': agent_input.metadata.get('error_message', ''),
            'test_cases': agent_input.metadata.get('test_cases', [])
        }
    
    def _plan_refactoring(self, agent_input: AgentInput) -> Dict[str, Any]:
        """Plan refactoring task."""
        return {
            'task_type': 'refactor',
            'steps': [
                'identify_smells',
                'suggest_improvements',
                'apply_refactoring',
                'validate_changes',
                'performance_check'
            ],
            'refactoring_type': agent_input.metadata.get('refactoring_type', 'general'),
            'preserve_behavior': agent_input.metadata.get('preserve_behavior', True)
        }
    
    def _plan_general(self, agent_input: AgentInput) -> Dict[str, Any]:
        """Plan general code processing task."""
        return {
            'task_type': 'general',
            'steps': ['analyze', 'process', 'report'],
            'custom_params': agent_input.metadata
        }
    
    def _detect_language(self, code: str) -> str:
        """Detect programming language from code."""
        # Simple language detection based on patterns
        patterns = {
            'python': [r'^import\s+\w+', r'^from\s+\w+\s+import', r'def\s+\w+\s*\(', r'class\s+\w+\s*\('],
            'javascript': [r'function\s+\w+\s*\(', r'const\s+\w+\s*=', r'let\s+\w+\s*=', r'var\s+\w+\s*='],
            'java': [r'public\s+class\s+\w+', r'import\s+java\.', r'public\s+static\s+void\s+main'],
            'cpp': [r'#include\s*<[^>]+>', r'int\s+main\s*\(', r'std::', r'using\s+namespace\s+std'],
            'go': [r'package\s+\w+', r'import\s+"[^"]+"', r'func\s+\w+\s*\('],
            'typescript': [r'interface\s+\w+', r'type\s+\w+\s*=', r'import\s+\{[^}]+\}\s+from']
        }
        
        for lang, lang_patterns in patterns.items():
            for pattern in lang_patterns:
                if re.search(pattern, code, re.MULTILINE):
                    return lang
        
        return 'unknown'
    
    def _analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity."""
        lines = code.splitlines()
        
        complexity = {
            'lines_of_code': len(lines),
            'cyclomatic_complexity': self._calculate_cyclomatic_complexity(code),
            'nesting_depth': self._calculate_nesting_depth(code),
            'function_count': len(re.findall(r'\b(def|function|func)\s+\w+', code)),
            'class_count': len(re.findall(r'\b(class|struct)\s+\w+', code))
        }
        
        return complexity
    
    def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity (simplified)."""
        complexity_indicators = [
            r'\bif\b', r'\belse\b', r'\belif\b', r'\bwhile\b',
            r'\bfor\b', r'\bcase\b', r'\bcatch\b', r'\band\b', r'\bor\b'
        ]
        
        complexity = 1
        for indicator in complexity_indicators:
            complexity += len(re.findall(indicator, code, re.IGNORECASE))
        
        return complexity
    
    def _calculate_nesting_depth(self, code: str) -> int:
        """Calculate maximum nesting depth."""
        # Simplified implementation
        lines = code.splitlines()
        max_depth = 0
        current_depth = 0
        
        for line in lines:
            stripped = line.strip()
            if any(stripped.endswith(char) for char in [':', '{', '(']):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif any(stripped.startswith(char) for char in ['}', ')', 'return', 'break']):
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def _analyze_language_specific(self, code: str, language: str) -> Dict[str, Any]:
        """Perform language-specific analysis."""
        if language == 'python':
            return self._analyze_python(code)
        elif language == 'javascript':
            return self._analyze_javascript(code)
        elif language == 'java':
            return self._analyze_java(code)
        else:
            return {'language_specific': f"Analysis for {language} not implemented"}
    
    def _analyze_python(self, code: str) -> Dict[str, Any]:
        """Analyze Python code."""
        try:
            tree = ast.parse(code)
            
            analysis = {
                'imports': [],
                'functions': [],
                'classes': [],
                'docstrings': 0,
                'type_hints': 0
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    analysis['imports'].extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis['imports'].append(node.module)
                elif isinstance(node, ast.FunctionDef):
                    analysis['functions'].append(node.name)
                    if ast.get_docstring(node):
                        analysis['docstrings'] += 1
                    if any(arg.annotation for arg in node.args.args):
                        analysis['type_hints'] += 1
                elif isinstance(node, ast.ClassDef):
                    analysis['classes'].append(node.name)
                    if ast.get_docstring(node):
                        analysis['docstrings'] += 1
            
            return analysis
        except SyntaxError as e:
            return {'error': f"Syntax error: {str(e)}"}
    
    def _analyze_javascript(self, code: str) -> Dict[str, Any]:
        """Analyze JavaScript code (basic)."""
        return {
            'functions': re.findall(r'function\s+(\w+)', code),
            'arrow_functions': len(re.findall(r'=>', code)),
            'classes': re.findall(r'class\s+(\w+)', code),
            'imports': re.findall(r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]', code)
        }
    
    def _analyze_java(self, code: str) -> Dict[str, Any]:
        """Analyze Java code (basic)."""
        return {
            'classes': re.findall(r'class\s+(\w+)', code),
            'methods': re.findall(r'public|private|protected\s+\w+\s+(\w+)\s*\(', code),
            'imports': re.findall(r'import\s+([\w.]+)', code)
        }
    
    def _find_issues(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Find potential issues in code."""
        issues = []
        
        # Common issues
        if 'TODO' in code:
            issues.append({
                'type': 'todo',
                'severity': 'info',
                'message': 'Found TODO comments',
                'line': None
            })
        
        if 'FIXME' in code:
            issues.append({
                'type': 'fixme',
                'severity': 'warning',
                'message': 'Found FIXME comments',
                'line': None
            })
        
        if language == 'python':
            issues.extend(self._find_python_issues(code))
        
        return issues
    
    def _find_python_issues(self, code: str) -> List[Dict[str, Any]]:
        """Find Python-specific issues."""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            # Check for unused variables
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Simple check for unused variables
                    pass
        except SyntaxError as e:
            issues.append({
                'type': 'syntax',
                'severity': 'error',
                'message': str(e),
                'line': getattr(e, 'lineno', None)
            })
        
        return issues
    
    def _find_bugs(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Find potential bugs in code."""
        bugs = []
        
        # Common bug patterns
        bug_patterns = [
            (r'==\s*None', 'Use "is None" instead of "== None"'),
            (r'!=\s*None', 'Use "is not None" instead of "!= None"'),
            (r'print\s*\([^)]*\+[^)]*\)', 'String concatenation in print statements'),
            (r'except\s*:', 'Bare except clause'),
            (r'import\s*\*', 'Wildcard import')
        ]
        
        for pattern, message in bug_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                bugs.append({
                    'type': 'bug',
                    'severity': 'warning',
                    'message': message,
                    'line': code[:match.start()].count('\n') + 1
                })
        
        return bugs
    
    def _suggest_refactoring(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Suggest refactoring improvements."""
        suggestions = []
        
        # General refactoring suggestions
        if len(code.splitlines()) > 100:
            suggestions.append({
                'type': 'complexity',
                'message': 'Consider breaking this into smaller functions',
                'priority': 'medium'
            })
        
        if language == 'python':
            if 'class ' in code and 'def __init__' not in code:
                suggestions.append({
                    'type': 'oop',
                    'message': 'Consider using classes for better organization',
                    'priority': 'low'
                })
        
        return suggestions
    
    def _analyze_code(self, code: str, language: str) -> Dict[str, Any]:
        """Perform comprehensive code analysis."""
        return {
            'structure': self._analyze_complexity(code),
            'language_specific': self._analyze_language_specific(code, language),
            'issues': self._find_issues(code, language),
            'metrics': {
                'lines_of_code': len(code.splitlines()),
                'characters': len(code),
                'words': len(code.split())
            }
        }
    
    def _generate_code(self, requirements: str, language: str) -> Dict[str, Any]:
        """Generate code based on requirements."""
        # Mock code generation
        if language == 'python':
            code = f"""# Generated Python code
# Requirements: {requirements}

def main():
    print("Hello, World!")
    # TODO: Implement based on requirements

if __name__ == "__main__":
    main()
"""
        else:
            code = f"// Generated {language} code\n// Requirements: {requirements}\n\n// TODO: Implement based on requirements"
        
        return {
            'generated_code': code,
            'language': language,
            'requirements': requirements,
            'confidence': 0.7
        }
    
    def _debug_code(self, code: str, language: str) -> Dict[str, Any]:
        """Debug code and suggest fixes."""
        issues = self._find_bugs(code, language)
        
        fixes = []
        for issue in issues:
            fix = self._suggest_fix(issue, code)
            if fix:
                fixes.append(fix)
        
        return {
            'issues': issues,
            'fixes': fixes,
            'debugged_code': code  # Mock - would apply fixes
        }
    
    def _suggest_fix(self, issue: Dict[str, Any], code: str) -> Optional[Dict[str, Any]]:
        """Suggest a fix for an issue."""
        # Mock fix suggestions
        fix_map = {
            'Use "is None" instead of "== None"': {
                'original': '== None',
                'fixed': 'is None',
                'description': 'Use identity check for None'
            },
            'Use "is not None" instead of "!= None"': {
                'original': '!= None',
                'fixed': 'is not None',
                'description': 'Use identity check for not None'
            }
        }
        
        return fix_map.get(issue['message'])
    
    def _refactor_code(self, code: str, language: str) -> Dict[str, Any]:
        """Refactor code based on best practices."""
        suggestions = self._suggest_refactoring(code, language)
        
        return {
            'original_code': code,
            'refactored_code': code,  # Mock - would apply refactoring
            'suggestions': suggestions,
            'improvements': [
                'Improved readability',
                'Better structure',
                'Performance optimization'
            ]
        }


if __name__ == "__main__":
    # Test the code agent
    import asyncio
    
    async def test_agent():
        agent = CodeAgent()
        
        test_code = """
def calculate_sum(a, b):
    return a + b

if __name__ == "__main__":
    result = calculate_sum(5, 3)
    print("Sum is: " + str(result))
"""
        
        test_input = AgentInput(
            data={'code': test_code},
            metadata={'task_type': 'analyze', 'language': 'python', 'focus_areas': ['style', 'complexity']}
        )
        
        result = await agent.execute(test_input)
        print(f"Code analysis completed: {result.success}")
        print(f"Language detected: {result.result.get('language')}")
        print(f"Issues found: {len(result.result.get('issues', []))}")
    
    # Run the async test function
    asyncio.run(test_agent())
