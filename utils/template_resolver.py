"""Template Variable Resolution Utility.

This module provides functionality to resolve template variables in DAG node inputs.
Template variables follow the pattern ${variable_name.field[*].subfield} and are
resolved using JSONPath expressions against available data sources.
"""

import re
import json
from typing import Any, Dict, List, Optional, Union
from jsonpath_ng import parse as jsonpath_parse
from jsonpath_ng.exceptions import JSONPathError

from scout_agent.custom_logging import get_logger


class TemplateResolver:
    """Resolves template variables in DAG node inputs."""
    
    def __init__(self):
        """Initialize the template resolver."""
        self.logger = get_logger("TemplateResolver")
        # Pattern to match ${variable} expressions
        self.template_pattern = re.compile(r'\$\{([^}]+)\}')
    
    def resolve_template_variables(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Recursively resolve template variables in data structure.
        
        Args:
            data: Data structure that may contain template variables
            context: Context containing available data for resolution
            
        Returns:
            Data structure with template variables resolved
        """
        if isinstance(data, str):
            return self._resolve_string_templates(data, context)
        elif isinstance(data, dict):
            return {k: self.resolve_template_variables(v, context) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.resolve_template_variables(item, context) for item in data]
        else:
            return data
    
    def _resolve_string_templates(self, text: str, context: Dict[str, Any]) -> Any:
        """
        Resolve template variables in a string.
        
        Args:
            text: String that may contain template variables
            context: Context containing available data for resolution
            
        Returns:
            Resolved value (may be string, list, dict, etc.)
        """
        # Find all template variables in the string
        matches = list(self.template_pattern.finditer(text))
        
        if not matches:
            return text
        
        # If the entire string is a single template variable, return the resolved value directly
        if len(matches) == 1 and matches[0].group(0) == text:
            variable_path = matches[0].group(1)
            return self._resolve_variable_path(variable_path, context)
        
        # Multiple variables or mixed content - substitute as strings
        result = text
        for match in reversed(matches):  # Reverse to maintain positions
            variable_path = match.group(1)
            resolved_value = self._resolve_variable_path(variable_path, context)
            
            # Convert to string for substitution
            if resolved_value is None:
                str_value = "None"
            elif isinstance(resolved_value, (list, dict)):
                str_value = json.dumps(resolved_value)
            else:
                str_value = str(resolved_value)
            
            result = result[:match.start()] + str_value + result[match.end():]
        
        return result
    
    def _resolve_variable_path(self, variable_path: str, context: Dict[str, Any]) -> Any:
        """
        Resolve a single variable path using JSONPath.
        
        Args:
            variable_path: Variable path like 'node_output.field[*].subfield'
            context: Context containing available data
            
        Returns:
            Resolved value or None if not found
        """
        try:
            # Parse the JSONPath expression
            jsonpath_expr = jsonpath_parse(f'$.{variable_path}')
            
            # Find matches in context
            matches = jsonpath_expr.find(context)
            
            if not matches:
                self.logger.warning(f"Could not resolve field path '{variable_path}': not found in context")
                self.logger.debug(f"Available context keys: {list(context.keys())}")
                return None
            
            # If single match, return the value directly
            if len(matches) == 1:
                return matches[0].value
            
            # Multiple matches - return as list
            return [match.value for match in matches]
            
        except JSONPathError as e:
            self.logger.error(f"Invalid JSONPath expression '{variable_path}': {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error resolving variable path '{variable_path}': {e}")
            return None
    
    def validate_template_variables(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Validate that all template variables in data can be resolved.
        
        Args:
            data: Data structure to validate
            context: Context for resolution
            
        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []
        self._collect_validation_errors(data, context, errors)
        return errors
    
    def _collect_validation_errors(self, data: Any, context: Dict[str, Any], errors: List[str]) -> None:
        """
        Recursively collect validation errors for template variables.
        
        Args:
            data: Data structure to validate
            context: Context for resolution
            errors: List to collect errors in
        """
        if isinstance(data, str):
            matches = self.template_pattern.findall(data)
            for variable_path in matches:
                try:
                    jsonpath_expr = jsonpath_parse(f'$.{variable_path}')
                    matches = jsonpath_expr.find(context)
                    if not matches:
                        errors.append(f"Template variable ${{{variable_path}}} cannot be resolved")
                except JSONPathError:
                    errors.append(f"Invalid JSONPath expression in ${{{variable_path}}}")
        elif isinstance(data, dict):
            for v in data.values():
                self._collect_validation_errors(v, context, errors)
        elif isinstance(data, list):
            for item in data:
                self._collect_validation_errors(item, context, errors)


# Global instance for convenience
_resolver = TemplateResolver()


def resolve_template_variables(data: Any, context: Dict[str, Any]) -> Any:
    """
    Convenience function to resolve template variables.
    
    Args:
        data: Data structure that may contain template variables
        context: Context containing available data for resolution
        
    Returns:
        Data structure with template variables resolved
    """
    return _resolver.resolve_template_variables(data, context)


def validate_template_variables(data: Any, context: Dict[str, Any]) -> List[str]:
    """
    Convenience function to validate template variables.
    
    Args:
        data: Data structure to validate
        context: Context for resolution
        
    Returns:
        List of validation errors (empty if all valid)
    """
    return _resolver.validate_template_variables(data, context)