"""Token counting and context management utilities for LLM backends."""

import re
from typing import List, Dict, Any, Optional, Tuple
from scout_agent.custom_logging import get_logger

logger = get_logger("llm.token_utils")

# Approximate token counts for different model families
MODEL_CONTEXT_LIMITS = {
    "deepseek-chat": 131072,
    "deepseek-coder": 131072,
    "deepseek-reasoner": 131072,
    "deepseek-v3": 131072,
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-3.5-turbo": 4096,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "gemini-pro": 32768,
}


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text using a simple heuristic.
    
    This is an approximation based on the rule that 1 token ≈ 4 characters
    for English text, but we use a more conservative estimate to be safe.
    
    Args:
        text: Input text to count tokens for
        
    Returns:
        Estimated number of tokens
    """
    if not text:
        return 0
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Conservative estimate: 1 token per 3 characters on average
    # This accounts for punctuation, special characters, and non-English text
    return max(1, len(text) // 3)


def estimate_message_tokens(messages: List[Dict[str, str]]) -> int:
    """
    Estimate total token count for a list of messages.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        Estimated total token count including message formatting overhead
    """
    total_tokens = 0
    
    for message in messages:
        # Count content tokens
        content = message.get('content', '')
        total_tokens += estimate_tokens(content)
        
        # Add overhead for message formatting (role, structure, etc.)
        total_tokens += 10  # Conservative estimate for message overhead
    
    # Add overhead for conversation structure
    total_tokens += 20
    
    return total_tokens


def get_model_context_limit(model_name: str) -> int:
    """
    Get the context limit for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Context limit in tokens, defaults to 4096 if unknown
    """
    # Check exact match first
    if model_name in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[model_name]
    
    # Check for partial matches
    for model_key, limit in MODEL_CONTEXT_LIMITS.items():
        if model_key in model_name.lower():
            return limit
    
    # Default conservative limit
    logger.warning(f"Unknown model {model_name}, using default context limit of 4096")
    return 4096


def truncate_text_to_tokens(text: str, max_tokens: int, preserve_end: bool = False) -> str:
    """
    Truncate text to fit within a token limit.
    
    Args:
        text: Input text to truncate
        max_tokens: Maximum number of tokens allowed
        preserve_end: If True, preserve the end of the text instead of the beginning
        
    Returns:
        Truncated text that fits within the token limit
    """
    if not text or max_tokens <= 0:
        return ""
    
    current_tokens = estimate_tokens(text)
    
    if current_tokens <= max_tokens:
        return text
    
    # Calculate approximate character limit
    # Use 3 chars per token as our conservative estimate
    char_limit = max_tokens * 3
    
    if preserve_end:
        # Keep the end of the text
        truncated = text[-char_limit:]
        # Find a good break point to avoid cutting words
        space_idx = truncated.find(' ')
        if space_idx > 0 and space_idx < len(truncated) // 4:
            truncated = truncated[space_idx + 1:]
        return "..." + truncated
    else:
        # Keep the beginning of the text
        truncated = text[:char_limit]
        # Find a good break point to avoid cutting words
        space_idx = truncated.rfind(' ')
        if space_idx > len(truncated) * 3 // 4:
            truncated = truncated[:space_idx]
        return truncated + "..."


def truncate_messages_to_context(
    messages: List[Dict[str, str]], 
    model_name: str, 
    max_output_tokens: int = 4096,
    system_prompt: Optional[str] = None
) -> Tuple[List[Dict[str, str]], bool]:
    """
    Truncate messages to fit within model context limits.
    
    Args:
        messages: List of message dictionaries
        model_name: Name of the model to get context limits for
        max_output_tokens: Maximum tokens reserved for output
        system_prompt: Optional system prompt that takes additional tokens
        
    Returns:
        Tuple of (truncated_messages, was_truncated)
    """
    context_limit = get_model_context_limit(model_name)
    
    # Reserve tokens for output and system prompt
    system_tokens = estimate_tokens(system_prompt) if system_prompt else 0
    available_tokens = context_limit - max_output_tokens - system_tokens - 100  # Safety buffer
    
    if available_tokens <= 0:
        logger.error(f"No tokens available for messages after reserving {max_output_tokens} for output")
        return [], True
    
    current_tokens = estimate_message_tokens(messages)
    
    if current_tokens <= available_tokens:
        return messages, False
    
    logger.warning(f"Messages exceed context limit ({current_tokens} > {available_tokens}), truncating")
    
    # Strategy: Keep the most recent messages and truncate older ones
    truncated_messages = []
    remaining_tokens = available_tokens
    
    # Process messages in reverse order (most recent first)
    for message in reversed(messages):
        message_tokens = estimate_tokens(message.get('content', '')) + 10  # Message overhead
        
        if message_tokens <= remaining_tokens:
            # Message fits, add it
            truncated_messages.insert(0, message)
            remaining_tokens -= message_tokens
        else:
            # Message doesn't fit, try to truncate its content
            if remaining_tokens > 50:  # Only if we have reasonable space left
                content_tokens = remaining_tokens - 10  # Account for message overhead
                truncated_content = truncate_text_to_tokens(
                    message.get('content', ''), 
                    content_tokens, 
                    preserve_end=True
                )
                
                if truncated_content:
                    truncated_message = message.copy()
                    truncated_message['content'] = truncated_content
                    truncated_messages.insert(0, truncated_message)
            
            # Stop processing older messages
            break
    
    return truncated_messages, True


def validate_request_tokens(
    messages: List[Dict[str, str]], 
    model_name: str, 
    max_output_tokens: int = 4096,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate that a request fits within model token limits.
    
    Args:
        messages: List of message dictionaries
        model_name: Name of the model
        max_output_tokens: Maximum tokens for output
        system_prompt: Optional system prompt
        
    Returns:
        Dictionary with validation results
    """
    context_limit = get_model_context_limit(model_name)
    input_tokens = estimate_message_tokens(messages)
    system_tokens = estimate_tokens(system_prompt) if system_prompt else 0
    total_input_tokens = input_tokens + system_tokens
    
    return {
        "valid": total_input_tokens + max_output_tokens <= context_limit,
        "context_limit": context_limit,
        "input_tokens": input_tokens,
        "system_tokens": system_tokens,
        "total_input_tokens": total_input_tokens,
        "max_output_tokens": max_output_tokens,
        "total_tokens": total_input_tokens + max_output_tokens,
        "available_tokens": context_limit - total_input_tokens,
        "needs_truncation": total_input_tokens + max_output_tokens > context_limit
    }