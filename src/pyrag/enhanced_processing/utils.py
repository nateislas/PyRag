"""Utility functions for enhanced processing."""

import json
import re
from typing import Any


def clean_llm_json_response(content: str) -> str:
    """Clean LLM response by removing markdown code blocks around JSON."""
    # Remove markdown code blocks if present
    cleaned_content = re.sub(r"^```json\s*", "", content)
    cleaned_content = re.sub(r"\s*```$", "", cleaned_content)
    return cleaned_content.strip()


def parse_llm_json_response(content: str) -> Any:
    """Parse LLM JSON response with proper error handling."""
    try:
        cleaned_content = clean_llm_json_response(content)
        return json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        # Try to extract JSON from the response if it's embedded in text
        json_match = re.search(r"\{.*\}", cleaned_content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # If all else fails, try to parse as a simple list
        list_match = re.search(r"\[.*\]", cleaned_content, re.DOTALL)
        if list_match:
            try:
                return json.loads(list_match.group(0))
            except json.JSONDecodeError:
                pass

        raise e
