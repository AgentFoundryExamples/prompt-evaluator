# Copyright 2025 John Brosnihan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
JSON Schema validation utilities for prompt evaluator.

This module provides functionality to load, validate, and cache JSON schemas
for validating LLM outputs.
"""

import json
import logging
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator, ValidationError

logger = logging.getLogger(__name__)

# Global cache for compiled validators to avoid re-compiling schemas
_validator_cache: dict[str, Draft7Validator] = {}


def load_json_schema(schema_path: Path) -> dict[str, Any]:
    """
    Load a JSON schema from a file.

    Args:
        schema_path: Path to the JSON schema file

    Returns:
        Parsed JSON schema as a dictionary

    Raises:
        FileNotFoundError: If schema file doesn't exist
        json.JSONDecodeError: If schema file contains invalid JSON
        ValueError: If schema file is empty or not a valid schema
    """
    if not schema_path.exists():
        raise FileNotFoundError(
            f"JSON schema file not found: {schema_path}. "
            "Please provide a valid schema file path."
        )

    if not schema_path.is_file():
        raise ValueError(
            f"Schema path points to a directory: {schema_path}. "
            "Please provide a path to a JSON schema file."
        )

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in schema file {schema_path}: {e.msg}",
            e.doc,
            e.pos,
        ) from e

    if not schema or not isinstance(schema, dict):
        raise ValueError(
            f"Schema file {schema_path} is empty or not a valid JSON object. "
            "Schema must be a non-empty JSON object."
        )

    return schema


def get_validator(schema: dict[str, Any], schema_path: Path | None = None) -> Draft7Validator:
    """
    Get a compiled JSON schema validator with caching.

    Args:
        schema: JSON schema dictionary
        schema_path: Optional path to schema file (used for cache key)

    Returns:
        Compiled Draft7Validator instance

    Raises:
        ValueError: If schema is invalid
    """
    # Use schema path as cache key if available, otherwise hash the schema
    if schema_path:
        cache_key = str(schema_path.resolve())
    else:
        cache_key = json.dumps(schema, sort_keys=True)

    # Check cache first
    if cache_key in _validator_cache:
        logger.debug("Using cached validator for schema: %s", cache_key)
        return _validator_cache[cache_key]

    # Validate the schema itself
    try:
        Draft7Validator.check_schema(schema)
    except Exception as e:
        raise ValueError(f"Invalid JSON schema: {str(e)}") from e

    # Create and cache validator
    validator = Draft7Validator(schema)
    _validator_cache[cache_key] = validator
    logger.debug("Compiled and cached validator for schema: %s", cache_key)

    return validator


def validate_json_output(
    output: str,
    schema: dict[str, Any],
    schema_path: Path | None = None,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """
    Validate JSON output against a schema.

    Args:
        output: String containing JSON output to validate
        schema: JSON schema to validate against
        schema_path: Optional path to schema file (for caching)

    Returns:
        Tuple of (is_valid, error_message, parsed_json):
            - is_valid: True if output is valid JSON matching schema
            - error_message: Error description if validation failed, None if valid
            - parsed_json: Parsed JSON object if valid, None if invalid
    """
    # First, try to parse as JSON
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON: {e.msg} at position {e.pos}"
        logger.debug("JSON parsing failed for output: %s", error_msg)
        return False, error_msg, None

    # Get or create validator
    try:
        validator = get_validator(schema, schema_path)
    except ValueError as e:
        # Schema itself is invalid - this should have been caught earlier
        error_msg = f"Schema validation error: {str(e)}"
        logger.error("Schema validation failed: %s", error_msg)
        return False, error_msg, None

    # Validate against schema
    try:
        validator.validate(parsed)
        logger.debug("JSON output validated successfully against schema")
        return True, None, parsed
    except ValidationError as e:
        # Build a readable error message
        error_path = ".".join(str(p) for p in e.path) if e.path else "root"
        error_msg = f"Schema validation failed at '{error_path}': {e.message}"
        logger.debug("Schema validation failed: %s", error_msg)
        return False, error_msg, parsed


def clear_validator_cache() -> None:
    """Clear the validator cache. Useful for testing or memory management."""
    global _validator_cache
    _validator_cache.clear()
    logger.debug("Validator cache cleared")
