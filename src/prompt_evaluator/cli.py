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
Command-line interface for the prompt evaluator.

This module provides the main CLI entry point and commands for running
prompt evaluations, managing configurations, and viewing results.
"""

import hashlib
import json
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from prompt_evaluator.config import (
    APIConfig,
    ConfigManager,
    PromptEvaluatorConfig,
    load_prompt_evaluator_config,
)
from prompt_evaluator.models import (
    GeneratorConfig,
    JudgeConfig,
    PromptRun,
    Rubric,
    Sample,
    SingleEvaluationRun,
    load_judge_prompt,
)
from prompt_evaluator.provider import (
    generate_completion,
    get_provider,
    judge_completion,
)

app = typer.Typer(
    name="prompt-evaluator",
    help="Evaluate and compare prompts across different LLM providers",
    add_completion=False,
)

# Shared config manager for caching configuration across CLI commands
_config_manager = ConfigManager()

# Configuration constants for evaluate-dataset command
DEFAULT_NUM_SAMPLES = 5  # Default number of samples per test case
QUICK_MODE_NUM_SAMPLES = 2  # Number of samples in quick mode
HIGH_STD_ABSOLUTE_THRESHOLD = 1.0  # Absolute std threshold for high variability warning
HIGH_STD_RELATIVE_THRESHOLD = 0.2  # Relative std/mean threshold for high variability warning


def resolve_prompt_path(
    prompt_input: str,
    app_config: PromptEvaluatorConfig | None = None
) -> Path:
    """
    Resolve a prompt path from user input, supporting both template keys and file paths.

    Args:
        prompt_input: User-provided prompt identifier - can be:
                     - A template key from config (e.g., "checkout_compiler")
                     - An absolute or relative file path
        app_config: Optional loaded application config with prompt templates

    Returns:
        Resolved absolute Path to the prompt file

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        ValueError: If prompt_input is empty or invalid
    """
    # Validate input
    if not prompt_input or not prompt_input.strip():
        raise ValueError("Prompt input cannot be empty")

    # First, check if the input is a valid file path (prioritize files over template keys)
    prompt_path = Path(prompt_input)

    # Try as absolute path
    if prompt_path.is_absolute() and prompt_path.exists():
        return prompt_path.resolve()

    # Try as relative path from current directory
    cwd_path = Path.cwd() / prompt_path
    if cwd_path.exists():
        return cwd_path.resolve()

    # If not a file, try to resolve as a template key if config is available
    if app_config is not None and prompt_input in app_config.prompt_templates:
        try:
            return app_config.get_prompt_template_path(prompt_input)
        except FileNotFoundError:
            # Template key found but file doesn't exist - let it raise
            raise

    # If we are here, it's not an existing file and not a valid template key
    # Provide helpful error message
    if app_config is not None and app_config.prompt_templates:
        available = ', '.join(sorted(app_config.prompt_templates.keys()))
        raise FileNotFoundError(
            f"Prompt file not found: '{prompt_input}' is not a valid file path and not a "
            f"known template key. Available template keys: {available}"
        )
    else:
        raise FileNotFoundError(f"Prompt file not found: {prompt_input}")


def resolve_dataset_path(
    dataset_input: str,
    app_config: PromptEvaluatorConfig | None = None
) -> Path:
    """
    Resolve a dataset path from user input, supporting both dataset keys and file paths.

    Args:
        dataset_input: User-provided dataset identifier - can be:
                      - A dataset key from config (e.g., "sample")
                      - An absolute or relative file path
        app_config: Optional loaded application config with dataset paths

    Returns:
        Resolved absolute Path to the dataset file

    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        ValueError: If dataset_input is empty or invalid
    """
    # Validate input
    if not dataset_input or not dataset_input.strip():
        raise ValueError("Dataset input cannot be empty")

    # First, check if the input is a valid file path (prioritize files over dataset keys)
    dataset_path = Path(dataset_input)

    # Try as absolute path
    if dataset_path.is_absolute() and dataset_path.exists():
        return dataset_path.resolve()

    # Try as relative path from current directory
    cwd_path = Path.cwd() / dataset_path
    if cwd_path.exists():
        return cwd_path.resolve()

    # If not a file, try to resolve as a dataset key if config is available
    if app_config is not None and dataset_input in app_config.dataset_paths:
        try:
            return app_config.get_dataset_path(dataset_input)
        except FileNotFoundError:
            # Dataset key found but file doesn't exist - let it raise
            raise

    # If we are here, it's not an existing file and not a valid dataset key
    # Provide helpful error message
    if app_config is not None and app_config.dataset_paths:
        available = ', '.join(sorted(app_config.dataset_paths.keys()))
        raise FileNotFoundError(
            f"Dataset file not found: '{dataset_input}' is not a valid file path and not a "
            f"known dataset key. Available dataset keys: {available}"
        )
    else:
        raise FileNotFoundError(f"Dataset file not found: {dataset_input}")


def compute_rubric_metadata(rubric: Rubric | None, rubric_path: Path | None) -> dict[str, Any]:
    """
    Compute rubric metadata including path, hash, and full definition.

    Args:
        rubric: Rubric object or None
        rubric_path: Path to rubric file or None

    Returns:
        Dictionary with rubric metadata
    """
    if rubric is None or rubric_path is None:
        return {}

    rubric_dict = asdict(rubric)

    # Compute hash of rubric content for change detection
    rubric_json = json.dumps(rubric_dict, sort_keys=True)
    rubric_hash = hashlib.sha256(rubric_json.encode()).hexdigest()

    return {
        "rubric_path": str(rubric_path),
        "rubric_hash": rubric_hash,
        "rubric_definition": rubric_dict,
    }


def compute_prompt_metadata(
    system_prompt_path: Path, prompt_version: str | None = None
) -> tuple[str, str]:
    """
    Compute prompt version metadata from system prompt file.

    Args:
        system_prompt_path: Path to system prompt file
        prompt_version: Optional user-provided prompt version string

    Returns:
        Tuple of (prompt_version_id, prompt_hash)

    Raises:
        FileNotFoundError: If prompt file doesn't exist
        ValueError: If prompt file is empty or unreadable
    """
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {system_prompt_path}")

    try:
        with open(system_prompt_path, "rb") as f:
            prompt_content = f.read()

        if not prompt_content:
            raise ValueError(f"System prompt file is empty: {system_prompt_path}")

        # Compute SHA-256 hash of prompt content
        prompt_hash = hashlib.sha256(prompt_content).hexdigest()

        # Use provided version or hash as version_id
        prompt_version_id = prompt_version if prompt_version else prompt_hash

        return prompt_version_id, prompt_hash

    except OSError as e:
        raise ValueError(f"Failed to read system prompt file {system_prompt_path}: {str(e)}") from e


def compute_aggregate_statistics(
    samples: list[Sample], rubric: Rubric | None = None
) -> dict[str, Any]:
    """
    Compute aggregate statistics from samples, including legacy and rubric-based metrics.

    This function computes:
    - Legacy statistics (mean/min/max score for backward compatibility)
    - Per-metric statistics (mean/min/max/count for each metric in rubric)
    - Per-flag statistics (count and proportion for each flag in rubric)

    Samples with status "judge_invalid_response" are excluded from aggregation.

    Args:
        samples: List of Sample objects with evaluation results
        rubric: Optional Rubric object defining metrics and flags

    Returns:
        Dictionary with aggregate statistics including:
        - Legacy fields: mean_score, min_score, max_score, num_successful, num_failed
        - metric_stats: Per-metric statistics (if rubric provided)
        - flag_stats: Per-flag statistics (if rubric provided)
    """
    # Filter out invalid samples (judge_invalid_response status)
    valid_samples = [s for s in samples if s.status != "judge_invalid_response"]

    # Compute legacy statistics for backward compatibility
    successful_scores = [
        s.judge_score
        for s in valid_samples
        if s.status == "completed" and s.judge_score is not None
    ]

    stats: dict[str, Any] = {}

    # Calculate num_successful based on "completed" status, which is mode-agnostic
    num_successful = sum(1 for s in samples if s.status == "completed")
    stats["num_successful"] = num_successful
    stats["num_failed"] = len(samples) - num_successful

    if successful_scores:
        stats["mean_score"] = sum(successful_scores) / len(successful_scores)
        stats["min_score"] = min(successful_scores)
        stats["max_score"] = max(successful_scores)
    else:
        stats["mean_score"] = None
        stats["min_score"] = None
        stats["max_score"] = None

    # If rubric is provided, compute per-metric and per-flag statistics
    if rubric is not None:
        # Compute per-metric statistics
        metric_stats: dict[str, dict[str, float | int | None]] = {}

        for metric in rubric.metrics:
            metric_name = metric.name
            # Collect scores for this metric from valid completed samples
            metric_scores = [
                s.judge_metrics[metric_name]["score"]
                for s in valid_samples
                if s.status == "completed"
                and metric_name in s.judge_metrics
                and "score" in s.judge_metrics[metric_name]
            ]

            if metric_scores:
                metric_stats[metric_name] = {
                    "mean": sum(metric_scores) / len(metric_scores),
                    "min": min(metric_scores),
                    "max": max(metric_scores),
                    "count": len(metric_scores),
                }
            else:
                metric_stats[metric_name] = {
                    "mean": None,
                    "min": None,
                    "max": None,
                    "count": 0,
                }

        stats["metric_stats"] = metric_stats

        # Compute per-flag statistics
        flag_stats: dict[str, dict[str, int | float]] = {}

        for flag in rubric.flags:
            flag_name = flag.name
            # Collect flag values from valid completed samples
            flag_values = [
                s.judge_flags[flag_name]
                for s in valid_samples
                if s.status == "completed" and flag_name in s.judge_flags
            ]

            true_count = sum(1 for v in flag_values if v)
            total_count = len(flag_values)

            flag_stats[flag_name] = {
                "true_count": true_count,
                "false_count": total_count - true_count,
                "total_count": total_count,
                "true_proportion": true_count / total_count if total_count > 0 else 0.0,
            }

        stats["flag_stats"] = flag_stats

    return stats


@app.command()
def version() -> None:
    """Display the version of prompt-evaluator."""
    from prompt_evaluator import __version__

    typer.echo(f"prompt-evaluator version {__version__}")


@app.command()
def generate(
    system_prompt: str = typer.Option(
        ..., "--system-prompt", "-s", help="Path to system prompt file or template key"
    ),
    input_path: str = typer.Option(
        ..., "--input", "-i", help="Path to input file (use '-' for stdin)"
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="Provider to use (openai, claude, anthropic, mock). Uses config default.",
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name override"),
    temperature: float | None = typer.Option(
        None, "--temperature", "-t", help="Temperature (0.0-2.0)"
    ),
    max_completion_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Maximum tokens to generate"
    ),
    seed: int | None = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o", help="Output directory for runs. Uses config default."
    ),
    config_file: str | None = typer.Option(
        None, "--config", "-c", help="Path to config file (YAML/TOML)"
    ),
    ab_test_system_prompt: bool = typer.Option(
        False,
        "--ab-test-system-prompt",
        help="Run A/B test: generate with and without system prompt. Doubles API calls.",
    ),
    json_schema: str | None = typer.Option(
        None,
        "--json-schema",
        help="Path to JSON schema file for validating generator outputs. Uses config default if not provided.",
    ),
) -> None:
    """
    Generate a completion from an LLM using system and user prompts.

    This command reads prompts from files, calls the LLM API, and saves
    the output along with metadata to the runs directory.

    System prompt can be either a file path or a template key defined in prompt_evaluator.yaml.
    Configuration defaults are used when CLI flags are omitted.
    """
    try:
        # Load configurations using shared config manager
        app_config = _config_manager.get_app_config(
            config_path=Path(config_file) if config_file else None,
            warn_if_missing=False
        )

        # Load API configuration using shared config manager
        config_path = Path(config_file) if config_file else None
        api_config = _config_manager.get_api_config(config_file_path=config_path)

        # Resolve system prompt path (supports template keys)
        try:
            system_prompt_path = resolve_prompt_path(system_prompt, app_config)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        system_prompt_content = system_prompt_path.read_text(encoding="utf-8")

        # Read user input (from file or stdin)
        if input_path == "-":
            # Read from stdin - for very large inputs, consider streaming
            user_prompt_content = sys.stdin.read()
            input_file_path = Path("<stdin>")
        else:
            input_file_path = Path(input_path)
            if not input_file_path.exists():
                typer.echo(f"Error: Input file not found: {input_path}", err=True)
                raise typer.Exit(1)
            user_prompt_content = input_file_path.read_text(encoding="utf-8")

        # Determine provider with proper precedence: CLI flag > app config > hardcoded default
        if provider is None and app_config is not None:
            provider = app_config.defaults.generator.provider
            typer.echo(f"Using provider from config: {provider}", err=True)
        elif provider is None:
            provider = "openai"
            typer.echo(f"Using default provider: {provider}", err=True)

        # Determine output directory with proper precedence
        if output_dir is None and app_config is not None:
            output_dir = app_config.defaults.run_directory
            typer.echo(f"Using output directory from config: {output_dir}", err=True)
        elif output_dir is None:
            output_dir = "runs"

        # Load JSON schema if provided (CLI flag > config default)
        schema_dict: dict[str, Any] | None = None
        schema_path: Path | None = None
        schema_path_str = json_schema
        if schema_path_str is None and app_config is not None:
            schema_path_str = app_config.defaults.json_schema
            if schema_path_str:
                typer.echo(f"Using JSON schema from config: {schema_path_str}", err=True)
        
        if schema_path_str:
            from prompt_evaluator.schema_validation import load_json_schema
            
            try:
                # Resolve schema path (may be relative to config)
                if app_config and not Path(schema_path_str).is_absolute():
                    schema_path = app_config.resolve_path(schema_path_str)
                else:
                    schema_path = Path(schema_path_str)
                
                schema_dict = load_json_schema(schema_path)
                typer.echo(f"✓ Loaded JSON schema from: {schema_path}", err=True)
            except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
                typer.echo(f"Error loading JSON schema: {e}", err=True)
                raise typer.Exit(1)

        # Build GeneratorConfig with proper precedence
        # Precedence: CLI flags > API config (env vars) > app config > hardcoded defaults
        app_gen_config = app_config.defaults.generator if app_config else None

        final_model_name = (
            model
            or api_config.model_name
            or (app_gen_config.model if app_gen_config else None)
            or "gpt-5.1"
        )
        final_temperature = (
            temperature
            if temperature is not None
            else (app_gen_config.temperature if app_gen_config else None)
            if (app_gen_config.temperature if app_gen_config else None) is not None
            else 0.7
        )
        final_max_tokens = (
            max_completion_tokens
            or (app_gen_config.max_completion_tokens if app_gen_config else None)
            or 1024
        )

        generator_config = GeneratorConfig(
            model_name=final_model_name,
            temperature=final_temperature,
            max_completion_tokens=final_max_tokens,
            seed=seed,
        )

        # Create provider
        # Providers handle API key detection internally via environment variables
        # Only pass api_key for OpenAI to support custom config
        provider_api_key = api_config.api_key if provider.lower() == "openai" else None

        try:
            provider_instance = get_provider(
                provider,
                api_key=provider_api_key,
                base_url=api_config.base_url if provider.lower() == "openai" else None
            )
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        # Warn about doubled API calls if A/B testing is enabled
        if ab_test_system_prompt:
            typer.echo(
                "⚠️  WARNING: A/B testing mode enabled. This will DOUBLE your API calls and costs.",
                err=True,
            )
            typer.echo(
                "    Generating TWO completions: one with system prompt, one without.",
                err=True,
            )
            typer.echo("", err=True)

        # Prepare output directory
        run_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        output_dir_path = Path(output_dir)

        # Validate output path is a directory
        if output_dir_path.exists() and not output_dir_path.is_dir():
            typer.echo(
                f"Error: Output path '{output_dir}' exists and is not a directory.", err=True
            )
            raise typer.Exit(1)

        run_dir = output_dir_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Determine variants to run
        variants = []
        if ab_test_system_prompt:
            variants = [
                ("with_prompt", system_prompt_content),
                ("no_prompt", ""),
            ]
        else:
            variants = [(None, system_prompt_content)]

        # Generate completions for each variant
        results = []
        for variant_name, variant_system_prompt in variants:
            variant_label = f" ({variant_name})" if variant_name else ""
            typer.echo(f"Generating completion{variant_label}...", err=True)

            try:
                response_text, metadata = generate_completion(
                    provider=provider_instance,
                    system_prompt=variant_system_prompt,
                    user_prompt=user_prompt_content,
                    model=generator_config.model_name,
                    temperature=generator_config.temperature,
                    max_completion_tokens=generator_config.max_completion_tokens,
                    seed=generator_config.seed,
                    json_schema=schema_dict,
                )

                # Validate against JSON schema if provided
                schema_validation_status = "not_validated"
                schema_validation_error = None
                if schema_dict:
                    from prompt_evaluator.schema_validation import validate_json_output
                    
                    is_valid, error_msg, parsed_json = validate_json_output(
                        response_text,
                        schema_dict,
                        schema_path if schema_path_str else None,
                    )
                    
                    if is_valid:
                        schema_validation_status = "valid"
                        typer.echo(f"  ✓ Output validated against JSON schema{variant_label}", err=True)
                    else:
                        schema_validation_status = "invalid_json" if "Invalid JSON" in error_msg else "schema_mismatch"
                        schema_validation_error = error_msg
                        typer.echo(
                            f"  ✗ Schema validation failed{variant_label}: {error_msg}",
                            err=True,
                        )

                results.append({
                    "variant": variant_name,
                    "response_text": response_text,
                    "metadata": metadata,
                    "status": "completed",
                    "schema_validation_status": schema_validation_status,
                    "schema_validation_error": schema_validation_error,
                })

            except Exception as e:
                typer.echo(
                    f"  ✗ Error generating{variant_label}: {e}",
                    err=True,
                )
                results.append({
                    "variant": variant_name,
                    "response_text": "",
                    "metadata": {},
                    "status": "generation_error",
                    "error": str(e),
                })

        # Validate that at least one variant succeeded
        if not any(r["status"] == "completed" for r in results):
            typer.echo(
                "\nError: All variants failed. No outputs generated.",
                err=True,
            )
            raise typer.Exit(1)

        # Write outputs for each variant
        for idx, result in enumerate(results):
            variant = result["variant"]
            response_text = result["response_text"]
            metadata = result["metadata"]

            # Determine file suffix
            if variant:
                suffix = f"_{variant}"
            else:
                suffix = ""

            # Write output
            output_file = run_dir / f"output{suffix}.txt"
            output_file.write_text(response_text, encoding="utf-8")

            # Write metadata
            prompt_run = PromptRun(
                id=run_id,
                timestamp=timestamp,
                system_prompt_path=system_prompt_path,
                input_path=input_file_path,
                model_config=generator_config,
                raw_output_path=output_file,
            )
            metadata_file = run_dir / f"metadata{suffix}.json"
            metadata_dict = prompt_run.to_dict()
            metadata_dict.update(metadata)
            if variant:
                metadata_dict["ab_variant"] = variant
            if result["status"] != "completed":
                metadata_dict["status"] = result["status"]
                if "error" in result:
                    metadata_dict["error"] = result["error"]
            # Add schema validation results if present
            if "schema_validation_status" in result:
                metadata_dict["schema_validation_status"] = result["schema_validation_status"]
            if "schema_validation_error" in result and result["schema_validation_error"]:
                metadata_dict["schema_validation_error"] = result["schema_validation_error"]
            if schema_path:
                metadata_dict["json_schema_path"] = str(schema_path)
            metadata_file.write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")

        # Print outputs to stdout (all variants)
        for result in results:
            if result["variant"]:
                typer.echo(f"\n{'='*60}")
                typer.echo(f"Variant: {result['variant']}")
                typer.echo('='*60)
            typer.echo(result["response_text"])

        # Print summary to stderr
        typer.echo("\n" + "=" * 60, err=True)
        typer.echo(f"Run ID: {run_id}", err=True)
        typer.echo(f"Model: {generator_config.model_name}", err=True)
        typer.echo(f"Temperature: {generator_config.temperature}", err=True)
        typer.echo(f"Max tokens: {generator_config.max_completion_tokens}", err=True)
        if generator_config.seed is not None:
            typer.echo(f"Seed: {generator_config.seed}", err=True)

        if ab_test_system_prompt:
            typer.echo(f"\nA/B Test Results:", err=True)
            for result in results:
                variant = result["variant"]
                status_icon = "✓" if result["status"] == "completed" else "✗"
                typer.echo(f"  {status_icon} {variant}:", err=True)
                if result["status"] == "completed":
                    output_file = run_dir / f"output_{variant}.txt"
                    metadata_file = run_dir / f"metadata_{variant}.json"
                    typer.echo(f"     Output: {output_file}", err=True)
                    typer.echo(f"     Metadata: {metadata_file}", err=True)
                    if result["metadata"].get("tokens_used"):
                        typer.echo(f"     Tokens: {result['metadata']['tokens_used']}", err=True)
                    typer.echo(f"     Latency: {result['metadata']['latency_ms']:.2f}ms", err=True)
                else:
                    typer.echo(f"     Error: {result.get('error', 'Unknown')}", err=True)
        else:
            output_file = run_dir / "output.txt"
            metadata_file = run_dir / "metadata.json"
            typer.echo(f"Output file: {output_file}", err=True)
            typer.echo(f"Metadata file: {metadata_file}", err=True)
            if results[0]["status"] == "completed":
                if results[0]["metadata"].get("tokens_used"):
                    typer.echo(f"Tokens used: {results[0]['metadata']['tokens_used']}", err=True)
                typer.echo(f"Latency: {results[0]['metadata']['latency_ms']:.2f}ms", err=True)

        typer.echo("=" * 60, err=True)

    except ValueError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(1)
    except FileNotFoundError as e:
        typer.echo(f"File error: {e}", err=True)
        raise typer.Exit(1)
    except (KeyboardInterrupt, SystemExit):
        # Allow system signals to propagate
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def evaluate_single(
    system_prompt: str = typer.Option(
        ..., "--system-prompt", "-s", help="Path to generator system prompt file or template key"
    ),
    input_path: str = typer.Option(..., "--input", "-i", help="Path to input file"),
    num_samples: int = typer.Option(
        ..., "--num-samples", "-n", help="Number of samples to generate"
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="Provider for both generator and judge (deprecated, use --generator-provider and --judge-provider). Uses config default.",
    ),
    generator_provider: str | None = typer.Option(
        None,
        "--generator-provider",
        help="Provider for generator (openai, claude, anthropic, mock). Overrides --provider. Uses config default.",
    ),
    judge_provider: str | None = typer.Option(
        None,
        "--judge-provider",
        help="Provider for judge (openai, claude, anthropic, mock). Overrides --provider. Uses config default.",
    ),
    generator_model: str | None = typer.Option(
        None, "--generator-model", help="Generator model name override"
    ),
    judge_model: str | None = typer.Option(None, "--judge-model", help="Judge model name override"),
    judge_max_tokens: int | None = typer.Option(
        None,
        "--judge-max-tokens",
        help="Maximum tokens for judge responses (default: 1024). Higher values allow more detailed evaluation rationales.",
        min=1,
        max=32000,
    ),
    judge_temperature: float | None = typer.Option(
        None,
        "--judge-temperature",
        help="Judge temperature (0.0-2.0). Default: 0.0 for deterministic judging.",
        min=0.0,
        max=2.0,
    ),
    judge_top_p: float | None = typer.Option(
        None,
        "--judge-top-p",
        help="Judge nucleus sampling parameter (0.0-1.0). Controls response diversity.",
        min=0.0,
        max=1.0,
    ),
    judge_system_prompt: str | None = typer.Option(
        None,
        "--judge-system-prompt",
        help="Path to custom judge system prompt file or template key",
    ),
    rubric: str | None = typer.Option(
        None,
        "--rubric",
        help=(
            "Rubric to use for evaluation. Can be a preset alias "
            "(default, content-quality, code-review) or a path to a rubric file "
            "(.yaml/.json). Uses config default if not specified."
        ),
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Random seed for generator reproducibility"
    ),
    temperature: float | None = typer.Option(
        None, "--temperature", "-t", help="Generator temperature (0.0-2.0)"
    ),
    max_completion_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Maximum tokens for generator"
    ),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o", help="Output directory for runs. Uses config default."
    ),
    config_file: str | None = typer.Option(
        None, "--config", "-c", help="Path to config file (YAML/TOML)"
    ),
    task_description: str | None = typer.Option(
        None, "--task-description", help="Optional task description for judge context"
    ),
    prompt_version: str | None = typer.Option(
        None,
        "--prompt-version",
        help=(
            "Version identifier for the prompt. If not provided, the SHA-256 hash "
            "of the system prompt file will be used as both prompt_version_id and prompt_hash."
        ),
    ),
    run_note: str | None = typer.Option(
        None, "--run-note", help="Optional note about this evaluation run"
    ),
    ab_test_system_prompt: bool = typer.Option(
        False,
        "--ab-test-system-prompt",
        help="Run A/B test: evaluate with and without system prompt. Doubles API calls.",
    ),
    json_schema: str | None = typer.Option(
        None,
        "--json-schema",
        help="Path to JSON schema file for validating generator outputs. Uses config default if not provided.",
    ),
) -> None:
    """
    Evaluate a prompt by generating N samples and judging each output.

    This command generates multiple completions for the same input, evaluates
    each output using a judge model, and produces aggregate statistics.

    System prompt and judge prompt can be either file paths or template keys defined
    in prompt_evaluator.yaml. Configuration defaults are used when CLI flags are omitted.
    """
    try:
        # Validate num_samples
        if num_samples <= 0:
            typer.echo("Error: --num-samples must be positive", err=True)
            raise typer.Exit(1)

        # Load configurations using shared config manager
        app_config = _config_manager.get_app_config(
            config_path=Path(config_file) if config_file else None,
            warn_if_missing=False
        )

        # Load API configuration using shared config manager
        config_path = Path(config_file) if config_file else None
        api_config = _config_manager.get_api_config(config_file_path=config_path)

        # Resolve system prompt path (supports template keys)
        try:
            system_prompt_path = resolve_prompt_path(system_prompt, app_config)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        system_prompt_content = system_prompt_path.read_text(encoding="utf-8")

        # Read user input
        input_file_path = Path(input_path)
        if not input_file_path.exists():
            typer.echo(f"Error: Input file not found: {input_path}", err=True)
            raise typer.Exit(1)

        user_prompt_content = input_file_path.read_text(encoding="utf-8")

        # Determine generator provider with proper precedence
        # Precedence: --generator-provider > --provider > app config > hardcoded default
        final_generator_provider = generator_provider or provider
        if final_generator_provider is None:
            if app_config is not None and app_config.defaults.generator.provider:
                final_generator_provider = app_config.defaults.generator.provider
                typer.echo(f"Using generator provider from config: {final_generator_provider}", err=True)
            else:
                final_generator_provider = "openai"
                typer.echo(f"Using default generator provider: {final_generator_provider}", err=True)
        
        # Determine judge provider with proper precedence
        # Precedence: --judge-provider > --provider > app config > hardcoded default
        final_judge_provider = judge_provider or provider
        if final_judge_provider is None:
            if app_config is not None and app_config.defaults.judge.provider:
                final_judge_provider = app_config.defaults.judge.provider
                typer.echo(f"Using judge provider from config: {final_judge_provider}", err=True)
            else:
                final_judge_provider = "openai"
                typer.echo(f"Using default judge provider: {final_judge_provider}", err=True)

        # Determine output directory with proper precedence
        if output_dir is None and app_config is not None:
            output_dir = app_config.defaults.run_directory
            typer.echo(f"Using output directory from config: {output_dir}", err=True)
        elif output_dir is None:
            output_dir = "runs"

        # Load rubric with proper precedence: CLI flag > app config > hardcoded default
        if rubric is None and app_config is not None and app_config.defaults.rubric is not None:
            rubric = app_config.defaults.rubric
            typer.echo(f"Using default rubric from config: {rubric}", err=True)

        try:
            from prompt_evaluator.config import load_rubric, resolve_rubric_path

            rubric_path = resolve_rubric_path(rubric)
            loaded_rubric = load_rubric(rubric_path)
            typer.echo(f"Using rubric: {rubric_path}", err=True)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error loading rubric: {e}", err=True)
            raise typer.Exit(1)

        # Load judge prompt (custom or default, supports template keys)
        try:
            if judge_system_prompt:
                judge_prompt_path = resolve_prompt_path(judge_system_prompt, app_config)
                judge_prompt_content = load_judge_prompt(judge_prompt_path)
            else:
                judge_prompt_content = load_judge_prompt()
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error loading judge prompt: {e}", err=True)
            raise typer.Exit(1)

        # Build GeneratorConfig with proper precedence
        # Precedence: CLI flags > API config (env vars) > app config > hardcoded defaults
        app_gen_config = app_config.defaults.generator if app_config else None

        final_model_name = (
            generator_model
            or api_config.model_name
            or (app_gen_config.model if app_gen_config else None)
            or "gpt-5.1"
        )
        final_temperature = (
            temperature
            if temperature is not None
            else (app_gen_config.temperature if app_gen_config else None)
            if (app_gen_config.temperature if app_gen_config else None) is not None
            else 0.7
        )
        final_max_tokens = (
            max_completion_tokens
            or (app_gen_config.max_completion_tokens if app_gen_config else None)
            or 1024
        )

        generator_config = GeneratorConfig(
            model_name=final_model_name,
            temperature=final_temperature,
            max_completion_tokens=final_max_tokens,
            seed=seed,
        )

        # Build JudgeConfig with proper precedence
        # Precedence: CLI flags > app config > hardcoded defaults
        app_judge_config = app_config.defaults.judge if app_config else None

        final_judge_model = (
            judge_model
            or (app_judge_config.model if app_judge_config else None)
            or "gpt-5.1"
        )

        final_judge_max_tokens = (
            judge_max_tokens
            or (app_judge_config.max_completion_tokens if app_judge_config else None)
            or 1024
        )
        
        final_judge_temperature = (
            judge_temperature
            if judge_temperature is not None
            else (app_judge_config.temperature if app_judge_config else 0.0)
        )
        
        final_judge_top_p = (
            judge_top_p
            if judge_top_p is not None
            else (app_judge_config.top_p if app_judge_config else None)
        )
        
        judge_config = JudgeConfig(
            model_name=final_judge_model,
            temperature=final_judge_temperature,
            max_completion_tokens=final_judge_max_tokens,
            top_p=final_judge_top_p,
        )

        # Create separate provider instances for generator and judge
        # Providers handle API key detection internally via environment variables
        # Only pass api_key for OpenAI to support custom config
        generator_api_key = api_config.api_key if final_generator_provider.lower() == "openai" else None
        judge_api_key = api_config.api_key if final_judge_provider.lower() == "openai" else None

        try:
            generator_provider_instance = get_provider(
                final_generator_provider,
                api_key=generator_api_key,
                base_url=api_config.base_url if final_generator_provider.lower() == "openai" else None
            )
            judge_provider_instance = get_provider(
                final_judge_provider,
                api_key=judge_api_key,
                base_url=api_config.base_url if final_judge_provider.lower() == "openai" else None
            )
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        # Create run metadata
        run_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        output_dir_path = Path(output_dir)

        # Validate output path is a directory
        if output_dir_path.exists() and not output_dir_path.is_dir():
            typer.echo(
                f"Error: Output path '{output_dir}' exists and is not a directory.", err=True
            )
            raise typer.Exit(1)

        run_dir = output_dir_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Warn about doubled API calls if A/B testing is enabled
        if ab_test_system_prompt:
            typer.echo(
                "⚠️  WARNING: A/B testing mode enabled. This will DOUBLE your API calls and costs.",
                err=True,
            )
            typer.echo(
                f"    Generating {num_samples * 2} total samples: {num_samples} with system prompt, {num_samples} without.",
                err=True,
            )
            typer.echo("", err=True)

        typer.echo(f"Starting evaluation run {run_id}...", err=True)

        if ab_test_system_prompt:
            typer.echo(f"Generating {num_samples} samples per variant (2 variants)...", err=True)
        else:
            typer.echo(f"Generating {num_samples} samples...", err=True)

        # Determine variants to run
        variants = []
        if ab_test_system_prompt:
            variants = [
                ("with_prompt", system_prompt_content),
                ("no_prompt", ""),
            ]
            total_samples = num_samples * 2
        else:
            variants = [(None, system_prompt_content)]
            total_samples = num_samples

        # Generate and judge samples
        samples: list[Sample] = []
        sample_idx = 0

        for variant_name, variant_system_prompt in variants:
            if variant_name:
                typer.echo(f"\nVariant: {variant_name}", err=True)

            for i in range(num_samples):
                sample_idx += 1
                if variant_name:
                    sample_id = f"{run_id}-{variant_name}-sample-{i + 1}"
                else:
                    sample_id = f"{run_id}-sample-{i + 1}"

                if variant_name:
                    typer.echo(f"  Sample {i + 1}/{num_samples} ({variant_name})...", err=True)
                else:
                    typer.echo(f"  Sample {sample_idx}/{num_samples}...", err=True)

                try:
                    # Generate completion with varied seed per sample
                    current_seed = generator_config.seed + i if generator_config.seed is not None else None
                    response_text, metadata = generate_completion(
                        provider=generator_provider_instance,
                        system_prompt=variant_system_prompt,
                        user_prompt=user_prompt_content,
                        model=generator_config.model_name,
                        temperature=generator_config.temperature,
                        max_completion_tokens=generator_config.max_completion_tokens,
                        seed=current_seed,
                    )

                    # Judge the output
                    judge_result = judge_completion(
                        provider=judge_provider_instance,
                        input_text=user_prompt_content,
                        generator_output=response_text,
                        judge_config=judge_config,
                        judge_system_prompt=judge_prompt_content,
                        task_description=task_description,
                        rubric=loaded_rubric,
                    )

                    # Create sample with judge results
                    sample = Sample(
                        sample_id=sample_id,
                        input_text=user_prompt_content,
                        generator_output=response_text,
                        judge_score=judge_result.get("judge_score"),
                        judge_rationale=judge_result.get("judge_rationale"),
                        judge_raw_response=judge_result.get("judge_raw_response"),
                        status=judge_result["status"],
                        task_description=task_description,
                        judge_metrics=judge_result.get("judge_metrics", {}),
                        judge_flags=judge_result.get("judge_flags", {}),
                        judge_overall_comment=judge_result.get("judge_overall_comment"),
                        ab_variant=variant_name,
                    )

                    samples.append(sample)

                    # Display progress
                    if sample.status == "completed":
                        if sample.judge_score is not None:
                            # Legacy single-metric mode
                            typer.echo(
                                f"    ✓ Generated and judged (score: {sample.judge_score:.1f}/5.0)",
                                err=True,
                            )
                        else:
                            # Rubric mode - show summary of metrics
                            num_metrics = len(sample.judge_metrics)
                            typer.echo(
                                f"    ✓ Generated and judged ({num_metrics} metrics evaluated)",
                                err=True,
                            )
                    elif sample.status == "judge_invalid_response":
                        error_msg = judge_result.get("error", "Invalid judge response")
                        typer.echo(f"    ⚠ Generated but judge response invalid: {error_msg}", err=True)
                    else:
                        error_msg = judge_result.get("error")
                        typer.echo(f"    ⚠ Generated but judge error: {error_msg}", err=True)

                except Exception as e:
                    typer.echo(f"    ✗ Error generating sample: {e}", err=True)
                    # Create a sample with error status
                    sample = Sample(
                        sample_id=sample_id,
                        input_text=user_prompt_content,
                        generator_output="",
                        status="generation_error",
                        task_description=task_description,
                        ab_variant=variant_name,
                    )
                    samples.append(sample)

        # Compute aggregate statistics with rubric awareness
        stats = compute_aggregate_statistics(samples, loaded_rubric)

        # If A/B testing, also compute per-variant statistics
        variant_stats = {}
        if ab_test_system_prompt:
            for variant_name, _ in variants:
                if variant_name:
                    variant_samples = [s for s in samples if s.ab_variant == variant_name]
                    variant_stats[variant_name] = compute_aggregate_statistics(
                        variant_samples, loaded_rubric
                    )

        # Compute rubric metadata
        rubric_metadata = compute_rubric_metadata(loaded_rubric, rubric_path)

        # Compute prompt metadata
        try:
            prompt_version_id, prompt_hash = compute_prompt_metadata(
                system_prompt_path, prompt_version
            )
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        # Create SingleEvaluationRun
        evaluation_run = SingleEvaluationRun(
            run_id=run_id,
            timestamp=timestamp,
            num_samples=total_samples if ab_test_system_prompt else num_samples,
            generator_config=generator_config,
            judge_config=judge_config,
            samples=samples,
            prompt_version_id=prompt_version_id,
            prompt_hash=prompt_hash,
            run_notes=run_note,
        )

        # Save evaluation results
        evaluation_file = run_dir / "evaluate-single.json"
        evaluation_dict = evaluation_run.to_dict()
        # Add aggregate statistics
        evaluation_dict["aggregate_stats"] = stats
        # Add variant statistics if A/B testing
        if ab_test_system_prompt:
            evaluation_dict["ab_test_enabled"] = True
            evaluation_dict["variant_stats"] = variant_stats
        # Add rubric metadata
        if rubric_metadata:
            evaluation_dict["rubric_metadata"] = rubric_metadata
        # Add metadata about prompts
        evaluation_dict["system_prompt_path"] = str(system_prompt_path)
        evaluation_dict["input_path"] = str(input_file_path)
        if judge_system_prompt:
            evaluation_dict["judge_system_prompt_path"] = str(Path(judge_system_prompt))

        evaluation_file.write_text(json.dumps(evaluation_dict, indent=2), encoding="utf-8")

        # Print summary to stderr
        typer.echo("\n" + "=" * 60, err=True)
        typer.echo("Evaluation Complete!", err=True)
        typer.echo(f"Run ID: {run_id}", err=True)
        typer.echo(f"Generator Model: {generator_config.model_name}", err=True)
        typer.echo(f"Judge Model: {judge_config.model_name}", err=True)
        if ab_test_system_prompt:
            typer.echo(f"A/B Test Mode: Enabled", err=True)
            typer.echo(f"Total Samples: {total_samples} ({num_samples} per variant)", err=True)
        else:
            typer.echo(f"Total Samples: {num_samples}", err=True)
        typer.echo(f"Successful: {stats['num_successful']}", err=True)
        typer.echo(f"Failed: {stats['num_failed']}", err=True)

        # Print variant statistics if A/B testing
        if ab_test_system_prompt:
            typer.echo("\nA/B Test Results:", err=True)
            for variant_name in ["with_prompt", "no_prompt"]:
                v_stats = variant_stats.get(variant_name, {})
                if not v_stats:
                    continue
                typer.echo(f"\n  Variant: {variant_name}", err=True)
                typer.echo(f"    Successful: {v_stats.get('num_successful', 0)}", err=True)
                typer.echo(f"    Failed: {v_stats.get('num_failed', 0)}", err=True)
                if v_stats.get("mean_score") is not None:
                    typer.echo(f"    Mean Score: {v_stats['mean_score']:.2f}/5.0", err=True)

        # Print legacy statistics if available (only for non-A/B mode)
        if not ab_test_system_prompt and stats["mean_score"] is not None:
            typer.echo("\nLegacy Aggregate Statistics:", err=True)
            typer.echo(f"  Mean Score: {stats['mean_score']:.2f}/5.0", err=True)
            typer.echo(f"  Min Score:  {stats['min_score']:.2f}/5.0", err=True)
            typer.echo(f"  Max Score:  {stats['max_score']:.2f}/5.0", err=True)

        # Print per-metric statistics if available (only for non-A/B mode to avoid clutter)
        if not ab_test_system_prompt and "metric_stats" in stats and stats["metric_stats"]:
            typer.echo("\nPer-Metric Statistics:", err=True)
            for metric_name, metric_data in stats["metric_stats"].items():
                if metric_data["count"] > 0:
                    typer.echo(
                        f"  {metric_name}: "
                        f"mean={metric_data['mean']:.2f}, "
                        f"min={metric_data['min']:.2f}, "
                        f"max={metric_data['max']:.2f}, "
                        f"count={metric_data['count']}",
                        err=True,
                    )
                else:
                    typer.echo(f"  {metric_name}: No valid scores", err=True)

        # Print per-flag statistics if available (only for non-A/B mode to avoid clutter)
        if not ab_test_system_prompt and "flag_stats" in stats and stats["flag_stats"]:
            typer.echo("\nPer-Flag Statistics:", err=True)
            for flag_name, flag_data in stats["flag_stats"].items():
                if flag_data["total_count"] > 0:
                    typer.echo(
                        f"  {flag_name}: "
                        f"true={flag_data['true_count']}, "
                        f"false={flag_data['false_count']}, "
                        f"proportion={flag_data['true_proportion']:.2%}",
                        err=True,
                    )
                else:
                    typer.echo(f"  {flag_name}: No samples evaluated", err=True)

        if stats.get("num_successful") == 0:
            typer.echo("\nNo successful samples to compute statistics.", err=True)

        typer.echo(f"\nResults saved to: {evaluation_file}", err=True)
        typer.echo("=" * 60, err=True)

        # Print JSON output to stdout for programmatic use
        typer.echo(json.dumps(evaluation_dict, indent=2))

    except ValueError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(1)
    except FileNotFoundError as e:
        typer.echo(f"File error: {e}", err=True)
        raise typer.Exit(1)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def show_rubric(
    rubric: str | None = typer.Option(
        None,
        "--rubric",
        help=(
            "Rubric to display. Can be a preset alias "
            "(default, content-quality, code-review) or a path to a rubric file "
            "(.yaml/.json). Defaults to 'default' if not specified."
        ),
    ),
) -> None:
    """
    Display the effective rubric as JSON without running an evaluation.

    This command loads and validates a rubric file, then outputs it as formatted
    JSON for inspection. Useful for validating rubric files before running evaluations.
    """
    try:
        from prompt_evaluator.config import load_rubric, resolve_rubric_path

        # Resolve and load the rubric
        rubric_path = resolve_rubric_path(rubric)
        loaded_rubric = load_rubric(rubric_path)

        # Convert rubric to dictionary for JSON output using dataclass serialization
        rubric_dict = asdict(loaded_rubric)
        rubric_dict["rubric_path"] = str(rubric_path)

        # Print the rubric as formatted JSON to stdout
        typer.echo(json.dumps(rubric_dict, indent=2))

    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error loading rubric: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def evaluate_dataset(
    dataset: str = typer.Option(
        ..., "--dataset", "-d", help="Path to dataset file (.yaml/.yml or .jsonl) or dataset key"
    ),
    system_prompt: str = typer.Option(
        ..., "--system-prompt", "-s", help="Path to generator system prompt file or template key"
    ),
    num_samples: int | None = typer.Option(
        None,
        "--num-samples",
        "-n",
        help="Number of samples to generate per test case (default: 5)",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="Provider for both generator and judge (deprecated, use --generator-provider and --judge-provider). Uses config default.",
    ),
    generator_provider: str | None = typer.Option(
        None,
        "--generator-provider",
        help="Provider for generator (openai, claude, anthropic, mock). Overrides --provider. Uses config default.",
    ),
    judge_provider: str | None = typer.Option(
        None,
        "--judge-provider",
        help="Provider for judge (openai, claude, anthropic, mock). Overrides --provider. Uses config default.",
    ),
    generator_model: str | None = typer.Option(
        None, "--generator-model", help="Generator model name override"
    ),
    judge_model: str | None = typer.Option(None, "--judge-model", help="Judge model name override"),
    judge_max_tokens: int | None = typer.Option(
        None,
        "--judge-max-tokens",
        help="Maximum tokens for judge responses (default: 1024). Higher values allow more detailed evaluation rationales.",
        min=1,
        max=32000,
    ),
    judge_temperature: float | None = typer.Option(
        None,
        "--judge-temperature",
        help="Judge temperature (0.0-2.0). Default: 0.0 for deterministic judging.",
        min=0.0,
        max=2.0,
    ),
    judge_top_p: float | None = typer.Option(
        None,
        "--judge-top-p",
        help="Judge nucleus sampling parameter (0.0-1.0). Controls response diversity.",
        min=0.0,
        max=1.0,
    ),
    judge_system_prompt: str | None = typer.Option(
        None,
        "--judge-system-prompt",
        help="Path to custom judge system prompt file or template key",
    ),
    rubric: str | None = typer.Option(
        None,
        "--rubric",
        help=(
            "Rubric to use for evaluation. Can be a preset alias "
            "(default, content-quality, code-review) or a path to a rubric file "
            "(.yaml/.json). Uses config default if not specified."
        ),
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Random seed for generator reproducibility"
    ),
    temperature: float | None = typer.Option(
        None, "--temperature", "-t", help="Generator temperature (0.0-2.0)"
    ),
    max_completion_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Maximum tokens for generator"
    ),
    case_ids: str | None = typer.Option(
        None, "--case-ids", help="Comma-separated list of test case IDs to evaluate"
    ),
    max_cases: int | None = typer.Option(
        None, "--max-cases", help="Maximum number of test cases to evaluate"
    ),
    quick: bool = typer.Option(
        False, "--quick", help="Quick mode: sets --num-samples=2 unless explicitly overridden"
    ),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o", help="Output directory for runs. Uses config default."
    ),
    config_file: str | None = typer.Option(
        None, "--config", "-c", help="Path to config file (YAML/TOML)"
    ),
    prompt_version: str | None = typer.Option(
        None,
        "--prompt-version",
        help=(
            "Version identifier for the prompt. If not provided, the SHA-256 hash "
            "of the system prompt file will be used as both prompt_version_id and prompt_hash."
        ),
    ),
    run_note: str | None = typer.Option(
        None, "--run-note", help="Optional note about this evaluation run"
    ),
    ab_test_system_prompt: bool = typer.Option(
        False,
        "--ab-test-system-prompt",
        help="Run A/B test: evaluate with and without system prompt. Doubles API calls per case.",
    ),
) -> None:
    """
    Evaluate a dataset of test cases with multiple samples per case.

    This command loads a dataset, generates N samples for each test case,
    judges each sample, and computes per-case and overall statistics.
    Results are streamed to disk as JSON files.

    Supports filtering by case IDs, limiting total cases, quick mode for testing,
    and resuming from interrupted runs.

    Dataset and system prompt can be specified as file paths or keys defined in
    prompt_evaluator.yaml.
    """
    try:
        # Load configurations using shared config manager
        app_config = _config_manager.get_app_config(
            config_path=Path(config_file) if config_file else None,
            warn_if_missing=False
        )

        # Handle --quick flag and num_samples interaction
        if quick and num_samples is not None:
            typer.echo(
                "Warning: Both --quick and --num-samples provided. "
                f"Using explicit --num-samples={num_samples}",
                err=True,
            )
        elif quick:
            num_samples = QUICK_MODE_NUM_SAMPLES
            typer.echo(f"Quick mode: Using --num-samples={num_samples}", err=True)
        elif num_samples is None:
            # Default if neither quick nor explicit num_samples provided
            num_samples = DEFAULT_NUM_SAMPLES
            typer.echo(f"Using default --num-samples={num_samples}", err=True)

        # Validate num_samples
        if num_samples <= 0:
            typer.echo("Error: --num-samples must be positive", err=True)
            raise typer.Exit(1)

        # Load API configuration using shared config manager
        config_path = Path(config_file) if config_file else None
        api_config = _config_manager.get_api_config(config_file_path=config_path)

        # Load dataset (supports dataset keys from config)
        from prompt_evaluator.config import load_dataset

        try:
            dataset_path = resolve_dataset_path(dataset, app_config)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Loading dataset from {dataset_path}...", err=True)
        test_cases, dataset_metadata = load_dataset(dataset_path)
        typer.echo(f"Loaded {len(test_cases)} test cases", err=True)

        # Apply case-ids filter if provided
        if case_ids:
            requested_ids = [cid.strip() for cid in case_ids.split(",")]
            test_case_map = {tc.id: tc for tc in test_cases}
            available_ids = set(test_case_map.keys())

            # Validate all requested IDs exist
            unknown_ids = [cid for cid in requested_ids if cid not in available_ids]
            if unknown_ids:
                typer.echo(f"Error: Unknown test case IDs: {', '.join(unknown_ids)}", err=True)
                typer.echo(f"Available IDs: {', '.join(sorted(available_ids))}", err=True)
                raise typer.Exit(1)

            # Filter test cases, preserving user-specified order
            test_cases = [test_case_map[cid] for cid in requested_ids]
            typer.echo(f"Filtered to {len(test_cases)} test cases by --case-ids", err=True)

        # Apply max-cases filter if provided
        if max_cases is not None:
            if max_cases <= 0:
                typer.echo("Error: --max-cases must be positive", err=True)
                raise typer.Exit(1)

            if len(test_cases) > max_cases:
                test_cases = test_cases[:max_cases]
                typer.echo(f"Limited to first {max_cases} test cases by --max-cases", err=True)

        # Resolve system prompt path (supports template keys)
        try:
            system_prompt_path = resolve_prompt_path(system_prompt, app_config)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        system_prompt_content = system_prompt_path.read_text(encoding="utf-8")

        # Determine generator provider with proper precedence
        # Precedence: --generator-provider > --provider > app config > hardcoded default
        final_generator_provider = generator_provider or provider
        if final_generator_provider is None:
            if app_config is not None and app_config.defaults.generator.provider:
                final_generator_provider = app_config.defaults.generator.provider
                typer.echo(f"Using generator provider from config: {final_generator_provider}", err=True)
            else:
                final_generator_provider = "openai"
                typer.echo(f"Using default generator provider: {final_generator_provider}", err=True)
        
        # Determine judge provider with proper precedence
        # Precedence: --judge-provider > --provider > app config > hardcoded default
        final_judge_provider = judge_provider or provider
        if final_judge_provider is None:
            if app_config is not None and app_config.defaults.judge.provider:
                final_judge_provider = app_config.defaults.judge.provider
                typer.echo(f"Using judge provider from config: {final_judge_provider}", err=True)
            else:
                final_judge_provider = "openai"
                typer.echo(f"Using default judge provider: {final_judge_provider}", err=True)

        # Determine output directory with proper precedence
        if output_dir is None and app_config is not None:
            output_dir = app_config.defaults.run_directory
            typer.echo(f"Using output directory from config: {output_dir}", err=True)
        elif output_dir is None:
            output_dir = "runs"

        # Load rubric (preset, custom, or default)
        # If rubric not specified and app_config has a default, use it
        if rubric is None and app_config is not None and app_config.defaults.rubric is not None:
            rubric = app_config.defaults.rubric
            typer.echo(f"Using default rubric from config: {rubric}", err=True)

        try:
            from prompt_evaluator.config import load_rubric, resolve_rubric_path

            rubric_path = resolve_rubric_path(rubric)
            loaded_rubric = load_rubric(rubric_path)
            typer.echo(f"Using rubric: {rubric_path}", err=True)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error loading rubric: {e}", err=True)
            raise typer.Exit(1)

        # Load judge prompt (custom or default, supports template keys)
        try:
            if judge_system_prompt:
                judge_prompt_path = resolve_prompt_path(judge_system_prompt, app_config)
                judge_prompt_content = load_judge_prompt(judge_prompt_path)
            else:
                judge_prompt_content = load_judge_prompt()
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error loading judge prompt: {e}", err=True)
            raise typer.Exit(1)

        # Build GeneratorConfig with proper precedence
        # Precedence: CLI flags > API config (env vars) > app config > hardcoded defaults
        app_gen_config = app_config.defaults.generator if app_config else None

        final_model_name = (
            generator_model
            or api_config.model_name
            or (app_gen_config.model if app_gen_config else None)
            or "gpt-5.1"
        )
        final_temperature = (
            temperature
            if temperature is not None
            else (app_gen_config.temperature if app_gen_config else None)
            if (app_gen_config.temperature if app_gen_config else None) is not None
            else 0.7
        )
        final_max_tokens = (
            max_completion_tokens
            or (app_gen_config.max_completion_tokens if app_gen_config else None)
            or 1024
        )

        generator_config = GeneratorConfig(
            model_name=final_model_name,
            temperature=final_temperature,
            max_completion_tokens=final_max_tokens,
            seed=seed,
        )

        # Build JudgeConfig with proper precedence
        # Precedence: CLI flags > app config > hardcoded defaults
        app_judge_config = app_config.defaults.judge if app_config else None

        final_judge_model = (
            judge_model
            or (app_judge_config.model if app_judge_config else None)
            or "gpt-5.1"
        )
        
        final_judge_max_tokens = (
            judge_max_tokens
            or (app_judge_config.max_completion_tokens if app_judge_config else None)
            or 1024
        )
        
        final_judge_temperature = (
            judge_temperature
            if judge_temperature is not None
            else (app_judge_config.temperature if app_judge_config else 0.0)
        )
        
        final_judge_top_p = (
            judge_top_p
            if judge_top_p is not None
            else (app_judge_config.top_p if app_judge_config else None)
        )

        judge_config = JudgeConfig(
            model_name=final_judge_model,
            temperature=final_judge_temperature,
            max_completion_tokens=final_judge_max_tokens,
            top_p=final_judge_top_p,
        )

        # Create separate provider instances for generator and judge
        # Providers handle API key detection internally via environment variables
        # Only pass api_key for OpenAI to support custom config
        generator_api_key = api_config.api_key if final_generator_provider.lower() == "openai" else None
        judge_api_key = api_config.api_key if final_judge_provider.lower() == "openai" else None

        try:
            generator_provider_instance = get_provider(
                final_generator_provider,
                api_key=generator_api_key,
                base_url=api_config.base_url if final_generator_provider.lower() == "openai" else None
            )
            judge_provider_instance = get_provider(
                final_judge_provider,
                api_key=judge_api_key,
                base_url=api_config.base_url if final_judge_provider.lower() == "openai" else None
            )
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        # Prepare output directory
        output_dir_path = Path(output_dir)
        if output_dir_path.exists() and not output_dir_path.is_dir():
            typer.echo(
                f"Error: Output path '{output_dir}' exists and is not a directory.", err=True
            )
            raise typer.Exit(1)

        # Compute rubric metadata
        rubric_metadata = compute_rubric_metadata(loaded_rubric, rubric_path)

        # Compute prompt metadata
        try:
            prompt_version_id, prompt_hash = compute_prompt_metadata(
                system_prompt_path, prompt_version
            )
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        # Warn about doubled API calls if A/B testing is enabled
        if ab_test_system_prompt:
            total_api_calls = len(test_cases) * num_samples * 2
            typer.echo(
                "⚠️  WARNING: A/B testing mode enabled. This will DOUBLE your API calls and costs.",
                err=True,
            )
            typer.echo(
                f"    Total API calls: ~{total_api_calls} "
                f"({len(test_cases)} cases × {num_samples} samples × 2 variants)",
                err=True,
            )
            typer.echo("", err=True)

        # Run dataset evaluation
        typer.echo("\n" + "=" * 60, err=True)
        typer.echo("Starting Dataset Evaluation", err=True)
        typer.echo("=" * 60, err=True)
        typer.echo(f"Dataset: {dataset_path}", err=True)
        typer.echo(f"Test Cases: {len(test_cases)}", err=True)
        if ab_test_system_prompt:
            typer.echo(f"A/B Test Mode: Enabled", err=True)
            typer.echo(f"Samples per Case per Variant: {num_samples}", err=True)
            typer.echo(f"Total Samples per Case: {num_samples * 2}", err=True)
        else:
            typer.echo(f"Samples per Case: {num_samples}", err=True)
        typer.echo(f"Generator Model: {generator_config.model_name}", err=True)
        typer.echo(f"Judge Model: {judge_config.model_name}", err=True)
        typer.echo("=" * 60 + "\n", err=True)

        from prompt_evaluator.dataset_evaluation import evaluate_dataset as run_dataset_evaluation

        evaluation_run = run_dataset_evaluation(
            generator_provider=generator_provider_instance,
            judge_provider=judge_provider_instance,
            test_cases=test_cases,
            dataset_metadata=dataset_metadata,
            system_prompt=system_prompt_content,
            system_prompt_path=system_prompt_path,
            num_samples_per_case=num_samples,
            generator_config=generator_config,
            judge_config=judge_config,
            judge_system_prompt=judge_prompt_content,
            rubric=loaded_rubric,
            rubric_metadata=rubric_metadata,
            output_dir=output_dir_path,
            progress_callback=typer.echo,
            prompt_version_id=prompt_version_id,
            prompt_hash=prompt_hash,
            run_notes=run_note,
            ab_test_system_prompt=ab_test_system_prompt,
        )

        # Print summary
        typer.echo("\n" + "=" * 60, err=True)
        typer.echo("Dataset Evaluation Complete!", err=True)
        typer.echo("=" * 60, err=True)
        typer.echo(f"Run ID: {evaluation_run.run_id}", err=True)
        typer.echo(f"Status: {evaluation_run.status}", err=True)

        # Count results by status
        num_completed = sum(
            1 for tc in evaluation_run.test_case_results if tc.status == "completed"
        )
        num_partial = sum(1 for tc in evaluation_run.test_case_results if tc.status == "partial")
        num_failed = sum(1 for tc in evaluation_run.test_case_results if tc.status == "failed")

        typer.echo(f"Test Cases Completed: {num_completed}/{len(test_cases)}", err=True)
        if num_partial > 0:
            typer.echo(f"Test Cases Partial: {num_partial}", err=True)
        if num_failed > 0:
            typer.echo(f"Test Cases Failed: {num_failed}", err=True)

        # Print per-case metric statistics with std highlighting
        if evaluation_run.test_case_results:
            typer.echo("\nPer-Case Metric Statistics:", err=True)
            for tc in evaluation_run.test_case_results:
                if tc.status in ("completed", "partial") and tc.per_metric_stats:
                    typer.echo(f"\n  Case: {tc.test_case_id}", err=True)
                    for metric_name, metric_data in tc.per_metric_stats.items():
                        count_val = metric_data.get("count", 0)
                        if isinstance(count_val, int) and count_val > 0:
                            mean = metric_data.get("mean")
                            std = metric_data.get("std")

                            # Highlight high standard deviation
                            std_warning = ""
                            if (
                                std is not None
                                and mean is not None
                                and isinstance(std, (int, float))
                                and isinstance(mean, (int, float))
                                and count_val > 1
                            ):
                                is_high_absolute = std > HIGH_STD_ABSOLUTE_THRESHOLD
                                is_high_relative = (
                                    mean != 0 and (std / abs(mean)) > HIGH_STD_RELATIVE_THRESHOLD
                                )
                                if is_high_absolute or is_high_relative:
                                    std_warning = " ⚠️ HIGH VARIABILITY"

                            std_str = f"std={std:.2f}" if std is not None else "std=N/A"
                            mean_str = f"mean={mean:.2f}" if mean is not None else "mean=N/A"

                            typer.echo(
                                f"    {metric_name}: {mean_str}, {std_str}{std_warning}", err=True
                            )

        # Print overall metric statistics
        if evaluation_run.overall_metric_stats:
            typer.echo("\nOverall Metric Statistics (mean of per-case means):", err=True)
            for metric_name, metric_data in evaluation_run.overall_metric_stats.items():
                num_cases = metric_data.get("num_cases", 0)
                if isinstance(num_cases, int) and num_cases > 0:
                    mean_of_means = metric_data.get("mean_of_means")
                    min_of_means = metric_data.get("min_of_means")
                    max_of_means = metric_data.get("max_of_means")
                    if (
                        mean_of_means is not None
                        and min_of_means is not None
                        and max_of_means is not None
                    ):
                        typer.echo(
                            f"  {metric_name}: "
                            f"mean={mean_of_means:.2f}, "
                            f"min={min_of_means:.2f}, "
                            f"max={max_of_means:.2f}, "
                            f"cases={num_cases}",
                            err=True,
                        )
                    else:
                        typer.echo(f"  {metric_name}: No valid results", err=True)
                else:
                    typer.echo(f"  {metric_name}: No valid results", err=True)

        # Print overall flag statistics
        if evaluation_run.overall_flag_stats:
            typer.echo("\nOverall Flag Statistics:", err=True)
            for flag_name, flag_data in evaluation_run.overall_flag_stats.items():
                if flag_data["total_count"] > 0:
                    typer.echo(
                        f"  {flag_name}: "
                        f"true={flag_data['true_count']}, "
                        f"false={flag_data['false_count']}, "
                        f"proportion={flag_data['true_proportion']:.2%}",
                        err=True,
                    )
                else:
                    typer.echo(f"  {flag_name}: No samples evaluated", err=True)

        output_file = output_dir_path / evaluation_run.run_id / "dataset_evaluation.json"
        typer.echo(f"\nResults saved to: {output_file}", err=True)
        typer.echo("=" * 60, err=True)

        # Print JSON output to stdout for programmatic use
        typer.echo(json.dumps(evaluation_run.to_dict(), indent=2))

    except ValueError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(1)
    except FileNotFoundError as e:
        typer.echo(f"File error: {e}", err=True)
        raise typer.Exit(1)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def compare_runs(
    baseline: str = typer.Option(
        ..., "--baseline", "-b", help="Path to baseline run artifact JSON file"
    ),
    candidate: str = typer.Option(
        ..., "--candidate", "-c", help="Path to candidate run artifact JSON file"
    ),
    metric_threshold: float = typer.Option(
        0.1,
        "--metric-threshold",
        help="Absolute threshold for metric regression (default: 0.1)",
    ),
    flag_threshold: float = typer.Option(
        0.05,
        "--flag-threshold",
        help="Absolute threshold for flag regression (default: 0.05)",
    ),
    output_file: str | None = typer.Option(
        None, "--output", "-o", help="Optional output file for comparison JSON"
    ),
) -> None:
    """
    Compare two evaluation runs and detect regressions.

    This command compares baseline and candidate run artifacts, computing
    metric and flag deltas. Regressions are flagged when:
    - A metric's mean decreases by more than metric_threshold
    - A flag's true proportion increases by more than flag_threshold

    The comparison result includes:
    - Metric deltas (mean, delta, percent change, regression flag)
    - Flag deltas (proportion, delta, percent change, regression flag)
    - Overall regression summary

    Results are printed to stdout as JSON and optionally saved to a file.
    A human-readable summary is printed to stderr.
    """
    try:
        from prompt_evaluator.comparison import compare_runs as run_comparison

        baseline_path = Path(baseline)
        candidate_path = Path(candidate)

        # Validate thresholds
        if metric_threshold < 0:
            typer.echo("Error: --metric-threshold must be non-negative", err=True)
            raise typer.Exit(1)
        if flag_threshold < 0:
            typer.echo("Error: --flag-threshold must be non-negative", err=True)
            raise typer.Exit(1)

        typer.echo("Loading run artifacts...", err=True)
        typer.echo(f"  Baseline: {baseline_path}", err=True)
        typer.echo(f"  Candidate: {candidate_path}", err=True)

        # Run comparison
        result = run_comparison(
            baseline_artifact=baseline_path,
            candidate_artifact=candidate_path,
            metric_threshold=metric_threshold,
            flag_threshold=flag_threshold,
        )

        # Convert to dictionary
        result_dict = result.to_dict()

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
            typer.echo(f"\nComparison saved to: {output_path}", err=True)

        # Print human-readable summary to stderr
        typer.echo("\n" + "=" * 60, err=True)
        typer.echo("Run Comparison Summary", err=True)
        typer.echo("=" * 60, err=True)
        typer.echo(f"Baseline Run ID: {result.baseline_run_id}", err=True)
        typer.echo(f"Candidate Run ID: {result.candidate_run_id}", err=True)

        if result.baseline_prompt_version or result.candidate_prompt_version:
            typer.echo(f"Baseline Prompt: {result.baseline_prompt_version or 'N/A'}", err=True)
            typer.echo(f"Candidate Prompt: {result.candidate_prompt_version or 'N/A'}", err=True)

        typer.echo("\nThresholds:", err=True)
        typer.echo(f"  Metric Threshold: {metric_threshold}", err=True)
        typer.echo(f"  Flag Threshold: {flag_threshold}", err=True)

        # Print metric deltas
        if result.metric_deltas:
            typer.echo(f"\n{'Metric Deltas':-^60}", err=True)
            for delta in result.metric_deltas:
                status = "🔴 REGRESSION" if delta.is_regression else "✓"
                typer.echo(f"\n  {delta.metric_name}: {status}", err=True)

                baseline_str = (
                    f"{delta.baseline_mean:.3f}" if delta.baseline_mean is not None else "N/A"
                )
                candidate_str = (
                    f"{delta.candidate_mean:.3f}" if delta.candidate_mean is not None else "N/A"
                )
                typer.echo(f"    Baseline:  {baseline_str}", err=True)
                typer.echo(f"    Candidate: {candidate_str}", err=True)

                if delta.delta is not None:
                    sign = "+" if delta.delta >= 0 else ""
                    typer.echo(f"    Delta: {sign}{delta.delta:.3f}", err=True)
                    if delta.percent_change is not None and abs(delta.percent_change) != float(
                        "inf"
                    ):
                        typer.echo(f"    Change: {sign}{delta.percent_change:.2f}%", err=True)

        # Print flag deltas
        if result.flag_deltas:
            typer.echo(f"\n{'Flag Deltas':-^60}", err=True)
            for flag_delta in result.flag_deltas:
                status = "🔴 REGRESSION" if flag_delta.is_regression else "✓"
                typer.echo(f"\n  {flag_delta.flag_name}: {status}", err=True)
                typer.echo(f"    Baseline:  {flag_delta.baseline_proportion:.2%}", err=True)
                typer.echo(f"    Candidate: {flag_delta.candidate_proportion:.2%}", err=True)

                sign = "+" if flag_delta.delta >= 0 else ""
                typer.echo(f"    Delta: {sign}{flag_delta.delta:.2%}", err=True)
                if (
                    flag_delta.percent_change is not None
                    and abs(flag_delta.percent_change) != float("inf")
                ):
                    typer.echo(f"    Change: {sign}{flag_delta.percent_change:.2f}%", err=True)

        # Print regression summary
        typer.echo(f"\n{'Summary':-^60}", err=True)
        if result.has_regressions:
            typer.echo(f"  🔴 {result.regression_count} regression(s) detected", err=True)
        else:
            typer.echo("  ✓ No regressions detected", err=True)

        typer.echo("=" * 60, err=True)

        # Print JSON to stdout
        typer.echo(json.dumps(result_dict, indent=2))

        # Exit with error code if regressions detected
        if result.has_regressions:
            raise typer.Exit(1)

    except FileNotFoundError as e:
        typer.echo(f"File error: {e}", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"Validation error: {e}", err=True)
        raise typer.Exit(1)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def render_report(
    run: str | None = typer.Option(
        None, "--run", help="Path to run directory containing artifact (for evaluation runs)"
    ),
    compare: str | None = typer.Option(
        None, "--compare", help="Path to comparison artifact JSON file (for comparison reports)"
    ),
    std_threshold: float = typer.Option(
        1.0,
        "--std-threshold",
        help="Standard deviation threshold for marking metrics as unstable (evaluation runs only)",
    ),
    weak_score_threshold: float = typer.Option(
        3.0,
        "--weak-threshold",
        help="Mean score threshold for marking metrics as weak (evaluation runs only)",
    ),
    qualitative_count: int = typer.Option(
        3,
        "--qualitative-count",
        help="Number of worst-case examples to include (evaluation runs only)",
    ),
    max_text_length: int = typer.Option(
        500, "--max-text-length", help="Maximum text length for truncation (evaluation runs only)"
    ),
    top_cases: int = typer.Option(
        5,
        "--top-cases",
        help="Number of top regressed/improved cases to show per metric (comparison reports only)",
    ),
    html: bool = typer.Option(False, "--html", help="Generate HTML report alongside Markdown"),
    output_name: str = typer.Option(
        "report.md", "--output", "-o", help="Output filename for Markdown report"
    ),
    html_output_name: str = typer.Option(
        "report.html", "--html-output", help="Output filename for HTML report"
    ),
) -> None:
    """
    Generate a Markdown (and optional HTML) report from an evaluation run or comparison.

    For evaluation runs, use --run to specify the run directory containing dataset_evaluation.json.
    The report includes:
    - Run metadata and configuration
    - Suite-level metric and flag statistics
    - Per-test-case summary with instability/weak-point annotations
    - Qualitative examples from worst-performing cases

    For comparison reports, use --compare to specify the comparison artifact JSON file.
    The report includes:
    - Comparison metadata (baseline vs candidate)
    - Suite-level metrics and flags comparison tables
    - Regressions and improvements sections
    - Top regressed/improved test cases per metric

    The report is written to the specified output location without modifying the original artifacts.
    """
    try:
        # Validate that exactly one of --run or --compare is provided
        if run and compare:
            typer.echo(
                "Error: Cannot specify both --run and --compare. Use one or the other.",
                err=True,
            )
            raise typer.Exit(1)

        if not run and not compare:
            typer.echo("Error: Must specify either --run or --compare.", err=True)
            raise typer.Exit(1)

        # Handle comparison report
        if compare:
            from prompt_evaluator.reporting import render_comparison_report

            compare_path = Path(compare)

            # Validate comparison artifact file
            if not compare_path.exists():
                typer.echo(f"Error: Comparison artifact not found: {compare}", err=True)
                raise typer.Exit(1)

            if not compare_path.is_file():
                typer.echo(f"Error: Path is not a file: {compare}", err=True)
                raise typer.Exit(1)

            # Validate top_cases
            if top_cases < 0:
                typer.echo("Error: --top-cases must be non-negative", err=True)
                raise typer.Exit(1)

            typer.echo(f"Generating comparison report for: {compare_path}", err=True)
            typer.echo(f"  Top cases per metric: {top_cases}", err=True)

            # Generate comparison report
            report_path = render_comparison_report(
                comparison_artifact_path=compare_path,
                top_cases_per_metric=top_cases,
                generate_html=html,
                output_name=output_name,
                html_output_name=html_output_name,
            )

            typer.echo(f"\n✓ Comparison report generated successfully: {report_path}", err=True)

            if html:
                html_path = compare_path.parent / html_output_name
                if html_path.exists():
                    typer.echo(f"✓ HTML report: {html_path}", err=True)
                else:
                    typer.echo(
                        "⚠ HTML report not generated (markdown library may not be installed)",
                        err=True,
                    )

        # Handle evaluation run report
        elif run:
            from prompt_evaluator.reporting import render_run_report

            run_path = Path(run)

            # Validate run directory
            if not run_path.exists():
                typer.echo(f"Error: Run directory not found: {run}", err=True)
                raise typer.Exit(1)

            if not run_path.is_dir():
                typer.echo(f"Error: Path is not a directory: {run}", err=True)
                raise typer.Exit(1)

            # Validate thresholds
            if std_threshold < 0:
                typer.echo("Error: --std-threshold must be non-negative", err=True)
                raise typer.Exit(1)

            if weak_score_threshold < 0:
                typer.echo("Error: --weak-threshold must be non-negative", err=True)
                raise typer.Exit(1)

            if qualitative_count < 0:
                typer.echo("Error: --qualitative-count must be non-negative", err=True)
                raise typer.Exit(1)

            if max_text_length < 1:
                typer.echo("Error: --max-text-length must be positive", err=True)
                raise typer.Exit(1)

            typer.echo(f"Generating report for run: {run_path}", err=True)
            typer.echo(f"  Std threshold: {std_threshold}", err=True)
            typer.echo(f"  Weak threshold: {weak_score_threshold}", err=True)
            typer.echo(f"  Qualitative samples: {qualitative_count}", err=True)

            # Generate report
            report_path = render_run_report(
                run_dir=run_path,
                std_threshold=std_threshold,
                weak_score_threshold=weak_score_threshold,
                qualitative_sample_count=qualitative_count,
                max_text_length=max_text_length,
                generate_html=html,
                output_name=output_name,
                html_output_name=html_output_name,
            )

            typer.echo(f"\n✓ Report generated successfully: {report_path}", err=True)

            if html:
                html_path = run_path / html_output_name
                if html_path.exists():
                    typer.echo(f"✓ HTML report: {html_path}", err=True)
                else:
                    typer.echo(
                        "⚠ HTML report not generated (markdown library may not be installed)",
                        err=True,
                    )

    except FileNotFoundError as e:
        typer.echo(f"File error: {e}", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"Validation error: {e}", err=True)
        raise typer.Exit(1)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version_flag: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
) -> None:
    """Main callback to handle version flag."""
    if version_flag:
        from prompt_evaluator import __version__

        typer.echo(f"prompt-evaluator version {__version__}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
