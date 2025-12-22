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

from prompt_evaluator.config import APIConfig
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
    OpenAIProvider,
    generate_completion,
    get_provider,
    judge_completion,
)

app = typer.Typer(
    name="prompt-evaluator",
    help="Evaluate and compare prompts across different LLM providers",
    add_completion=False,
)

# Configuration constants for evaluate-dataset command
DEFAULT_NUM_SAMPLES = 5  # Default number of samples per test case
QUICK_MODE_NUM_SAMPLES = 2  # Number of samples in quick mode
HIGH_STD_ABSOLUTE_THRESHOLD = 1.0  # Absolute std threshold for high variability warning
HIGH_STD_RELATIVE_THRESHOLD = 0.2  # Relative std/mean threshold for high variability warning


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
                if s.status == "completed"
                and flag_name in s.judge_flags
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
        ..., "--system-prompt", "-s", help="Path to system prompt file"
    ),
    input_path: str = typer.Option(
        ..., "--input", "-i", help="Path to input file (use '-' for stdin)"
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name override"),
    temperature: float | None = typer.Option(
        None, "--temperature", "-t", help="Temperature (0.0-2.0)"
    ),
    max_completion_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Maximum tokens to generate"
    ),
    seed: int | None = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    output_dir: str = typer.Option("runs", "--output-dir", "-o", help="Output directory for runs"),
    config_file: str | None = typer.Option(
        None, "--config", "-c", help="Path to config file (YAML/TOML)"
    ),
) -> None:
    """
    Generate a completion from an LLM using system and user prompts.

    This command reads prompts from files, calls the LLM API, and saves
    the output along with metadata to the runs directory.
    """
    try:
        # Load API configuration
        config_path = Path(config_file) if config_file else None
        api_config = APIConfig(config_file_path=config_path)

        # Read system prompt
        system_prompt_path = Path(system_prompt)
        if not system_prompt_path.exists():
            typer.echo(f"Error: System prompt file not found: {system_prompt}", err=True)
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

        # Build GeneratorConfig with CLI overrides
        # Precedence: CLI > config file > defaults
        # Start with model defaults, override with API config, then with CLI args
        base_config = GeneratorConfig(model_name=api_config.model_name)

        # Create overrides dict, filtering out None values
        cli_overrides: dict[str, Any] = {}
        if model is not None:
            cli_overrides["model_name"] = model
        if temperature is not None:
            cli_overrides["temperature"] = temperature
        if max_completion_tokens is not None:
            cli_overrides["max_completion_tokens"] = max_completion_tokens
        if seed is not None:
            cli_overrides["seed"] = seed

        # Apply overrides to base config
        generator_config = GeneratorConfig(
            model_name=cli_overrides.get("model_name", base_config.model_name),
            temperature=cli_overrides.get("temperature", base_config.temperature),
            max_completion_tokens=cli_overrides.get(
                "max_completion_tokens", base_config.max_completion_tokens
            ),
            seed=cli_overrides.get("seed", base_config.seed),
        )

        # Create provider
        provider = get_provider("openai", api_key=api_config.api_key, base_url=api_config.base_url)
        # Type assertion: get_provider with "openai" always returns OpenAIProvider
        assert isinstance(provider, OpenAIProvider)

        # Generate completion
        typer.echo("Generating completion...", err=True)
        response_text, metadata = generate_completion(
            provider=provider,
            system_prompt=system_prompt_content,
            user_prompt=user_prompt_content,
            model=generator_config.model_name,
            temperature=generator_config.temperature,
            max_completion_tokens=generator_config.max_completion_tokens,
            seed=generator_config.seed,
        )

        # Create run record
        run_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        output_dir_path = Path(output_dir)

        # Validate output path is a directory
        if output_dir_path.exists() and not output_dir_path.is_dir():
            typer.echo(
                f"Error: Output path '{output_dir}' exists and is not a directory.",
                err=True
            )
            raise typer.Exit(1)

        run_dir = output_dir_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write output
        output_file = run_dir / "output.txt"
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
        metadata_file = run_dir / "metadata.json"
        metadata_dict = prompt_run.to_dict()
        metadata_dict.update(metadata)
        metadata_file.write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")

        # Print output to stdout
        typer.echo(response_text)

        # Print summary to stderr
        typer.echo("\n" + "=" * 60, err=True)
        typer.echo(f"Run ID: {run_id}", err=True)
        typer.echo(f"Model: {generator_config.model_name}", err=True)
        typer.echo(f"Temperature: {generator_config.temperature}", err=True)
        typer.echo(f"Max tokens: {generator_config.max_completion_tokens}", err=True)
        if generator_config.seed is not None:
            typer.echo(f"Seed: {generator_config.seed}", err=True)
        typer.echo(f"Output file: {output_file}", err=True)
        typer.echo(f"Metadata file: {metadata_file}", err=True)
        if metadata.get("tokens_used"):
            typer.echo(f"Tokens used: {metadata['tokens_used']}", err=True)
        typer.echo(f"Latency: {metadata['latency_ms']:.2f}ms", err=True)
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
        ..., "--system-prompt", "-s", help="Path to generator system prompt file"
    ),
    input_path: str = typer.Option(
        ..., "--input", "-i", help="Path to input file"
    ),
    num_samples: int = typer.Option(
        ..., "--num-samples", "-n", help="Number of samples to generate"
    ),
    generator_model: str | None = typer.Option(
        None, "--generator-model", help="Generator model name override"
    ),
    judge_model: str | None = typer.Option(
        None, "--judge-model", help="Judge model name override"
    ),
    judge_system_prompt: str | None = typer.Option(
        None, "--judge-system-prompt", help="Path to custom judge system prompt file"
    ),
    rubric: str | None = typer.Option(
        None,
        "--rubric",
        help=(
            "Rubric to use for evaluation. Can be a preset alias "
            "(default, content-quality, code-review) or a path to a rubric file "
            "(.yaml/.json). Defaults to 'default' if not specified."
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
    output_dir: str = typer.Option(
        "runs", "--output-dir", "-o", help="Output directory for evaluation runs"
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
) -> None:
    """
    Evaluate a prompt by generating N samples and judging each output.

    This command generates multiple completions for the same input, evaluates
    each output using a judge model, and produces aggregate statistics.
    """
    try:
        # Validate num_samples
        if num_samples <= 0:
            typer.echo("Error: --num-samples must be positive", err=True)
            raise typer.Exit(1)

        # Load API configuration
        config_path = Path(config_file) if config_file else None
        api_config = APIConfig(config_file_path=config_path)

        # Read system prompt
        system_prompt_path = Path(system_prompt)
        if not system_prompt_path.exists():
            typer.echo(f"Error: System prompt file not found: {system_prompt}", err=True)
            raise typer.Exit(1)

        system_prompt_content = system_prompt_path.read_text(encoding="utf-8")

        # Read user input
        input_file_path = Path(input_path)
        if not input_file_path.exists():
            typer.echo(f"Error: Input file not found: {input_path}", err=True)
            raise typer.Exit(1)

        user_prompt_content = input_file_path.read_text(encoding="utf-8")

        # Load rubric (preset, custom, or default)
        try:
            from prompt_evaluator.config import load_rubric, resolve_rubric_path

            rubric_path = resolve_rubric_path(rubric)
            loaded_rubric = load_rubric(rubric_path)
            typer.echo(f"Using rubric: {rubric_path}", err=True)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error loading rubric: {e}", err=True)
            raise typer.Exit(1)

        # Load judge prompt (custom or default)
        try:
            if judge_system_prompt:
                judge_prompt_path = Path(judge_system_prompt)
                judge_prompt_content = load_judge_prompt(judge_prompt_path)
            else:
                judge_prompt_content = load_judge_prompt()
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error loading judge prompt: {e}", err=True)
            raise typer.Exit(1)

        # Build GeneratorConfig
        base_gen_config = GeneratorConfig(model_name=api_config.model_name)
        cli_overrides: dict[str, Any] = {}
        if generator_model is not None:
            cli_overrides["model_name"] = generator_model
        if temperature is not None:
            cli_overrides["temperature"] = temperature
        if max_completion_tokens is not None:
            cli_overrides["max_completion_tokens"] = max_completion_tokens
        if seed is not None:
            cli_overrides["seed"] = seed

        generator_config = GeneratorConfig(
            model_name=cli_overrides.get("model_name", base_gen_config.model_name),
            temperature=cli_overrides.get("temperature", base_gen_config.temperature),
            max_completion_tokens=cli_overrides.get(
                "max_completion_tokens", base_gen_config.max_completion_tokens
            ),
            seed=cli_overrides.get("seed", base_gen_config.seed),
        )

        # Build JudgeConfig
        judge_config = JudgeConfig(
            model_name=judge_model if judge_model else api_config.model_name,
            temperature=0.0,  # Use deterministic judge
        )

        # Create provider
        provider = get_provider("openai", api_key=api_config.api_key, base_url=api_config.base_url)
        assert isinstance(provider, OpenAIProvider)

        # Create run metadata
        run_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        output_dir_path = Path(output_dir)

        # Validate output path is a directory
        if output_dir_path.exists() and not output_dir_path.is_dir():
            typer.echo(
                f"Error: Output path '{output_dir}' exists and is not a directory.",
                err=True
            )
            raise typer.Exit(1)

        run_dir = output_dir_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        typer.echo(f"Starting evaluation run {run_id}...", err=True)
        typer.echo(f"Generating {num_samples} samples...", err=True)

        # Generate and judge samples
        samples: list[Sample] = []
        for i in range(num_samples):
            sample_id = f"{run_id}-sample-{i+1}"
            typer.echo(f"  Sample {i+1}/{num_samples}...", err=True)

            try:
                # Generate completion
                response_text, metadata = generate_completion(
                    provider=provider,
                    system_prompt=system_prompt_content,
                    user_prompt=user_prompt_content,
                    model=generator_config.model_name,
                    temperature=generator_config.temperature,
                    max_completion_tokens=generator_config.max_completion_tokens,
                    seed=generator_config.seed,
                )

                # Judge the output
                judge_result = judge_completion(
                    provider=provider,
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
                )

                samples.append(sample)

                # Display progress
                if sample.status == "completed":
                    if sample.judge_score is not None:
                        # Legacy single-metric mode
                        typer.echo(
                            f"    ✓ Generated and judged (score: {sample.judge_score:.1f}/5.0)",
                            err=True
                        )
                    else:
                        # Rubric mode - show summary of metrics
                        num_metrics = len(sample.judge_metrics)
                        typer.echo(
                            f"    ✓ Generated and judged ({num_metrics} metrics evaluated)",
                            err=True
                        )
                elif sample.status == "judge_invalid_response":
                    error_msg = judge_result.get('error', 'Invalid judge response')
                    typer.echo(f"    ⚠ Generated but judge response invalid: {error_msg}", err=True)
                else:
                    error_msg = judge_result.get('error')
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
                )
                samples.append(sample)

        # Compute aggregate statistics with rubric awareness
        stats = compute_aggregate_statistics(samples, loaded_rubric)

        # Compute rubric metadata
        rubric_metadata = compute_rubric_metadata(loaded_rubric, rubric_path)

        # Compute prompt metadata
        prompt_version_id, prompt_hash = compute_prompt_metadata(system_prompt_path, prompt_version)

        # Create SingleEvaluationRun
        evaluation_run = SingleEvaluationRun(
            run_id=run_id,
            timestamp=timestamp,
            num_samples=num_samples,
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
        typer.echo(f"Total Samples: {num_samples}", err=True)
        typer.echo(f"Successful: {stats['num_successful']}", err=True)
        typer.echo(f"Failed: {stats['num_failed']}", err=True)

        # Print legacy statistics if available
        if stats["mean_score"] is not None:
            typer.echo("\nLegacy Aggregate Statistics:", err=True)
            typer.echo(f"  Mean Score: {stats['mean_score']:.2f}/5.0", err=True)
            typer.echo(f"  Min Score:  {stats['min_score']:.2f}/5.0", err=True)
            typer.echo(f"  Max Score:  {stats['max_score']:.2f}/5.0", err=True)

        # Print per-metric statistics if available
        if "metric_stats" in stats and stats["metric_stats"]:
            typer.echo("\nPer-Metric Statistics:", err=True)
            for metric_name, metric_data in stats["metric_stats"].items():
                if metric_data["count"] > 0:
                    typer.echo(
                        f"  {metric_name}: "
                        f"mean={metric_data['mean']:.2f}, "
                        f"min={metric_data['min']:.2f}, "
                        f"max={metric_data['max']:.2f}, "
                        f"count={metric_data['count']}",
                        err=True
                    )
                else:
                    typer.echo(f"  {metric_name}: No valid scores", err=True)

        # Print per-flag statistics if available
        if "flag_stats" in stats and stats["flag_stats"]:
            typer.echo("\nPer-Flag Statistics:", err=True)
            for flag_name, flag_data in stats["flag_stats"].items():
                if flag_data["total_count"] > 0:
                    typer.echo(
                        f"  {flag_name}: "
                        f"true={flag_data['true_count']}, "
                        f"false={flag_data['false_count']}, "
                        f"proportion={flag_data['true_proportion']:.2%}",
                        err=True
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
        ..., "--dataset", "-d", help="Path to dataset file (.yaml/.yml or .jsonl)"
    ),
    system_prompt: str = typer.Option(
        ..., "--system-prompt", "-s", help="Path to generator system prompt file"
    ),
    num_samples: int | None = typer.Option(
        None, "--num-samples", "-n", help="Number of samples to generate per test case (default: 5)"
    ),
    generator_model: str | None = typer.Option(
        None, "--generator-model", help="Generator model name override"
    ),
    judge_model: str | None = typer.Option(
        None, "--judge-model", help="Judge model name override"
    ),
    judge_system_prompt: str | None = typer.Option(
        None, "--judge-system-prompt", help="Path to custom judge system prompt file"
    ),
    rubric: str | None = typer.Option(
        None,
        "--rubric",
        help=(
            "Rubric to use for evaluation. Can be a preset alias "
            "(default, content-quality, code-review) or a path to a rubric file "
            "(.yaml/.json). Defaults to 'default' if not specified."
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
    output_dir: str = typer.Option(
        "runs", "--output-dir", "-o", help="Output directory for evaluation runs"
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
) -> None:
    """
    Evaluate a dataset of test cases with multiple samples per case.

    This command loads a dataset, generates N samples for each test case,
    judges each sample, and computes per-case and overall statistics.
    Results are streamed to disk as JSON files.

    Supports filtering by case IDs, limiting total cases, quick mode for testing,
    and resuming from interrupted runs.
    """
    try:
        # Handle --quick flag and num_samples interaction
        if quick and num_samples is not None:
            typer.echo(
                "Warning: Both --quick and --num-samples provided. "
                f"Using explicit --num-samples={num_samples}",
                err=True
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

        # Load API configuration
        config_path = Path(config_file) if config_file else None
        api_config = APIConfig(config_file_path=config_path)

        # Load dataset
        from prompt_evaluator.config import load_dataset

        dataset_path = Path(dataset)
        if not dataset_path.exists():
            typer.echo(f"Error: Dataset file not found: {dataset}", err=True)
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
                typer.echo(
                    f"Error: Unknown test case IDs: {', '.join(unknown_ids)}",
                    err=True
                )
                typer.echo(
                    f"Available IDs: {', '.join(sorted(available_ids))}",
                    err=True
                )
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

        # Read system prompt
        system_prompt_path = Path(system_prompt)
        if not system_prompt_path.exists():
            typer.echo(f"Error: System prompt file not found: {system_prompt}", err=True)
            raise typer.Exit(1)

        system_prompt_content = system_prompt_path.read_text(encoding="utf-8")

        # Load rubric (preset, custom, or default)
        try:
            from prompt_evaluator.config import load_rubric, resolve_rubric_path

            rubric_path = resolve_rubric_path(rubric)
            loaded_rubric = load_rubric(rubric_path)
            typer.echo(f"Using rubric: {rubric_path}", err=True)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error loading rubric: {e}", err=True)
            raise typer.Exit(1)

        # Load judge prompt (custom or default)
        try:
            if judge_system_prompt:
                judge_prompt_path = Path(judge_system_prompt)
                judge_prompt_content = load_judge_prompt(judge_prompt_path)
            else:
                judge_prompt_content = load_judge_prompt()
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error loading judge prompt: {e}", err=True)
            raise typer.Exit(1)

        # Build GeneratorConfig
        base_gen_config = GeneratorConfig(model_name=api_config.model_name)
        cli_overrides: dict[str, Any] = {}
        if generator_model is not None:
            cli_overrides["model_name"] = generator_model
        if temperature is not None:
            cli_overrides["temperature"] = temperature
        if max_completion_tokens is not None:
            cli_overrides["max_completion_tokens"] = max_completion_tokens
        if seed is not None:
            cli_overrides["seed"] = seed

        generator_config = GeneratorConfig(
            model_name=cli_overrides.get("model_name", base_gen_config.model_name),
            temperature=cli_overrides.get("temperature", base_gen_config.temperature),
            max_completion_tokens=cli_overrides.get(
                "max_completion_tokens", base_gen_config.max_completion_tokens
            ),
            seed=cli_overrides.get("seed", base_gen_config.seed),
        )

        # Build JudgeConfig
        judge_config = JudgeConfig(
            model_name=judge_model if judge_model else api_config.model_name,
            temperature=0.0,  # Use deterministic judge
        )

        # Create provider
        provider = get_provider("openai", api_key=api_config.api_key, base_url=api_config.base_url)
        assert isinstance(provider, OpenAIProvider)

        # Prepare output directory
        output_dir_path = Path(output_dir)
        if output_dir_path.exists() and not output_dir_path.is_dir():
            typer.echo(
                f"Error: Output path '{output_dir}' exists and is not a directory.",
                err=True
            )
            raise typer.Exit(1)

        # Compute rubric metadata
        rubric_metadata = compute_rubric_metadata(loaded_rubric, rubric_path)

        # Compute prompt metadata
        prompt_version_id, prompt_hash = compute_prompt_metadata(system_prompt_path, prompt_version)

        # Run dataset evaluation
        typer.echo("\n" + "=" * 60, err=True)
        typer.echo("Starting Dataset Evaluation", err=True)
        typer.echo("=" * 60, err=True)
        typer.echo(f"Dataset: {dataset_path}", err=True)
        typer.echo(f"Test Cases: {len(test_cases)}", err=True)
        typer.echo(f"Samples per Case: {num_samples}", err=True)
        typer.echo(f"Generator Model: {generator_config.model_name}", err=True)
        typer.echo(f"Judge Model: {judge_config.model_name}", err=True)
        typer.echo("=" * 60 + "\n", err=True)

        from prompt_evaluator.dataset_evaluation import evaluate_dataset as run_dataset_evaluation

        evaluation_run = run_dataset_evaluation(
            provider=provider,
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
        num_partial = sum(
            1 for tc in evaluation_run.test_case_results if tc.status == "partial"
        )
        num_failed = sum(
            1 for tc in evaluation_run.test_case_results if tc.status == "failed"
        )

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
                                f"    {metric_name}: {mean_str}, {std_str}{std_warning}",
                                err=True
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
                        err=True
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
