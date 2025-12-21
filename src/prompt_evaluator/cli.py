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
        help="Rubric to use for evaluation. Can be a preset alias (default, content-quality, code-review) or a path to a rubric file (.yaml/.json). Defaults to 'default' if not specified.",
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

        # Compute aggregate statistics
        successful_scores = [
            s.judge_score for s in samples
            if s.status == "completed" and s.judge_score is not None
        ]

        stats: dict[str, float | int | None]
        if successful_scores:
            stats = {
                "mean_score": sum(successful_scores) / len(successful_scores),
                "min_score": min(successful_scores),
                "max_score": max(successful_scores),
                "num_successful": len(successful_scores),
                "num_failed": len(samples) - len(successful_scores),
            }
        else:
            stats = {
                "mean_score": None,
                "min_score": None,
                "max_score": None,
                "num_successful": 0,
                "num_failed": len(samples),
            }

        # Create SingleEvaluationRun
        evaluation_run = SingleEvaluationRun(
            run_id=run_id,
            timestamp=timestamp,
            num_samples=num_samples,
            generator_config=generator_config,
            judge_config=judge_config,
            samples=samples,
        )

        # Save evaluation results
        evaluation_file = run_dir / "evaluate-single.json"
        evaluation_dict = evaluation_run.to_dict()
        # Add aggregate statistics
        evaluation_dict["aggregate_stats"] = stats
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

        if stats["mean_score"] is not None:
            typer.echo("\nAggregate Statistics:", err=True)
            typer.echo(f"  Mean Score: {stats['mean_score']:.2f}/5.0", err=True)
            typer.echo(f"  Min Score:  {stats['min_score']:.2f}/5.0", err=True)
            typer.echo(f"  Max Score:  {stats['max_score']:.2f}/5.0", err=True)
        else:
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
        help="Rubric to display. Can be a preset alias (default, content-quality, code-review) or a path to a rubric file (.yaml/.json). Defaults to 'default' if not specified.",
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
