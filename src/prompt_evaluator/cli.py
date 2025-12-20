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
from datetime import datetime, timezone
from pathlib import Path

import typer

from prompt_evaluator.config import APIConfig
from prompt_evaluator.models import GeneratorConfig, PromptRun
from prompt_evaluator.provider import generate_completion, get_provider

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
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Maximum tokens to generate"),
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
        # Use defaults from GeneratorConfig if not provided via CLI
        default_config = GeneratorConfig()
        model_name = model or api_config.model_name
        config_temp = temperature if temperature is not None else default_config.temperature
        config_max_tokens = max_tokens if max_tokens is not None else default_config.max_tokens

        generator_config = GeneratorConfig(
            model_name=model_name,
            temperature=config_temp,
            max_tokens=config_max_tokens,
            seed=seed,
        )

        # Create provider
        provider = get_provider("openai", api_key=api_config.api_key, base_url=api_config.base_url)

        # Generate completion
        typer.echo("Generating completion...", err=True)
        response_text, metadata = generate_completion(
            provider=provider,
            system_prompt=system_prompt_content,
            user_prompt=user_prompt_content,
            model=generator_config.model_name,
            temperature=generator_config.temperature,
            max_tokens=generator_config.max_tokens,
            seed=generator_config.seed,
        )

        # Create run record
        run_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        output_dir_path = Path(output_dir)
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
        typer.echo(f"Max tokens: {generator_config.max_tokens}", err=True)
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
