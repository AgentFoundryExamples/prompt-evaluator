"""
Command-line interface for the prompt evaluator.

This module provides the main CLI entry point and commands for running
prompt evaluations, managing configurations, and viewing results.
"""

import typer

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


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version_flag: bool = typer.Option(
        False, "--version", "-v", help="Show version and exit"
    ),
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
