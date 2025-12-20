# Prompt Evaluator

A Python tool for evaluating and comparing prompts across different LLM providers. This project enables systematic testing and comparison of prompt variations to help optimize prompt engineering workflows.

## Overview

The Prompt Evaluator provides a structured approach to:
- Define and manage prompt templates
- Configure multiple LLM providers (OpenAI, etc.)
- Run evaluations with different prompt variations
- Compare and analyze results

## Project Structure

```
prompt-evaluator/
├── src/
│   └── prompt_evaluator/
│       ├── __init__.py       # Package initialization
│       ├── cli.py            # Command-line interface
│       ├── config.py         # Configuration management
│       ├── models.py         # Data models and schemas
│       └── provider.py       # LLM provider integrations
├── tests/                    # Test suite
├── examples/                 # Example configurations (gitignored)
├── runs/                     # Evaluation run outputs (gitignored)
└── pyproject.toml           # Project metadata and dependencies
```

## Requirements

- Python >= 3.10
- Dependencies are managed via `pyproject.toml`

## Installation

```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

After installation, try the tool with the provided example files:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-api-key-here"

# Run a basic generation with example prompts
prompt-evaluator generate \
  --system-prompt examples/system_prompt.txt \
  --input examples/input.txt

# View the generated output
cat runs/<run-id>/output.txt
```

The tool will generate a completion, print it to stdout, and save both the output and metadata to a `runs/` directory with a unique run ID.

## Usage

The CLI entry point is available as `prompt-evaluator` after installation:

```bash
prompt-evaluator --help
```

### Configuration

The prompt evaluator requires API credentials to connect to LLM providers. Configuration can be provided through environment variables or a config file.

#### Required Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key (required)

#### Optional Environment Variables

- `OPENAI_BASE_URL` - Custom base URL for OpenAI API (optional, uses default if not set)
- `OPENAI_MODEL` - Default model to use (optional, defaults to `gpt-3.5-turbo`)

#### Example Setup

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
export OPENAI_MODEL="gpt-4"
```

#### Config File Override

You can optionally provide a config file (YAML or TOML format) to override environment variables:

**config.yaml:**
```yaml
api_key: sk-your-api-key-here
base_url: https://api.openai.com/v1
model_name: gpt-4
```

**config.toml:**
```toml
api_key = "sk-your-api-key-here"
base_url = "https://api.openai.com/v1"
model_name = "gpt-4"
```

Config file values take precedence over environment variables. If the config file is missing, the tool will gracefully fall back to environment variables and defaults.

### Generate Command

The `generate` command performs a single LLM completion with system and user prompts:

```bash
# Basic usage with example files
prompt-evaluator generate \
  --system-prompt examples/system_prompt.txt \
  --input examples/input.txt

# With model and parameter overrides
prompt-evaluator generate \
  --system-prompt examples/system_prompt.txt \
  --input examples/input.txt \
  --model gpt-4 \
  --temperature 0.5 \
  --max-tokens 500 \
  --seed 42

# Using stdin for input (useful for piping or interactive use)
echo "What is Python?" | prompt-evaluator generate \
  --system-prompt examples/system_prompt.txt \
  --input -

# With custom output directory
prompt-evaluator generate \
  --system-prompt examples/system_prompt.txt \
  --input examples/input.txt \
  --output-dir my-runs
```

#### How it Works

The command will:
1. Read system and user prompts from files (or stdin when input is `-`)
2. Call the OpenAI API with the specified model and parameters
3. Print the completion to stdout (can be redirected or piped)
4. Save the completion and metadata to the output directory (default: `runs/<run_id>/`)
5. Display a summary with run details to stderr

#### Output Artifacts

Each run generates artifacts in a unique run directory:
- `runs/<run_id>/output.txt` - Raw completion text
- `runs/<run_id>/metadata.json` - Run metadata including prompts, config, tokens, and latency

**Note:** When using `--output-dir`, the directory will be created if it doesn't exist. Each run gets a unique UUID-based subdirectory to prevent conflicts.

#### Using Stdin

When `--input -` is specified, the tool reads user input from stdin. This is useful for:
- Piping output from other commands
- Interactive testing with echo or heredocs
- Processing dynamic inputs in scripts

```bash
# Pipe from echo
echo "Explain quantum computing" | prompt-evaluator generate \
  --system-prompt examples/system_prompt.txt \
  --input -

# Use heredoc for multi-line input
prompt-evaluator generate \
  --system-prompt examples/system_prompt.txt \
  --input - << EOF
Please explain the following concepts:
1. Machine learning
2. Neural networks
3. Deep learning
EOF
```

### Error Handling

The tool provides clear error messages for common issues:

- **Missing API Key**: If `OPENAI_API_KEY` is not set, the tool will exit with an error message
- **Missing Files**: If system prompt or input files don't exist, the tool reports the specific missing file
- **Invalid Parameters**: Configuration errors (e.g., temperature out of range) are caught early with descriptive messages
- **API Errors**: OpenAI API errors are caught and reported with relevant details

Exit codes:
- `0` - Success
- `1` - Error (configuration, file not found, API failure, etc.)

Example error handling:
```bash
# Check exit code
prompt-evaluator generate \
  --system-prompt examples/system_prompt.txt \
  --input examples/input.txt

if [ $? -eq 0 ]; then
  echo "Generation successful"
else
  echo "Generation failed"
fi
```

## Roadmap

- [x] Project scaffolding and structure
- [x] Configuration file format and loading
- [x] CLI commands for running evaluations
- [x] LLM provider integrations (OpenAI, etc.)
- [ ] Evaluation metrics and reporting
- [ ] Result comparison and analysis tools

## Development

### Testing

```bash
pytest
```

### Linting

```bash
ruff check .
ruff format .
```

### Type Checking

```bash
mypy src/
```

# Permanents (License, Contributing, Author)

Do not change any of the below sections

## License

This Agent Foundry Project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Contributing

Feel free to submit issues and enhancement requests!

## Author

Created by Agent Foundry and John Brosnihan
