# Prompt Evaluator

A Python tool for evaluating and comparing prompts across different LLM providers. This project enables systematic testing and comparison of prompt variations to help optimize prompt engineering workflows. Currently the api is setup to handle gpt-5+ which is the most recent set of openAI models. This api ddeprecates max_tokens in favor of max_completion_tokens.

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

## Understanding Generator and Judge Roles

The prompt evaluator uses a two-model architecture to assess prompt quality:

### Generator Model

The **generator** is the LLM that produces text completions based on your prompts. Its role is to:
- Process the system prompt and user input
- Generate natural language responses
- Potentially use a seed for reproducibility (when supported by the provider)
- Apply sampling parameters (temperature, max tokens, etc.)

Generator configuration is controlled via `GeneratorConfig` and CLI flags like `--generator-model`, `--temperature`, `--seed`, and `--max-tokens`.

### Judge Model

The **judge** is a separate LLM that evaluates the quality of generator outputs. Its role is to:
- Assess how well the generated output preserves the semantic meaning of the input
- Assign a numeric score on a 1-5 scale
- Provide a rationale explaining the score
- Return structured feedback as JSON

Judge configuration is controlled via `JudgeConfig` and CLI flags like `--judge-model` and `--judge-system-prompt`.

### Semantic Fidelity Scoring

The default judge evaluates **semantic fidelity** - how faithfully the output preserves the meaning and intent of the input. The scoring scale is:

- **5.0** - Completely faithful: Perfect preservation of semantic meaning and intent
- **4.0** - Mostly faithful: Minor deviations but core semantics preserved
- **3.0** - Partially faithful: Some key information preserved but with notable gaps
- **2.0** - Mostly unfaithful: Major semantic deviations or omissions
- **1.0** - Completely unfaithful: Output contradicts or has no relation to the input

### Judge Response Format

The judge returns a JSON object with exactly two fields:

```json
{
  "semantic_fidelity": 4.5,
  "rationale": "The output accurately captures the main concepts with minor elaboration that enhances clarity."
}
```

The `semantic_fidelity` field is extracted and stored as `judge_score` in the evaluation results. This structured format allows for programmatic analysis of evaluation results. If the judge fails to return valid JSON or the score is out of range, the sample is marked with a `judge_error` status and the raw response is preserved for debugging.

### Evaluate-Single Command

The `evaluate-single` command runs multiple generation samples for a single prompt/input pair, judges each output, and produces aggregate statistics. This is useful for assessing the consistency and quality of a prompt across multiple generations.

```bash
# Basic usage with 5 samples
prompt-evaluator evaluate-single \
  --system-prompt examples/system_prompt.txt \
  --input examples/input.txt \
  --num-samples 5

# With custom models and parameters
prompt-evaluator evaluate-single \
  --system-prompt examples/system_prompt.txt \
  --input examples/input.txt \
  --num-samples 10 \
  --generator-model gpt-4 \
  --judge-model gpt-4 \
  --seed 42 \
  --temperature 0.7 \
  --task-description "Explain programming concepts clearly"

# With custom judge prompt
prompt-evaluator evaluate-single \
  --system-prompt examples/system_prompt.txt \
  --input examples/input.txt \
  --num-samples 3 \
  --judge-system-prompt custom_judge.txt
```

#### Command Parameters

**Required Parameters:**

- `--system-prompt`, `-s`: Path to the generator's system prompt file
- `--input`, `-i`: Path to the input/user prompt file
- `--num-samples`, `-n`: Number of samples to generate and evaluate (must be positive)

**Optional Parameters:**

- `--generator-model`: Override the generator model (default: `gpt-5.1`, or from `OPENAI_MODEL` environment variable, or from config file)
- `--judge-model`: Override the judge model (default: same as generator model)
- `--judge-system-prompt`: Path to custom judge prompt file (default: uses built-in semantic fidelity prompt). **Note:** Custom judge prompts must still return a JSON object with `semantic_fidelity` (numeric score 1-5) and `rationale` (string) keys to be parsed correctly.
- `--seed`: Random seed for generator reproducibility (default: no seed)
- `--temperature`, `-t`: Generator sampling temperature 0.0-2.0 (default: `0.7`)
- `--max-tokens`: Maximum completion tokens for generator (default: `1024`)
- `--task-description`: Optional description providing context to the judge about the task
- `--output-dir`, `-o`: Directory for run artifacts (default: `runs/`)
- `--config`, `-c`: Path to config file for API credentials (default: reads from environment)

**Note on Stdin:** Unlike the `generate` command, `evaluate-single` does not support stdin input (`--input -`). You must provide a file path for reproducible multi-sample evaluations.

#### Configuration Defaults

The command uses the following default configurations:

**Generator defaults:**
- Model: `gpt-5.1` (or from `OPENAI_MODEL` environment variable)
- Temperature: `0.7`
- Max completion tokens: `1024`
- Seed: `null` (non-deterministic)

**Judge defaults:**
- Model: Same as generator model (or from `--judge-model`)
- Temperature: `0.0` (deterministic judging)
- Max completion tokens: `512`
- Seed: `null`
- System prompt: Built-in semantic fidelity evaluator

Defaults can be overridden via CLI flags or config file. CLI flags take precedence over config file values.

#### Deterministic Runs with --seed

The `--seed` parameter enables reproducible generator outputs when the LLM provider supports seeding:

```bash
# Deterministic generation (when supported)
prompt-evaluator evaluate-single \
  --system-prompt examples/system_prompt.txt \
  --input examples/input.txt \
  --num-samples 5 \
  --seed 42
```

**Important notes about seeding:**

- **Provider support varies:** Not all LLM providers honor the seed parameter. OpenAI's GPT models support seeding for deterministic outputs, but results may still vary across different API versions or model updates.
- **Judge is always deterministic:** The judge uses `temperature=0.0` by default to ensure consistent scoring, regardless of the `--seed` parameter (which only affects the generator).
- **No guarantee:** Even with a seed, providers may ignore it or return slightly different results due to internal changes. Always verify reproducibility for your specific model and provider.
- **Debugging tip:** If you get different outputs with the same seed, check the provider's documentation for seeding support and any known limitations.

#### Artifact Output

Each evaluation run creates a unique directory under `runs/<run_id>/` containing:

**Main artifact:**
- `evaluate-single.json` - Complete evaluation results with all samples, scores, and metadata

**Location:**
- Default: `runs/<uuid>/evaluate-single.json`
- Custom: `<output-dir>/<uuid>/evaluate-single.json` (when using `--output-dir`)

The run directory is created automatically if it doesn't exist. Each run gets a unique UUID-based identifier to prevent conflicts.

#### How it Works

The command will:
1. Read system and user prompts from files
2. Generate N completions using the specified generator model
3. Evaluate each completion using the judge model
4. Compute aggregate statistics (mean, min, max scores)
5. Save detailed results to `runs/<run_id>/evaluate-single.json`
6. Print summary to stderr and full JSON to stdout

#### Output Structure

Each evaluation run generates a JSON file with:
- **run_id**: Unique identifier for this evaluation
- **timestamp**: When the evaluation was started
- **num_samples**: Number of samples generated
- **generator_config**: Configuration used for generation (model, temperature, etc.)
- **judge_config**: Configuration used for judging
- **samples**: Array of all samples with inputs, outputs, and judge scores
- **aggregate_stats**: Computed statistics (mean/min/max scores, success/failure counts)

Example output structure:
```json
{
  "run_id": "abc123...",
  "timestamp": "2025-12-21T06:00:00+00:00",
  "num_samples": 3,
  "generator_config": {
    "model_name": "gpt-4",
    "temperature": 0.7,
    "max_completion_tokens": 1024,
    "seed": 42
  },
  "judge_config": {
    "model_name": "gpt-4",
    "temperature": 0.0,
    "max_completion_tokens": 512,
    "seed": null
  },
  "samples": [
    {
      "sample_id": "abc123-sample-1",
      "input_text": "What is Python?",
      "generator_output": "Python is a programming language...",
      "judge_score": 4.5,
      "judge_rationale": "Excellent semantic preservation...",
      "status": "completed"
    }
  ],
  "aggregate_stats": {
    "mean_score": 4.33,
    "min_score": 4.0,
    "max_score": 4.5,
    "num_successful": 3,
    "num_failed": 0
  }
}
```

#### Interpreting Results

**Sample Status Values:**

Each sample in the output has a `status` field indicating its evaluation state:

- `"completed"` - Both generation and judging succeeded; `judge_score` and `judge_rationale` are populated
- `"judge_error"` - Generation succeeded but judge failed to return valid JSON or encountered an error; check `judge_raw_response` for debugging
- `"generation_error"` - Generator failed to produce output; `generator_output` will be empty
- `"pending"` - Sample not yet processed (should not appear in final results)

**Aggregate Statistics:**

The `aggregate_stats` object summarizes results across all samples:

- `mean_score` - Average semantic fidelity score across successful samples
- `min_score` - Lowest score among successful samples
- `max_score` - Highest score among successful samples
- `num_successful` - Count of samples with `status: "completed"`
- `num_failed` - Count of samples with `judge_error` or `generation_error`

**When Statistics are Null:**

If all samples fail (no successful completions), the statistics fields are set to `null`:

```json
{
  "aggregate_stats": {
    "mean_score": null,
    "min_score": null,
    "max_score": null,
    "num_successful": 0,
    "num_failed": 5
  }
}
```

This indicates a systemic issue - check your API credentials, model availability, or prompt configuration.

**Inspecting Judge Errors:**

When a sample has `status: "judge_error"`, examine the `judge_raw_response` field in the artifact JSON:

```json
{
  "sample_id": "abc123-sample-2",
  "status": "judge_error",
  "judge_score": null,
  "judge_rationale": null,
  "judge_raw_response": "I think the output is good but I cannot provide a score in JSON format."
}
```

Common causes of judge errors:
- Judge returned text instead of JSON
- JSON structure is missing required fields (`semantic_fidelity` or `rationale`)
- Network or API errors during judge request

To recover, consider:
- Using a more capable judge model (e.g., `gpt-4` instead of `gpt-3.5-turbo`)
- Adjusting the judge system prompt to emphasize JSON formatting
- **Note:** The judge's `max_completion_tokens` is set to 512 by default and cannot be adjusted via CLI flags. If you need more tokens for verbose rationales, you would need to modify the `JudgeConfig` in code or request this as a feature enhancement.

#### Error Handling and Resilience

The evaluate-single command is designed to handle failures gracefully:

**Generation Failures:**
- If the generator fails for a sample, it's recorded with `status: "generation_error"`
- Remaining samples continue to process
- Failed samples are counted in `num_failed` but excluded from score statistics

**Judge Failures:**
- If judging fails (e.g., invalid JSON response), the sample is marked as `status: "judge_error"`
- The generator output is preserved but scores remain `null`
- Raw judge response is saved in `judge_raw_response` for debugging
- Failed samples are counted in `num_failed` but excluded from score statistics

**Partial Success:**
- Aggregate statistics are computed only from successful samples
- If all samples fail, statistics are set to null with clear messaging

**Exit Codes:**
- `0` - Evaluation completed (even if some samples failed)
- `1` - Configuration error, missing files, or unrecoverable failure

The command prioritizes completing as many samples as possible, even when individual samples encounter errors.

#### Current Limitations

**Single Input Per Run:**
- The `evaluate-single` command processes one input file per execution
- To evaluate multiple inputs, run the command multiple times with different `--input` files
- Each run generates a separate artifact with its own `run_id`
- Future versions may support batch evaluation of multiple inputs

**Semantic Fidelity Only:**
- The default judge evaluates only **semantic fidelity** (meaning preservation)
- Other dimensions (fluency, coherence, factuality, safety) are not currently assessed
- Custom judge prompts (via `--judge-system-prompt`) can evaluate different criteria, but the JSON schema remains the same
- Future versions may support multi-dimensional scoring and custom rubrics

**No Stdin Support:**
- Unlike `generate`, the `evaluate-single` command does not accept stdin input (`--input -`)
- This is intentional to ensure reproducibility - all samples must use the same input file
- Workaround: Write stdin content to a temporary file, then pass that file path

**Model and Provider Constraints:**
- Currently optimized for OpenAI API (GPT-4, GPT-3.5, etc.)
- Other providers may work but are not actively tested
- Seeding behavior depends on provider support and may not be consistent

#### Future Extensibility

The architecture is designed for future enhancements:
- **Multi-dimensional scoring:** Support for evaluating multiple quality dimensions simultaneously (fluency, factuality, etc.)
- **Batch evaluation:** Process multiple input files in a single run with comparative analysis
- **Custom rubrics:** User-defined evaluation criteria and scoring schemas
- **Provider diversity:** Expanded testing and support for Anthropic, Hugging Face, and other LLM providers
- **Progress streaming:** Real-time progress updates and partial result streaming for long-running evaluations
- **Retry mechanisms:** Configurable retry logic for transient API failures

Contributions and feedback on prioritization are welcome!

## Roadmap

- [x] Project scaffolding and structure
- [x] Configuration file format and loading
- [x] CLI commands for running evaluations
- [x] LLM provider integrations (OpenAI, etc.)
- [x] Judge models and evaluation data structures for semantic fidelity scoring
- [x] Evaluation command for running multiple samples with aggregate statistics
- [ ] Result comparison and analysis tools

## Judge Models

The prompt evaluator includes built-in support for evaluating generated outputs using a judge model. The judge assesses semantic fidelity on a 1-5 scale and provides structured feedback.

### Key Features

- **JudgeConfig**: Separate configuration for judge models (model name, temperature, tokens, seed)
- **Sample**: Data structure for storing input, output, and judge scoring results
- **SingleEvaluationRun**: Container for multiple samples with generator and judge configurations
- **Default Judge Prompt**: Built-in system prompt for semantic fidelity evaluation with JSON schema enforcement
- **Custom Prompts**: Load custom judge prompts from files for specialized evaluation criteria
- **Robust Parsing**: Handles JSON extraction from responses with extra text, score clamping, and graceful error handling

### Usage Example

```python
from prompt_evaluator.models import JudgeConfig, Sample, load_judge_prompt
from prompt_evaluator.provider import OpenAIProvider, judge_completion

# Initialize provider and configs
provider = OpenAIProvider(api_key="your-key")
judge_config = JudgeConfig(model_name="gpt-4", temperature=0.0)

# Load default or custom judge prompt
judge_prompt = load_judge_prompt()  # Uses DEFAULT_JUDGE_SYSTEM_PROMPT
# Or load custom: judge_prompt = load_judge_prompt(Path("custom_prompt.txt"))

# Evaluate a sample
result = judge_completion(
    provider=provider,
    input_text="What is Python?",
    generator_output="Python is a programming language.",
    judge_config=judge_config,
    judge_system_prompt=judge_prompt,
    task_description="Explain programming concepts clearly"  # Optional
)

# Check result
if result["status"] == "completed":
    print(f"Score: {result['judge_score']}/5")
    print(f"Rationale: {result['judge_rationale']}")
else:
    print(f"Error: {result['error']}")
```

### Error Handling

The judge completion function gracefully handles various error scenarios:
- Invalid JSON or missing fields → `judge_error` status with raw response preserved
- Out-of-range scores → Automatic clamping to 1.0-5.0 range
- API exceptions → `judge_error` status with error details
- Extra text around JSON → Automatic extraction of JSON object

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
