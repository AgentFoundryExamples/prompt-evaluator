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
- `--rubric`: Rubric to use for evaluation. Can be a preset alias (`default`, `content-quality`, `code-review`) or a path to a rubric file (`.yaml`/`.json`). Defaults to `default` if not specified. See [Using Rubrics with the CLI](#using-rubrics-with-the-cli) for details.
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
- **aggregate_stats**: Computed statistics (mean/min/max scores, success/failure counts, per-metric stats, per-flag stats)
- **rubric_metadata** (when using rubric): Rubric path, hash, and full definition

##### Sample Fields

Each sample in the `samples` array includes:
- **sample_id**: Unique identifier for the sample
- **input_text**: Original input text
- **generator_output**: Text generated by the model
- **status**: Evaluation status (see below for values)
- **task_description**: Optional task description provided to judge
- **judge_score** (legacy): Semantic fidelity score 1-5 (deprecated, use judge_metrics)
- **judge_rationale** (legacy): Explanation for judge_score (deprecated, use judge_metrics)
- **judge_metrics**: Dict of metric results (when using rubric), each containing:
  - `score`: Numeric score within metric's min/max range
  - `rationale`: Explanation for the score
- **judge_flags**: Dict of flag results (when using rubric), each containing boolean value
- **judge_overall_comment**: Overall assessment from judge (when using rubric)
- **judge_raw_response**: Raw response from judge model for debugging

##### Aggregate Statistics

The `aggregate_stats` object includes:
- **Legacy fields** (for backward compatibility):
  - `mean_score`: Average semantic fidelity score across successful samples
  - `min_score`: Lowest score among successful samples
  - `max_score`: Highest score among successful samples
  - `num_successful`: Count of samples with `status: "completed"`
  - `num_failed`: Count of samples with errors
- **Per-metric statistics** (when using rubric):
  - `metric_stats`: Dict keyed by metric name, each containing:
    - `mean`: Average score for this metric
    - `min`: Minimum score for this metric
    - `max`: Maximum score for this metric
    - `count`: Number of valid samples with this metric
- **Per-flag statistics** (when using rubric):
  - `flag_stats`: Dict keyed by flag name, each containing:
    - `true_count`: Number of samples where flag is true
    - `false_count`: Number of samples where flag is false
    - `total_count`: Total number of samples evaluated for this flag
    - `true_proportion`: Proportion of samples where flag is true (0.0-1.0)

**Important:** Samples with `status: "judge_invalid_response"` are automatically excluded from all aggregation calculations.

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
      "status": "completed",
      "judge_score": null,
      "judge_rationale": null,
      "judge_metrics": {
        "semantic_fidelity": {
          "score": 4.5,
          "rationale": "Excellent semantic preservation"
        },
        "decomposition_quality": {
          "score": 4.0,
          "rationale": "Clear logical structure"
        },
        "constraint_adherence": {
          "score": 5.0,
          "rationale": "All constraints followed"
        }
      },
      "judge_flags": {
        "invented_constraints": false,
        "omitted_constraints": false
      },
      "judge_overall_comment": "High-quality response with good structure",
      "judge_raw_response": "{...}"
    }
  ],
  "aggregate_stats": {
    "mean_score": null,
    "min_score": null,
    "max_score": null,
    "num_successful": 3,
    "num_failed": 0,
    "metric_stats": {
      "semantic_fidelity": {
        "mean": 4.33,
        "min": 4.0,
        "max": 4.5,
        "count": 3
      },
      "decomposition_quality": {
        "mean": 3.83,
        "min": 3.5,
        "max": 4.0,
        "count": 3
      },
      "constraint_adherence": {
        "mean": 4.67,
        "min": 4.0,
        "max": 5.0,
        "count": 3
      }
    },
    "flag_stats": {
      "invented_constraints": {
        "true_count": 0,
        "false_count": 3,
        "total_count": 3,
        "true_proportion": 0.0
      },
      "omitted_constraints": {
        "true_count": 1,
        "false_count": 2,
        "total_count": 3,
        "true_proportion": 0.33
      }
    }
  },
  "rubric_metadata": {
    "rubric_path": "/path/to/examples/rubrics/default.yaml",
    "rubric_hash": "a1b2c3d4...",
    "rubric_definition": {
      "metrics": [...],
      "flags": [...]
    }
  }
}
```

#### Interpreting Results

**Sample Status Values:**

Each sample in the output has a `status` field indicating its evaluation state:

- `"completed"` - Both generation and judging succeeded; `judge_score` and `judge_rationale` are populated (legacy mode) or `judge_metrics`, `judge_flags`, and `judge_overall_comment` are populated (rubric mode)
- `"judge_error"` - Generation succeeded but judge encountered a fatal error (e.g., API failure); check `judge_raw_response` for debugging
- `"judge_invalid_response"` - Generation succeeded but judge returned unparseable or invalid response; sample is **excluded from aggregation**; check `judge_raw_response` for debugging
- `"generation_error"` - Generator failed to produce output; `generator_output` will be empty
- `"pending"` - Sample not yet processed (should not appear in final results)

**Aggregate Statistics:**

The `aggregate_stats` object summarizes results across all samples:

**Legacy fields (for backward compatibility):**
- `mean_score` - Average semantic fidelity score across successful samples (null when using rubrics)
- `min_score` - Lowest score among successful samples (null when using rubrics)
- `max_score` - Highest score among successful samples (null when using rubrics)
- `num_successful` - Count of samples with `status: "completed"`
- `num_failed` - Count of samples with `judge_error`, `judge_invalid_response`, or `generation_error`

**Rubric-aware fields (when using `--rubric`):**
- `metric_stats` - Per-metric statistics:
  - Each metric has `mean`, `min`, `max`, and `count` fields
  - Computed only from samples with `status: "completed"`
  - Samples with `judge_invalid_response` are excluded
- `flag_stats` - Per-flag statistics:
  - Each flag has `true_count`, `false_count`, `total_count`, and `true_proportion` fields
  - Computed only from samples with `status: "completed"`
  - Flags that are never true still appear with zero counts
  - `true_proportion` is a float between 0.0 and 1.0

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
- **Batch evaluation:** Process multiple input files in a single run with comparative analysis
- **Provider diversity:** Expanded testing and support for Anthropic, Hugging Face, and other LLM providers
- **Progress streaming:** Real-time progress updates and partial result streaming for long-running evaluations
- **Retry mechanisms:** Configurable retry logic for transient API failures
- **Weighted scoring:** Support for weighted aggregation of multiple metrics

Contributions and feedback on prioritization are welcome!

### Evaluate-Dataset Command

The `evaluate-dataset` command evaluates multiple test cases from a dataset file, generating N samples per case and computing per-case and overall statistics. This is ideal for systematic prompt testing across diverse inputs.

```bash
# Basic usage with a dataset file
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt examples/system_prompt.txt \
  --num-samples 5

# Quick testing mode (2 samples per case)
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt examples/system_prompt.txt \
  --quick

# Filter to specific test cases
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt examples/system_prompt.txt \
  --num-samples 3 \
  --case-ids test-001,test-003

# Limit number of test cases evaluated
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt examples/system_prompt.txt \
  --num-samples 3 \
  --max-cases 5

# With custom models and rubric
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt examples/system_prompt.txt \
  --num-samples 10 \
  --generator-model gpt-4 \
  --judge-model gpt-4 \
  --rubric content-quality \
  --temperature 0.7 \
  --seed 42
```

#### Command Parameters

**Required Parameters:**

- `--dataset`, `-d`: Path to dataset file (`.yaml`, `.yml`, or `.jsonl`)
- `--system-prompt`, `-s`: Path to the generator's system prompt file

**Optional Parameters:**

- `--num-samples`, `-n`: Number of samples to generate per test case (default: 5)
- `--quick`: Quick mode flag - sets num-samples to 2 for fast testing (overridden by explicit `--num-samples`)
- `--case-ids`: Comma-separated list of test case IDs to evaluate (filters dataset to specific cases)
- `--max-cases`: Maximum number of test cases to evaluate (applies after --case-ids filter)
- `--generator-model`: Override the generator model (default: from config or `gpt-5.1`)
- `--judge-model`: Override the judge model (default: same as generator model)
- `--judge-system-prompt`: Path to custom judge prompt file (default: built-in rubric-aware prompt)
- `--rubric`: Rubric to use (preset alias or file path, default: `default`)
- `--seed`: Random seed for generator reproducibility
- `--temperature`, `-t`: Generator temperature 0.0-2.0 (default: 0.7)
- `--max-tokens`: Maximum completion tokens for generator (default: 1024)
- `--output-dir`, `-o`: Directory for run artifacts (default: `runs/`)
- `--config`, `-c`: Path to config file for API credentials

#### Filtering Options

**Filter by Case IDs (`--case-ids`):**

Select specific test cases to evaluate:

```bash
# Evaluate only test-001 and test-003
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt examples/system_prompt.txt \
  --case-ids test-001,test-003 \
  --num-samples 5
```

The command will:
- Validate all case IDs exist in the dataset
- Fail fast with a list of unknown IDs if any are invalid
- Show available case IDs in the error message

**Limit Total Cases (`--max-cases`):**

Evaluate only the first N test cases (useful for quick runs or sampling):

```bash
# Evaluate first 5 test cases from dataset
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt examples/system_prompt.txt \
  --max-cases 5 \
  --num-samples 3
```

**Combining Filters:**

Filters are applied in order: `--case-ids` first, then `--max-cases`:

```bash
# Filter to specific cases, then limit to first 2
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt examples/system_prompt.txt \
  --case-ids test-001,test-002,test-003 \
  --max-cases 2 \
  --num-samples 3
```

This evaluates test-001 and test-002 (first 2 after filtering).

#### Quick Mode

The `--quick` flag is designed for rapid testing and iteration:

```bash
# Quick mode: 2 samples per case
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt examples/system_prompt.txt \
  --quick
```

**Behavior:**
- Sets `--num-samples=2` automatically
- Can be combined with `--max-cases` for even faster testing
- If explicit `--num-samples` is provided, it takes precedence and a warning is shown

**Warning Example:**
```bash
# Both --quick and --num-samples provided
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt examples/system_prompt.txt \
  --quick \
  --num-samples 10

# Output: Warning: Both --quick and --num-samples provided. Using explicit --num-samples=10
```

#### Output and Progress

**Progress Output:**

The command prints detailed progress to stderr:

```
Loading dataset from examples/datasets/sample.yaml...
Loaded 3 test cases
Using default --num-samples=5
Using rubric: /path/to/examples/rubrics/default.yaml

============================================================
Starting Dataset Evaluation
============================================================
Dataset: examples/datasets/sample.yaml
Test Cases: 3
Samples per Case: 5
Generator Model: gpt-5.1
Judge Model: gpt-5.1
============================================================

Evaluating test case 1/3: test-001...
  Completed 5/5 samples successfully
Evaluating test case 2/3: test-002...
  Completed 5/5 samples successfully
Evaluating test case 3/3: test-003...
  Completed 4/5 samples successfully

============================================================
Dataset Evaluation Complete!
============================================================
Run ID: abc123-def456-...
Status: partial
Test Cases Completed: 2/3
Test Cases Partial: 1

Per-Case Metric Statistics:

  Case: test-001
    semantic_fidelity: mean=4.20, std=0.45
    clarity: mean=4.50, std=1.20 ⚠️ HIGH VARIABILITY

  Case: test-002
    semantic_fidelity: mean=3.80, std=0.30
    clarity: mean=4.00, std=0.50

  Case: test-003
    semantic_fidelity: mean=4.00, std=1.50 ⚠️ HIGH VARIABILITY
    clarity: mean=3.75, std=0.85

Overall Metric Statistics (mean of per-case means):
  semantic_fidelity: mean=4.00, min=3.80, max=4.20, cases=3
  clarity: mean=4.08, min=3.75, max=4.50, cases=3

Results saved to: runs/abc123-def456-.../dataset_evaluation.json
============================================================
```

**High Variability Warning:**

The command highlights metrics with high standard deviation (>1.0 or >20% of mean) with a ⚠️ warning. This helps identify:
- Inconsistent prompt behavior across samples
- Metrics sensitive to temperature or sampling variation
- Cases requiring additional investigation

#### Artifact Output

Each dataset evaluation creates artifacts in a unique run directory:

**Main artifact:**
- `dataset_evaluation.json` - Complete evaluation results with all test cases and samples

**Per-case artifacts:**
- `test_case_<id>.json` - Individual results for each test case (streamed during evaluation)

**Location:**
- Default: `runs/<run_id>/dataset_evaluation.json`
- Custom: `<output-dir>/<run_id>/dataset_evaluation.json` (when using `--output-dir`)

**Output Structure:**

The `dataset_evaluation.json` file includes:

```json
{
  "run_id": "abc123-...",
  "dataset_path": "/path/to/dataset.yaml",
  "dataset_hash": "sha256-hash",
  "dataset_count": 3,
  "num_samples_per_case": 5,
  "status": "completed",
  "timestamp_start": "2025-12-21T10:00:00+00:00",
  "timestamp_end": "2025-12-21T10:15:00+00:00",
  "generator_config": { ... },
  "judge_config": { ... },
  "rubric_metadata": { ... },
  "test_case_results": [
    {
      "test_case_id": "test-001",
      "status": "completed",
      "samples": [ ... ],
      "per_metric_stats": {
        "semantic_fidelity": {
          "mean": 4.2,
          "std": 0.45,
          "min": 3.5,
          "max": 5.0,
          "count": 5
        }
      },
      "per_flag_stats": { ... }
    }
  ],
  "overall_metric_stats": {
    "semantic_fidelity": {
      "mean_of_means": 4.0,
      "min_of_means": 3.8,
      "max_of_means": 4.2,
      "num_cases": 3
    }
  },
  "overall_flag_stats": { ... }
}
```

#### Error Handling

**Missing Dataset File:**
```bash
$ prompt-evaluator evaluate-dataset -d missing.yaml -s sys.txt
Error: Dataset file not found: missing.yaml
```

**Unknown Case IDs:**
```bash
$ prompt-evaluator evaluate-dataset -d data.yaml -s sys.txt --case-ids unknown
Error: Unknown test case IDs: unknown
Available IDs: test-001, test-002, test-003
```

**Invalid Filters:**
```bash
$ prompt-evaluator evaluate-dataset -d data.yaml -s sys.txt --max-cases 0
Error: --max-cases must be positive
```

**Partial Failures:**

If some samples fail during evaluation:
- The run continues for remaining test cases
- Status is set to "partial" if at least one case succeeds
- Failed samples are recorded with error details
- Aggregate statistics exclude failed samples

#### Current Limitations

**No Resume Support (Yet):**
- If a run is interrupted, you must start over
- Future versions will support resuming from partial artifacts

**Sequential Processing:**
- Test cases are evaluated sequentially, not in parallel
- Large datasets may take significant time

**No Progress Bar:**
- Progress is shown as text messages, not a visual progress bar
- Consider using `--max-cases` for quick testing

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

## Evaluation Rubrics

The prompt evaluator supports configurable evaluation rubrics that define metrics and flags for assessing prompt outputs. Rubrics provide a structured, machine-readable way to specify evaluation criteria beyond the default semantic fidelity scoring.

### Rubric Schema

A rubric consists of:
- **Metrics**: Scored evaluation dimensions with numeric ranges (e.g., 1-5 scale)
- **Flags**: Binary checks for specific conditions (true/false)

#### RubricMetric Fields

Each metric defines a scored evaluation dimension:

- `name` (required): Unique identifier for the metric (e.g., "semantic_fidelity")
- `description` (required): Human-readable description of what this metric measures
- `min_score` (required): Minimum score value (inclusive), must be numeric
- `max_score` (required): Maximum score value (inclusive), must be numeric and >= min_score
- `guidelines` (required): Detailed scoring guidelines explaining how to assign scores

#### RubricFlag Fields

Each flag defines a binary check:

- `name` (required): Unique identifier for the flag (e.g., "invented_constraints")
- `description` (required): Human-readable description of what this flag indicates
- `default` (optional): Default value for this flag (defaults to `false`)

### File Format

Rubrics can be defined in YAML or JSON format. Both formats are supported and validated identically.

**YAML Example (`rubric.yaml`):**

```yaml
metrics:
  - name: semantic_fidelity
    description: How well the output preserves the semantic meaning of the input
    min_score: 1
    max_score: 5
    guidelines: |
      Score 1: Completely unfaithful - Output contradicts or has no relation to input
      Score 2: Mostly unfaithful - Major semantic deviations or omissions
      Score 3: Partially faithful - Some key information preserved but with notable gaps
      Score 4: Mostly faithful - Minor deviations but core semantics preserved
      Score 5: Completely faithful - Perfect preservation of semantic meaning

  - name: clarity
    description: How clear and understandable the output is
    min_score: 1.0
    max_score: 5.0
    guidelines: "Rate from 1 (confusing) to 5 (perfectly clear)"

flags:
  - name: invented_constraints
    description: Output introduces constraints not present in the input
    default: false

  - name: requires_verification
    description: Output contains claims that should be verified
    default: true
```

**JSON Example (`rubric.json`):**

```json
{
  "metrics": [
    {
      "name": "code_correctness",
      "description": "Whether the code is syntactically correct and logically sound",
      "min_score": 1,
      "max_score": 5,
      "guidelines": "Score 1: Broken code\nScore 5: Perfect correctness"
    }
  ],
  "flags": [
    {
      "name": "uses_deprecated_apis",
      "description": "Code uses deprecated or outdated APIs",
      "default": false
    }
  ]
}
```

### Loading Rubrics

Use the `load_rubric()` function to load and validate rubric files:

```python
from pathlib import Path
from prompt_evaluator.config import load_rubric

# Load from file
rubric = load_rubric(Path("examples/rubrics/default.yaml"))

# Access metrics
for metric in rubric.metrics:
    print(f"{metric.name}: {metric.min_score}-{metric.max_score}")

# Access flags
for flag in rubric.flags:
    print(f"{flag.name}: default={flag.default}")
```

### Validation Rules

The loader enforces strict validation:

1. **Required Fields**: All required fields must be present and non-empty
2. **Numeric Ranges**: `min_score` must be ≤ `max_score`
3. **Unique Names**: Metric and flag names must be unique (case-insensitive)
4. **No Overlaps**: The same name cannot be used for both a metric and a flag
5. **At Least One Metric**: Every rubric must define at least one metric
6. **Type Validation**: Scores must be numeric, defaults must be boolean

### Validation Errors

The loader provides descriptive error messages for common issues:

```python
# Empty metrics
ValueError: Rubric must contain at least one metric

# Duplicate names
ValueError: Rubric contains duplicate metric names: {'quality'}

# Invalid range
ValueError: Metric 'quality' min_score (10) cannot be greater than max_score (5)

# Missing field
ValueError: Metric at index 0 is missing required field: guidelines

# Type error
ValueError: Metric 'quality' min_score must be numeric, got str
```

### Packaged Rubrics

The package includes several preset rubrics in `examples/rubrics/`:

- **`default.yaml`**: Standard rubric with semantic_fidelity, decomposition_quality, and constraint_adherence metrics
- **`code_review.json`**: Code evaluation rubric with correctness, clarity, and efficiency metrics
- **`content_quality.yaml`**: Content assessment with factual_accuracy, completeness, and clarity metrics

These can be used as-is or as templates for custom rubrics:

```python
from pathlib import Path
from prompt_evaluator.config import load_rubric

# Load default rubric
default = load_rubric(Path("examples/rubrics/default.yaml"))

# Load code review rubric
code_rubric = load_rubric(Path("examples/rubrics/code_review.json"))
```

### Using Rubrics with the CLI

The CLI provides built-in support for using rubrics with evaluation commands through the `--rubric` option and the `show-rubric` command.

#### The --rubric Option

The `--rubric` option on the `evaluate-single` command allows you to specify which rubric to use for evaluation:

```bash
# Use the default rubric (automatic if --rubric is omitted)
prompt-evaluator evaluate-single \
  --system-prompt examples/system_prompt.txt \
  --input examples/input.txt \
  --num-samples 5

# Use a preset rubric by alias
prompt-evaluator evaluate-single \
  --system-prompt examples/system_prompt.txt \
  --input examples/input.txt \
  --num-samples 5 \
  --rubric content-quality

# Use a custom rubric file (absolute or relative path)
prompt-evaluator evaluate-single \
  --system-prompt examples/system_prompt.txt \
  --input examples/input.txt \
  --num-samples 5 \
  --rubric path/to/my_rubric.yaml
```

**Preset Aliases:**

The following preset aliases are available and resolve to bundled rubric files:

- `default` → `examples/rubrics/default.yaml` (semantic fidelity, decomposition quality, constraint adherence)
- `content-quality` → `examples/rubrics/content_quality.yaml` (factual accuracy, completeness, clarity)
- `code-review` → `examples/rubrics/code_review.json` (code correctness, clarity, efficiency)

**File Paths:**

You can also provide a path to your own rubric file:
- Absolute paths: `/absolute/path/to/rubric.yaml`
- Relative paths: `my-rubrics/custom.yaml` (resolved from current working directory)
- Supported formats: `.yaml`, `.yml`, or `.json`

**Default Behavior:**

If `--rubric` is not specified, the CLI automatically uses the `default` preset rubric (`examples/rubrics/default.yaml`).

#### The show-rubric Command

The `show-rubric` command displays the effective rubric as formatted JSON for inspection and validation. This is useful for:
- Verifying rubric files before running evaluations
- Understanding the structure of preset rubrics
- Debugging custom rubrics
- Documenting evaluation criteria

```bash
# Show the default rubric
prompt-evaluator show-rubric

# Show a preset rubric
prompt-evaluator show-rubric --rubric content-quality

# Show a custom rubric file
prompt-evaluator show-rubric --rubric my-rubrics/custom.yaml
```

**Example Output:**

```json
{
  "rubric_path": "/path/to/examples/rubrics/default.yaml",
  "metrics": [
    {
      "name": "semantic_fidelity",
      "description": "How well the output preserves the semantic meaning and intent of the input",
      "min_score": 1,
      "max_score": 5,
      "guidelines": "Score 1: Completely unfaithful...\nScore 5: Completely faithful..."
    },
    {
      "name": "decomposition_quality",
      "description": "How effectively the output breaks down complex concepts...",
      "min_score": 1,
      "max_score": 5,
      "guidelines": "Score 1: Poor decomposition...\nScore 5: Excellent decomposition..."
    }
  ],
  "flags": [
    {
      "name": "invented_constraints",
      "description": "Output introduces constraints or requirements not present in the input",
      "default": false
    }
  ]
}
```

**Key Features:**

- **No API Key Required**: The `show-rubric` command does not require API credentials - it only loads and displays the rubric file
- **Validation**: The command validates the rubric file and reports errors if the rubric is malformed
- **JSON Output**: Output is always formatted JSON, suitable for piping to other tools (e.g., `jq`)

#### Error Handling

The CLI provides clear error messages for rubric-related issues:

**Invalid Preset:**
```bash
$ prompt-evaluator show-rubric --rubric invalid-preset
Error loading rubric: Rubric file not found: /path/to/invalid-preset. 
Please provide a valid file path or use a preset: code-review, content-quality, default
```

**Missing File:**
```bash
$ prompt-evaluator evaluate-single --rubric missing.yaml -s sys.txt -i in.txt -n 1
Error loading rubric: Rubric file not found: /path/to/missing.yaml. 
Please provide a valid file path or use a preset: code-review, content-quality, default
```

**Directory Instead of File:**
```bash
$ prompt-evaluator show-rubric --rubric examples/rubrics/
Error loading rubric: Rubric path points to a directory: /path/to/examples/rubrics. 
Please provide a path to a rubric file (.yaml, .yml, or .json)
```

**Malformed Rubric File:**
```bash
$ prompt-evaluator show-rubric --rubric bad-rubric.yaml
Error loading rubric: Rubric must contain at least one metric
```

### Edge Cases

The rubric system handles various edge cases:

- **Empty Metrics Array**: Rejected with validation error
- **No Flags Section**: Valid - rubric can have metrics only
- **Negative Score Ranges**: Allowed (e.g., min_score=-10, max_score=10)
- **Equal Min/Max**: Valid for fixed-score metrics
- **Whitespace-Only Fields**: Treated as empty and rejected
- **Case Sensitivity**: Name uniqueness is case-insensitive

### Future Extensions

The rubric system is designed for future enhancements:

- Integration with judge prompts for multi-dimensional scoring
- Automatic rubric-based evaluation report generation
- Weighted metric aggregation
- Custom validation rules per metric type
- Rubric versioning and migration support

## Datasets

The Prompt Evaluator supports dataset-driven evaluation, allowing you to define reusable collections of test cases for systematic prompt testing.

### Dataset Schema

Each test case in a dataset includes:

**Required Fields:**
- `id` (string): Unique identifier for the test case
- `input` (string): Input text to be evaluated

**Optional Fields:**
- `description` (string): Description of the test case
- `task` (string): Task description for evaluation context
- `expected_constraints` (string): Constraints that should be satisfied
- `reference` (string): Reference output or expected result
- `metadata` (dict): Any additional custom fields are automatically preserved here

### Supported Formats

Datasets can be provided in two formats with identical semantics:

**JSONL Format** (`.jsonl`): One JSON object per line

```jsonl
{"id": "test-001", "input": "Explain what Python is.", "task": "Explain programming language", "difficulty": "easy"}
{"id": "test-002", "input": "Write a factorial function.", "task": "Code generation", "topic": "algorithms"}
```

**YAML Format** (`.yaml` or `.yml`): List of objects

```yaml
- id: test-001
  input: Explain what Python is.
  task: Explain programming language
  difficulty: easy

- id: test-002
  input: Write a factorial function.
  task: Code generation
  topic: algorithms
```

### Loading Datasets

```python
from pathlib import Path
from prompt_evaluator.config import load_dataset

# Load dataset
test_cases, metadata = load_dataset(Path("examples/datasets/sample.yaml"))

# Access test cases
for test_case in test_cases:
    print(f"ID: {test_case.id}")
    print(f"Input: {test_case.input}")
    print(f"Custom fields: {test_case.metadata}")

# Access dataset metadata
print(f"Path: {metadata['path']}")
print(f"Hash: {metadata['hash']}")
print(f"Count: {metadata['count']}")
```

### Key Features

- **Validation**: Enforces unique IDs, non-empty required fields, and rejects unsupported formats
- **Metadata Passthrough**: Custom fields (e.g., `difficulty`, `topic`, `priority`) are preserved in the `metadata` dict
- **Error Context**: Validation errors reference specific line numbers (JSONL) or indices (YAML)
- **Efficient**: Supports streaming for large datasets (200+ test cases)
- **Order Preservation**: Test cases are processed in file order

### Example Datasets

Sample datasets are provided in `examples/datasets/`:
- `sample.jsonl` - JSONL format example
- `sample.yaml` - YAML format example

### Detailed Documentation

For comprehensive information about dataset formats, schema, validation rules, and best practices, see [docs/datasets.md](docs/datasets.md).

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
