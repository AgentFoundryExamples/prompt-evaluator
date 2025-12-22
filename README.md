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

For detailed artifact schema with annotated examples, see `examples/run-artifacts/run-sample.json`.

#### Runtime Expectations and Performance

**Typical Runtime Estimates:**

The evaluation time depends on test cases, samples per case, and API latency:

- **Small dataset (10 cases, 5 samples):** ~5-10 minutes
- **Medium dataset (50 cases, 5 samples):** ~25-50 minutes  
- **Large dataset (200 cases, 5 samples):** ~100-200 minutes (1.5-3.5 hours)

**Runtime calculation:**
```
Total API calls = test_cases × num_samples × 2 (generator + judge)
Estimated time ≈ Total calls × (2-3 seconds per call)
```

For example, 50 test cases with 5 samples = 500 API calls ≈ 16-25 minutes.

**Factors affecting runtime:**
- **API latency:** OpenAI response times vary by load (typically 1-5 seconds per call)
- **Model selection:** Larger models (e.g., GPT-4) are slower than GPT-3.5
- **Token count:** Longer outputs increase generation time
- **Network conditions:** API round-trip times vary by location and connection

**Rate Limits and API Quotas:**

⚠️ **Warning:** Large evaluations can hit API rate limits, causing failures or slowdowns.

**OpenAI rate limits (typical):**
- **Requests per minute (RPM):** 3,500 (GPT-3.5), 500 (GPT-4)
- **Tokens per minute (TPM):** 90,000 (GPT-3.5), 10,000 (GPT-4)

**Recommendations for avoiding rate limits:**
1. **Start with smoke tests:** Use `--max-cases 5 --quick` to test ~10 API calls before full runs
2. **Use filters strategically:** Test critical cases first with `--case-ids`
3. **Monitor progress:** Watch for rate limit errors in output
4. **Tier-based planning:**
   - Tier 1 (3 RPM): Max ~90 samples/hour → use `--quick` mode only
   - Tier 3 (3,500 RPM): Can handle full datasets with 5-10 samples per case
   - Tier 5 (10,000 RPM): Can run large evaluations with high sample counts

**Example smoke test workflow:**
```bash
# Step 1: Quick smoke test (2 samples × 5 cases = 20 API calls ≈ 40 seconds)
prompt-evaluator evaluate-dataset \
  --dataset dataset.yaml \
  --system-prompt prompt.txt \
  --max-cases 5 \
  --quick

# Step 2: If successful, run on more cases with more samples
prompt-evaluator evaluate-dataset \
  --dataset dataset.yaml \
  --system-prompt prompt.txt \
  --num-samples 5
```

#### Interpreting Stability Metrics

Dataset evaluation provides per-case and overall statistics to help assess prompt consistency and quality.

**Understanding Mean and Standard Deviation:**

For each metric, the evaluation reports:
- **mean:** Average score across samples
- **std (standard deviation):** Measure of score variability
- **min/max:** Score range
- **count:** Number of valid samples (excludes failed/invalid samples)

**What standard deviation tells you:**

- **Low std (<0.5):** Prompt produces consistent outputs → stable and reliable
- **Moderate std (0.5-1.0):** Some variation but generally predictable → acceptable for most uses
- **High std (>1.0 or >20% of mean):** ⚠️ High variability → prompt behavior is inconsistent

**Example interpretation:**

```
semantic_fidelity: mean=4.2, std=0.3, min=3.8, max=4.6
  ✓ Low std (0.3) indicates stable, consistent prompt behavior
  
clarity: mean=4.5, std=1.2, min=2.5, max=5.0  ⚠️ HIGH VARIABILITY
  ⚠️ High std (1.2) and wide range (2.5-5.0) indicate inconsistent clarity
     → Prompt may need refinement to produce more consistent outputs
```

**Why high variance matters:**

High variance indicates that the prompt's effectiveness depends on factors beyond your control:
- **Temperature sensitivity:** Higher temperature (0.7+) increases randomness
- **Prompt ambiguity:** Vague prompts allow model to interpret differently each time
- **Input sensitivity:** Prompt may work well for some inputs but poorly for others

**When to rerun with more samples:**

Use sample count to build confidence:
- **2-5 samples:** Good for smoke testing, but std may be unreliable
- **5-10 samples:** Reasonable confidence for most metrics (std within ~15% of true value)
- **10-20 samples:** High confidence (std within ~10% of true value)
- **50+ samples:** Very high confidence (std within ~5% of true value) → use for critical prompts

**Rule of thumb:** If std > 1.0, consider:
1. **Lower temperature:** Try 0.3 instead of 0.7 to reduce randomness
2. **More samples:** Increase to 10-20 samples to confirm variance is real, not sampling noise
3. **Prompt refinement:** Add constraints or examples to reduce ambiguity
4. **Input analysis:** Check if high variance occurs on specific test cases

**Flag rate interpretation:**

Flag statistics show how often specific conditions occur:
- **true_proportion = 0.0:** Flag never triggered → good if flag indicates a problem
- **true_proportion = 0.1-0.3:** Flag occasionally triggered → investigate specific cases
- **true_proportion > 0.5:** Flag frequently triggered → systemic issue with prompt

Example:
```
omitted_constraints: true_proportion=0.15 (15% of samples)
  → Prompt occasionally misses constraints
  → Review the 15% of samples where flag was true
  → Consider adding explicit reminders in system prompt
```

**Overall vs. Per-Case Statistics:**

- **Per-case stats:** Help identify which specific inputs cause problems
- **Overall stats (mean of means):** Summarize prompt performance across all inputs

If overall mean is good but some per-case means are poor:
- Prompt works well for most inputs but fails on specific cases
- Use per-case breakdown to identify problematic input patterns

#### Incomplete Runs and Future Resume Support

**Detecting Incomplete Runs:**

A run is considered incomplete if:
- **status = "aborted":** Run was interrupted (Ctrl+C, system shutdown, etc.)
- **status = "partial":** Some test cases succeeded but others failed
- **timestamp_end = null:** Run never completed normally

Check the `status` field in `dataset_evaluation.json`:
```json
{
  "status": "aborted",
  "timestamp_end": null,
  "test_case_results": [
    { "test_case_id": "test-001", "status": "completed" },
    { "test_case_id": "test-002", "status": "completed" },
    { "test_case_id": "test-003", "status": "pending" }
  ]
}
```

**What happens on interruption:**

When a run is interrupted:
1. The main `dataset_evaluation.json` file may be incomplete or missing
2. Per-case artifacts (`test_case_<id>.json`) are saved as each case completes
3. Completed test cases have valid results, pending cases have no artifact

**Current Limitation - No Resume:**

⚠️ **Resume functionality is not yet implemented.** If a run is interrupted, you must restart from the beginning or manually determine which cases completed and run the remaining ones.

**Manual workaround strategies:**

These are manual approaches to handle interrupted runs, not built-in resume commands:

1. **Manually filter to remaining cases:**
   ```bash
   # Original run was interrupted after test-001 and test-002
   # Inspect the partial results to identify completed cases
   # Then manually run remaining cases with --case-ids
   prompt-evaluator evaluate-dataset \
     --dataset dataset.yaml \
     --system-prompt prompt.txt \
     --case-ids test-003,test-004,test-005 \
     --num-samples 5
   ```

2. **Use smaller batches from the start:**
   ```bash
   # Instead of running all 100 cases at once, split into batches
   prompt-evaluator evaluate-dataset --case-ids test-001,...,test-020 --num-samples 5
   prompt-evaluator evaluate-dataset --case-ids test-021,...,test-040 --num-samples 5
   # ... etc
   ```

3. **Checkpoint with smaller batches:**
   ```bash
   # Process in chunks of 10 cases to minimize risk
   for i in {0..9}; do
     prompt-evaluator evaluate-dataset \
       --max-cases 10 \
       --num-samples 5 \
       --output-dir runs/batch-$i
   done
   ```

**Note:** These workarounds require manual tracking of completed cases and cannot automatically merge results. You must manually inspect partial artifacts to determine which cases need to be rerun.

**Tracking Multiple Runs:**

To track and compare multiple evaluation runs, you can manually create an index file. See `examples/run-artifacts/index.json` for an example of tracking run metadata, configurations, and results across multiple evaluation campaigns.

**Hashing and Reproducibility:**

The evaluation tracks dataset and rubric changes using SHA-256 hashes:
- **dataset_hash:** Hash of dataset file content (not path) → detects if dataset changed
- **rubric_hash:** Hash of rubric file content → detects if evaluation criteria changed

Same hash = same content = reproducible comparison across runs.
Different hash = content changed = results may not be directly comparable.

**Use cases for hashing:**
1. **Detect dataset drift:** Compare `dataset_hash` across runs to ensure using same test cases
2. **Track rubric changes:** If `rubric_hash` differs, evaluation criteria have changed
3. **Filesystem portability:** Hashes are content-based, so paths can differ across systems while ensuring same content

**Note:** Hashes are computed from file content including whitespace and line endings. Identical content on different operating systems (Unix vs. Windows line endings) may produce different hashes. Normalize line endings if exact hash matching is critical.

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


## Prompt Versioning and Run Tracking

The Prompt Evaluator provides built-in support for tracking prompt versions and associating metadata with evaluation runs. This enables systematic comparison of prompt iterations and tracking of prompt evolution over time.

### Prompt Version Metadata

Every evaluation run can be tagged with:
- **Prompt Version ID**: A human-readable version identifier (e.g., `v1.0`, `baseline`, `experiment-2025-01-15`)
- **Prompt Hash**: An automatic SHA-256 hash of the system prompt file content
- **Run Notes**: Free-form notes about the evaluation run

These metadata fields appear in all run artifacts and enable tracking which prompt version produced which results.

#### Using --prompt-version

The `--prompt-version` flag allows you to explicitly tag runs with a version identifier:

```bash
# Tag a baseline evaluation
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt prompts/system-v1.txt \
  --num-samples 5 \
  --prompt-version "v1.0-baseline" \
  --run-note "Initial baseline evaluation"

# Tag a candidate evaluation with different prompt
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt prompts/system-v2.txt \
  --num-samples 5 \
  --prompt-version "v2.0-clarity-improvements" \
  --run-note "Added explicit clarity guidelines to system prompt"
```

**Both `evaluate-single` and `evaluate-dataset` commands support `--prompt-version` and `--run-note` flags.**

#### Default Hashing Behavior

If you don't provide `--prompt-version`, the tool automatically computes a SHA-256 hash of your system prompt file and uses it as both the `prompt_version_id` and `prompt_hash`:

```bash
# Without explicit version - hash is used automatically
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt prompts/system.txt \
  --num-samples 5
```

In the resulting artifact:
```json
{
  "prompt_version_id": "a1b2c3d4...",  // Hash used as version ID
  "prompt_hash": "a1b2c3d4...",        // Same hash for reproducibility
  "run_notes": null
}
```

With explicit version:
```json
{
  "prompt_version_id": "v1.0-baseline",  // Your custom version
  "prompt_hash": "a1b2c3d4...",          // Hash still computed for tracking
  "run_notes": "Initial baseline evaluation"
}
```

#### Prompt Metadata in Artifacts

All evaluation artifacts include these fields in the JSON output:

- `prompt_version_id`: The version identifier (user-provided or hash)
- `prompt_hash`: SHA-256 hash of the system prompt file content (always computed)
- `run_notes`: Optional notes about the run (null if not provided)

**Why both fields?**
- `prompt_version_id`: Human-readable identifier for easy reference
- `prompt_hash`: Content-based hash ensures you can detect if the prompt file changed between runs with the same version ID
- Together they provide both convenience and safety

### Tagging Runs for Comparison

When preparing to compare two runs, tag them clearly:

```bash
# Step 1: Create baseline with current production prompt
prompt-evaluator evaluate-dataset \
  --dataset datasets/prod_test_cases.yaml \
  --system-prompt prompts/production.txt \
  --num-samples 10 \
  --prompt-version "v1.2-production" \
  --run-note "Production baseline before optimization" \
  --output-dir runs/baselines

# Step 2: Make changes to your prompt and create candidate run
# Edit prompts/production.txt with improvements...

prompt-evaluator evaluate-dataset \
  --dataset datasets/prod_test_cases.yaml \
  --system-prompt prompts/production.txt \
  --num-samples 10 \
  --prompt-version "v1.3-candidate-clarity" \
  --run-note "Added structured output format and clarity guidelines" \
  --output-dir runs/candidates

# Step 3: Compare the runs (see Compare Runs Command section)
prompt-evaluator compare-runs \
  --baseline runs/baselines/<run-id>/dataset_evaluation.json \
  --candidate runs/candidates/<run-id>/dataset_evaluation.json
```

### Best Practices for Version Tracking

**1. Use Semantic Versioning:**
```bash
--prompt-version "v1.0"      # Initial version
--prompt-version "v1.1"      # Minor improvements
--prompt-version "v2.0"      # Major redesign
```

**2. Include Descriptive Labels:**
```bash
--prompt-version "v1.2-baseline"           # Baseline for comparison
--prompt-version "v1.3-add-examples"       # What changed
--prompt-version "v2.0-structured-output"  # Key feature
```

**3. Use Run Notes for Context:**
```bash
--run-note "Baseline evaluation before Q4 optimization project"
--run-note "Testing reduced temperature (0.3 vs 0.7) for consistency"
--run-note "Candidate with explicit constraint checklist added to prompt"
```

**4. Keep Version IDs Consistent with Git Tags:**
```bash
# Tag your prompt file in git
git tag -a prompt-v1.2-prod -m "Production prompt version 1.2"
git push --tags

# Use same version in evaluation
--prompt-version "v1.2-prod"
```

**5. Track Hashes for Reproducibility:**

Check if prompt files actually differ:
```bash
# Extract hashes from two run artifacts
cat runs/run1/dataset_evaluation.json | jq -r '.prompt_hash'
cat runs/run2/dataset_evaluation.json | jq -r '.prompt_hash'

# Different hashes = different prompt content = not directly comparable
# Same hash = identical prompts = fair comparison
```

### Troubleshooting Version Tracking

**Problem: Same prompt_version_id but different prompt_hash**

This means someone changed the prompt file without updating the version:

```json
// Run 1: v1.0-baseline
{
  "prompt_version_id": "v1.0-baseline",
  "prompt_hash": "abc123..."
}

// Run 2: v1.0-baseline (but prompt was edited!)
{
  "prompt_version_id": "v1.0-baseline",
  "prompt_hash": "def456..."  // Different hash!
}
```

**Solution:** Always update `--prompt-version` when you change the prompt file.

**Problem: Can't remember which prompt file a run used**

Use `prompt_hash` to verify:
```bash
# Compute hash of suspected prompt file
sha256sum prompts/system-v1.txt

# Compare with hash in artifact
cat runs/<run-id>/dataset_evaluation.json | jq -r '.prompt_hash'

# If they match, you found the right file
```

**Problem: Need to find all runs using a specific prompt version**

Create a manual index file (see `examples/run-artifacts/index.json` for a template):
```bash
# Search through run artifacts
find runs -name 'dataset_evaluation.json' -exec jq -r \
  'select(.prompt_version_id == "v1.2-prod") | .run_id' {} \;
```

### Compare Runs Command

The `compare-runs` command compares two evaluation runs to detect performance regressions. It computes deltas for metrics and flags, and flags regressions based on configurable thresholds.

```bash
# Basic comparison
prompt-evaluator compare-runs \
  --baseline runs/baseline-run-id/dataset_evaluation.json \
  --candidate runs/candidate-run-id/dataset_evaluation.json

# With custom thresholds
prompt-evaluator compare-runs \
  --baseline runs/baseline-run-id/dataset_evaluation.json \
  --candidate runs/candidate-run-id/dataset_evaluation.json \
  --metric-threshold 0.2 \
  --flag-threshold 0.1

# Save comparison to file
prompt-evaluator compare-runs \
  --baseline runs/baseline-run-id/dataset_evaluation.json \
  --candidate runs/candidate-run-id/dataset_evaluation.json \
  --output comparison-results.json
```

#### Command Parameters

**Required Parameters:**

- `--baseline`, `-b`: Path to baseline run artifact JSON file (e.g., `dataset_evaluation.json` or `evaluate-single.json`)
- `--candidate`, `-c`: Path to candidate run artifact JSON file

**Optional Parameters:**

- `--metric-threshold`: Absolute threshold for metric regression (default: 0.1)
  - A regression is flagged if `candidate_mean < baseline_mean - threshold`
- `--flag-threshold`: Absolute threshold for flag regression (default: 0.05)
  - A regression is flagged if `candidate_proportion > baseline_proportion + threshold`
- `--output`, `-o`: Optional output file for comparison JSON

#### How it Works

The compare-runs command:

1. Loads both baseline and candidate run artifacts
2. Extracts `overall_metric_stats` and `overall_flag_stats` from each run
3. Computes deltas for each metric and flag present in either run
4. Checks each delta against the respective threshold to detect regressions
5. Outputs a comparison summary to stderr and full JSON to stdout
6. Exits with code 1 if any regressions are detected, 0 otherwise

#### Regression Detection

**Metric Regressions:**
- A metric shows regression when the candidate's mean score decreases by more than the threshold
- Formula: `delta = candidate_mean - baseline_mean`
- Regression if: `delta < 0` and `|delta| > metric_threshold`

**Flag Regressions:**
- A flag shows regression when the candidate's true proportion increases by more than the threshold
- Formula: `delta = candidate_proportion - baseline_proportion`
- Regression if: `delta > 0` and `delta > flag_threshold`

#### Output Format

**Human-Readable Summary (stderr):**

```
============================================================
Run Comparison Summary
============================================================
Baseline Run ID: baseline-abc123
Candidate Run ID: candidate-def456
Baseline Prompt: v1.0
Candidate Prompt: v2.0

Thresholds:
  Metric Threshold: 0.1
  Flag Threshold: 0.05

-----------------------Metric Deltas------------------------

  semantic_fidelity: ✓
    Baseline:  4.000
    Candidate: 4.300
    Delta: +0.300
    Change: +7.50%

  clarity: 🔴 REGRESSION
    Baseline:  4.200
    Candidate: 3.800
    Delta: -0.400
    Change: -9.52%

------------------------Flag Deltas-------------------------

  invented_constraints: ✓
    Baseline:  10.00%
    Candidate: 5.00%
    Delta: -5.00%
    Change: -50.00%

--------------------------Summary---------------------------
  🔴 1 regression(s) detected
============================================================
```

**JSON Output (stdout):**

```json
{
  "baseline_run_id": "baseline-abc123",
  "candidate_run_id": "candidate-def456",
  "baseline_prompt_version": "v1.0",
  "candidate_prompt_version": "v2.0",
  "metric_deltas": [
    {
      "metric_name": "semantic_fidelity",
      "baseline_mean": 4.0,
      "candidate_mean": 4.3,
      "delta": 0.3,
      "percent_change": 7.5,
      "is_regression": false,
      "threshold_used": 0.1
    },
    {
      "metric_name": "clarity",
      "baseline_mean": 4.2,
      "candidate_mean": 3.8,
      "delta": -0.4,
      "percent_change": -9.52,
      "is_regression": true,
      "threshold_used": 0.1
    }
  ],
  "flag_deltas": [
    {
      "flag_name": "invented_constraints",
      "baseline_proportion": 0.1,
      "candidate_proportion": 0.05,
      "delta": -0.05,
      "percent_change": -50.0,
      "is_regression": false,
      "threshold_used": 0.05
    }
  ],
  "has_regressions": true,
  "regression_count": 1,
  "comparison_timestamp": "2025-12-22T02:00:00.000000+00:00",
  "thresholds_config": {
    "metric_threshold": 0.1,
    "flag_threshold": 0.05
  }
}
```

#### Use Cases

**1. CI/CD Integration:**

```bash
# Compare new prompt against baseline in CI pipeline
prompt-evaluator compare-runs \
  --baseline baseline/dataset_evaluation.json \
  --candidate runs/latest/dataset_evaluation.json \
  --metric-threshold 0.05 \
  --flag-threshold 0.03

# Exit code 1 if regressions detected, fails the CI build
if [ $? -ne 0 ]; then
  echo "Regressions detected! Blocking deployment."
  exit 1
fi
```

**2. A/B Testing:**

Compare two different prompt versions to determine which performs better:

```bash
prompt-evaluator compare-runs \
  --baseline runs/prompt-v1/dataset_evaluation.json \
  --candidate runs/prompt-v2/dataset_evaluation.json \
  --output prompt-v1-vs-v2-comparison.json
```

**3. Threshold Tuning:**

Experiment with different thresholds to find the right sensitivity:

```bash
# Strict thresholds (catch small regressions)
prompt-evaluator compare-runs \
  --baseline baseline.json \
  --candidate candidate.json \
  --metric-threshold 0.05 \
  --flag-threshold 0.01

# Relaxed thresholds (only catch major regressions)
prompt-evaluator compare-runs \
  --baseline baseline.json \
  --candidate candidate.json \
  --metric-threshold 0.3 \
  --flag-threshold 0.1
```

**4. Tracking Improvements:**

Compare runs to validate that changes actually improved performance:

```bash
# If no regressions and positive deltas, the change was successful
prompt-evaluator compare-runs \
  --baseline runs/before-optimization/dataset_evaluation.json \
  --candidate runs/after-optimization/dataset_evaluation.json
```

#### Understanding Deltas

**Positive Delta (Improvement):**
- Metrics: Higher scores are better → positive delta is good
- Flags: Lower proportions are better (fewer problems) → negative delta is good

**Negative Delta (Potential Regression):**
- Metrics: Lower scores → negative delta indicates regression
- Flags: Higher proportions (more problems) → positive delta indicates regression

**Edge Cases:**

- Missing metrics/flags in one run are included in comparison with `None` values
- Zero baseline values result in `inf` percent change (handled gracefully)
- Asymmetric metric/flag sets between runs are supported

#### Exit Codes

- `0` - Comparison successful, no regressions detected
- `1` - Regressions detected OR comparison failed (file not found, invalid JSON, etc.)

Use exit codes in scripts for automated decision-making:

```bash
if prompt-evaluator compare-runs -b baseline.json -c candidate.json; then
  echo "✓ Safe to deploy"
else
  echo "✗ Regressions detected or comparison failed"
fi
```

#### Current Limitations

**Single Comparison:**
- Compares only two runs at a time
- For comparing multiple candidates, run the command multiple times

**Overall Statistics Only:**
- Compares only `overall_metric_stats` and `overall_flag_stats`
- Does not perform per-case comparisons (may be added in future)

**Static Thresholds:**
- Same threshold applies to all metrics/flags
- Future versions may support per-metric/per-flag thresholds

**No Model Compatibility Enforcement:**
- The tool does not validate or enforce model consistency between baseline and candidate runs
- You can compare runs with different generator or judge models, but the tool will not warn you
- Cross-model comparisons are **technically possible but methodologically problematic** because you cannot isolate prompt changes from model differences
- **Strong recommendation:** Always use identical models (same version) for baseline and candidate to ensure valid comparisons

**No Statistical Significance Testing:**
- The tool does not perform statistical hypothesis testing (t-tests, p-values, etc.)
- Regressions are detected purely by threshold-based delta comparison
- Users must determine appropriate sample sizes for statistical confidence

**No Built-in Visualization:**
- Comparison results are text and JSON only
- For visualizations, export JSON and use external tools (e.g., plotting libraries, dashboards)

## Run Comparison Workflows

This section provides end-to-end guidance on comparing prompt versions, choosing thresholds, and interpreting comparison results.

### Baseline vs Candidate Workflow

The recommended workflow for systematic prompt improvement:

#### Step 1: Establish a Baseline

Run an evaluation with your current prompt and tag it as the baseline:

```bash
# Establish baseline with production prompt
prompt-evaluator evaluate-dataset \
  --dataset datasets/test_suite.yaml \
  --system-prompt prompts/production-v1.txt \
  --num-samples 10 \
  --prompt-version "v1.0-baseline" \
  --run-note "Production baseline - current stable version" \
  --seed 42 \
  --temperature 0.7 \
  --output-dir runs/baselines

# Note the run ID from output for later comparison
# e.g., runs/baselines/abc123-def456-...
```

**Baseline recommendations:**
- Use at least 5-10 samples per test case for reliable statistics
- Use a seed if you want reproducible baselines
- Document generator settings (model, temperature, seed) in run notes
- Store baseline artifacts in a dedicated directory structure

#### Step 2: Make Prompt Changes

Edit your system prompt with improvements:

```bash
# Make changes to prompts/production-v1.txt
# Examples:
# - Add clarity guidelines
# - Include few-shot examples
# - Adjust constraint specifications
# - Restructure instructions

# Save as new version (or edit in place and track via version control)
cp prompts/production-v1.txt prompts/production-v2.txt
# ... make edits to v2 ...
```

#### Step 3: Run Candidate Evaluation

Evaluate with the new prompt using **identical dataset and settings**:

```bash
# Run candidate evaluation with new prompt
prompt-evaluator evaluate-dataset \
  --dataset datasets/test_suite.yaml \
  --system-prompt prompts/production-v2.txt \
  --num-samples 10 \
  --prompt-version "v2.0-candidate-clarity" \
  --run-note "Added structured output format and explicit clarity guidelines" \
  --seed 42 \
  --temperature 0.7 \
  --output-dir runs/candidates

# Note the run ID for comparison
```

**Critical: Keep evaluation settings consistent:**
- ✅ Same dataset file (tool validates matching `dataset_hash`)
- ✅ Same number of samples per test case
- ✅ Same generator model and temperature
- ✅ Same judge model and rubric
- ✅ Same seed for reproducible comparison (optional but recommended)

#### Step 4: Compare Runs

Use `compare-runs` to detect regressions:

```bash
# Compare baseline vs candidate
prompt-evaluator compare-runs \
  --baseline runs/baselines/<baseline-run-id>/dataset_evaluation.json \
  --candidate runs/candidates/<candidate-run-id>/dataset_evaluation.json \
  --metric-threshold 0.1 \
  --flag-threshold 0.05 \
  --output comparison-v1-vs-v2.json
```

The comparison will output:
- Human-readable summary to stderr (shown in terminal)
- Full JSON results to stdout
- Optional saved file if `--output` provided

#### Step 5: Interpret Results

Read the comparison summary carefully:

**If no regressions detected (exit code 0):**
```
✓ 0 regression(s) detected
```
→ Candidate is safe to deploy (or proceed with further testing)

**If regressions detected (exit code 1):**
```
🔴 1 regression(s) detected

  clarity: 🔴 REGRESSION
    Baseline:  4.200
    Candidate: 3.800
    Delta: -0.400
```
→ Review changes and decide:
1. Fix the prompt to address regression
2. Accept the regression if tradeoff is worthwhile
3. Adjust threshold if regression is acceptable variance

#### Step 6: Decision Making

Based on comparison results, choose next action:

**Scenario A: No Regressions, Positive Improvements**
```bash
# All metrics improved or unchanged
# → Deploy the candidate prompt
cp prompts/production-v2.txt prompts/production.txt
git commit -m "Deploy v2.0 - clarity improvements with no regressions"
```

**Scenario B: Minor Regressions in Non-Critical Metrics**
```bash
# Small regression in one metric, big improvement in another
# → Decide if tradeoff is acceptable
# → Document decision in run notes or git commit
```

**Scenario C: Major Regressions**
```bash
# Go back to baseline or iterate on candidate
# → Review what changed in the prompt
# → Make targeted fixes
# → Re-run Step 3-4 with new candidate version
```

### Choosing Regression Thresholds

Thresholds define what counts as a regression. Choose based on your risk tolerance and metric scales.

#### Default Thresholds

The tool provides sensible defaults:
- **Metric threshold: 0.1** (10% of a 1-5 scale = ~2% relative change)
- **Flag threshold: 0.05** (5 percentage points, e.g., 10% → 15%)

These work well for most use cases with 1-5 scale metrics and binary flags.

#### Strict Thresholds (Catch Small Regressions)

Use when:
- Deploying to production with high quality requirements
- Small regressions could have business impact
- You have high confidence in measurement accuracy (10+ samples per case)

```bash
prompt-evaluator compare-runs \
  --baseline baseline.json \
  --candidate candidate.json \
  --metric-threshold 0.05 \  # Half the default
  --flag-threshold 0.02      # <2 percentage point changes flagged
```

**Tradeoff:** More false positives (flagging acceptable variance as regressions).

#### Relaxed Thresholds (Only Catch Major Regressions)

Use when:
- Experimenting with prompts in early iterations
- Minor regressions are acceptable for other gains
- Sample sizes are small (5 or fewer samples) and variance is high

```bash
prompt-evaluator compare-runs \
  --baseline baseline.json \
  --candidate candidate.json \
  --metric-threshold 0.2 \   # Double the default
  --flag-threshold 0.1       # Only flag 10+ percentage point changes
```

**Tradeoff:** May miss smaller regressions that accumulate over time.

#### Threshold Selection Guidelines

**For metrics on 1-5 scale:**
| Threshold | Meaning | When to Use |
|-----------|---------|-------------|
| 0.05 | Very strict - 1% of scale | Production deploys, high sample counts (20+) |
| 0.1 | Moderate - 2% of scale | **Default** - good balance for most cases |
| 0.2 | Relaxed - 4% of scale | Early experiments, low sample counts (<5) |
| 0.3 | Very relaxed - 6% of scale | Initial prototyping, accepting higher variance |

**For flags (proportion changes):**
| Threshold | Meaning | When to Use |
|-----------|---------|-------------|
| 0.01-0.02 | Very strict - 1-2% point change | Critical flags (security, correctness) |
| 0.05 | Moderate - 5% point change | **Default** - good balance |
| 0.1 | Relaxed - 10% point change | Non-critical flags or exploratory work |

#### Calculating Threshold for Your Metrics

**Important:** Thresholds represent **absolute changes** on the metric scale, not relative percentages.

The default threshold of 0.1 on a 1-5 scale means:
- Absolute change: 0.1 points (e.g., 4.0 → 3.9)
- Relative change: ~2% of scale range (0.1 / 5.0)

If your metrics use a different scale, you should scale thresholds to maintain the same **relative sensitivity**:

```python
# For a 0-10 scale (maintaining ~2% sensitivity):
metric_threshold = 0.1 * (10 / 5)  # = 0.2 (2% of 10-point scale)

# For a 0-100 scale (maintaining ~2% sensitivity):
metric_threshold = 0.1 * (100 / 5)  # = 2.0 (2% of 100-point scale)
```

**General formula (preserves relative sensitivity):**
```
threshold = default_threshold * (your_scale_max / 5)
```

**Alternative approach (absolute point changes):**
If you prefer to think in absolute terms (e.g., "flag anything worse by 2 points regardless of scale"), use the same absolute threshold across all scales. However, this means a 2-point drop is more significant on a 1-5 scale (40% drop) than on a 0-100 scale (2% drop).

#### Recommended Threshold Strategies

**Strategy 1: Conservative (Minimize Risk)**
- Start with strict thresholds (0.05 for metrics)
- Gradually relax if you get too many false positives
- Good for production systems and mature prompts

**Strategy 2: Iterative (Balance Speed and Safety)**
- Use default thresholds (0.1 for metrics, 0.05 for flags)
- Review flagged regressions case-by-case
- Good for active development and prompt optimization

**Strategy 3: Exploratory (Maximize Learning)**
- Use relaxed thresholds (0.2 for metrics, 0.1 for flags)
- Focus on major improvements rather than avoiding minor regressions
- Good for research, prototyping, and early iterations

### Interpreting Comparison Results

The comparison output shows three possible states for each metric/flag:

#### Improved (✓)

**Metric:**
```
  semantic_fidelity: ✓
    Baseline:  4.000
    Candidate: 4.300
    Delta: +0.300
    Change: +7.50%
```

**Interpretation:**
- Candidate score is higher than baseline
- No regression - this is an improvement
- Safe to proceed (this metric improved)

**Flag:**
```
  invented_constraints: ✓
    Baseline:  10.00%
    Candidate: 5.00%
    Delta: -5.00%
    Change: -50.00%
```

**Interpretation:**
- Flag occurs less frequently in candidate
- Fewer problems - this is an improvement
- Negative delta for flags is good (lower proportion of issues)

#### Unchanged (✓)

```
  clarity: ✓
    Baseline:  4.500
    Candidate: 4.520
    Delta: +0.020
    Change: +0.44%
```

**Interpretation:**
- Delta is below threshold (0.02 < 0.1)
- Effectively unchanged - within acceptable variance
- Not flagged as regression

#### Regressed (🔴 REGRESSION)

**Metric:**
```
  clarity: 🔴 REGRESSION
    Baseline:  4.200
    Candidate: 3.800
    Delta: -0.400
    Change: -9.52%
```

**Interpretation:**
- Candidate score decreased by more than threshold
- Regression detected - requires attention
- Exit code 1 will be returned

**Flag:**
```
  omitted_constraints: 🔴 REGRESSION
    Baseline:  5.00%
    Candidate: 12.00%
    Delta: +7.00%
    Change: +140.00%
```

**Interpretation:**
- Flag occurs more frequently in candidate
- More problems detected - this is a regression
- Positive delta for flags is bad (higher proportion of issues)

#### Edge Cases in Interpretation

**Missing Metrics:**
```json
{
  "metric_name": "new_metric",
  "baseline_mean": null,
  "candidate_mean": 4.2,
  "delta": null,
  "is_regression": false
}
```
- Metric exists only in candidate (or baseline)
- No regression flagged (can't compare without both values)
- Review manually to assess new metric

**Zero Baseline:**
```json
{
  "metric_name": "clarity",
  "baseline_mean": 0.0,
  "candidate_mean": 4.2,
  "delta": 4.2,
  "percent_change": null
}
```
- Percent change is null or infinite (division by zero)
- Delta is still computed
- Regression detection uses absolute delta only

**Very Small Changes:**
```json
{
  "delta": -0.001,
  "is_regression": false,
  "threshold_used": 0.1
}
```
- Delta is negative but smaller than threshold
- Not flagged as regression
- Likely measurement noise or sampling variance

### Dataset and Model Compatibility

For valid comparisons, baseline and candidate must meet compatibility requirements:

#### Dataset Compatibility

**✅ Same dataset (not enforced, user responsibility):**
```bash
# The tool does NOT validate that datasets match
# You must manually ensure baseline and candidate use the same dataset
```

The compare-runs command will compare any two run artifacts without validation. Results are only meaningful if:
- Same test cases (verify by checking matching `dataset_hash` in both artifacts)
- Same number of samples per case
- Same rubric (verify by checking matching `rubric_hash` if present)

**❌ Different datasets:**
```bash
# Baseline used dataset-v1.yaml (50 test cases)
# Candidate used dataset-v2.yaml (100 test cases)
# → Comparison is not valid! Tool will not prevent this.
```

Even if some test cases overlap, comparing different datasets produces meaningless results:
- Overall statistics aggregate different test cases
- You can't isolate prompt changes from dataset changes
- Per-case deltas are not computed

**Solution:** Always manually verify the exact same dataset file was used for baseline and candidate. Check `dataset_hash` fields in both artifacts to confirm.

#### Model Compatibility

**✅ Same models (strongly recommended):**
```bash
# Baseline: gpt-5.1 generator, gpt-5.1 judge
# Candidate: gpt-5.1 generator, gpt-5.1 judge
# → Valid comparison
```

**⚠️ Different models (not blocked but problematic):**
```bash
# Baseline: gpt-4 generator, gpt-4 judge
# Candidate: gpt-5.1 generator, gpt-5.1 judge
# → Tool will not prevent this comparison, but results are confounded
```

When models differ, you cannot isolate:
- Prompt improvements vs. model improvements
- Regression due to prompt vs. different model behavior

The tool does **not** validate or enforce model consistency. It's the user's responsibility to ensure models match.

**Best practice:**
- Always use identical generator and judge models for valid prompt comparisons
- If models must differ, document it clearly in run notes and interpret results with extreme caution
- Consider running separate experiments: same prompt with different models

**When cross-model comparison might be acceptable:**
- You're intentionally testing a combined prompt + model upgrade (not just the prompt)
- You're doing exploratory analysis (not production decisions)
- You clearly document that results reflect both prompt AND model changes

#### Rubric Compatibility

**✅ Same rubric (validated by hash):**
```bash
# Both runs use examples/rubrics/default.yaml
# rubric_hash matches → metrics are directly comparable
```

**❌ Different rubrics:**
```bash
# Baseline: default.yaml (semantic_fidelity, clarity)
# Candidate: content_quality.yaml (factual_accuracy, completeness)
# → Metrics have different names and definitions!
```

If rubrics differ:
- Metric names may not match
- Comparison includes only overlapping metrics
- Results may be misleading if definitions changed

**Solution:** Use the same rubric file for baseline and candidate. If you must change the rubric, run a new baseline with the new rubric before comparing.

### Parsing Comparison Artifacts

Comparison results are output as JSON for programmatic consumption:

#### Structure of Comparison JSON

```json
{
  "baseline_run_id": "abc123...",
  "candidate_run_id": "def456...",
  "baseline_prompt_version": "v1.0",
  "candidate_prompt_version": "v2.0",
  "metric_deltas": [ ... ],
  "flag_deltas": [ ... ],
  "has_regressions": true,
  "regression_count": 2,
  "comparison_timestamp": "2025-12-22T10:00:00+00:00",
  "thresholds_config": {
    "metric_threshold": 0.1,
    "flag_threshold": 0.05
  }
}
```

For a detailed comparison artifact schema with annotated examples and interpretation guidance, see `examples/run-artifacts/comparison-sample.json`.

#### Extracting Key Information

**Check if any regressions detected:**
```bash
cat comparison.json | jq '.has_regressions'
# Output: true or false

cat comparison.json | jq '.regression_count'
# Output: number of regressions
```

**List all regressed metrics:**
```bash
cat comparison.json | jq -r '.metric_deltas[] | select(.is_regression) | .metric_name'
# Output: List of metric names with regressions
```

**Find largest regression:**
```bash
cat comparison.json | jq -r '.metric_deltas | sort_by(.delta) | .[0] | "\(.metric_name): \(.delta)"'
# Output: metric_name: -0.42
```

**Extract baseline vs candidate prompt versions:**
```bash
cat comparison.json | jq -r '"\(.baseline_prompt_version) → \(.candidate_prompt_version)"'
# Output: v1.0 → v2.0
```

**Get all delta details as CSV:**
```bash
cat comparison.json | jq -r '.metric_deltas[] | [.metric_name, .baseline_mean, .candidate_mean, .delta, .is_regression] | @csv'
# Output: CSV format for spreadsheet analysis
```

#### Building Automation

**Example: CI/CD Script**

```bash
#!/bin/bash
# compare-and-deploy.sh

BASELINE=$1
CANDIDATE=$2
OUTPUT="comparison-result.json"

# Run comparison
prompt-evaluator compare-runs \
  --baseline "$BASELINE" \
  --candidate "$CANDIDATE" \
  --metric-threshold 0.1 \
  --flag-threshold 0.05 \
  --output "$OUTPUT"

EXIT_CODE=$?

# Parse results
REGRESSIONS=$(jq -r '.regression_count' "$OUTPUT")
PROMPT_VERSION=$(jq -r '.candidate_prompt_version' "$OUTPUT")

if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ No regressions detected for $PROMPT_VERSION"
  echo "Safe to deploy candidate prompt"
  exit 0
else
  echo "❌ $REGRESSIONS regression(s) detected for $PROMPT_VERSION"
  echo "Review comparison results before deploying:"
  jq '.metric_deltas[] | select(.is_regression)' "$OUTPUT"
  exit 1
fi
```

**Example: Slack Notification**

```python
import json
import requests

with open("comparison-result.json") as f:
    result = json.load(f)

message = f"""
Prompt Comparison: {result['baseline_prompt_version']} → {result['candidate_prompt_version']}
Regressions: {result['regression_count']}
Status: {"🔴 BLOCKED" if result['has_regressions'] else "✅ APPROVED"}
"""

# Send to Slack webhook
requests.post(SLACK_WEBHOOK_URL, json={"text": message})
```

### Limitations and Caveats

Be aware of these limitations when interpreting comparison results:

#### No Statistical Significance Testing

The tool does **not** perform statistical hypothesis testing:
- No t-tests, p-values, or confidence intervals
- Regressions are detected by simple threshold comparison
- Small sample sizes may produce unreliable comparisons

**Recommendation:**
- Use at least 10 samples per test case for reliable comparisons
- For critical decisions, use 20-50 samples for statistical confidence
- Consider running multiple evaluations and comparing distributions

#### No Visualization Features

Comparison results are text and JSON only:
- No built-in charts or graphs
- No distribution visualizations
- No trend analysis across multiple runs

**Workaround:**
- Export JSON and use external tools (matplotlib, plotly, Tableau, etc.)
- Create custom dashboards for your team
- Track results over time in spreadsheets or databases

#### No Per-Case Comparison

The compare-runs command only compares overall statistics:
- Does not show which specific test cases regressed
- Cannot identify if regression is isolated to certain inputs
- Per-case analysis requires manual inspection of artifacts

**Workaround:**
```bash
# Compare per-case stats manually
diff <(jq '.test_case_results[].per_metric_stats' baseline.json) \
     <(jq '.test_case_results[].per_metric_stats' candidate.json)
```

#### No Model Compatibility Validation

The tool does not validate or warn about model mismatches:
- You can compare runs with different models without any error or warning
- The tool won't tell you if models differ between baseline and candidate
- User must manually check `generator_config.model_name` and `judge_config.model_name` in both artifacts
- No `--allow-different-models` or `--require-same-models` flags exist

**Implication:**
Cross-model comparisons are technically possible but scientifically problematic - you cannot determine whether deltas are due to prompt changes or model differences.

**Workaround:**
- Always document models in run notes
- Manually verify model names match before trusting comparison results
- Use same models for baseline and candidate to ensure valid prompt comparisons


## Roadmap

- [x] Project scaffolding and structure
- [x] Configuration file format and loading
- [x] CLI commands for running evaluations
- [x] LLM provider integrations (OpenAI, etc.)
- [x] Judge models and evaluation data structures for semantic fidelity scoring
- [x] Evaluation command for running multiple samples with aggregate statistics
- [x] Result comparison and analysis tools

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

## Evaluation Reporting

The Prompt Evaluator includes comprehensive support for generating human-readable evaluation reports from JSON artifacts. These reports transform raw evaluation data into structured Markdown documents (with optional HTML conversion) for reviewing prompt quality, stability, and regressions.

### Generate Report Command

Generate a formatted report from a dataset evaluation run or comparison:

```bash
# Single-run report generation
prompt-evaluator render-report --run runs/<run_id>

# Comparison report generation
prompt-evaluator render-report --compare runs/comparison.json

# With custom options
prompt-evaluator render-report \
  --run runs/<run_id> \
  --std-threshold 1.0 \
  --weak-threshold 3.0 \
  --qualitative-count 3 \
  --html \
  --output custom-report.md

# Comparison report with custom options
prompt-evaluator render-report \
  --compare runs/comparison.json \
  --top-cases 5 \
  --html \
  --output comparison-report.md
```

**Common Options:**
- `--html`: Generate HTML report alongside Markdown
- `--output`: Output filename for Markdown report (default: report.md)
- `--html-output`: Output filename for HTML report (default: report.html)

**Single-Run Report Options:**
- `--run`: Path to run directory containing `dataset_evaluation.json` (required)
- `--std-threshold`: Standard deviation threshold for marking metrics as unstable (default: 1.0)
- `--weak-threshold`: Mean score threshold for marking metrics as weak (default: 3.0)
- `--qualitative-count`: Number of worst-case examples to include (default: 3)
- `--max-text-length`: Maximum text length for truncation (default: 500)

**Comparison Report Options:**
- `--compare`: Path to comparison artifact JSON file (required)
- `--top-cases`: Number of top regressed/improved cases to show per metric (default: 5)

### Report Structure

**Single-Run Reports include:**
- **Metadata**: Run ID, status, timestamps, dataset info, model configuration
- **Suite-Level Metrics**: Mean, std, min, max across all test cases
- **Suite-Level Flags**: Flag occurrence rates and proportions
- **Per-Test-Case Summary**: Status, sample counts, with unstable/weak annotations
- **Qualitative Examples**: Worst-performing cases with inputs, outputs, and judge rationales

**Comparison Reports include:**
- **Comparison Metadata**: Baseline/candidate run IDs, prompt versions, regression summary
- **Suite-Level Metrics Comparison**: Baseline, candidate, delta, % change, and status for each metric
- **Suite-Level Flags Comparison**: Baseline, candidate, delta, % change, and status for each flag
- **Regressions Detected**: List of metrics and flags that regressed, sorted by severity
- **Improvements Detected**: List of metrics and flags that improved, sorted by magnitude

Reports automatically:
- Flag metrics with high variance (std > threshold) as **UNSTABLE** (single-run)
- Flag metrics with low scores (mean < threshold) as **WEAK** (single-run)
- Detect regressions and improvements based on configurable thresholds (comparison)
- Show deltas with appropriate signs and formatting
- Handle missing data gracefully with N/A values
- Link back to raw JSON artifacts
- Truncate long text for readability

### Report Types

**Single-Run Reports:**
- Present results from a single `evaluate-dataset` execution
- Include metadata, overall statistics, per-case breakdowns, and qualitative examples
- Annotate unstable metrics (high std deviation) and weak performance (low scores)
- Highlight high flag occurrence rates

**Compare-Runs Reports:**
- Compare baseline vs candidate runs to detect regressions and improvements
- Show metric and flag deltas with regression detection based on thresholds
- Provide detailed breakdowns of improvements and regressions with severity levels
- Display suite-level comparison tables with baseline, candidate, delta, and % change columns
- Highlight regressions (🔴) and improvements (✓) with clear visual indicators
- Support metrics/flags present in only one run (displayed as N/A)

### Key Features

- **Pure Consumer**: Reports are generated from existing JSON artifacts without modifying schemas
- **Fully Configurable**: All thresholds, limits, and formatting options are adjustable via CLI or config
- **Edge Case Handling**: Gracefully handles missing metrics, small datasets, failed test cases, and asymmetric comparisons
- **HTML Conversion**: Optional HTML output with clean styling for sharing (requires `markdown` package)
- **Artifact Links**: Reports include links back to raw JSON files for detailed inspection
- **Deterministic Sorting**: Ties in regression/improvement lists sorted alphabetically for consistency

### Reporting Workflow

The typical workflow for using evaluation reports in prompt engineering:

#### 1. Run Evaluation

First, evaluate your prompt with a dataset:

```bash
# Run evaluation with your prompt
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt prompts/my-prompt.txt \
  --num-samples 5 \
  --prompt-version "v1.0" \
  --output-dir runs/
```

This creates artifacts in `runs/<run_id>/dataset_evaluation.json`.

#### 2. Generate Report

Create a human-readable report from the raw artifacts:

```bash
# Generate Markdown report
prompt-evaluator render-report \
  --run runs/<run_id> \
  --output report.md

# Or generate both Markdown and HTML
prompt-evaluator render-report \
  --run runs/<run_id> \
  --output report.md \
  --html
# HTML output will be named report.html by default (derived from --output)
# To specify a different HTML filename, use --html-output:
#   --html --output report.md --html-output custom-name.html
```

**Tip**: On Windows, use backslashes for paths or quotes: `--run "runs\<run_id>"` or `--run runs/<run_id>` (forward slashes work in most shells).

#### 3. Review Report

Open the generated `report.md` in any Markdown viewer or the HTML file in a browser. The report highlights:

- **🔴 WEAK** metrics: Scores below threshold (default: 3.0 on 1-5 scale)
- **⚠️ UNSTABLE** metrics: High standard deviation (default: >1.0 or >20% of mean)
- **⚠️ High flag rates**: Flags occurring >20% of the time (configurable)
- **Qualitative examples**: Worst-performing samples with judge explanations

#### 4. Iterate on Prompt

Based on report insights:

1. **Identify weak areas**: Look for metrics flagged as WEAK or UNSTABLE
2. **Review examples**: Examine worst-performing samples to understand failure modes
3. **Adjust prompt**: Add constraints, examples, or clarifications
4. **Rerun evaluation**: Test improvements with the same dataset

#### 5. Compare Runs

Compare baseline vs improved prompt:

```bash
# Run with improved prompt
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt prompts/my-prompt-v2.txt \
  --num-samples 5 \
  --prompt-version "v2.0-improvements" \
  --output-dir runs/

# Compare runs
prompt-evaluator compare-runs \
  --baseline runs/<baseline-run-id>/dataset_evaluation.json \
  --candidate runs/<candidate-run-id>/dataset_evaluation.json \
  --output comparison.json

# Generate comparison report
prompt-evaluator render-report \
  --compare comparison.json \
  --output comparison-report.md \
  --html
```

The comparison report shows:
- **🔴 REGRESSION**: Metrics/flags that got worse
- **✅ Improved**: Metrics/flags that got better
- **Delta and % change**: Magnitude of changes

### Interpreting Reports

#### Single-Run Report Sections

1. **Run Summary**: Overview of test cases, samples, models, and completion status
2. **Overall Metric Statistics**: Suite-level aggregates (mean-of-means across test cases)
3. **Overall Flag Statistics**: Suite-level flag occurrence rates with warnings for high rates
4. **Test Case Details**: Per-case breakdown with stability annotations
5. **Qualitative Examples**: Worst-performing samples showing failure patterns

#### Understanding Instability

**What it means**: A metric is marked **⚠️ UNSTABLE** when:
- Standard deviation > 1.0 (absolute threshold), OR
- Standard deviation > 20% of mean (relative threshold)

**Why it matters**: High variance indicates inconsistent prompt behavior. The same input produces very different outputs across samples.

**Example**:
```
clarity: mean=4.5, std=1.2 ⚠️ UNSTABLE
  → Some outputs are very clear (5.0), others confusing (2.5-3.0)
```

**Action**: 
- Lower temperature (try 0.3 instead of 0.7)
- Add more specific constraints or examples to prompt
- Increase sample count to confirm variance is real

#### Understanding Weakness

**What it means**: A metric is marked **🔴 WEAK** when:
- Mean score < 3.0 (default threshold on 1-5 scale)

**Why it matters**: Average performance is below acceptable level.

**Example**:
```
semantic_fidelity: mean=2.8 🔴 WEAK
  → Outputs often fail to preserve input meaning
```

**Action**:
- Review qualitative examples to understand why scores are low
- Adjust prompt to better address the evaluation criteria
- Check if rubric expectations match prompt intent

#### Understanding Flag Rates

**What it means**: A flag is marked **⚠️** when:
- True proportion > 20% (default threshold)

**Why it matters**: The issue occurs frequently across samples.

**Example**:
```
omitted_constraints: 35% ⚠️
  → Over 1/3 of outputs miss required constraints
```

**Action**:
- Add explicit constraint checklist to prompt
- Include examples showing constraint compliance
- Consider adding structured output format

#### Comparison Report Indicators

**🔴 REGRESSION**: 
- Metric decreased by more than threshold (default: 0.1 on 1-5 scale)
- Flag rate increased by more than threshold (default: 5 percentage points)
- Requires attention before deploying prompt

**✅ Improved**:
- Metric increased (any positive delta)
- Flag rate decreased (fewer problems)
- Good sign, but check for tradeoffs

**✅ Unchanged**:
- Delta is below regression threshold
- Effectively no meaningful change
- Acceptable variance

### Sample Reports

Example reports are available in `examples/run-artifacts/`:
- `report-sample.md` - Sample single-run evaluation report
- `comparison-report-sample.md` - Sample comparison report

These demonstrate the output format and annotation styles.

### Report Constraints and Features

**Offline-Friendly:**
- Markdown reports require no external dependencies
- HTML reports (optional) embed all CSS inline - no external assets
- Can be viewed without internet connection

**No JavaScript:**
- HTML reports use pure CSS for styling
- Works in any browser without JavaScript enabled
- Printable and accessible

**Links to Raw Artifacts:**
- Reports include relative links to source JSON files
- Navigate to detailed data for deeper investigation
- Links work in both Markdown viewers and HTML

**Cross-Platform Paths:**
- CLI accepts both Unix (`/`) and Windows (`\`) path separators
- Reports use relative paths when possible
- Absolute paths displayed for clarity in metadata sections

**Adjustable Thresholds:**
- All warning thresholds are configurable via CLI flags
- No hardcoded constants - tune for your use case
- Thresholds documented in Configuration Reference section of each report

### When to Generate Reports

**Generate single-run reports when:**
- Completing initial prompt evaluation
- Assessing stability and quality metrics
- Identifying weak areas or high-variance behaviors
- Preparing for prompt review meetings
- Documenting evaluation results

**Generate comparison reports when:**
- Comparing baseline vs candidate prompts
- Validating prompt improvements
- Detecting regressions before deployment
- A/B testing different prompt variations
- Tracking prompt evolution over time

**Tip**: Generate both Markdown and HTML (`--html` flag) when sharing with non-technical stakeholders. HTML is easier to view in browsers.

### Optional HTML Conversion

HTML generation requires the `markdown` Python package:

```bash
# Install markdown support
pip install markdown

# Generate HTML report
# By default, HTML filename is derived from --output (report.md -> report.html)
prompt-evaluator render-report \
  --run runs/<run_id> \
  --html \
  --output report.md

# Or specify custom HTML output filename
prompt-evaluator render-report \
  --run runs/<run_id> \
  --html \
  --output report.md \
  --html-output custom-report.html
```

If the `markdown` package is not installed, the HTML report will be skipped with a warning. Markdown output always works.

**HTML Features:**
- Clean, responsive design
- Embedded CSS (no external dependencies)
- Syntax highlighting for code blocks
- Works offline
- Printable

### Reporting Specification

For the complete specification including section structures, table formats, configuration parameters, and edge case handling, see [docs/reporting.md](docs/reporting.md).

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
