# Dataset Format Guide

This guide describes the dataset format used for dataset-driven evaluation in the Prompt Evaluator.

## Overview

Datasets allow you to define reusable collections of test cases for evaluating prompts. Each test case consists of required fields (id and input) plus optional fields for providing additional context and metadata.

## Supported Formats

Datasets can be provided in two formats with identical semantics:

- **JSONL** (`.jsonl`): One JSON object per line
- **YAML** (`.yaml` or `.yml`): A list of objects

Both formats support the same schema and are validated identically.

## Schema

### Required Fields

Every test case must include:

- **`id`** (string, non-empty): Unique identifier for the test case
  - Must be unique across the entire dataset
  - Used to track results and identify specific test cases
  - Example: `"test-001"`, `"scenario-alpha"`, `"edge-case-42"`

- **`input`** (string, non-empty): The input text for the test case
  - This is the primary content that will be evaluated
  - Example: `"Explain what Python is in simple terms."`

### Optional Fields

The following optional fields provide additional context:

- **`description`** (string, optional): Human-readable description of the test case
  - Provides context about what this test case is testing
  - Example: `"Basic Python explanation for beginners"`

- **`task`** (string, optional): Description of the task to be performed
  - Can be passed to the judge model for evaluation context
  - Example: `"Explain programming language"`

- **`expected_constraints`** (string, optional): Constraints that should be satisfied
  - Describes requirements or constraints for the output
  - Example: `"Keep it under 100 words, use simple language"`

- **`reference`** (string, optional): Reference output or expected result
  - Provides a baseline or ideal output for comparison
  - Example: `"Python is a popular programming language..."`

### Metadata Passthrough

Any additional fields not explicitly defined in the schema are automatically preserved in the `metadata` dictionary. This allows you to add custom fields for your own use without validation errors.

**Example custom fields:**
- `difficulty`: `"easy"`, `"medium"`, `"hard"`
- `topic`: `"programming"`, `"science"`, `"math"`
- `category`: `"code-generation"`, `"summarization"`
- `priority`: `1`, `2`, `3`
- Any other domain-specific fields you need

These custom fields are accessible via the `metadata` attribute on each `TestCase` object.

## File Formats

### JSONL Format

JSONL files contain one JSON object per line. Each line represents a single test case.

**File: `dataset.jsonl`**

```jsonl
{"id": "test-001", "input": "Explain what Python is in simple terms.", "description": "Basic Python explanation", "task": "Explain programming language", "expected_constraints": "Keep it under 100 words, use simple language", "reference": "Python is a popular programming language known for its readability and ease of use.", "difficulty": "easy", "topic": "programming"}
{"id": "test-002", "input": "Write a function to calculate factorial of a number.", "task": "Code generation", "expected_constraints": "Use recursion, include docstring", "custom_tag": "algorithms"}
{"id": "test-003", "input": "Summarize the water cycle.", "description": "Science education test case", "metadata_field": "example"}
```

**Key features:**
- One test case per line
- Empty lines are ignored
- Preserves order of test cases
- Efficient for large datasets (streaming support)

### YAML Format

YAML files contain a list of objects, where each object is a test case.

**File: `dataset.yaml`**

```yaml
- id: test-001
  input: Explain what Python is in simple terms.
  description: Basic Python explanation
  task: Explain programming language
  expected_constraints: Keep it under 100 words, use simple language
  reference: Python is a popular programming language known for its readability and ease of use.
  difficulty: easy
  topic: programming

- id: test-002
  input: Write a function to calculate factorial of a number.
  task: Code generation
  expected_constraints: Use recursion, include docstring
  custom_tag: algorithms

- id: test-003
  input: Summarize the water cycle.
  description: Science education test case
  metadata_field: example
```

**Key features:**
- Human-readable and editable
- Comments supported (lines starting with `#`)
- Blank lines allowed between entries
- Preserves order of test cases
- Multi-line strings supported with `|` or `>` syntax

## Loading Datasets

Use the `load_dataset()` function to load and validate datasets:

```python
from pathlib import Path
from prompt_evaluator.config import load_dataset

# Load dataset
test_cases, metadata = load_dataset(Path("examples/datasets/sample.yaml"))

# Access test cases
for test_case in test_cases:
    print(f"ID: {test_case.id}")
    print(f"Input: {test_case.input}")
    print(f"Task: {test_case.task}")
    print(f"Custom metadata: {test_case.metadata}")

# Access dataset metadata
print(f"Dataset path: {metadata['path']}")
print(f"Dataset hash: {metadata['hash']}")
print(f"Test case count: {metadata['count']}")
print(f"Format: {metadata['format']}")
```

### Return Values

The `load_dataset()` function returns a tuple:

1. **List of TestCase objects**: Parsed and validated test cases
2. **Dataset metadata dictionary** with:
   - `path`: Absolute path to the dataset file
   - `hash`: SHA-256 hash of the file content (for tracking changes)
   - `count`: Number of test cases in the dataset
   - `format`: File extension (`.jsonl`, `.yaml`, or `.yml`)

## Validation

The loader enforces strict validation to catch errors early:

### Duplicate IDs

```python
# This will raise ValueError
# dataset.jsonl:
# {"id": "test-001", "input": "First case"}
# {"id": "test-001", "input": "Duplicate ID!"}

# Error: Duplicate test case ID 'test-001' found at line 2
```

### Missing Required Fields

```python
# This will raise ValueError
# dataset.jsonl:
# {"input": "Missing ID field"}

# Error: Record at line 1 is missing required field: id
```

### Empty Required Fields

```python
# This will raise ValueError
# dataset.yaml:
# - id: ""
#   input: "Empty ID string"

# Error: Invalid test case at index 0: id field validation failed
```

### Unsupported File Extensions

```python
# This will raise ValueError
load_dataset(Path("dataset.csv"))

# Error: Unsupported dataset file format: .csv. Supported formats: .jsonl, .yaml, .yml
```

## Tips for Curating Datasets

### 1. Use Descriptive IDs

Choose IDs that make it easy to identify test cases in results:

```yaml
# Good
- id: code-factorial-recursive
- id: explain-python-beginner
- id: edge-case-empty-input

# Less helpful
- id: test-1
- id: tc-2
- id: case-3
```

### 2. Leverage Optional Fields

Use optional fields to provide context that helps with evaluation:

```yaml
- id: api-doc-example
  input: Generate API documentation for a REST endpoint
  task: Documentation generation
  expected_constraints: Include parameters, return values, and example usage
  reference: |
    ## POST /api/users
    Creates a new user account.
    
    **Parameters:**
    - username (string, required)
    - email (string, required)
```

### 3. Use Metadata for Organization

Add custom fields to organize and filter test cases:

```yaml
- id: test-001
  input: Explain recursion
  difficulty: medium
  topic: algorithms
  language: python
  priority: 1
```

### 4. Both Description and Task Can Coexist

Don't worry about redundancy - use both fields if helpful:

```yaml
- id: test-complex
  input: Implement a binary search tree
  description: Data structure implementation with core operations
  task: Write a BST class with insert, search, and delete methods
```

### 5. Preserve Order

Test cases are processed in the order they appear in the file. Use this to:
- Start with simple cases and build complexity
- Group related test cases together
- Place high-priority cases first

### 6. Handle Large Datasets

For datasets with 200+ test cases:
- JSONL format is more efficient for streaming
- Consider splitting into multiple files by category
- Use descriptive IDs to track which file a case came from

## Edge Cases

The loader handles various edge cases gracefully:

### Blank Lines (JSONL)

Empty lines are automatically skipped:

```jsonl
{"id": "test-001", "input": "First case"}

{"id": "test-002", "input": "Second case"}

```

### Comments (YAML)

YAML comments are supported:

```yaml
# Production test cases
- id: prod-001
  input: Test case for production

# Experimental test cases
- id: exp-001
  input: Experimental test case
```

### Mixed Field Types

Custom metadata fields can have any JSON-compatible type:

```yaml
- id: test-001
  input: Test input
  priority: 1           # integer
  tags: [a, b, c]       # list
  config:               # nested object
    strict: true
    timeout: 30
```

## Example Datasets

Sample datasets are provided in the `examples/datasets/` directory:

- `sample.jsonl` - JSONL format example
- `sample.yaml` - YAML format example

These demonstrate the schema and can serve as templates for creating your own datasets.

## Dataset Evaluation Workflow

The `evaluate-dataset` command evaluates all test cases in a dataset, producing comprehensive statistics and artifacts.

### Basic Workflow

```bash
# 1. Prepare your dataset
# Create or use an existing dataset file (YAML or JSONL format)

# 2. Run a smoke test first (2 samples × 5 cases = ~20 API calls)
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt prompts/system.txt \
  --max-cases 5 \
  --quick

# 3. Review smoke test results
# Check runs/<run_id>/dataset_evaluation.json for issues

# 4. Run full evaluation if smoke test passes
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt prompts/system.txt \
  --num-samples 5

# 5. Analyze results and iterate
# Review per-case statistics to identify problematic inputs
# Adjust prompt or parameters based on variance and flag rates
```

### Command Overview

**Required flags:**
- `--dataset`, `-d`: Path to dataset file (`.yaml`, `.yml`, or `.jsonl`)
- `--system-prompt`, `-s`: Path to system prompt file

**Key optional flags:**
- `--num-samples`, `-n`: Samples per test case (default: 5)
- `--quick`: Fast mode with 2 samples per case
- `--case-ids`: Filter to specific test cases (comma-separated IDs)
- `--max-cases`: Limit total number of test cases evaluated
- `--rubric`: Evaluation rubric (default: `default`)
- `--temperature`, `-t`: Generator temperature (default: 0.7)
- `--seed`: Random seed for reproducibility
- `--output-dir`, `-o`: Output directory (default: `runs/`)

For complete command reference, see the README.md section on `evaluate-dataset`.

### Runtime Expectations

Evaluation time depends on dataset size, samples per case, and API performance:

**Typical runtimes:**
- **50 cases × 5 samples:** ~25-50 minutes
- **100 cases × 5 samples:** ~50-100 minutes
- **200 cases × 5 samples:** ~100-200 minutes (1.5-3.5 hours)

**Formula:** `test_cases × num_samples × 2 (gen + judge) × 2-3 seconds per call`

**Recommendation:** Always start with `--max-cases 5 --quick` for smoke testing (~40 seconds) before committing to multi-hour runs.

### API Rate Limits

⚠️ **Warning:** Large datasets can hit API rate limits, causing failures or significant slowdowns.

**Strategies to avoid rate limits:**
1. **Smoke test first:** Use `--max-cases 5 --quick` to verify prompt works
2. **Filter strategically:** Test critical cases with `--case-ids case-001,case-002,...`
3. **Batch processing:** Split large datasets using `--max-cases` in multiple runs
4. **Monitor tier limits:** OpenAI Tier 1 (3 RPM) requires `--quick` mode; Tier 3+ (500+ RPM) handles full evaluations

Example smoke test before full run:
```bash
# Quick check: 2 samples × 5 cases = 20 API calls ≈ 40 seconds
prompt-evaluator evaluate-dataset -d data.yaml -s prompt.txt --max-cases 5 --quick

# If successful, run full evaluation: 5 samples × 50 cases = 500 API calls ≈ 16-25 minutes  
prompt-evaluator evaluate-dataset -d data.yaml -s prompt.txt --num-samples 5
```

## Interpreting Stability Metrics

Dataset evaluation provides per-metric statistics to assess prompt consistency and identify issues.

### Understanding Statistics

For each metric on each test case, you get:
- **mean:** Average score across samples
- **std (standard deviation):** Measure of score variability/consistency
- **min/max:** Score range (lowest and highest sample scores)
- **count:** Number of valid samples (excludes failed samples)

### What Standard Deviation Means

Standard deviation (std) indicates how consistently the prompt performs:

**Low std (<0.5):**
- Prompt produces consistent outputs across samples
- Behavior is stable and predictable
- ✓ Good sign - prompt is reliable

**Moderate std (0.5-1.0):**
- Some variation, but generally acceptable
- Minor differences between samples
- Usually acceptable for most applications

**High std (>1.0 or >20% of mean):**
- ⚠️ High variability in prompt behavior
- Outputs are inconsistent across samples
- Indicates potential issues requiring investigation

### Example Interpretation

```
Case: test-001
  semantic_fidelity: mean=4.20, std=0.30, min=3.8, max=4.6
    ✓ Low std (0.30) - consistent semantic preservation
  
  clarity: mean=4.50, std=1.20, min=2.5, max=5.0  ⚠️ HIGH VARIABILITY
    ⚠️ High std (1.20) and wide range (2.5-5.0)
    → Clarity is inconsistent - some outputs very clear, others confusing
    → Action: Review samples with low clarity scores
    → Consider: Lower temperature or add clarity guidelines to prompt
```

### Why High Variance Matters

High standard deviation indicates your prompt's effectiveness is unpredictable:

**Common causes:**
1. **High temperature:** `temp=0.7` or higher increases randomness → try `temp=0.3`
2. **Vague prompt:** Ambiguous instructions allow varied interpretations → add examples or constraints
3. **Input sensitivity:** Prompt works well for some inputs but poorly for others → analyze per-case breakdown
4. **Judge inconsistency:** Judge model may be unreliable (rare if using temp=0.0)

**When to investigate:**
- Std > 1.0 or std > 20% of mean
- Wide min/max range (e.g., 2.0-5.0 on a 1-5 scale)
- High variance on critical metrics (semantic_fidelity, accuracy, etc.)

### Sample Size and Confidence

More samples → more reliable statistics:

| Samples | Confidence Level | Use Case |
|---------|------------------|----------|
| 2-5 | Low | Smoke testing, rapid iteration |
| 5-10 | Moderate | Standard evaluation |
| 10-20 | High | Critical prompts |
| 50+ | Very High | Production validation |

**Note:** The confidence levels above are general guidelines. Actual statistical precision depends on the underlying distribution of metric scores and the inherent variability in model outputs. These estimates assume reasonably well-behaved distributions.

**Rule of thumb:** If std is high, increase sample count to confirm it's real variance, not sampling noise.

### When to Rerun

Rerun with more samples if:
1. **High std with low sample count:** 2-5 samples may not capture true variance
2. **Borderline metrics:** Mean is close to acceptance threshold
3. **Critical prompts:** Production systems require high-confidence validation
4. **After prompt changes:** Verify changes reduced variance as expected

**Example progression:**
```bash
# Initial: Quick test with 2 samples (std may be unreliable)
prompt-evaluator evaluate-dataset -d data.yaml -s prompt.txt --quick

# If std > 1.0, increase to 10 samples to confirm
prompt-evaluator evaluate-dataset -d data.yaml -s prompt.txt --num-samples 10

# If still high std, adjust temperature and retest
prompt-evaluator evaluate-dataset -d data.yaml -s prompt.txt --num-samples 10 --temperature 0.3
```

### Flag Rate Interpretation

Flag statistics show how often specific conditions occur:

```json
{
  "per_flag_stats": {
    "omitted_constraints": {
      "true_count": 12,
      "false_count": 88,
      "total_count": 100,
      "true_proportion": 0.12
    }
  }
}
```

**Interpreting true_proportion:**
- **0.0 (0%):** Flag never triggered → good if flag indicates a problem
- **0.05-0.20 (5-20%):** Flag occasionally triggered → review specific samples, may be acceptable
- **0.20-0.50 (20-50%):** Flag frequently triggered → likely systemic issue
- **>0.50 (>50%):** Flag triggered on majority → serious prompt problem

**Example:**
```
omitted_constraints: true_proportion=0.15 (15%)
  → 15% of samples omit required constraints
  → Action: Review the 15 samples where flag=true
  → Fix: Add explicit constraint checklist to system prompt
```

### Overall vs. Per-Case Statistics

**Per-case statistics:**
- Show performance on each individual test case
- Help identify which inputs cause problems
- Useful for debugging specific failure modes

**Overall statistics:**
- Summarize prompt performance across all test cases
- "mean_of_means" is the average of per-case means
- Provides big-picture view of prompt quality

**When to use each:**
1. **Overall mean is good but some per-case means are poor:**
   - Prompt works well overall but fails on specific input types
   - Use per-case breakdown to find problematic patterns
   
2. **Overall mean is poor:**
   - Systemic prompt issue affecting most inputs
   - Focus on improving core prompt before addressing per-case variance

## Artifact Structure and Schema

### Output Files

Each `evaluate-dataset` run creates a directory with multiple artifacts:

**Directory structure:**
```
runs/
└── <run_id>/
    ├── dataset_evaluation.json    # Main consolidated results
    └── test_case_<id>.json        # Per-case results (streamed)
```

**Main artifact:** `dataset_evaluation.json`
- Complete evaluation results with all test cases and samples
- Per-case and overall statistics
- Generator and judge configurations
- Dataset and rubric metadata with hashes

**Per-case artifacts:** `test_case_<id>.json`
- Individual test case results
- Streamed to disk as each case completes
- Useful for recovering partial results if run is interrupted

### JSON Schema

The `dataset_evaluation.json` file contains:

```json
{
  "run_id": "uuid",                    // Unique run identifier
  "dataset_path": "absolute/path",      // Path to dataset file
  "dataset_hash": "sha256:...",         // Dataset content hash
  "dataset_count": 100,                 // Total test cases in dataset
  "num_samples_per_case": 5,            // Samples per test case
  "status": "completed",                // Run status (see below)
  "timestamp_start": "ISO 8601",        // When run started (UTC)
  "timestamp_end": "ISO 8601",          // When run ended (null if incomplete)
  "system_prompt_path": "path",         // System prompt file path
  "generator_config": { ... },          // Generator model settings
  "judge_config": { ... },              // Judge model settings
  "rubric_metadata": { ... },           // Rubric with hash for reproducibility
  "test_case_results": [ ... ],         // Array of test case results
  "overall_metric_stats": { ... },      // Mean of per-case means
  "overall_flag_stats": { ... }         // Aggregated flag counts
}
```

**Status field values:**
- `"completed"`: All test cases succeeded
- `"partial"`: Some test cases succeeded, some failed
- `"failed"`: All test cases failed
- `"running"`: Evaluation in progress
- `"aborted"`: Run was interrupted

**Sample status values (per sample within test case):**
- `"completed"`: Generation and judging succeeded
- `"judge_error"`: Generation succeeded but judge failed
- `"judge_invalid_response"`: Judge returned unparseable response (excluded from stats)
- `"generation_error"`: Generation failed
- `"pending"`: Not yet processed

For detailed schema with examples, see `examples/run-artifacts/run-sample.json`.

### Content Hashing Rationale

The evaluation tracks content changes using SHA-256 hashes:

**dataset_hash:**
- Hash of dataset file content (not path)
- Detects if test cases have changed between runs
- Allows comparing results across runs with confidence

**rubric_hash:**
- Hash of rubric file content
- Detects if evaluation criteria changed
- Ensures reproducible comparisons

**Why hashing matters:**
1. **Reproducibility:** Verify two runs used identical inputs
2. **Change detection:** Quickly identify when dataset or rubric was modified
3. **Filesystem portability:** Hashes are content-based, so results are comparable even if file paths differ across systems

**Important:** Hashes include whitespace and line endings. Identical content with different line endings (Unix `\n` vs Windows `\r\n`) will produce different hashes. Normalize line endings if exact hash matching is critical.

### Incomplete Runs and Resume

**Detecting incomplete runs:**

Check the `status` and `timestamp_end` fields:

```json
{
  "status": "aborted",        // Run was interrupted
  "timestamp_end": null,       // No end timestamp
  "test_case_results": [
    { "status": "completed" }, // These cases finished
    { "status": "completed" },
    { "status": "pending" }    // This case never started
  ]
}
```

**What gets saved on interruption:**
- Per-case artifacts (`test_case_<id>.json`) for completed cases
- Partial `dataset_evaluation.json` (may be incomplete)
- Overall stats computed only from completed cases

**⚠️ Resume Not Yet Supported:**

Currently, if a run is interrupted:
- You must restart from the beginning
- Or manually filter to remaining cases using `--case-ids`
- Or process in smaller batches using `--max-cases`

**Workaround example:**
```bash
# Original run interrupted after test-001, test-002
# Check which cases completed:
cat runs/<run_id>/dataset_evaluation.json | jq '.test_case_results[].test_case_id'

# Manually run remaining cases:
prompt-evaluator evaluate-dataset \
  --dataset dataset.yaml \
  --system-prompt prompt.txt \
  --case-ids test-003,test-004,test-005 \
  --num-samples 5
```

**Future support:**
- Resume functionality is planned for future versions
- Will allow continuing from last completed test case
- Will merge partial results into final artifact

### Tracking Multiple Runs

To track and compare multiple evaluation runs across time:

1. **Manual tracking:** Create an index file with run summaries
   - See `examples/run-artifacts/index.json` for example structure
   - Track run_id, configuration, metrics, and notes

2. **Version control:** Commit artifacts to git (or LFS for large files)
   - Tag important baseline runs
   - Compare metrics across branches or versions

3. **Compare hashes:** Use dataset_hash and rubric_hash to ensure fair comparison
   - Same hashes = identical evaluation setup
   - Different hashes = results may not be directly comparable

Example index structure:
```json
{
  "runs": [
    {
      "run_id": "uuid-1",
      "description": "Baseline with temp=0.7",
      "dataset_hash": "sha256:...",
      "overall_metrics": { "semantic_fidelity": { "mean_of_means": 4.2 } }
    },
    {
      "run_id": "uuid-2", 
      "description": "Optimized with temp=0.3",
      "dataset_hash": "sha256:...",  // Same hash = same dataset
      "overall_metrics": { "semantic_fidelity": { "mean_of_means": 4.5 } }
    }
  ]
}
```

## Best Practices

1. **Version Control**: Keep datasets in version control to track changes over time
2. **Consistent IDs**: Use a consistent ID naming scheme across datasets
3. **Document Custom Fields**: Document what custom metadata fields mean in a README
4. **Validate Early**: Use `load_dataset()` to validate datasets before running evaluations
5. **Start Small**: Begin with a small dataset (5-10 cases) and expand as needed
6. **Use Both Formats**: YAML for human editing, JSONL for machine-generated datasets

## Troubleshooting

### Parser Errors

If you get a YAML parsing error, check for:
- Incorrect indentation (YAML requires consistent indentation)
- Missing colons after field names
- Unquoted strings with special characters

If you get a JSON parsing error, check for:
- Missing commas between fields
- Unquoted strings
- Trailing commas (not allowed in JSON)

### Validation Errors

If you get validation errors:
- Check that all test cases have unique IDs
- Ensure `id` and `input` fields are present and non-empty
- Verify the file extension is `.jsonl`, `.yaml`, or `.yml`

### Performance Issues

For large datasets (1000+ test cases):
- Prefer JSONL format for better streaming performance
- Consider splitting into multiple smaller datasets
- Use filters based on custom metadata fields to run subsets
