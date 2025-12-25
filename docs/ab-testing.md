# A/B Testing System Prompt Mode

## Overview

The A/B testing mode enables systematic comparison of LLM outputs with and without system prompts. This feature helps quantify the influence of system prompts on model behavior by running paired generations and evaluations.

## Key Features

- **Paired Execution**: Automatically generates two variants for each request:
  - `with_prompt`: Uses the configured system prompt
  - `no_prompt`: Runs without a system prompt (empty string)
  
- **Variant Tagging**: All outputs are tagged with variant metadata for easy filtering and comparison

- **Separate Storage**: Outputs are stored in separate files to prevent collisions:
  - `output_with_prompt.txt` and `output_no_prompt.txt` for generate command
  - Samples tagged with `ab_variant` field in evaluation results

- **Cost Awareness**: Clear warnings about doubled API calls before execution

- **Variant Statistics**: Separate aggregate statistics computed for each variant

## Usage

### Generate Command

Run a single generation with A/B testing:

```bash
prompt-evaluator generate \
  --system-prompt prompts/assistant.txt \
  --input inputs/question.txt \
  --ab-test-system-prompt
```

**Output Structure:**
```
runs/
  <run-id>/
    output_with_prompt.txt      # Output using system prompt
    output_no_prompt.txt         # Output without system prompt
    metadata_with_prompt.json    # Metadata for with_prompt variant
    metadata_no_prompt.json      # Metadata for no_prompt variant
```

### Evaluate-Single Command

Run single-input evaluation with multiple samples per variant:

```bash
prompt-evaluator evaluate-single \
  --system-prompt prompts/assistant.txt \
  --input inputs/task.txt \
  --num-samples 5 \
  --ab-test-system-prompt
```

**Behavior:**
- `--num-samples 5` generates **10 total samples**: 5 with system prompt, 5 without
- Each sample is tagged with its variant (`with_prompt` or `no_prompt`)
- Separate statistics are computed for each variant

**Output Structure:**
```json
{
  "run_id": "...",
  "num_samples": 10,
  "samples": [
    {
      "sample_id": "...-with_prompt-sample-1",
      "ab_variant": "with_prompt",
      ...
    },
    {
      "sample_id": "...-no_prompt-sample-1",
      "ab_variant": "no_prompt",
      ...
    }
  ],
  "aggregate_stats": { ... },
  "variant_stats": {
    "with_prompt": {
      "mean_score": 4.2,
      "num_successful": 5,
      ...
    },
    "no_prompt": {
      "mean_score": 3.1,
      "num_successful": 5,
      ...
    }
  }
}
```

### Evaluate-Dataset Command

Run dataset evaluation with A/B testing across all test cases:

```bash
prompt-evaluator evaluate-dataset \
  --dataset datasets/qa_set.yaml \
  --system-prompt prompts/assistant.txt \
  --num-samples 3 \
  --ab-test-system-prompt
```

**Behavior:**
- Each test case generates samples for both variants
- Total samples = num_test_cases × num_samples × 2
- Example: 10 test cases × 3 samples = **60 total samples** (30 per variant)

**Warning Output:**
```
⚠️  WARNING: A/B testing mode enabled. This will DOUBLE your API calls and costs.
    Total API calls: ~60 (10 cases × 3 samples × 2 variants)
```

## Cost Considerations

### API Call Multiplication

A/B testing **doubles** the number of API calls for both generation and judging:

| Command | Normal Mode | A/B Mode | Multiplier |
|---------|-------------|----------|------------|
| `generate` | 1 call | 2 calls | 2× |
| `evaluate-single --num-samples 5` | 10 calls (5 gen + 5 judge) | 20 calls | 2× |
| `evaluate-dataset` (10 cases, 3 samples) | 60 calls | 120 calls | 2× |

### Cost Estimation

Before running A/B tests, estimate costs:

```bash
# For evaluate-dataset:
# Total API calls = test_cases × samples × 2 × 2
#                   (test cases × samples × variants × (generation + judging))

# Example: 50 test cases, 5 samples per case
# = 50 × 5 × 2 × 2 = 1,000 API calls
```

### Best Practices

1. **Start Small**: Test with `--num-samples 2 --quick` or `--max-cases 5` first
2. **Use `--quick` Mode**: For dataset evaluation, this sets `--num-samples=2` automatically
3. **Filter Test Cases**: Use `--case-ids` to run A/B tests on specific cases only
4. **Monitor Costs**: Check token usage in output metadata files

## Understanding Results

### Variant Comparison

When A/B testing is enabled, results include:

1. **Overall Statistics**: Combined stats across all samples
2. **Variant Statistics**: Separate stats for each variant

Example CLI Output:
```
A/B Test Results:

  Variant: with_prompt
    Successful: 5
    Failed: 0
    Mean Score: 4.20/5.0

  Variant: no_prompt
    Successful: 5
    Failed: 0
    Mean Score: 3.10/5.0
```

### Interpreting Differences

- **Higher scores with system prompt**: System prompt is providing valuable context
- **Similar scores**: System prompt may not be significantly impacting outputs
- **Lower scores with system prompt**: System prompt may be adding unnecessary constraints

### Statistical Significance

For meaningful comparisons:
- Use sufficient samples (recommended: ≥10 per variant)
- Consider variance/std deviation in results
- Run multiple independent evaluations if results are inconclusive

## Integration with Existing Workflows

### Comparison Reports

A/B test results can be compared using the existing `compare-runs` command:

```bash
# Compare two A/B test runs
prompt-evaluator compare-runs \
  --baseline runs/<run-id-1>/evaluate-single.json \
  --candidate runs/<run-id-2>/evaluate-single.json
```

**Note**: Standard comparison shows overall differences. For variant-specific comparisons, filter samples by `ab_variant` in post-processing.

### Programmatic Access

Load and analyze A/B test results programmatically:

```python
import json

# Load evaluation results
with open('runs/<run-id>/evaluate-single.json') as f:
    data = json.load(f)

# Filter samples by variant
with_prompt_samples = [s for s in data['samples'] if s['ab_variant'] == 'with_prompt']
no_prompt_samples = [s for s in data['samples'] if s['ab_variant'] == 'no_prompt']

# Access variant statistics
with_prompt_stats = data['variant_stats']['with_prompt']
no_prompt_stats = data['variant_stats']['no_prompt']

# Compute delta
score_delta = with_prompt_stats['mean_score'] - no_prompt_stats['mean_score']
print(f"System prompt impact: {score_delta:+.2f} points")
```

## Edge Cases and Limitations

### Empty System Prompts

Some providers may behave differently with empty system prompts:
- **OpenAI**: Accepts empty strings, treated as no system message
- **Anthropic**: Accepts empty strings, falls back to default behavior
- **Mock Provider**: Accepts empty strings for testing

If a provider rejects empty system prompts, the `no_prompt` variant will fail and be marked as `generation_error`.

### Partial Failures

If one variant fails while the other succeeds:
- The successful variant's output is saved normally
- The failed variant is marked with `status: "generation_error"`
- Variant statistics reflect only successful samples
- Overall statistics include all samples for comparison

Example output with partial failure:
```
A/B Test Results:
  ✓ with_prompt:
     Output: runs/.../output_with_prompt.txt
     Tokens: 150
  ✗ no_prompt:
     Error: API rate limit exceeded
```

### Dataset Custom System Prompts

**Current Behavior**: The `--ab-test-system-prompt` flag uses the global system prompt provided via `--system-prompt`. If individual test cases have custom system prompts defined in the dataset, those are **not** used in A/B testing.

**Workaround**: For datasets with per-case system prompts, run separate evaluations:
1. First run: Use dataset's system prompts (without A/B flag)
2. Second run: Use `--ab-test-system-prompt` with a standard prompt
3. Compare results manually

## Examples

### Example 1: Quick A/B Test on Small Dataset

```bash
# Test first 5 cases with 2 samples each
prompt-evaluator evaluate-dataset \
  --dataset datasets/qa_small.yaml \
  --system-prompt prompts/assistant.txt \
  --quick \
  --max-cases 5 \
  --ab-test-system-prompt
```

**Output Summary:**
```
⚠️  WARNING: A/B testing mode enabled.
    Total API calls: ~40 (5 cases × 2 samples × 2 variants × 2 (gen+judge))

Starting Dataset Evaluation
Test Cases: 5
A/B Test Mode: Enabled
Samples per Case per Variant: 2
Total Samples per Case: 4
```

### Example 2: Comprehensive Single-Input Analysis

```bash
# Generate 20 samples per variant for robust comparison
prompt-evaluator evaluate-single \
  --system-prompt prompts/coder.txt \
  --input tasks/refactor.txt \
  --num-samples 20 \
  --ab-test-system-prompt \
  --output-dir results/ab_test
```

**Statistical Power**: 20 samples per variant provides good statistical power for detecting differences.

### Example 3: Comparing Two System Prompts

```bash
# Run 1: Prompt A vs no prompt
prompt-evaluator evaluate-single \
  --system-prompt prompts/prompt_a.txt \
  --input test_input.txt \
  --num-samples 10 \
  --ab-test-system-prompt

# Run 2: Prompt B vs no prompt
prompt-evaluator evaluate-single \
  --system-prompt prompts/prompt_b.txt \
  --input test_input.txt \
  --num-samples 10 \
  --ab-test-system-prompt

# Compare the with_prompt variants from both runs
# to see which system prompt performs better
```

## Troubleshooting

### High Variance in Results

If you see high variance between samples:
- Increase `--num-samples` for more stable statistics
- Check if `--seed` is set (may reduce variance but also creativity)
- Review judge consistency (may need rubric adjustments)

### Unexpected Equal Performance

If both variants perform identically:
- Verify system prompt is actually being used (check metadata files)
- Review the task: simple tasks may not benefit from system prompts
- Consider if the user prompt already contains all necessary context

### API Rate Limits

If hitting rate limits with A/B testing:
- Add delays between requests (future enhancement)
- Use `--max-cases` to limit dataset size
- Split large datasets into batches
- Increase `--num-samples` to get more data per test case

## Future Enhancements

Potential improvements for A/B testing mode:

1. **Multi-Variant Testing**: Compare 3+ system prompts simultaneously
2. **Custom Variant Labels**: Allow user-defined variant names
3. **Statistical Tests**: Automatic t-tests or other significance tests
4. **Visualization**: Built-in charts comparing variant performance
5. **Progressive Testing**: Stop early if one variant clearly dominates
6. **Variant-Aware Comparison**: Enhanced `compare-runs` with variant filtering
7. **Dataset per-case prompts**: Support A/B testing with dataset-defined system prompts

## See Also

- [Datasets Documentation](datasets.md) - Dataset structure and format
- [Reporting Documentation](reporting.md) - Generating and interpreting reports
- [README](../README.md) - General usage and configuration
