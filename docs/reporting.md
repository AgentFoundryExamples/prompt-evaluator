# Evaluation Reporting Specification

## Overview

This document specifies the structure and format for human-readable evaluation reports generated from dataset evaluation runs and run comparisons. These reports transform JSON artifacts into formatted Markdown (and optionally HTML) documents that engineers can review to assess prompt quality, stability, and regressions.

**Key Principles:**
- Reports are **consumers** of existing JSON artifacts - they do not modify or require changes to artifact schemas
- All thresholds, limits, and formatting options are **configurable** via CLI flags or configuration files
- Reports support both single-run analysis and comparative analysis (baseline vs candidate)
- Missing or optional data is handled gracefully with clear fallback messaging

## JSON Artifact Sources

Reports consume the following JSON artifacts produced by evaluation commands:

1. **Single Evaluation Run**: `dataset_evaluation.json`
   - Produced by `evaluate-dataset` command
   - Location: `examples/run-artifacts/run-sample.json` (sample schema)
   - Contains: test case results, per-metric stats, per-flag stats, overall aggregates

2. **Run Comparison**: Comparison artifact JSON
   - Produced by `compare-runs` command  
   - Location: `examples/run-artifacts/comparison-sample.json` (sample schema)
   - Contains: metric deltas, flag deltas, regression flags, baseline/candidate metadata

**Important**: Report generation does NOT modify these JSON schemas. Any enhancements to reporting must work with existing artifact structures.

## Single-Run Report Structure

A single-run report presents results from one `evaluate-dataset` execution. The report follows a hierarchical structure optimized for reviewing prompt quality and identifying issues.

### Report Sections (In Order)

1. **Header and Metadata**
2. **Run Summary**
3. **Overall Metric Statistics**
4. **Overall Flag Statistics**
5. **Test Case Details** (with instability/weakness annotations)
6. **Qualitative Examples**
7. **Configuration Reference**

### 1. Header and Metadata

The report begins with identification and context:

```markdown
# Evaluation Report: [Prompt Version ID]

**Run ID**: `a1b2c3d4-e5f6-7890-abcd-ef1234567890`  
**Timestamp**: 2025-12-21 10:00:00 UTC (completed at 10:15:30 UTC)  
**Status**: ✅ Completed / ⚠️ Partial / ❌ Failed  
**Dataset**: `examples/datasets/sample.yaml` (hash: `sha256:1a2b3c4d...`)  
**Prompt**: `v1.0-baseline` (hash: `1a2b3c4d...`)  
**Run Notes**: Testing new prompt variation for improved clarity  
```

**Fields:**
- **Prompt Version ID**: From `prompt_version_id` field (user-provided or hash)
- **Run ID**: Unique identifier for this evaluation (`run_id`)
- **Timestamp**: Start and end times in UTC (`timestamp_start`, `timestamp_end`)
- **Status**: Visual indicator based on `status` field:
  - ✅ `completed` - All test cases succeeded
  - ⚠️ `partial` - Some test cases succeeded, some failed
  - ❌ `failed` - All test cases failed
- **Dataset**: Path and hash for reproducibility tracking
- **Prompt**: Version ID and content hash
- **Run Notes**: Optional user-provided context (`run_notes`, displayed only if non-null)

**Configuration Knobs:**
- None (metadata section always included)

### 2. Run Summary

High-level statistics about the evaluation:

```markdown
## Run Summary

- **Test Cases Evaluated**: 3 total (2 completed, 1 partial, 0 failed)
- **Samples Per Case**: 5
- **Total Samples**: 15 (13 successful, 2 failed)
- **Generator Model**: gpt-5.1 (temperature: 0.7, seed: 42)
- **Judge Model**: gpt-5.1 (temperature: 0.0)
- **Rubric**: `examples/rubrics/default.yaml` (hash: `sha256:abcdef...`)
```

**Fields:**
- **Test Cases Evaluated**: Breakdown by status (`completed`, `partial`, `failed`)
- **Samples Per Case**: From `num_samples_per_case`
- **Total Samples**: Count of successful vs failed samples across all cases
- **Generator Model**: From `generator_config` (model, temperature, seed)
- **Judge Model**: From `judge_config` (model, temperature)
- **Rubric**: From `rubric_metadata` (path, hash)

**Configuration Knobs:**
- None (summary section always included)

### 3. Overall Metric Statistics

Suite-level aggregate statistics showing mean-of-means across all test cases:

```markdown
## Overall Metric Statistics

Aggregate statistics computed from per-case means (test cases with status `completed` or `partial` only).

| Metric | Mean | Min | Max | Cases |
|--------|------|-----|-----|-------|
| semantic_fidelity | 4.39 | 4.10 | 4.67 | 2 |
| clarity | 4.59 | 4.50 | 4.67 | 2 |
| decomposition_quality | 3.85 | 3.50 | 4.20 | 2 |

**Interpretation Guide:**
- **Mean**: Average of per-case means (mean-of-means)
- **Min/Max**: Range of per-case means across test cases
- **Cases**: Number of test cases with valid statistics (excludes failed cases)
```

**Data Source**: `overall_metric_stats` object in artifact

**Table Format:**
- Column 1: Metric name
- Column 2: `mean_of_means` (formatted to 2 decimal places)
- Column 3: `min_of_means` (formatted to 2 decimal places)
- Column 4: `max_of_means` (formatted to 2 decimal places)
- Column 5: `num_cases` (integer)

**Edge Cases:**
- **No valid metrics** (all cases failed): Display message "No valid metric statistics available. All test cases failed."
- **Null statistics**: If `mean_of_means` is null, display "N/A" in table cells

**Configuration Knobs:**
- None (section always included if metrics exist)

### 4. Overall Flag Statistics

Suite-level flag occurrence rates:

```markdown
## Overall Flag Statistics

Aggregate flag statistics computed across all samples with status `completed`.

| Flag | True Count | False Count | Total | True Proportion |
|------|------------|-------------|-------|-----------------|
| invented_constraints | 0 | 8 | 8 | 0.00 (0%) |
| omitted_constraints | 1 | 7 | 8 | 0.13 (13%) |
| requires_verification | 3 | 5 | 8 | 0.38 (38%) ⚠️ |

**Interpretation Guide:**
- **True Count**: Number of samples where flag was set to true
- **False Count**: Number of samples where flag was set to false
- **Total**: Total samples evaluated for this flag
- **True Proportion**: Percentage of samples where flag was true

**⚠️ High Flag Rate**: Flags with true_proportion > 20% are highlighted (configurable threshold).
```

**Data Source**: `overall_flag_stats` object in artifact

**Table Format:**
- Column 1: Flag name
- Column 2: `true_count` (integer)
- Column 3: `false_count` (integer)
- Column 4: `total_count` (integer)
- Column 5: `true_proportion` formatted as decimal (2 places) and percentage with warning if exceeds threshold

**Warning Annotation:**
- Add ⚠️ symbol to True Proportion column if `true_proportion > flag_warning_threshold`
- Default threshold: **0.20** (20%)

**Edge Cases:**
- **No flags in rubric**: Display message "No flags defined in evaluation rubric."
- **Zero samples**: If `total_count` is 0, display "N/A" for proportion

**Configuration Knobs:**
- `flag_warning_threshold` (default: 0.20): Threshold for highlighting problematic flag rates

### 5. Test Case Details

Per-test-case breakdown with stability annotations:

```markdown
## Test Case Details

Detailed statistics for each test case, with annotations for instability and weak performance.

### Test Case: test-001

**Input**: Explain what Python is in simple terms.  
**Status**: ✅ Completed  
**Samples**: 5 successful, 0 failed  
**Metadata**: difficulty=easy, topic=programming

#### Per-Metric Statistics

| Metric | Mean | Std Dev | Min | Max | Count |
|--------|------|---------|-----|-----|-------|
| semantic_fidelity | 4.10 | 0.39 | 3.5 | 4.5 | 5 |
| clarity | 4.50 | 0.45 | 4.0 | 5.0 | 5 |

#### Per-Flag Statistics

| Flag | True Count | False Count | Total | True Proportion |
|------|------------|-------------|-------|-----------------|
| invented_constraints | 0 | 5 | 5 | 0.00 (0%) |
| omitted_constraints | 0 | 5 | 5 | 0.00 (0%) |

---

### Test Case: test-002

**Input**: Write a factorial function in Python.  
**Status**: ⚠️ Partial  
**Samples**: 3 successful, 2 failed  
**Metadata**: difficulty=medium, topic=algorithms

#### Per-Metric Statistics

| Metric | Mean | Std Dev | Min | Max | Count |
|--------|------|---------|-----|-----|-------|
| semantic_fidelity | 4.67 | 0.58 | 4.0 | 5.0 | 3 |
| clarity | 4.67 | 1.10 ⚠️ UNSTABLE | 4.5 | 5.0 | 3 |

**⚠️ UNSTABLE**: clarity has high standard deviation (1.10 > threshold of 1.0).

#### Per-Flag Statistics

| Flag | True Count | False Count | Total | True Proportion |
|------|------------|-------------|-------|-----------------|
| invented_constraints | 0 | 3 | 3 | 0.00 (0%) |
| omitted_constraints | 1 | 2 | 3 | 0.33 (33%) ⚠️ |

**⚠️ HIGH FLAG RATE**: omitted_constraints flag rate (33%) exceeds threshold.

---

### Test Case: test-003

**Input**: Summarize the water cycle.  
**Status**: ❌ Failed  
**Samples**: 0 successful, 5 failed  
**Error**: All samples failed

No statistics available (all samples failed).
```

**Data Source**: `test_case_results` array in artifact

**Per-Case Structure:**
1. **Test Case Header**: test_case_id
2. **Basic Info**: Input text, status, sample counts, metadata
3. **Per-Metric Statistics Table**: From `per_metric_stats`
4. **Per-Flag Statistics Table**: From `per_flag_stats`
5. **Annotations**: Instability and high flag rate warnings

**Status Indicators:**
- ✅ `completed` - All samples succeeded
- ⚠️ `partial` - Some samples succeeded, some failed
- ❌ `failed` - All samples failed

**Instability Detection:**
For each metric, check if:
- `std > instability_threshold_abs` (absolute threshold), OR
- `std > mean * instability_threshold_rel` (relative threshold)

If either condition is met, add "⚠️ UNSTABLE" annotation next to Std Dev value and include explanation below table.

**Default Thresholds:**
- `instability_threshold_abs`: **1.0** (absolute std deviation)
- `instability_threshold_rel`: **0.20** (20% of mean)

**Weakness Detection:**
For each metric, check if:
- `mean < weakness_threshold`

If condition is met, add "🔴 WEAK" annotation next to Mean value and include explanation below table.

**Default Threshold:**
- `weakness_threshold`: **3.0** (on 1-5 scale, configurable per-metric)

**Edge Cases:**
- **Missing Metadata**: If `test_case_metadata` is empty or all fields are null, omit the Metadata line
- **No Samples Successful**: Display "No statistics available" message instead of tables
- **Count Mismatch**: Use `count` field from stats (not `num_samples`) as it reflects valid samples only

**Configuration Knobs:**
- `instability_threshold_abs` (default: 1.0): Absolute std deviation threshold for instability
- `instability_threshold_rel` (default: 0.20): Relative std deviation threshold (as proportion of mean)
- `weakness_threshold` (default: 3.0): Mean score threshold for weakness annotation
- `weakness_threshold_per_metric` (optional): Override weakness threshold for specific metrics

### 6. Qualitative Examples

Selected samples showcasing best and worst performance:

```markdown
## Qualitative Examples

Representative samples selected to illustrate prompt performance. Samples are chosen based on judge scores across all metrics.

**Selection Criteria:**
- **Best Performance**: Samples with highest average metric scores
- **Worst Performance**: Samples with lowest average metric scores
- **Ties**: If multiple samples have the same average score, they are sorted by sample_id in ascending lexicographical order to ensure deterministic selection

### Best Performance Examples

#### Example 1: test-001, sample-1

**Input**: Explain what Python is in simple terms.

**Generator Output**:
```
Python is a popular programming language known for its readability and ease of use. It's widely used for web development, data science, automation, and more.
```

**Judge Evaluation**:
- **semantic_fidelity**: 4.5/5.0 - "Excellent preservation of semantic meaning."
- **clarity**: 4.0/5.0 - "Clear and easy to understand, though could be slightly more structured."
- **Flags**: invented_constraints=false, omitted_constraints=false
- **Overall Comment**: "Strong response that effectively explains Python in accessible terms."

---

#### Example 2: test-002, sample-1

**Input**: Write a factorial function in Python.

**Generator Output**:
```python
def factorial(n):
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

**Judge Evaluation**:
- **semantic_fidelity**: 5.0/5.0 - "Perfect implementation meeting all requirements."
- **clarity**: 5.0/5.0 - "Clear, well-structured code with docstring."
- **Flags**: invented_constraints=false, omitted_constraints=false
- **Overall Comment**: "Excellent recursive implementation."

---

### Worst Performance Examples

#### Example 1: test-001, sample-5

**Input**: Explain what Python is in simple terms.

**Generator Output**:
```
Python is a programming language.
```

**Judge Evaluation**:
- **semantic_fidelity**: 3.5/5.0 - "Technically accurate but lacks detail expected from 'simple terms' explanation."
- **clarity**: 5.0/5.0 - "Extremely clear, though very minimal."
- **Flags**: invented_constraints=false, omitted_constraints=false
- **Overall Comment**: "Too brief but not incorrect."

---

#### Example 2: test-002, sample-3

**Input**: Write a factorial function in Python.

**Generator Output**:
```python
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
```

**Judge Evaluation**:
- **semantic_fidelity**: 4.0/5.0 - "Correct implementation but uses iteration instead of recursion as requested."
- **clarity**: 4.5/5.0 - "Clear and easy to understand, though missing docstring."
- **Flags**: invented_constraints=false, omitted_constraints=true ⚠️
- **Overall Comment**: "Functional but doesn't follow specified constraints."

---
```

**Data Source**: 
- Extract all samples from `test_case_results[].samples[]`
- Compute average score per sample across all metrics
- Sort by average score (descending for best, ascending for worst)
- Select top N for each category

**Selection Algorithm:**
1. For each sample with `status == "completed"`:
   - Compute average score: `avg = sum(metric.score for metric in judge_metrics.values()) / len(judge_metrics)`
   - Store tuple: `(avg, sample_id, sample_object)`
2. Sort by average score descending for best examples
3. Sort by average score ascending for worst examples
4. Select top N from each list
5. **Tie-breaking**: If multiple samples have the same average score, they are sorted by `sample_id` in ascending lexicographical order to ensure deterministic selection.

**Default Limits:**
- `max_best_examples`: **3** (configurable)
- `max_worst_examples`: **3** (configurable)

**Edge Cases:**
- **Fewer samples than limit**: Display all available samples
- **No valid samples**: Display message "No valid samples available for qualitative analysis."
- **No metrics** (legacy mode): Use `judge_score` field instead of averaging metrics
- **Missing rationale**: Display "No rationale provided" if `judge_metrics[metric].rationale` is null

**Configuration Knobs:**
- `max_best_examples` (default: 3): Maximum number of best performance examples
- `max_worst_examples` (default: 3): Maximum number of worst performance examples
- `include_qualitative_examples` (default: true): Whether to include this section at all

### 7. Configuration Reference

Summary of thresholds and settings used for report generation:

```markdown
## Configuration Reference

This report was generated with the following configuration:

**Instability Detection:**
- Absolute threshold: 1.0 (std deviation)
- Relative threshold: 0.20 (20% of mean)

**Weakness Detection:**
- Default threshold: 3.0 (on 1-5 scale)

**Flag Warnings:**
- High flag rate threshold: 0.20 (20%)

**Qualitative Examples:**
- Maximum best examples: 3
- Maximum worst examples: 3

**Report Format:**
- Output format: Markdown
- HTML conversion: Disabled

---

**Artifact Paths:**
- Run artifact: `runs/a1b2c3d4.../dataset_evaluation.json`
- Generated report: `runs/a1b2c3d4.../evaluation_report.md`

---

_Report generated by prompt-evaluator version 0.1.0_
```

**Configuration Knobs:**
- All thresholds and limits documented above
- `output_format` (default: "markdown"): Output format (markdown or html)
- `enable_html_conversion` (default: false): Whether to generate HTML alongside Markdown

## Compare-Runs Report Structure

A compare-runs report presents the comparison between two dataset evaluation runs (baseline vs candidate). The report focuses on identifying regressions and improvements.

### Report Sections (In Order)

1. **Header and Metadata**
2. **Comparison Summary**
3. **Metric Delta Summary**
4. **Flag Delta Summary**
5. **Regression Details**
6. **Improvement Details**
7. **Top Regressed Cases** (per metric) - *Future Enhancement*
8. **Top Improved Cases** (per metric) - *Future Enhancement*
9. **Configuration Reference**

### 1. Header and Metadata

```markdown
# Run Comparison Report

**Baseline Run**: `v1.0-baseline` (ID: `a1b2c3d4-...`)  
**Candidate Run**: `v2.0-clarity-improvements` (ID: `b2c3d4e5-...`)  
**Comparison Timestamp**: 2025-12-22 03:00:00 UTC  
**Comparison Result**: 🔴 **REGRESSIONS DETECTED** / ✅ **NO REGRESSIONS**

## Run Metadata

| Property | Baseline | Candidate |
|----------|----------|-----------|
| Prompt Version | v1.0-baseline | v2.0-clarity-improvements |
| Prompt Hash | 1a2b3c4d... | 2b3c4d5e... |
| Dataset Hash | sha256:abc123... | sha256:abc123... ✅ |
| Generator Model | gpt-5.1 | gpt-5.1 ✅ |
| Judge Model | gpt-5.1 | gpt-5.1 ✅ |
| Rubric Hash | sha256:def456... | sha256:def456... ✅ |

**✅ Match**: Property values are identical between runs (valid comparison)  
**Dataset Hash Match**: Essential for valid comparison (same test cases)  
**Model Match**: Recommended for isolating prompt effects
```

**Data Source**: Comparison artifact fields

**Comparison Result:**
- 🔴 **REGRESSIONS DETECTED** if `has_regressions == true`
- ✅ **NO REGRESSIONS** if `has_regressions == false`

**Metadata Table:**
- Compare key properties between baseline and candidate
- Add ✅ checkmark if values match
- Highlight dataset hash match as critical for validity

**Configuration Knobs:**
- None (header always included)

### 2. Comparison Summary

```markdown
## Comparison Summary

**Regressions Detected**: 2  
**Metrics Compared**: 4  
**Flags Compared**: 3  

**Thresholds Used:**
- Metric regression threshold: 0.10 (absolute delta)
- Flag regression threshold: 0.05 (absolute proportion change)

**Overall Assessment**: 
The candidate run shows 1 metric regression (clarity) and 1 flag regression (omitted_constraints). Review detailed sections below to assess whether regressions are acceptable tradeoffs for other improvements.
```

**Data Source**: Comparison artifact summary fields

**Configuration Knobs:**
- None (summary always included)

### 3. Metric Delta Summary

```markdown
## Metric Delta Summary

Comparison of overall metric statistics between baseline and candidate runs.

| Metric | Baseline | Candidate | Delta | % Change | Status |
|--------|----------|-----------|-------|----------|--------|
| semantic_fidelity | 4.00 | 4.30 | +0.30 | +7.5% | ✅ Improved |
| clarity | 4.20 | 3.80 | -0.40 | -9.5% | 🔴 **REGRESSION** |
| decomposition_quality | 4.50 | 4.52 | +0.02 | +0.4% | ✅ Unchanged |
| constraint_adherence | 3.90 | 3.85 | -0.05 | -1.3% | ✅ Unchanged |

**Legend:**
- ✅ **Improved**: Positive delta, candidate score is higher
- ✅ **Unchanged**: Delta below regression threshold (not significant)
- 🔴 **REGRESSION**: Negative delta exceeding threshold
```

**Data Source**: `metric_deltas` array in comparison artifact

**Table Format:**
- Column 1: `metric_name`
- Column 2: `baseline_mean` (formatted to 2 decimal places)
- Column 3: `candidate_mean` (formatted to 2 decimal places)
- Column 4: `delta` with sign (formatted to 2 decimal places)
- Column 5: `percent_change` with sign and % symbol (formatted to 1 decimal place)
- Column 6: Status based on `is_regression` flag

**Status Determination:**
- 🔴 **REGRESSION**: `is_regression == true`
- ✅ **Improved**: `delta > 0`
- ⚠️ **Degraded**: `delta < 0 and is_regression == false` (negative delta below threshold)
- ✅ **Unchanged**: `delta == 0`

**Edge Cases:**
- **Null baseline or candidate**: Display "N/A" for mean, delta, and percent change
- **New metric**: Display "New Metric" in Status column if `baseline_mean == null`
- **Removed metric**: Display "Removed Metric" in Status column if `candidate_mean == null`
- **Infinite percent change**: Display "N/A" if `baseline_mean == 0`

**Configuration Knobs:**
- None (section always included if metrics exist)

### 4. Flag Delta Summary

```markdown
## Flag Delta Summary

Comparison of overall flag statistics between baseline and candidate runs.

| Flag | Baseline | Candidate | Delta | % Change | Status |
|------|----------|-----------|-------|----------|--------|
| invented_constraints | 10.0% | 5.0% | -5.0pp | -50.0% | ✅ Improved |
| omitted_constraints | 5.0% | 12.0% | +7.0pp | +140.0% | 🔴 **REGRESSION** |
| requires_verification | 30.0% | 32.0% | +2.0pp | +6.7% | ✅ Unchanged |

**Legend:**
- ✅ **Improved**: Negative delta (fewer problems in candidate)
- ✅ **Unchanged**: Delta below regression threshold (not significant)
- 🔴 **REGRESSION**: Positive delta exceeding threshold (more problems in candidate)

**Note**: For flags, positive delta indicates regression (more occurrences of the issue).
```

**Data Source**: `flag_deltas` array in comparison artifact

**Table Format:**
- Column 1: `flag_name`
- Column 2: `baseline_proportion` as percentage (formatted to 1 decimal place)
- Column 3: `candidate_proportion` as percentage (formatted to 1 decimal place)
- Column 4: `delta` with sign and "pp" suffix (percentage points)
- Column 5: `percent_change` with sign and % symbol (formatted to 1 decimal place)
- Column 6: Status based on `is_regression` flag

**Status Determination:**
- 🔴 **REGRESSION**: `is_regression == true`
- ✅ **Improved**: `delta < 0 and is_regression == false` (fewer occurrences)
- ✅ **Unchanged**: `delta >= 0 and is_regression == false` (below threshold)

**Edge Cases:**
- **Null baseline or candidate**: Display "N/A" for proportion, delta, and percent change
- **New flag**: Display "New Flag" in Status column if `baseline_proportion == null`
- **Removed flag**: Display "Removed Flag" in Status column if `candidate_proportion == null`
- **Zero baseline**: Display "N/A" for percent change if `baseline_proportion == 0`

**Configuration Knobs:**
- None (section always included if flags exist)

### 5. Regression Details

```markdown
## Regression Details

Detailed information about detected regressions.

### Metric Regressions

#### clarity: -0.40 (-9.5%)

- **Baseline**: 4.20
- **Candidate**: 3.80
- **Delta**: -0.40
- **Threshold Used**: 0.10
- **Assessment**: Candidate mean decreased by 0.40, exceeding the regression threshold. This indicates a measurable decline in clarity.

---

### Flag Regressions

#### omitted_constraints: +7.0pp (+140.0%)

- **Baseline**: 5.0% (5 occurrences)
- **Candidate**: 12.0% (12 occurrences)
- **Delta**: +7.0 percentage points
- **Threshold Used**: 0.05
- **Assessment**: Flag occurrence rate increased by 7 percentage points, exceeding the regression threshold. More samples are omitting constraints in the candidate run.

---
```

**Data Source**: Filter `metric_deltas` and `flag_deltas` where `is_regression == true`

**Structure:**
- Group by type (metrics, then flags)
- For each regression, display detailed breakdown with context

**Edge Cases:**
- **No regressions**: Display message "No regressions detected. All metrics and flags meet acceptance criteria."

**Configuration Knobs:**
- None (section included only if regressions exist)

### 6. Improvement Details

```markdown
## Improvement Details

Metrics and flags that improved in the candidate run.

### Metric Improvements

#### semantic_fidelity: +0.30 (+7.5%)

- **Baseline**: 4.00
- **Candidate**: 4.30
- **Delta**: +0.30
- **Assessment**: Candidate mean increased by 0.30, indicating improved semantic fidelity.

---

### Flag Improvements

#### invented_constraints: -5.0pp (-50.0%)

- **Baseline**: 10.0% (10 occurrences)
- **Candidate**: 5.0% (5 occurrences)
- **Delta**: -5.0 percentage points
- **Assessment**: Flag occurrence rate decreased by 5 percentage points. Fewer samples are inventing constraints in the candidate run.

---
```

**Data Source**: 
- Metrics: Filter `metric_deltas` where `delta > 0 and is_regression == false`
- Flags: Filter `flag_deltas` where `delta < 0 and is_regression == false`

**Structure:**
- Group by type (metrics, then flags)
- For each improvement, display detailed breakdown

**Edge Cases:**
- **No improvements**: Omit this section entirely (not an error, just no improvements to report)

**Configuration Knobs:**
- `include_improvements` (default: true): Whether to include this section

### 7. Top Regressed Cases (Per Metric)

```markdown
## Top Regressed Cases (Per Metric)

Test cases showing the largest per-case regressions for each metric. Useful for identifying which inputs are most affected by prompt changes.

**Note**: This section requires per-case comparison, which is not currently implemented in the JSON comparison artifact. This is a future enhancement.

**Placeholder Structure** (for future implementation):

### semantic_fidelity: Top 3 Regressed Cases

| Test Case | Baseline Mean | Candidate Mean | Delta |
|-----------|---------------|----------------|-------|
| test-042 | 4.8 | 3.2 | -1.6 |
| test-017 | 4.5 | 3.3 | -1.2 |
| test-089 | 4.9 | 3.9 | -1.0 |

---

### clarity: Top 3 Regressed Cases

| Test Case | Baseline Mean | Candidate Mean | Delta |
|-----------|---------------|----------------|-------|
| test-023 | 4.7 | 3.1 | -1.6 |
| test-055 | 4.3 | 3.2 | -1.1 |
| test-091 | 4.6 | 3.6 | -1.0 |

---
```

**Current Status**: 
- **NOT IMPLEMENTED** in current JSON schema
- Comparison artifact only contains overall statistics, not per-case breakdowns
- Future enhancement: Extend comparison artifact to include per-case deltas

**Future Data Source**:
- Would require new field: `per_case_metric_deltas` in comparison artifact
- Structure: `{ test_case_id: string, metric_name: string, baseline_mean: float, candidate_mean: float, delta: float }`

**Configuration Knobs** (future):
- `max_regressed_cases_per_metric` (default: 3): Number of top regressed cases to display per metric
- `min_regression_delta` (default: 0.5): Minimum delta to consider for top regressed cases

### 8. Top Improved Cases (Per Metric)

```markdown
## Top Improved Cases (Per Metric)

Test cases showing the largest per-case improvements for each metric.

**Note**: This section requires per-case comparison, which is not currently implemented in the JSON comparison artifact. This is a future enhancement.

**Placeholder Structure** (for future implementation):

### semantic_fidelity: Top 3 Improved Cases

| Test Case | Baseline Mean | Candidate Mean | Delta |
|-----------|---------------|----------------|-------|
| test-015 | 3.2 | 4.8 | +1.6 |
| test-067 | 3.5 | 4.6 | +1.1 |
| test-033 | 3.8 | 4.8 | +1.0 |

---
```

**Current Status**: 
- **NOT IMPLEMENTED** in current JSON schema
- Same limitation as Top Regressed Cases

**Future Data Source**:
- Same as Top Regressed Cases, but sorted by positive delta descending

**Configuration Knobs** (future):
- `max_improved_cases_per_metric` (default: 3): Number of top improved cases to display per metric
- `min_improvement_delta` (default: 0.5): Minimum delta to consider for top improved cases

### 9. Configuration Reference

```markdown
## Configuration Reference

This comparison report was generated with the following configuration:

**Regression Thresholds:**
- Metric threshold: 0.10 (absolute delta)
- Flag threshold: 0.05 (absolute proportion change)

**Report Format:**
- Output format: Markdown
- HTML conversion: Disabled

**Sections Included:**
- Metric Delta Summary: Yes
- Flag Delta Summary: Yes
- Regression Details: Yes
- Improvement Details: Yes
- Top Regressed/Improved Cases: No (not yet implemented)

---

**Artifact Paths:**
- Baseline run: `runs/a1b2c3d4.../dataset_evaluation.json`
- Candidate run: `runs/b2c3d4e5.../dataset_evaluation.json`
- Comparison artifact: `comparison-result.json`
- Generated report: `comparison-report.md`

---

_Report generated by prompt-evaluator version 0.1.0_
```

**Configuration Knobs:**
- All thresholds and settings documented above

## HTML Conversion

Reports can optionally be converted to HTML for richer viewing and sharing.

### Conversion Options

**Default Behavior:**
- Reports are generated as Markdown (`.md` files)
- HTML conversion is **disabled by default**

**Enable HTML Conversion:**
```bash
prompt-evaluator report-single \
  --run runs/<run_id>/dataset_evaluation.json \
  --output report.md \
  --enable-html
```

**Output:**
- Markdown file: `report.md`
- HTML file: `report.html` (generated automatically when `--enable-html` is set)

### HTML Styling

HTML conversion uses a minimal, clean stylesheet with:
- Responsive layout (works on mobile and desktop)
- Syntax highlighting for code blocks
- Clear visual hierarchy (headings, tables, lists)
- Accessibility features (semantic HTML, ARIA labels)

**Customization:**
- HTML template and CSS are embedded in the report generator
- Users can override with `--html-template` flag pointing to custom template file

### HTML Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report: [Prompt Version]</title>
    <style>
        /* Embedded CSS for styling */
    </style>
</head>
<body>
    <!-- Converted Markdown content -->
</body>
</html>
```

## Configuration Knobs Reference

All thresholds, limits, and options that control report generation:

### Single-Run Report Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instability_threshold_abs` | float | 1.0 | Absolute std deviation threshold for instability annotation |
| `instability_threshold_rel` | float | 0.20 | Relative std deviation threshold (as proportion of mean) |
| `weakness_threshold` | float | 3.0 | Mean score threshold for weakness annotation (1-5 scale) |
| `weakness_threshold_per_metric` | dict | {} | Override weakness threshold for specific metrics (e.g., `{"clarity": 3.5}`) |
| `flag_warning_threshold` | float | 0.20 | True proportion threshold for highlighting high flag rates |
| `max_best_examples` | int | 3 | Maximum number of best performance qualitative examples |
| `max_worst_examples` | int | 3 | Maximum number of worst performance qualitative examples |
| `include_qualitative_examples` | bool | true | Whether to include qualitative examples section |
| `output_format` | str | "markdown" | Output format ("markdown" or "html") |
| `enable_html_conversion` | bool | false | Whether to generate HTML alongside Markdown |
| `html_template` | str | null | Path to custom HTML template (optional) |

### Compare-Runs Report Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric_threshold` | float | 0.10 | Absolute threshold for metric regression detection |
| `flag_threshold` | float | 0.05 | Absolute threshold for flag regression detection |
| `include_improvements` | bool | true | Whether to include improvement details section |
| `max_regressed_cases_per_metric` | int | 3 | Top N regressed cases per metric (future) |
| `max_improved_cases_per_metric` | int | 3 | Top N improved cases per metric (future) |
| `min_regression_delta` | float | 0.5 | Minimum delta for top regressed cases (future) |
| `min_improvement_delta` | float | 0.5 | Minimum delta for top improved cases (future) |
| `output_format` | str | "markdown" | Output format ("markdown" or "html") |
| `enable_html_conversion` | bool | false | Whether to generate HTML alongside Markdown |
| `html_template` | str | null | Path to custom HTML template (optional) |

## Edge Case Handling

### Missing or Null Data

**Scenario**: Some optional metrics or flags are missing from the run artifact.

**Handling**:
- **Missing metrics**: Omit from metric tables entirely (do not show empty rows)
- **Null statistics**: Display "N/A" in table cells for null values
- **No metrics at all**: Display message "No metrics available in this run."
- **No flags defined**: Display message "No flags defined in evaluation rubric."

**Example**:
```markdown
## Overall Metric Statistics

No metrics available in this run.
```

### Small Datasets

**Scenario**: Dataset has very few test cases (e.g., 1-3 cases).

**Handling**:
- Render all sections normally
- Include note in summary: "Note: Small dataset (N test cases). Statistics may have high variance."
- All tables and examples still render with available data

**Minimum Requirements**:
- At least 1 test case with status `completed` or `partial` for statistics
- If all cases failed, display "No valid statistics" message

### Tie-Breaking in Qualitative Examples

**Scenario**: Multiple samples have identical average scores.

**Handling**:
- Sort by average score first (descending for best, ascending for worst)
- Break ties by sorting by `sample_id` lexicographically (deterministic)
- This ensures consistent selection across multiple report generations

**Example**:
```python
# Pseudocode for sorting
samples_with_scores = [
    (4.5, "test-001-sample-1", sample_obj_1),
    (4.5, "test-001-sample-2", sample_obj_2),  # Same score
    (4.5, "test-001-sample-3", sample_obj_3),  # Same score
]

# Sort by score descending, then by sample_id ascending
sorted_samples = sorted(samples_with_scores, key=lambda x: (-x[0], x[1]))

# Result: sample-1, sample-2, sample-3 (deterministic order)
```

### Missing Metrics in Comparison

**Scenario**: Baseline and candidate runs have different metrics (one metric present in only one run).

**Handling**:
- Include all metrics found in either run
- For metrics only in baseline: Display baseline value, "N/A" for candidate, null delta
- For metrics only in candidate: Display "N/A" for baseline, candidate value, null delta
- Status column: "New Metric" or "Removed Metric"
- Do not flag as regression (no comparison possible)

**Example**:
| Metric | Baseline | Candidate | Delta | % Change | Status |
|--------|----------|-----------|-------|----------|--------|
| semantic_fidelity | 4.00 | 4.30 | +0.30 | +7.5% | ✅ Improved |
| new_metric | N/A | 4.50 | N/A | N/A | New Metric |
| removed_metric | 3.80 | N/A | N/A | N/A | Removed Metric |

### Failed Test Cases

**Scenario**: One or more test cases have status `failed` (all samples failed).

**Handling**:
- Include test case in Test Case Details section
- Mark with ❌ Failed status
- Display error message: "No statistics available (all samples failed)."
- Omit per-metric and per-flag tables for failed cases
- Exclude from overall statistics calculations

### Large Number of Test Cases

**Scenario**: Dataset has many test cases (e.g., 200+ cases).

**Handling**:
- Render all test cases in Test Case Details section
- No truncation or pagination (Markdown supports long documents)
- Consider adding a "Jump to" navigation menu at the top (optional, configurable)
- For HTML output, include table of contents with anchor links

**Configuration Knob**:
- `include_toc` (default: false): Whether to include table of contents for navigation

## JSON Schema Guarantees

**Critical Principle**: Report generation is a **pure consumer** of existing JSON artifacts. No changes to artifact schemas are required or permitted.

### Schema Stability Guarantees

1. **No New Fields Required**: Reports work with existing `dataset_evaluation.json` and comparison artifact schemas as documented in `examples/run-artifacts/`.

2. **Backward Compatibility**: If artifact schemas are extended in the future, report generation must continue to work with older artifacts.

3. **Optional Field Handling**: All report logic gracefully handles missing or null optional fields.

4. **Forward Compatibility**: Future report enhancements (e.g., per-case comparison) may consume new artifact fields if/when they are added, but current reports must work with current schemas.

### Future Schema Extensions

If artifact schemas are extended to support richer reporting:

**Allowed Extensions** (future consideration):
- Add per-case comparison fields to comparison artifact for "Top Regressed/Improved Cases" sections
- Add additional metadata fields for enhanced report context
- Add per-sample timing information for performance analysis

**Not Allowed**:
- Modifying existing field types or semantics
- Removing fields that reports depend on
- Changing field names (breaks existing reports)

### Artifact Schema References

- **Single Run Artifact**: `examples/run-artifacts/run-sample.json`
- **Comparison Artifact**: `examples/run-artifacts/comparison-sample.json`

All report specifications must align with these schemas.

## CLI Integration

Report generation is exposed via CLI commands (future implementation):

### Generate Single-Run Report

```bash
prompt-evaluator report-single \
  --run runs/<run_id>/dataset_evaluation.json \
  --output report.md \
  --instability-threshold-abs 1.0 \
  --instability-threshold-rel 0.20 \
  --weakness-threshold 3.0 \
  --flag-warning-threshold 0.20 \
  --max-best-examples 3 \
  --max-worst-examples 3 \
  --enable-html
```

### Generate Compare-Runs Report

```bash
prompt-evaluator report-compare \
  --baseline runs/<baseline_id>/dataset_evaluation.json \
  --candidate runs/<candidate_id>/dataset_evaluation.json \
  --output comparison-report.md \
  --metric-threshold 0.10 \
  --flag-threshold 0.05 \
  --enable-html
```

### Configuration File Support

Thresholds and settings can also be provided via configuration file:

**report-config.yaml:**
```yaml
single_run:
  instability_threshold_abs: 1.0
  instability_threshold_rel: 0.20
  weakness_threshold: 3.0
  flag_warning_threshold: 0.20
  max_best_examples: 3
  max_worst_examples: 3
  
compare_runs:
  metric_threshold: 0.10
  flag_threshold: 0.05
```

**Usage:**
```bash
prompt-evaluator report-single \
  --run runs/<run_id>/dataset_evaluation.json \
  --config report-config.yaml
```

CLI flags take precedence over config file values.

## Links to Raw Artifacts

Reports should include clickable links back to raw JSON artifacts for detailed inspection:

### Single-Run Report Links

```markdown
---

**Raw Artifacts:**
- [Run artifact (JSON)](../runs/<run_id>/dataset_evaluation.json)
- [Per-case artifacts](../runs/<run_id>/)

---
```

### Compare-Runs Report Links

```markdown
---

**Raw Artifacts:**
- [Baseline run (JSON)](../runs/<baseline_id>/dataset_evaluation.json)
- [Candidate run (JSON)](../runs/<candidate_id>/dataset_evaluation.json)
- [Comparison artifact (JSON)](comparison-result.json)

---
```

**Path Handling:**
- Links should be relative to report location if possible
- Absolute paths used if relative paths cannot be resolved
- Links work in both Markdown viewers and HTML output

## Specification Maintenance

This specification defines the contract for report generation. Future implementations must:

1. **Preserve Section Order**: Maintain the documented section ordering for consistency
2. **Honor Configuration Knobs**: Support all documented configuration parameters
3. **Handle Edge Cases**: Gracefully handle all edge cases documented above
4. **Maintain Schema Compatibility**: Work with existing JSON artifact schemas
5. **Update Documentation**: Update this spec when adding new features or configuration options

**Versioning**: This specification is version 1.0. Changes to report structure or configuration should be documented as version increments.

## Future Enhancements

Potential future additions to this specification:

1. **Per-Case Comparison**: Extend comparison artifacts and reports to show per-case regressions/improvements
2. **Trend Analysis**: Multi-run comparison showing metrics over time
3. **Interactive HTML**: JavaScript-enhanced HTML reports with filtering and sorting
4. **Custom Sections**: Plugin architecture for custom report sections
5. **Export Formats**: CSV, PDF, or other export formats
6. **Dashboard Integration**: JSON output optimized for dashboard consumption

These enhancements should be added as extensions to this spec, maintaining backward compatibility.

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-22  
**Specification Status**: Final
