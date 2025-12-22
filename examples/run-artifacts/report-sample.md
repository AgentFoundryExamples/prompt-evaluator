# Evaluation Report: v1.0-baseline

> **Note**: This is a sample evaluation report generated from synthetic data for documentation purposes.

**Run ID**: `a1b2c3d4-e5f6-7890-abcd-ef1234567890`  
**Timestamp**: 2025-12-21 10:00:00 UTC (completed at 10:15:30 UTC)  
**Status**: ⚠️ Partial (some test cases had failed samples)  
**Dataset**: `examples/datasets/sample.yaml` (hash: `sha256:1a2b3c4d...`)  
**Prompt**: `v1.0-baseline` (hash: `1a2b3c4d...`)  
**Run Notes**: Testing new prompt variation for improved clarity  

---

## Run Summary

- **Test Cases Evaluated**: 3 total (2 completed, 1 partial, 0 failed)
- **Samples Per Case**: 5
- **Total Samples**: 13 successful, 2 failed
- **Generator Model**: gpt-5.1 (temperature: 0.7, seed: 42)
- **Judge Model**: gpt-5.1 (temperature: 0.0)
- **Rubric**: `examples/rubrics/default.yaml` (hash: `sha256:abcdef...`)

---

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

---

## Overall Flag Statistics

Aggregate flag statistics computed across all samples with status `completed`.

| Flag | True Count | False Count | Total | True Proportion |
|------|------------|-------------|-------|-----------------|
| invented_constraints | 0 | 8 | 8 | 0.00 (0%) |
| omitted_constraints | 1 | 7 | 8 | 0.13 (13%) |

**Interpretation Guide:**
- **True Count**: Number of samples where flag was set to true
- **False Count**: Number of samples where flag was set to false
- **Total**: Total samples evaluated for this flag
- **True Proportion**: Percentage of samples where flag was true

---

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
| decomposition_quality | 3.50 | 0.50 | 3.0 | 4.0 | 5 |

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
| clarity | 4.67 | 1.10 ⚠️ UNSTABLE | 3.5 | 5.0 | 3 |
| decomposition_quality | 4.20 | 0.40 | 3.8 | 4.6 | 3 |

**⚠️ UNSTABLE**: clarity has high standard deviation (1.10 > threshold of 1.0), indicating inconsistent outputs across samples.

#### Per-Flag Statistics

| Flag | True Count | False Count | Total | True Proportion |
|------|------------|-------------|-------|-----------------|
| invented_constraints | 0 | 3 | 3 | 0.00 (0%) |
| omitted_constraints | 1 | 2 | 3 | 0.33 (33%) ⚠️ |

**⚠️ HIGH FLAG RATE**: omitted_constraints flag rate (33%) exceeds threshold of 20%.

---

### Test Case: test-003

**Input**: Summarize the concept of recursion with a simple example.  
**Status**: ❌ Failed  
**Samples**: 0 successful, 5 failed  
**Metadata**: difficulty=hard, topic=programming-concepts

#### Per-Metric Statistics

| Metric | Mean | Std Dev | Min | Max | Count |
|--------|------|---------|-----|-----|-------|
| semantic_fidelity | N/A | N/A | N/A | N/A | 0 |
| clarity | N/A | N/A | N/A | N/A | 0 |
| decomposition_quality | N/A | N/A | N/A | N/A | 0 |

**Note**: This test case has no valid metric statistics because all samples failed during generation or judging.

---

## Qualitative Examples

Representative samples selected to illustrate prompt performance. Samples are chosen based on judge scores across all metrics.

**Selection Criteria:**
- **Worst Performance**: Samples with lowest average metric scores
- **Ties**: If multiple samples have the same average score, sorted by sample_id for deterministic selection

### Worst Performance Examples

#### Example 1: test-001, sample-5

**Input**: Explain what Python is in simple terms.

**Generator Output**:
```
Python is a programming language.
```

**Judge Evaluation**:
- **semantic_fidelity**: 3.5/5.0 - "Technically accurate but lacks detail expected from 'simple terms' explanation. Too terse."
- **clarity**: 5.0/5.0 - "Extremely clear, though very minimal in content."
- **decomposition_quality**: 3.0/5.0 - "No decomposition of concepts provided."
- **Flags**: invented_constraints=false, omitted_constraints=false
- **Overall Comment**: "Response is correct but too brief to be helpful. Needs more elaboration on what Python is used for and why it matters."

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
- **semantic_fidelity**: 4.0/5.0 - "Correct implementation but uses iteration instead of recursion. Specification didn't explicitly require recursion, but it's commonly expected for factorial examples."
- **clarity**: 4.5/5.0 - "Clear and easy to understand code, though missing docstring."
- **decomposition_quality**: 4.0/5.0 - "Logical structure is good but could benefit from type hints and error handling."
- **Flags**: invented_constraints=false, omitted_constraints=true ⚠️
- **Overall Comment**: "Functional implementation but omits best practices like docstrings, type hints, and error handling for edge cases (e.g., negative numbers)."

---

#### Example 3: test-002, sample-2

**Input**: Write a factorial function in Python.

**Generator Output**:
```python
def factorial(n):
    """Calculate factorial of n using recursion."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

**Judge Evaluation**:
- **semantic_fidelity**: 5.0/5.0 - "Perfect recursive implementation meeting all expectations."
- **clarity**: 3.5/5.0 - "Code is clear but docstring could be more detailed. Missing type hints."
- **decomposition_quality**: 4.6/5.0 - "Good recursive decomposition with base case. Could improve with type hints."
- **Flags**: invented_constraints=false, omitted_constraints=false
- **Overall Comment**: "Strong recursive solution with docstring. Would benefit from type annotations for completeness."

---

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
- Maximum worst examples: 3
- Text truncation: 500 characters

**Report Format:**
- Output format: Markdown
- HTML conversion: Disabled

---

**Artifact Paths:**
- Run artifact: `runs/a1b2c3d4.../dataset_evaluation.json`
- Generated report: `runs/a1b2c3d4.../evaluation_report.md`

**Raw Artifacts:**
- [Run artifact (JSON)](run-sample.json)
- [Per-case artifacts directory](.)

---

_Report generated by prompt-evaluator v0.1.0 on 2025-12-22_
