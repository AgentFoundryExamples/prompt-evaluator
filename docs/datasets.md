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

## Future CLI Integration

Datasets will be used by evaluation commands to run batch evaluations:

```bash
# Future usage (not yet implemented)
prompt-evaluator evaluate-dataset \
  --dataset examples/datasets/sample.yaml \
  --system-prompt prompts/system.txt \
  --num-samples 5
```

This will:
1. Load all test cases from the dataset
2. Generate and evaluate outputs for each test case
3. Aggregate results across all test cases
4. Produce a comprehensive evaluation report

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
