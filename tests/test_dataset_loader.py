# Copyright 2025 John Brosnihan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for TestCase model and dataset loading functionality.
"""

from pathlib import Path

import pytest

from prompt_evaluator.config import load_dataset
from prompt_evaluator.models import TestCase


class TestTestCaseModel:
    """Tests for TestCase data model."""

    def test_testcase_with_required_fields_only(self):
        """Test that TestCase can be created with only required fields."""
        test_case = TestCase(id="test-001", input="What is Python?")

        assert test_case.id == "test-001"
        assert test_case.input == "What is Python?"
        assert test_case.description is None
        assert test_case.task is None
        assert test_case.expected_constraints is None
        assert test_case.reference is None
        assert test_case.metadata == {}

    def test_testcase_with_all_fields(self):
        """Test that TestCase can be created with all explicit fields."""
        test_case = TestCase(
            id="test-001",
            input="What is Python?",
            description="Test case for Python explanation",
            task="Explain programming language",
            expected_constraints="Keep it simple",
            reference="Python is a programming language.",
        )

        assert test_case.id == "test-001"
        assert test_case.input == "What is Python?"
        assert test_case.description == "Test case for Python explanation"
        assert test_case.task == "Explain programming language"
        assert test_case.expected_constraints == "Keep it simple"
        assert test_case.reference == "Python is a programming language."
        assert test_case.metadata == {}

    def test_testcase_empty_id_raises_error(self):
        """Test that empty ID raises validation error."""
        with pytest.raises(ValueError, match="at least 1 character"):
            TestCase(id="", input="What is Python?")

    def test_testcase_empty_input_raises_error(self):
        """Test that empty input raises validation error."""
        with pytest.raises(ValueError, match="at least 1 character"):
            TestCase(id="test-001", input="")

    def test_testcase_whitespace_only_id_raises_error(self):
        """Test that whitespace-only ID raises validation error."""
        with pytest.raises(ValueError):
            TestCase(id="   ", input="What is Python?")

    def test_testcase_whitespace_only_input_raises_error(self):
        """Test that whitespace-only input raises validation error."""
        with pytest.raises(ValueError):
            TestCase(id="test-001", input="   ")

    def test_testcase_preserves_extra_fields_in_metadata(self):
        """Test that extra fields are preserved in metadata dict."""
        test_case = TestCase(
            id="test-001",
            input="What is Python?",
            difficulty="easy",
            topic="programming",
            priority=1,
        )

        assert test_case.id == "test-001"
        assert test_case.input == "What is Python?"
        assert "difficulty" in test_case.metadata
        assert test_case.metadata["difficulty"] == "easy"
        assert "topic" in test_case.metadata
        assert test_case.metadata["topic"] == "programming"
        assert "priority" in test_case.metadata
        assert test_case.metadata["priority"] == 1

    def test_testcase_both_description_and_task(self):
        """Test that description and task can coexist without conflict."""
        test_case = TestCase(
            id="test-001",
            input="What is Python?",
            description="Test case for Python explanation",
            task="Explain programming language",
        )

        assert test_case.description == "Test case for Python explanation"
        assert test_case.task == "Explain programming language"

    def test_testcase_metadata_can_be_dict(self):
        """Test that metadata can be provided as a dict."""
        test_case = TestCase(
            id="test-001",
            input="What is Python?",
            metadata={"custom": "value", "nested": {"key": "val"}},
        )

        assert test_case.metadata["custom"] == "value"
        assert test_case.metadata["nested"]["key"] == "val"

    def test_testcase_to_dict(self):
        """Test that TestCase can be serialized to dict."""
        test_case = TestCase(
            id="test-001",
            input="What is Python?",
            description="Test case",
            difficulty="easy",
        )

        test_dict = test_case.model_dump()

        assert test_dict["id"] == "test-001"
        assert test_dict["input"] == "What is Python?"
        assert test_dict["description"] == "Test case"
        assert "difficulty" in test_dict["metadata"]
        assert test_dict["metadata"]["difficulty"] == "easy"


class TestLoadDatasetJSONL:
    """Tests for loading JSONL datasets."""

    def test_load_jsonl_with_minimal_fields(self, tmp_path):
        """Test loading JSONL with only required fields."""
        dataset_path = tmp_path / "test.jsonl"
        dataset_path.write_text(
            '{"id": "test-001", "input": "What is Python?"}\n'
            '{"id": "test-002", "input": "What is Java?"}\n'
        )

        test_cases, metadata = load_dataset(dataset_path)

        assert len(test_cases) == 2
        assert test_cases[0].id == "test-001"
        assert test_cases[0].input == "What is Python?"
        assert test_cases[1].id == "test-002"
        assert test_cases[1].input == "What is Java?"

        # Check metadata
        assert metadata["count"] == 2
        assert metadata["format"] == ".jsonl"
        assert "path" in metadata
        assert "hash" in metadata

    def test_load_jsonl_with_all_fields(self, tmp_path):
        """Test loading JSONL with all explicit and custom fields."""
        dataset_path = tmp_path / "test.jsonl"
        dataset_path.write_text(
            '{"id": "test-001", "input": "What is Python?", "description": "Test", '
            '"task": "Explain", "expected_constraints": "Simple", "reference": "A language", '
            '"difficulty": "easy", "topic": "programming"}\n'
        )

        test_cases, metadata = load_dataset(dataset_path)

        assert len(test_cases) == 1
        test_case = test_cases[0]
        assert test_case.id == "test-001"
        assert test_case.description == "Test"
        assert test_case.task == "Explain"
        assert test_case.expected_constraints == "Simple"
        assert test_case.reference == "A language"
        assert test_case.metadata["difficulty"] == "easy"
        assert test_case.metadata["topic"] == "programming"

    def test_load_jsonl_ignores_empty_lines(self, tmp_path):
        """Test that empty lines in JSONL are ignored."""
        dataset_path = tmp_path / "test.jsonl"
        dataset_path.write_text(
            '{"id": "test-001", "input": "What is Python?"}\n'
            "\n"
            '{"id": "test-002", "input": "What is Java?"}\n'
            "\n"
        )

        test_cases, metadata = load_dataset(dataset_path)

        assert len(test_cases) == 2
        assert metadata["count"] == 2

    def test_load_jsonl_preserves_order(self, tmp_path):
        """Test that JSONL preserves test case order."""
        dataset_path = tmp_path / "test.jsonl"
        dataset_path.write_text(
            '{"id": "test-003", "input": "Third"}\n'
            '{"id": "test-001", "input": "First"}\n'
            '{"id": "test-002", "input": "Second"}\n'
        )

        test_cases, metadata = load_dataset(dataset_path)

        assert test_cases[0].id == "test-003"
        assert test_cases[1].id == "test-001"
        assert test_cases[2].id == "test-002"

    def test_load_jsonl_duplicate_id_raises_error(self, tmp_path):
        """Test that duplicate IDs raise error with line context."""
        dataset_path = tmp_path / "test.jsonl"
        dataset_path.write_text(
            '{"id": "test-001", "input": "First"}\n'
            '{"id": "test-001", "input": "Duplicate"}\n'
        )

        with pytest.raises(ValueError, match="Duplicate test case ID 'test-001' found at line 2"):
            load_dataset(dataset_path)

    def test_load_jsonl_missing_id_raises_error(self, tmp_path):
        """Test that missing ID raises error with line context."""
        dataset_path = tmp_path / "test.jsonl"
        dataset_path.write_text('{"input": "Missing ID"}\n')

        with pytest.raises(ValueError, match="Record at line 1 is missing required field: id"):
            load_dataset(dataset_path)

    def test_load_jsonl_missing_input_raises_error(self, tmp_path):
        """Test that missing input raises error with line context."""
        dataset_path = tmp_path / "test.jsonl"
        dataset_path.write_text('{"id": "test-001"}\n')

        with pytest.raises(ValueError, match="Record at line 1 is missing required field: input"):
            load_dataset(dataset_path)

    def test_load_jsonl_invalid_json_raises_error(self, tmp_path):
        """Test that invalid JSON raises error with line context."""
        dataset_path = tmp_path / "test.jsonl"
        dataset_path.write_text('{"id": "test-001", invalid json}\n')

        with pytest.raises(ValueError, match="Invalid JSON at line 1"):
            load_dataset(dataset_path)

    def test_load_jsonl_non_object_raises_error(self, tmp_path):
        """Test that non-object JSON raises error."""
        dataset_path = tmp_path / "test.jsonl"
        dataset_path.write_text('"just a string"\n')

        with pytest.raises(ValueError, match="Record at line 1 must be a JSON object"):
            load_dataset(dataset_path)

    def test_load_jsonl_empty_id_raises_error(self, tmp_path):
        """Test that empty ID string raises validation error."""
        dataset_path = tmp_path / "test.jsonl"
        dataset_path.write_text('{"id": "", "input": "What is Python?"}\n')

        with pytest.raises(ValueError, match="Invalid test case at line 1"):
            load_dataset(dataset_path)

    def test_load_jsonl_empty_input_raises_error(self, tmp_path):
        """Test that empty input string raises validation error."""
        dataset_path = tmp_path / "test.jsonl"
        dataset_path.write_text('{"id": "test-001", "input": ""}\n')

        with pytest.raises(ValueError, match="Invalid test case at line 1"):
            load_dataset(dataset_path)


class TestLoadDatasetYAML:
    """Tests for loading YAML datasets."""

    def test_load_yaml_with_minimal_fields(self, tmp_path):
        """Test loading YAML with only required fields."""
        dataset_path = tmp_path / "test.yaml"
        dataset_path.write_text(
            """
- id: test-001
  input: What is Python?
- id: test-002
  input: What is Java?
"""
        )

        test_cases, metadata = load_dataset(dataset_path)

        assert len(test_cases) == 2
        assert test_cases[0].id == "test-001"
        assert test_cases[0].input == "What is Python?"
        assert test_cases[1].id == "test-002"
        assert test_cases[1].input == "What is Java?"

        # Check metadata
        assert metadata["count"] == 2
        assert metadata["format"] == ".yaml"
        assert "path" in metadata
        assert "hash" in metadata

    def test_load_yaml_with_all_fields(self, tmp_path):
        """Test loading YAML with all explicit and custom fields."""
        dataset_path = tmp_path / "test.yaml"
        dataset_path.write_text(
            """
- id: test-001
  input: What is Python?
  description: Test case
  task: Explain programming language
  expected_constraints: Keep it simple
  reference: Python is a language
  difficulty: easy
  topic: programming
"""
        )

        test_cases, metadata = load_dataset(dataset_path)

        assert len(test_cases) == 1
        test_case = test_cases[0]
        assert test_case.id == "test-001"
        assert test_case.description == "Test case"
        assert test_case.task == "Explain programming language"
        assert test_case.expected_constraints == "Keep it simple"
        assert test_case.reference == "Python is a language"
        assert test_case.metadata["difficulty"] == "easy"
        assert test_case.metadata["topic"] == "programming"

    def test_load_yaml_with_yml_extension(self, tmp_path):
        """Test that .yml extension is supported."""
        dataset_path = tmp_path / "test.yml"
        dataset_path.write_text(
            """
- id: test-001
  input: What is Python?
"""
        )

        test_cases, metadata = load_dataset(dataset_path)

        assert len(test_cases) == 1
        assert metadata["format"] == ".yml"

    def test_load_yaml_preserves_order(self, tmp_path):
        """Test that YAML preserves test case order."""
        dataset_path = tmp_path / "test.yaml"
        dataset_path.write_text(
            """
- id: test-003
  input: Third
- id: test-001
  input: First
- id: test-002
  input: Second
"""
        )

        test_cases, metadata = load_dataset(dataset_path)

        assert test_cases[0].id == "test-003"
        assert test_cases[1].id == "test-001"
        assert test_cases[2].id == "test-002"

    def test_load_yaml_with_comments(self, tmp_path):
        """Test that YAML comments are ignored."""
        dataset_path = tmp_path / "test.yaml"
        dataset_path.write_text(
            """
# Test dataset
- id: test-001
  input: What is Python?
  # This is a comment

# Another comment
- id: test-002
  input: What is Java?
"""
        )

        test_cases, metadata = load_dataset(dataset_path)

        assert len(test_cases) == 2
        assert test_cases[0].id == "test-001"
        assert test_cases[1].id == "test-002"

    def test_load_yaml_duplicate_id_raises_error(self, tmp_path):
        """Test that duplicate IDs raise error with index context."""
        dataset_path = tmp_path / "test.yaml"
        dataset_path.write_text(
            """
- id: test-001
  input: First
- id: test-001
  input: Duplicate
"""
        )

        with pytest.raises(ValueError, match="Duplicate test case ID 'test-001' found at index 1"):
            load_dataset(dataset_path)

    def test_load_yaml_missing_id_raises_error(self, tmp_path):
        """Test that missing ID raises error with index context."""
        dataset_path = tmp_path / "test.yaml"
        dataset_path.write_text(
            """
- input: Missing ID
"""
        )

        with pytest.raises(ValueError, match="Record at index 0 is missing required field: id"):
            load_dataset(dataset_path)

    def test_load_yaml_missing_input_raises_error(self, tmp_path):
        """Test that missing input raises error with index context."""
        dataset_path = tmp_path / "test.yaml"
        dataset_path.write_text(
            """
- id: test-001
"""
        )

        with pytest.raises(
            ValueError, match="Record at index 0 is missing required field: input"
        ):
            load_dataset(dataset_path)

    def test_load_yaml_not_a_list_raises_error(self, tmp_path):
        """Test that non-list YAML raises error."""
        dataset_path = tmp_path / "test.yaml"
        dataset_path.write_text(
            """
id: test-001
input: Not a list
"""
        )

        with pytest.raises(ValueError, match="YAML dataset must contain a list of objects"):
            load_dataset(dataset_path)

    def test_load_yaml_non_object_entry_raises_error(self, tmp_path):
        """Test that non-object entries raise error."""
        dataset_path = tmp_path / "test.yaml"
        dataset_path.write_text(
            """
- just a string
- id: test-001
  input: Valid
"""
        )

        with pytest.raises(ValueError, match="Record at index 0 must be an object"):
            load_dataset(dataset_path)

    def test_load_yaml_empty_file(self, tmp_path):
        """Test that empty YAML file returns empty list."""
        dataset_path = tmp_path / "test.yaml"
        dataset_path.write_text("")

        test_cases, metadata = load_dataset(dataset_path)

        assert len(test_cases) == 0
        assert metadata["count"] == 0

    def test_load_yaml_empty_id_raises_error(self, tmp_path):
        """Test that empty ID string raises validation error."""
        dataset_path = tmp_path / "test.yaml"
        dataset_path.write_text(
            """
- id: ""
  input: What is Python?
"""
        )

        with pytest.raises(ValueError, match="Invalid test case at index 0"):
            load_dataset(dataset_path)

    def test_load_yaml_empty_input_raises_error(self, tmp_path):
        """Test that empty input string raises validation error."""
        dataset_path = tmp_path / "test.yaml"
        dataset_path.write_text(
            """
- id: test-001
  input: ""
"""
        )

        with pytest.raises(ValueError, match="Invalid test case at index 0"):
            load_dataset(dataset_path)


class TestLoadDatasetGeneral:
    """Tests for general dataset loading behavior."""

    def test_load_dataset_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_dataset(Path("/nonexistent/file.jsonl"))

    def test_load_dataset_unsupported_extension_raises_error(self, tmp_path):
        """Test that unsupported file extension raises error."""
        dataset_path = tmp_path / "test.csv"
        dataset_path.write_text("id,input\ntest-001,What is Python?\n")

        with pytest.raises(
            ValueError, match="Unsupported dataset file format: .csv.*Supported formats"
        ):
            load_dataset(dataset_path)

    def test_load_dataset_metadata_includes_hash(self, tmp_path):
        """Test that metadata includes SHA-256 hash of file content."""
        dataset_path = tmp_path / "test.jsonl"
        dataset_path.write_text('{"id": "test-001", "input": "What is Python?"}\n')

        test_cases, metadata = load_dataset(dataset_path)

        assert "hash" in metadata
        assert len(metadata["hash"]) == 64  # SHA-256 hash is 64 hex chars
        assert isinstance(metadata["hash"], str)

    def test_load_dataset_metadata_hash_changes_with_content(self, tmp_path):
        """Test that hash changes when file content changes."""
        dataset_path = tmp_path / "test.jsonl"

        # First version
        dataset_path.write_text('{"id": "test-001", "input": "What is Python?"}\n')
        _, metadata1 = load_dataset(dataset_path)
        hash1 = metadata1["hash"]

        # Second version with different content
        dataset_path.write_text('{"id": "test-001", "input": "What is Java?"}\n')
        _, metadata2 = load_dataset(dataset_path)
        hash2 = metadata2["hash"]

        assert hash1 != hash2

    def test_load_dataset_metadata_includes_absolute_path(self, tmp_path):
        """Test that metadata includes absolute path."""
        dataset_path = tmp_path / "test.jsonl"
        dataset_path.write_text('{"id": "test-001", "input": "What is Python?"}\n')

        test_cases, metadata = load_dataset(dataset_path)

        assert "path" in metadata
        assert Path(metadata["path"]).is_absolute()
        assert Path(metadata["path"]) == dataset_path.absolute()

    def test_load_dataset_large_dataset(self, tmp_path):
        """Test loading a large dataset (edge case for streaming)."""
        dataset_path = tmp_path / "test.jsonl"

        # Create a dataset with 250 test cases
        with open(dataset_path, "w") as f:
            for i in range(250):
                f.write(f'{{"id": "test-{i:03d}", "input": "Test case {i}"}}\n')

        test_cases, metadata = load_dataset(dataset_path)

        assert len(test_cases) == 250
        assert metadata["count"] == 250
        assert test_cases[0].id == "test-000"
        assert test_cases[249].id == "test-249"

    def test_load_dataset_complex_metadata_fields(self, tmp_path):
        """Test that complex metadata fields are preserved."""
        dataset_path = tmp_path / "test.jsonl"
        dataset_path.write_text(
            '{"id": "test-001", "input": "Test", "priority": 1, "tags": ["a", "b"], '
            '"config": {"strict": true, "timeout": 30}}\n'
        )

        test_cases, metadata = load_dataset(dataset_path)

        test_case = test_cases[0]
        assert test_case.metadata["priority"] == 1
        assert test_case.metadata["tags"] == ["a", "b"]
        assert test_case.metadata["config"]["strict"] is True
        assert test_case.metadata["config"]["timeout"] == 30

    def test_load_dataset_jsonl_and_yaml_equivalent(self, tmp_path):
        """Test that JSONL and YAML produce equivalent results."""
        # JSONL version
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text(
            '{"id": "test-001", "input": "What is Python?", "difficulty": "easy"}\n'
        )

        # YAML version
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(
            """
- id: test-001
  input: What is Python?
  difficulty: easy
"""
        )

        jsonl_cases, jsonl_meta = load_dataset(jsonl_path)
        yaml_cases, yaml_meta = load_dataset(yaml_path)

        # Compare test cases
        assert len(jsonl_cases) == len(yaml_cases)
        assert jsonl_cases[0].id == yaml_cases[0].id
        assert jsonl_cases[0].input == yaml_cases[0].input
        assert jsonl_cases[0].metadata["difficulty"] == yaml_cases[0].metadata["difficulty"]

        # Metadata should differ only in path and hash
        assert jsonl_meta["count"] == yaml_meta["count"]
        assert jsonl_meta["format"] != yaml_meta["format"]
