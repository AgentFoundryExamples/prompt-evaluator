"""
Tests for rubric models and configuration loading.

Tests validate RubricMetric, RubricFlag, Rubric classes and load_rubric functionality.
"""

from pathlib import Path

import pytest

from prompt_evaluator.config import load_rubric
from prompt_evaluator.models import Rubric, RubricFlag, RubricMetric


class TestRubricMetric:
    """Tests for RubricMetric dataclass."""

    def test_rubric_metric_creation(self):
        """Test that RubricMetric can be created with valid fields."""
        metric = RubricMetric(
            name="semantic_fidelity",
            description="How well the output preserves meaning",
            min_score=1.0,
            max_score=5.0,
            guidelines="Score 1-5 based on semantic preservation",
        )

        assert metric.name == "semantic_fidelity"
        assert metric.description == "How well the output preserves meaning"
        assert metric.min_score == 1.0
        assert metric.max_score == 5.0
        assert metric.guidelines == "Score 1-5 based on semantic preservation"

    def test_rubric_metric_with_integer_scores(self):
        """Test that RubricMetric accepts integer scores."""
        metric = RubricMetric(
            name="clarity",
            description="Clarity of output",
            min_score=1,
            max_score=10,
            guidelines="Rate clarity 1-10",
        )

        assert metric.min_score == 1
        assert metric.max_score == 10

    def test_rubric_metric_empty_name(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Metric name cannot be empty"):
            RubricMetric(
                name="",
                description="Test",
                min_score=1,
                max_score=5,
                guidelines="Test",
            )

    def test_rubric_metric_whitespace_name(self):
        """Test that whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="Metric name cannot be empty"):
            RubricMetric(
                name="   ",
                description="Test",
                min_score=1,
                max_score=5,
                guidelines="Test",
            )

    def test_rubric_metric_empty_description(self):
        """Test that empty description raises ValueError."""
        with pytest.raises(ValueError, match="must have a non-empty description"):
            RubricMetric(
                name="test",
                description="",
                min_score=1,
                max_score=5,
                guidelines="Test",
            )

    def test_rubric_metric_empty_guidelines(self):
        """Test that empty guidelines raises ValueError."""
        with pytest.raises(ValueError, match="must have non-empty guidelines"):
            RubricMetric(
                name="test",
                description="Test description",
                min_score=1,
                max_score=5,
                guidelines="",
            )

    def test_rubric_metric_min_greater_than_max(self):
        """Test that min_score > max_score raises ValueError."""
        with pytest.raises(ValueError, match="min_score.*cannot be greater than max_score"):
            RubricMetric(
                name="test",
                description="Test",
                min_score=10,
                max_score=5,
                guidelines="Test",
            )

    def test_rubric_metric_non_numeric_min_score(self):
        """Test that non-numeric min_score raises ValueError."""
        with pytest.raises(ValueError, match="min_score must be numeric"):
            RubricMetric(
                name="test",
                description="Test",
                min_score="one",  # type: ignore[arg-type]
                max_score=5,
                guidelines="Test",
            )

    def test_rubric_metric_non_numeric_max_score(self):
        """Test that non-numeric max_score raises ValueError."""
        with pytest.raises(ValueError, match="max_score must be numeric"):
            RubricMetric(
                name="test",
                description="Test",
                min_score=1,
                max_score="five",  # type: ignore[arg-type]
                guidelines="Test",
            )

    def test_rubric_metric_equal_min_max(self):
        """Test that min_score == max_score is valid."""
        metric = RubricMetric(
            name="binary",
            description="Binary check",
            min_score=0,
            max_score=0,
            guidelines="Always 0",
        )
        assert metric.min_score == metric.max_score

    def test_rubric_metric_negative_scores(self):
        """Test that negative scores are allowed."""
        metric = RubricMetric(
            name="delta",
            description="Change metric",
            min_score=-10,
            max_score=10,
            guidelines="Rate from -10 to 10",
        )
        assert metric.min_score == -10
        assert metric.max_score == 10


class TestRubricFlag:
    """Tests for RubricFlag dataclass."""

    def test_rubric_flag_creation_default_false(self):
        """Test that RubricFlag can be created with default=False."""
        flag = RubricFlag(
            name="invented_constraints",
            description="Output adds constraints not in input",
        )

        assert flag.name == "invented_constraints"
        assert flag.description == "Output adds constraints not in input"
        assert flag.default is False

    def test_rubric_flag_creation_default_true(self):
        """Test that RubricFlag can be created with default=True."""
        flag = RubricFlag(
            name="requires_review",
            description="Output needs human review",
            default=True,
        )

        assert flag.name == "requires_review"
        assert flag.default is True

    def test_rubric_flag_empty_name(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Flag name cannot be empty"):
            RubricFlag(name="", description="Test")

    def test_rubric_flag_empty_description(self):
        """Test that empty description raises ValueError."""
        with pytest.raises(ValueError, match="must have a non-empty description"):
            RubricFlag(name="test", description="")

    def test_rubric_flag_non_boolean_default(self):
        """Test that non-boolean default raises ValueError."""
        with pytest.raises(ValueError, match="default must be boolean"):
            RubricFlag(
                name="test",
                description="Test",
                default="false",  # type: ignore[arg-type]
            )


class TestRubric:
    """Tests for Rubric dataclass."""

    def test_rubric_creation_with_metrics_only(self):
        """Test that Rubric can be created with metrics only."""
        metric = RubricMetric(
            name="quality",
            description="Output quality",
            min_score=1,
            max_score=5,
            guidelines="Rate 1-5",
        )
        rubric = Rubric(metrics=[metric])

        assert len(rubric.metrics) == 1
        assert len(rubric.flags) == 0
        assert rubric.metrics[0].name == "quality"

    def test_rubric_creation_with_metrics_and_flags(self):
        """Test that Rubric can be created with both metrics and flags."""
        metric = RubricMetric(
            name="quality",
            description="Output quality",
            min_score=1,
            max_score=5,
            guidelines="Rate 1-5",
        )
        flag = RubricFlag(name="needs_review", description="Requires review")
        rubric = Rubric(metrics=[metric], flags=[flag])

        assert len(rubric.metrics) == 1
        assert len(rubric.flags) == 1

    def test_rubric_empty_metrics_list(self):
        """Test that empty metrics list raises ValueError."""
        with pytest.raises(ValueError, match="Rubric must contain at least one metric"):
            Rubric(metrics=[])

    def test_rubric_none_metrics(self):
        """Test that None metrics raises ValueError."""
        with pytest.raises(ValueError, match="Rubric must contain at least one metric"):
            Rubric(metrics=None)  # type: ignore[arg-type]

    def test_rubric_duplicate_metric_names(self):
        """Test that duplicate metric names raise ValueError."""
        metric1 = RubricMetric(
            name="quality",
            description="Quality 1",
            min_score=1,
            max_score=5,
            guidelines="Test",
        )
        metric2 = RubricMetric(
            name="quality",
            description="Quality 2",
            min_score=1,
            max_score=5,
            guidelines="Test",
        )

        with pytest.raises(ValueError, match="duplicate metric names"):
            Rubric(metrics=[metric1, metric2])

    def test_rubric_duplicate_metric_names_case_insensitive(self):
        """Test that duplicate metric names are case-insensitive."""
        metric1 = RubricMetric(
            name="Quality",
            description="Quality 1",
            min_score=1,
            max_score=5,
            guidelines="Test",
        )
        metric2 = RubricMetric(
            name="quality",
            description="Quality 2",
            min_score=1,
            max_score=5,
            guidelines="Test",
        )

        with pytest.raises(ValueError, match="duplicate metric names"):
            Rubric(metrics=[metric1, metric2])

    def test_rubric_duplicate_flag_names(self):
        """Test that duplicate flag names raise ValueError."""
        metric = RubricMetric(
            name="quality",
            description="Quality",
            min_score=1,
            max_score=5,
            guidelines="Test",
        )
        flag1 = RubricFlag(name="needs_review", description="Review 1")
        flag2 = RubricFlag(name="needs_review", description="Review 2")

        with pytest.raises(ValueError, match="duplicate flag names"):
            Rubric(metrics=[metric], flags=[flag1, flag2])

    def test_rubric_duplicate_flag_names_case_insensitive(self):
        """Test that duplicate flag names are case-insensitive."""
        metric = RubricMetric(
            name="quality",
            description="Quality",
            min_score=1,
            max_score=5,
            guidelines="Test",
        )
        flag1 = RubricFlag(name="NeedsReview", description="Review 1")
        flag2 = RubricFlag(name="needsreview", description="Review 2")

        with pytest.raises(ValueError, match="duplicate flag names"):
            Rubric(metrics=[metric], flags=[flag1, flag2])

    def test_rubric_name_overlap_between_metrics_and_flags(self):
        """Test that same name in metric and flag raises ValueError."""
        metric = RubricMetric(
            name="quality",
            description="Quality metric",
            min_score=1,
            max_score=5,
            guidelines="Test",
        )
        flag = RubricFlag(name="quality", description="Quality flag")

        with pytest.raises(ValueError, match="names used in both metrics and flags"):
            Rubric(metrics=[metric], flags=[flag])

    def test_rubric_multiple_metrics_unique_names(self):
        """Test that multiple unique metrics work correctly."""
        metric1 = RubricMetric(
            name="quality",
            description="Quality",
            min_score=1,
            max_score=5,
            guidelines="Test",
        )
        metric2 = RubricMetric(
            name="clarity",
            description="Clarity",
            min_score=1,
            max_score=5,
            guidelines="Test",
        )
        metric3 = RubricMetric(
            name="completeness",
            description="Completeness",
            min_score=1,
            max_score=5,
            guidelines="Test",
        )

        rubric = Rubric(metrics=[metric1, metric2, metric3])
        assert len(rubric.metrics) == 3

    def test_rubric_empty_flags_list_is_valid(self):
        """Test that empty flags list is valid."""
        metric = RubricMetric(
            name="quality",
            description="Quality",
            min_score=1,
            max_score=5,
            guidelines="Test",
        )
        rubric = Rubric(metrics=[metric], flags=[])
        assert len(rubric.flags) == 0


class TestLoadRubric:
    """Tests for load_rubric function."""

    def test_load_rubric_yaml_valid(self, tmp_path):
        """Test loading a valid YAML rubric file."""
        rubric_file = tmp_path / "test.yaml"
        rubric_file.write_text(
            """
metrics:
  - name: semantic_fidelity
    description: Semantic preservation
    min_score: 1
    max_score: 5
    guidelines: Rate 1-5 based on meaning preservation

  - name: clarity
    description: Output clarity
    min_score: 1.0
    max_score: 5.0
    guidelines: Rate clarity

flags:
  - name: needs_review
    description: Requires human review
    default: false
"""
        )

        rubric = load_rubric(rubric_file)

        assert len(rubric.metrics) == 2
        assert len(rubric.flags) == 1
        assert rubric.metrics[0].name == "semantic_fidelity"
        assert rubric.metrics[1].name == "clarity"
        assert rubric.flags[0].name == "needs_review"
        assert rubric.flags[0].default is False

    def test_load_rubric_json_valid(self, tmp_path):
        """Test loading a valid JSON rubric file."""
        rubric_file = tmp_path / "test.json"
        rubric_file.write_text(
            """
{
  "metrics": [
    {
      "name": "quality",
      "description": "Output quality",
      "min_score": 0,
      "max_score": 10,
      "guidelines": "Rate 0-10"
    }
  ],
  "flags": [
    {
      "name": "has_issues",
      "description": "Has quality issues",
      "default": true
    }
  ]
}
"""
        )

        rubric = load_rubric(rubric_file)

        assert len(rubric.metrics) == 1
        assert rubric.metrics[0].name == "quality"
        assert rubric.metrics[0].max_score == 10
        assert len(rubric.flags) == 1
        assert rubric.flags[0].default is True

    def test_load_rubric_file_not_found(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        non_existent = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Rubric file not found"):
            load_rubric(non_existent)

    def test_load_rubric_unsupported_format(self, tmp_path):
        """Test that unsupported file format raises ValueError."""
        rubric_file = tmp_path / "test.txt"
        rubric_file.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported rubric file format"):
            load_rubric(rubric_file)

    def test_load_rubric_invalid_yaml(self, tmp_path):
        """Test that malformed YAML raises ValueError."""
        rubric_file = tmp_path / "invalid.yaml"
        rubric_file.write_text("metrics: [invalid: yaml: content")

        with pytest.raises(ValueError, match="Failed to parse YAML"):
            load_rubric(rubric_file)

    def test_load_rubric_not_dict(self, tmp_path):
        """Test that non-dict root raises ValueError."""
        rubric_file = tmp_path / "list.yaml"
        rubric_file.write_text("- item1\n- item2")

        with pytest.raises(ValueError, match="must contain a dictionary"):
            load_rubric(rubric_file)

    def test_load_rubric_missing_metrics_key(self, tmp_path):
        """Test that missing metrics key raises ValueError."""
        rubric_file = tmp_path / "no_metrics.yaml"
        rubric_file.write_text("flags: []")

        with pytest.raises(ValueError, match="Rubric must contain at least one metric"):
            load_rubric(rubric_file)

    def test_load_rubric_empty_metrics_array(self, tmp_path):
        """Test that empty metrics array raises ValueError."""
        rubric_file = tmp_path / "empty_metrics.yaml"
        rubric_file.write_text("metrics: []")

        with pytest.raises(ValueError, match="Rubric must contain at least one metric"):
            load_rubric(rubric_file)

    def test_load_rubric_metrics_not_list(self, tmp_path):
        """Test that metrics as non-list raises ValueError."""
        rubric_file = tmp_path / "bad_metrics.yaml"
        rubric_file.write_text("metrics: 'not a list'")

        with pytest.raises(ValueError, match="Metrics must be a list"):
            load_rubric(rubric_file)

    def test_load_rubric_metric_missing_name(self, tmp_path):
        """Test that metric missing name raises ValueError."""
        rubric_file = tmp_path / "no_name.yaml"
        rubric_file.write_text(
            """
metrics:
  - description: Test
    min_score: 1
    max_score: 5
    guidelines: Test
"""
        )

        with pytest.raises(ValueError, match="missing required fields: name"):
            load_rubric(rubric_file)

    def test_load_rubric_metric_missing_description(self, tmp_path):
        """Test that metric missing description raises ValueError."""
        rubric_file = tmp_path / "no_desc.yaml"
        rubric_file.write_text(
            """
metrics:
  - name: test
    min_score: 1
    max_score: 5
    guidelines: Test
"""
        )

        with pytest.raises(ValueError, match="missing required fields: description"):
            load_rubric(rubric_file)

    def test_load_rubric_metric_missing_min_score(self, tmp_path):
        """Test that metric missing min_score raises ValueError."""
        rubric_file = tmp_path / "no_min.yaml"
        rubric_file.write_text(
            """
metrics:
  - name: test
    description: Test
    max_score: 5
    guidelines: Test
"""
        )

        with pytest.raises(ValueError, match="missing required fields: min_score"):
            load_rubric(rubric_file)

    def test_load_rubric_metric_missing_max_score(self, tmp_path):
        """Test that metric missing max_score raises ValueError."""
        rubric_file = tmp_path / "no_max.yaml"
        rubric_file.write_text(
            """
metrics:
  - name: test
    description: Test
    min_score: 1
    guidelines: Test
"""
        )

        with pytest.raises(ValueError, match="missing required fields: max_score"):
            load_rubric(rubric_file)

    def test_load_rubric_metric_missing_guidelines(self, tmp_path):
        """Test that metric missing guidelines raises ValueError."""
        rubric_file = tmp_path / "no_guidelines.yaml"
        rubric_file.write_text(
            """
metrics:
  - name: test
    description: Test
    min_score: 1
    max_score: 5
"""
        )

        with pytest.raises(ValueError, match="missing required fields: guidelines"):
            load_rubric(rubric_file)

    def test_load_rubric_metric_invalid_type(self, tmp_path):
        """Test that non-dict metric raises ValueError."""
        rubric_file = tmp_path / "bad_metric.yaml"
        rubric_file.write_text("metrics:\n  - 'not a dict'")

        with pytest.raises(ValueError, match="Metric at index 0 must be a dictionary"):
            load_rubric(rubric_file)

    def test_load_rubric_min_greater_than_max(self, tmp_path):
        """Test that min > max in loaded metric raises ValueError."""
        rubric_file = tmp_path / "bad_range.yaml"
        rubric_file.write_text(
            """
metrics:
  - name: test
    description: Test
    min_score: 10
    max_score: 5
    guidelines: Test
"""
        )

        with pytest.raises(ValueError, match="Invalid metric at index 0"):
            load_rubric(rubric_file)

    def test_load_rubric_duplicate_metric_names(self, tmp_path):
        """Test that duplicate metric names raise ValueError."""
        rubric_file = tmp_path / "dupe_metrics.yaml"
        rubric_file.write_text(
            """
metrics:
  - name: quality
    description: Quality 1
    min_score: 1
    max_score: 5
    guidelines: Test
  - name: quality
    description: Quality 2
    min_score: 1
    max_score: 5
    guidelines: Test
"""
        )

        with pytest.raises(ValueError, match="duplicate metric names"):
            load_rubric(rubric_file)

    def test_load_rubric_flags_not_list(self, tmp_path):
        """Test that flags as non-list raises ValueError."""
        rubric_file = tmp_path / "bad_flags.yaml"
        rubric_file.write_text(
            """
metrics:
  - name: quality
    description: Quality
    min_score: 1
    max_score: 5
    guidelines: Test
flags: 'not a list'
"""
        )

        with pytest.raises(ValueError, match="Flags must be a list"):
            load_rubric(rubric_file)

    def test_load_rubric_flag_missing_name(self, tmp_path):
        """Test that flag missing name raises ValueError."""
        rubric_file = tmp_path / "flag_no_name.yaml"
        rubric_file.write_text(
            """
metrics:
  - name: quality
    description: Quality
    min_score: 1
    max_score: 5
    guidelines: Test
flags:
  - description: Test flag
    default: false
"""
        )

        with pytest.raises(ValueError, match="Flag at index 0 is missing required field: name"):
            load_rubric(rubric_file)

    def test_load_rubric_flag_missing_description(self, tmp_path):
        """Test that flag missing description raises ValueError."""
        rubric_file = tmp_path / "flag_no_desc.yaml"
        rubric_file.write_text(
            """
metrics:
  - name: quality
    description: Quality
    min_score: 1
    max_score: 5
    guidelines: Test
flags:
  - name: test_flag
    default: false
"""
        )

        with pytest.raises(
            ValueError, match="Flag at index 0 is missing required field: description"
        ):
            load_rubric(rubric_file)

    def test_load_rubric_flag_default_optional(self, tmp_path):
        """Test that flag default is optional and defaults to False."""
        rubric_file = tmp_path / "flag_no_default.yaml"
        rubric_file.write_text(
            """
metrics:
  - name: quality
    description: Quality
    min_score: 1
    max_score: 5
    guidelines: Test
flags:
  - name: test_flag
    description: Test flag
"""
        )

        rubric = load_rubric(rubric_file)
        assert rubric.flags[0].default is False

    def test_load_rubric_flag_invalid_type(self, tmp_path):
        """Test that non-dict flag raises ValueError."""
        rubric_file = tmp_path / "bad_flag.yaml"
        rubric_file.write_text(
            """
metrics:
  - name: quality
    description: Quality
    min_score: 1
    max_score: 5
    guidelines: Test
flags:
  - 'not a dict'
"""
        )

        with pytest.raises(ValueError, match="Flag at index 0 must be a dictionary"):
            load_rubric(rubric_file)

    def test_load_rubric_duplicate_flag_names(self, tmp_path):
        """Test that duplicate flag names raise ValueError."""
        rubric_file = tmp_path / "dupe_flags.yaml"
        rubric_file.write_text(
            """
metrics:
  - name: quality
    description: Quality
    min_score: 1
    max_score: 5
    guidelines: Test
flags:
  - name: needs_review
    description: Review 1
  - name: needs_review
    description: Review 2
"""
        )

        with pytest.raises(ValueError, match="duplicate flag names"):
            load_rubric(rubric_file)

    def test_load_rubric_without_flags_section(self, tmp_path):
        """Test that rubric without flags section works."""
        rubric_file = tmp_path / "no_flags.yaml"
        rubric_file.write_text(
            """
metrics:
  - name: quality
    description: Quality
    min_score: 1
    max_score: 5
    guidelines: Test
"""
        )

        rubric = load_rubric(rubric_file)
        assert len(rubric.metrics) == 1
        assert len(rubric.flags) == 0

    def test_load_rubric_default_rubric_file(self):
        """Test loading the packaged default rubric file."""
        # Use relative path from project root
        default_rubric_path = Path("examples/rubrics/default.yaml")

        if not default_rubric_path.exists():
            pytest.skip("Default rubric file not found")

        rubric = load_rubric(default_rubric_path)

        # Verify expected metrics
        assert len(rubric.metrics) == 3
        metric_names = [m.name for m in rubric.metrics]
        assert "semantic_fidelity" in metric_names
        assert "decomposition_quality" in metric_names
        assert "constraint_adherence" in metric_names

        # Verify expected flags
        assert len(rubric.flags) == 2
        flag_names = [f.name for f in rubric.flags]
        assert "invented_constraints" in flag_names
        assert "omitted_constraints" in flag_names

    def test_load_rubric_code_review_preset(self):
        """Test loading the code review preset rubric."""
        code_review_path = Path("examples/rubrics/code_review.json")

        if not code_review_path.exists():
            pytest.skip("Code review rubric file not found")

        rubric = load_rubric(code_review_path)

        assert len(rubric.metrics) >= 1
        assert len(rubric.flags) >= 0

    def test_load_rubric_content_quality_preset(self):
        """Test loading the content quality preset rubric."""
        content_quality_path = Path("examples/rubrics/content_quality.yaml")

        if not content_quality_path.exists():
            pytest.skip("Content quality rubric file not found")

        rubric = load_rubric(content_quality_path)

        assert len(rubric.metrics) >= 1
        assert len(rubric.flags) >= 0
