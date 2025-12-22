"""
Report generator for run comparisons.

This module generates Markdown (and optionally HTML) reports from comparison
artifacts, highlighting metric/flag deltas, regressions, and improvements.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_evaluator.reporting.run_report import (
    convert_markdown_to_html,
    escape_html_for_markdown,
)

logger = logging.getLogger(__name__)


@dataclass
class CompareReportConfig:
    """Configuration for comparison report generation."""

    top_cases_per_metric: int = 5  # Number of top regressed/improved cases to show per metric
    generate_html: bool = False  # Whether to generate HTML alongside Markdown


def load_comparison_artifact(artifact_path: Path) -> dict[str, Any]:
    """
    Load and validate a comparison artifact JSON file.

    Args:
        artifact_path: Path to the comparison artifact JSON file

    Returns:
        Parsed JSON dictionary

    Raises:
        FileNotFoundError: If artifact file doesn't exist
        ValueError: If artifact file is not valid JSON or missing required fields
    """
    if not artifact_path.exists():
        raise FileNotFoundError(f"Comparison artifact not found: {artifact_path}")

    try:
        with open(artifact_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in artifact {artifact_path}: {e}") from e

    # Validate required fields
    required_fields = ["baseline_run_id", "candidate_run_id", "metric_deltas", "flag_deltas"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(
            f"Comparison artifact {artifact_path} missing required fields: "
            f"{', '.join(missing_fields)}"
        )

    return data


def format_delta_sign(value: float | None) -> str:
    """
    Format a numeric delta with appropriate sign prefix.

    Args:
        value: Delta value to format

    Returns:
        Formatted string with sign (e.g., "+0.5", "-0.3", "N/A")
    """
    if value is None:
        return "N/A"
    if value > 0:
        return f"+{value:.3f}"
    return f"{value:.3f}"


def format_percentage(value: float | None) -> str:
    """
    Format a percentage value with sign and % suffix.

    Args:
        value: Percentage value to format

    Returns:
        Formatted string (e.g., "+12.5%", "-5.0%", "N/A")
    """
    if value is None:
        return "N/A"
    if abs(value) == float("inf"):
        return "∞" if value > 0 else "-∞"
    if value > 0:
        return f"+{value:.2f}%"
    return f"{value:.2f}%"


def render_comparison_metadata_section(comparison_data: dict[str, Any]) -> str:
    """
    Render the metadata section of the comparison report.

    Args:
        comparison_data: Comparison artifact dictionary

    Returns:
        Markdown-formatted metadata section
    """
    lines = [
        "# Run Comparison Report\n",
        "## Comparison Metadata\n",
        f"- **Baseline Run ID**: `{comparison_data.get('baseline_run_id', 'N/A')}`",
        f"- **Candidate Run ID**: `{comparison_data.get('candidate_run_id', 'N/A')}`",
    ]

    # Add prompt version info if available
    baseline_version = comparison_data.get("baseline_prompt_version")
    candidate_version = comparison_data.get("candidate_prompt_version")
    if baseline_version or candidate_version:
        lines.append(
            f"- **Baseline Prompt**: `{baseline_version if baseline_version else 'N/A'}`"
        )
        lines.append(
            f"- **Candidate Prompt**: `{candidate_version if candidate_version else 'N/A'}`"
        )

    # Add timestamp
    timestamp = comparison_data.get("comparison_timestamp", "N/A")
    lines.append(f"- **Comparison Time**: {timestamp}")

    # Add threshold configuration
    thresholds = comparison_data.get("thresholds_config", {})
    metric_threshold = thresholds.get("metric_threshold", "N/A")
    flag_threshold = thresholds.get("flag_threshold", "N/A")
    lines.append(f"- **Metric Regression Threshold**: {metric_threshold}")
    lines.append(f"- **Flag Regression Threshold**: {flag_threshold}")

    # Add regression summary
    has_regressions = comparison_data.get("has_regressions", False)
    regression_count = comparison_data.get("regression_count", 0)
    if has_regressions:
        lines.append(f"\n⚠️ **{regression_count} regression(s) detected**\n")
    else:
        lines.append("\n✓ **No regressions detected**\n")

    return "\n".join(lines)


def render_suite_comparison_table(comparison_data: dict[str, Any]) -> str:
    """
    Render suite-level metrics comparison table.

    Args:
        comparison_data: Comparison artifact dictionary

    Returns:
        Markdown-formatted comparison table
    """
    lines = [
        "## Suite-Level Metrics Comparison\n",
        "| Metric | Baseline | Candidate | Delta | % Change | Status |",
        "|--------|----------|-----------|-------|----------|--------|",
    ]

    metric_deltas = comparison_data.get("metric_deltas", [])
    if not metric_deltas:
        lines.append("| *No metrics available* | - | - | - | - | - |")
    else:
        for delta in metric_deltas:
            metric_name = escape_html_for_markdown(delta.get("metric_name", "unknown"))
            baseline = delta.get("baseline_mean")
            candidate = delta.get("candidate_mean")
            delta_val = delta.get("delta")
            percent = delta.get("percent_change")
            is_regression = delta.get("is_regression", False)

            baseline_str = f"{baseline:.2f}" if baseline is not None else "N/A"
            candidate_str = f"{candidate:.2f}" if candidate is not None else "N/A"
            delta_str = format_delta_sign(delta_val)
            percent_str = format_percentage(percent)

            if is_regression:
                status = "🔴 Regressed"
            elif delta_val is not None and delta_val > 0:
                status = "✓ Improved"
            elif delta_val is not None and delta_val == 0:
                status = "→ Unchanged"
            elif delta_val is not None and delta_val < 0:
                status = "↓ Decreased"
            else:
                status = "- N/A"

            lines.append(
                f"| {metric_name} | {baseline_str} | {candidate_str} | "
                f"{delta_str} | {percent_str} | {status} |"
            )

    return "\n".join(lines) + "\n"


def render_suite_flags_comparison_table(comparison_data: dict[str, Any]) -> str:
    """
    Render suite-level flags comparison table.

    Args:
        comparison_data: Comparison artifact dictionary

    Returns:
        Markdown-formatted comparison table
    """
    lines = [
        "## Suite-Level Flags Comparison\n",
        "| Flag | Baseline | Candidate | Delta | % Change | Status |",
        "|------|----------|-----------|-------|----------|--------|",
    ]

    flag_deltas = comparison_data.get("flag_deltas", [])
    if not flag_deltas:
        lines.append("| *No flags available* | - | - | - | - | - |")
    else:
        for delta in flag_deltas:
            flag_name = escape_html_for_markdown(delta.get("flag_name", "unknown"))
            baseline = delta.get("baseline_proportion")
            candidate = delta.get("candidate_proportion")
            delta_val = delta.get("delta")
            percent = delta.get("percent_change")
            is_regression = delta.get("is_regression", False)

            # Format proportions as percentages
            baseline_str = f"{baseline:.1%}" if baseline is not None else "N/A"
            candidate_str = f"{candidate:.1%}" if candidate is not None else "N/A"

            # Format delta as percentage points
            if delta_val is not None:
                delta_str = format_delta_sign(delta_val * 100) + " pp"
            else:
                delta_str = "N/A"

            percent_str = format_percentage(percent)

            if is_regression:
                status = "🔴 Regressed"
            elif delta_val is not None and delta_val < 0:
                status = "✓ Improved"
            elif delta_val is not None and delta_val == 0:
                status = "→ Unchanged"
            elif delta_val is not None and delta_val > 0:
                status = "↑ Increased"
            else:
                status = "- N/A"

            lines.append(
                f"| {flag_name} | {baseline_str} | {candidate_str} | "
                f"{delta_str} | {percent_str} | {status} |"
            )

    return "\n".join(lines) + "\n"


def render_regressions_section(comparison_data: dict[str, Any]) -> str:
    """
    Render section highlighting regressions.

    Args:
        comparison_data: Comparison artifact dictionary

    Returns:
        Markdown-formatted regressions section
    """
    lines = ["## Regressions Detected\n"]

    # Find regressed metrics
    regressed_metrics = [
        d for d in comparison_data.get("metric_deltas", []) if d.get("is_regression", False)
    ]

    # Find regressed flags
    regressed_flags = [
        d for d in comparison_data.get("flag_deltas", []) if d.get("is_regression", False)
    ]

    if not regressed_metrics and not regressed_flags:
        lines.append("*No regressions detected. All metrics and flags are stable or improved.*\n")
        return "\n".join(lines)

    if regressed_metrics:
        lines.append("### Regressed Metrics\n")
        # Sort by severity (absolute delta, descending)
        regressed_metrics_sorted = sorted(
            regressed_metrics,
            key=lambda d: (abs(d.get("delta", 0)), d.get("metric_name", "")),
            reverse=True,
        )
        for delta in regressed_metrics_sorted:
            metric_name = escape_html_for_markdown(delta.get("metric_name", "unknown"))
            delta_val = delta.get("delta")
            percent = delta.get("percent_change")
            baseline = delta.get("baseline_mean")
            candidate = delta.get("candidate_mean")

            delta_str = format_delta_sign(delta_val)
            percent_str = format_percentage(percent)
            severity = "**High**" if abs(delta_val or 0) > 0.5 else "**Medium**"

            lines.append(
                f"- **{metric_name}**: {baseline:.2f} → {candidate:.2f} "
                f"({delta_str}, {percent_str}) - Severity: {severity}"
            )
        lines.append("")

    if regressed_flags:
        lines.append("### Regressed Flags\n")
        # Sort by severity (absolute delta, descending)
        regressed_flags_sorted = sorted(
            regressed_flags,
            key=lambda d: (abs(d.get("delta", 0)), d.get("flag_name", "")),
            reverse=True,
        )
        for delta in regressed_flags_sorted:
            flag_name = escape_html_for_markdown(delta.get("flag_name", "unknown"))
            delta_val = delta.get("delta")
            percent = delta.get("percent_change")
            baseline = delta.get("baseline_proportion", 0)
            candidate = delta.get("candidate_proportion", 0)

            if delta_val is not None:
                delta_str = format_delta_sign(delta_val * 100) + " pp"
            else:
                delta_str = "N/A"
            percent_str = format_percentage(percent)
            severity = "**High**" if abs(delta_val or 0) > 0.2 else "**Medium**"

            lines.append(
                f"- **{flag_name}**: {baseline:.1%} → {candidate:.1%} "
                f"({delta_str}, {percent_str}) - Severity: {severity}"
            )
        lines.append("")

    return "\n".join(lines)


def render_improvements_section(comparison_data: dict[str, Any]) -> str:
    """
    Render section highlighting improvements.

    Args:
        comparison_data: Comparison artifact dictionary

    Returns:
        Markdown-formatted improvements section
    """
    lines = ["## Improvements Detected\n"]

    # Find improved metrics (positive delta, not null, not regression)
    improved_metrics = [
        d
        for d in comparison_data.get("metric_deltas", [])
        if d.get("delta") is not None
        and d.get("delta") > 0
        and not d.get("is_regression", False)
    ]

    # Find improved flags (negative delta, not null, not regression)
    improved_flags = [
        d
        for d in comparison_data.get("flag_deltas", [])
        if d.get("delta") is not None
        and d.get("delta") < 0
        and not d.get("is_regression", False)
    ]

    if not improved_metrics and not improved_flags:
        lines.append("*No improvements detected.*\n")
        return "\n".join(lines)

    if improved_metrics:
        lines.append("### Improved Metrics\n")
        # Sort by magnitude (absolute delta, descending)
        improved_metrics_sorted = sorted(
            improved_metrics,
            key=lambda d: (abs(d.get("delta", 0)), d.get("metric_name", "")),
            reverse=True,
        )
        for delta in improved_metrics_sorted:
            metric_name = escape_html_for_markdown(delta.get("metric_name", "unknown"))
            delta_val = delta.get("delta")
            percent = delta.get("percent_change")
            baseline = delta.get("baseline_mean")
            candidate = delta.get("candidate_mean")

            delta_str = format_delta_sign(delta_val)
            percent_str = format_percentage(percent)

            lines.append(
                f"- **{metric_name}**: {baseline:.2f} → {candidate:.2f} "
                f"({delta_str}, {percent_str})"
            )
        lines.append("")

    if improved_flags:
        lines.append("### Improved Flags\n")
        # Sort by magnitude (absolute delta, descending)
        improved_flags_sorted = sorted(
            improved_flags,
            key=lambda d: (abs(d.get("delta", 0)), d.get("flag_name", "")),
            reverse=True,
        )
        for delta in improved_flags_sorted:
            flag_name = escape_html_for_markdown(delta.get("flag_name", "unknown"))
            delta_val = delta.get("delta")
            percent = delta.get("percent_change")
            baseline = delta.get("baseline_proportion", 0)
            candidate = delta.get("candidate_proportion", 0)

            if delta_val is not None:
                delta_str = format_delta_sign(delta_val * 100) + " pp"
            else:
                delta_str = "N/A"
            percent_str = format_percentage(percent)

            lines.append(
                f"- **{flag_name}**: {baseline:.1%} → {candidate:.1%} "
                f"({delta_str}, {percent_str})"
            )
        lines.append("")

    return "\n".join(lines)


def render_markdown_comparison_report(
    comparison_data: dict[str, Any], config: CompareReportConfig
) -> str:
    """
    Render a complete Markdown comparison report.

    Args:
        comparison_data: Comparison artifact dictionary
        config: Report configuration

    Returns:
        Complete Markdown report as string
    """
    # Render sections
    sections = [
        render_comparison_metadata_section(comparison_data),
        render_suite_comparison_table(comparison_data),
        render_suite_flags_comparison_table(comparison_data),
        render_regressions_section(comparison_data),
        render_improvements_section(comparison_data),
    ]

    return "\n".join(sections)


def render_comparison_report(
    comparison_artifact_path: Path,
    top_cases_per_metric: int = 5,
    generate_html: bool = False,
    output_name: str = "comparison_report.md",
    html_output_name: str = "comparison_report.html",
) -> Path:
    """
    Generate a Markdown (and optional HTML) report from a comparison artifact.

    Args:
        comparison_artifact_path: Path to comparison artifact JSON file
        top_cases_per_metric: Number of top regressed/improved cases to show per metric
        generate_html: Whether to generate HTML alongside Markdown
        output_name: Filename for Markdown report
        html_output_name: Filename for HTML report

    Returns:
        Path to generated Markdown report file

    Raises:
        FileNotFoundError: If comparison artifact not found
        ValueError: If artifact is invalid
    """
    # Validate comparison artifact file
    if not comparison_artifact_path.exists():
        raise FileNotFoundError(f"Comparison artifact not found: {comparison_artifact_path}")

    if not comparison_artifact_path.is_file():
        raise ValueError(f"Path is not a file: {comparison_artifact_path}")

    # Load comparison artifact
    logger.info(f"Loading comparison artifact from {comparison_artifact_path}")
    comparison_data = load_comparison_artifact(comparison_artifact_path)

    # Create config
    config = CompareReportConfig(
        top_cases_per_metric=top_cases_per_metric,
        generate_html=generate_html,
    )

    # Render Markdown report
    logger.info("Rendering Markdown comparison report")
    markdown_report = render_markdown_comparison_report(comparison_data, config)

    # Write Markdown file in same directory as comparison artifact
    output_dir = comparison_artifact_path.parent
    markdown_path = output_dir / output_name
    markdown_path.write_text(markdown_report, encoding="utf-8")
    logger.info(f"Markdown comparison report written to {markdown_path}")

    # Optionally generate HTML
    if generate_html:
        html_content = convert_markdown_to_html(markdown_report)
        if html_content:
            html_path = output_dir / html_output_name
            html_path.write_text(html_content, encoding="utf-8")
            logger.info(f"HTML comparison report written to {html_path}")
        else:
            logger.warning("HTML generation skipped (converter unavailable)")

    return markdown_path
