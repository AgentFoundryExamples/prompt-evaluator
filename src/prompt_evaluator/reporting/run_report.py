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
Report generator for dataset evaluation runs.

This module generates Markdown (and optionally HTML) reports from evaluation
run artifacts, including metrics, flags, and qualitative examples.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    std_threshold: float = 1.0  # Threshold for marking metric as unstable
    weak_score_threshold: float = 3.0  # Threshold for marking metric as weak
    qualitative_sample_count: int = 3  # Number of worst-case examples to include
    max_text_length: int = 500  # Maximum length for truncated text
    generate_html: bool = False  # Whether to generate HTML alongside Markdown


def load_run_artifact(run_path: Path) -> dict[str, Any]:
    """
    Load run artifact JSON from a run directory.

    Args:
        run_path: Path to run directory containing dataset_evaluation.json

    Returns:
        Parsed JSON artifact as dictionary

    Raises:
        FileNotFoundError: If artifact file not found
        ValueError: If JSON is invalid
    """
    # Try standard dataset evaluation artifact name
    artifact_path = run_path / "dataset_evaluation.json"
    
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Run artifact not found at {artifact_path}. "
            f"Expected dataset_evaluation.json in run directory."
        )
    
    try:
        with open(artifact_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in artifact file: {e}") from e


def identify_unstable_metrics(
    test_case_results: list[dict[str, Any]], std_threshold: float
) -> dict[str, list[str]]:
    """
    Identify metrics with high variance (instability) per test case.

    Args:
        test_case_results: List of test case result dictionaries
        std_threshold: Standard deviation threshold for instability

    Returns:
        Dictionary mapping test_case_id to list of unstable metric names
    """
    unstable_by_case: dict[str, list[str]] = {}
    
    for tc in test_case_results:
        test_case_id = tc.get("test_case_id", "unknown")
        per_metric_stats = tc.get("per_metric_stats", {})
        unstable_metrics = []
        
        for metric_name, stats in per_metric_stats.items():
            # Skip comment fields (keys starting with _)
            if metric_name.startswith("_"):
                continue
            
            # Skip if stats is not a dictionary
            if not isinstance(stats, dict):
                continue
            
            count = stats.get("count", 0)
            std = stats.get("std")
            
            # Only flag as unstable if we have multiple samples (n > 1)
            if count > 1 and std is not None and std > std_threshold:
                unstable_metrics.append(metric_name)
        
        if unstable_metrics:
            unstable_by_case[test_case_id] = unstable_metrics
    
    return unstable_by_case


def identify_weak_metrics(
    test_case_results: list[dict[str, Any]], weak_threshold: float
) -> dict[str, list[str]]:
    """
    Identify metrics with low mean scores (weak points) per test case.

    Args:
        test_case_results: List of test case result dictionaries
        weak_threshold: Mean score threshold for weak points

    Returns:
        Dictionary mapping test_case_id to list of weak metric names
    """
    weak_by_case: dict[str, list[str]] = {}
    
    for tc in test_case_results:
        test_case_id = tc.get("test_case_id", "unknown")
        per_metric_stats = tc.get("per_metric_stats", {})
        weak_metrics = []
        
        for metric_name, stats in per_metric_stats.items():
            # Skip comment fields (keys starting with _)
            if metric_name.startswith("_"):
                continue
            
            # Skip if stats is not a dictionary
            if not isinstance(stats, dict):
                continue
            
            mean = stats.get("mean")
            
            if mean is not None and mean < weak_threshold:
                weak_metrics.append(metric_name)
        
        if weak_metrics:
            weak_by_case[test_case_id] = weak_metrics
    
    return weak_by_case


def select_qualitative_samples(
    run_data: dict[str, Any], count: int, fallback_metric: str = "semantic_fidelity"
) -> list[dict[str, Any]]:
    """
    Select worst-performing test cases for qualitative analysis.

    Selects cases with lowest mean score for primary metric (or fallback).

    Args:
        run_data: Full run artifact dictionary
        count: Number of samples to select
        fallback_metric: Metric to use if primary not available

    Returns:
        List of test case dictionaries with samples, sorted by mean score (worst first)
    """
    test_case_results = run_data.get("test_case_results", [])
    
    # Score each test case by its mean on the fallback metric
    scored_cases = []
    for tc in test_case_results:
        per_metric_stats = tc.get("per_metric_stats", {})
        
        # Try to get the fallback metric's mean score
        metric_stats = per_metric_stats.get(fallback_metric, {})
        mean_score = metric_stats.get("mean")
        
        if mean_score is not None:
            scored_cases.append((mean_score, tc))
    
    # Sort by score ascending (worst first)
    scored_cases.sort(key=lambda x: x[0])
    
    # Return top N cases
    return [tc for _, tc in scored_cases[:count]]


def truncate_text(text: str, max_length: int) -> str:
    """
    Truncate text to maximum length with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text with "..." if truncated
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def escape_markdown(text: str) -> str:
    """
    Escape special Markdown characters in text.

    Args:
        text: Text to escape

    Returns:
        Escaped text safe for Markdown
    """
    # Escape common Markdown special characters
    replacements = {
        "\\": "\\\\",
        "`": "\\`",
        "*": "\\*",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "[": "\\[",
        "]": "\\]",
        "(": "\\(",
        ")": "\\)",
        "#": "\\#",
        "+": "\\+",
        "-": "\\-",
        ".": "\\.",
        "!": "\\!",
        "|": "\\|",
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    return text


def render_metadata_section(run_data: dict[str, Any]) -> str:
    """
    Render the metadata section of the report.

    Args:
        run_data: Run artifact dictionary

    Returns:
        Markdown-formatted metadata section
    """
    lines = [
        "# Evaluation Run Report\n",
        "## Metadata\n",
        f"- **Run ID**: `{run_data.get('run_id', 'N/A')}`",
        f"- **Status**: {run_data.get('status', 'N/A')}",
        f"- **Started**: {run_data.get('timestamp_start', 'N/A')}",
        f"- **Ended**: {run_data.get('timestamp_end', 'N/A')}",
        f"- **Dataset**: `{run_data.get('dataset_path', 'N/A')}`",
        f"- **Dataset Hash**: `{run_data.get('dataset_hash', 'N/A')[:16]}...`"
        if run_data.get('dataset_hash')
        else "- **Dataset Hash**: N/A",
        f"- **Test Cases**: {run_data.get('dataset_count', 0)}",
        f"- **Samples per Case**: {run_data.get('num_samples_per_case', 0)}",
        f"- **System Prompt**: `{run_data.get('system_prompt_path', 'N/A')}`",
        f"- **Prompt Version**: `{run_data.get('prompt_version_id', 'N/A')}`",
        f"- **Generator Model**: {run_data.get('generator_config', {}).get('model_name', 'N/A')}",
        f"- **Judge Model**: {run_data.get('judge_config', {}).get('model_name', 'N/A')}",
    ]
    
    if run_data.get('run_notes'):
        lines.append(f"- **Notes**: {run_data['run_notes']}")
    
    # Add link to artifact
    lines.append(f"\n[View Raw Artifact JSON](./dataset_evaluation.json)\n")
    
    return "\n".join(lines)


def render_suite_metrics_table(run_data: dict[str, Any]) -> str:
    """
    Render the suite-level metrics table.

    Args:
        run_data: Run artifact dictionary

    Returns:
        Markdown-formatted metrics table
    """
    lines = [
        "## Suite-Level Metrics\n",
        "| Metric | Mean | Std | Min | Max | Cases |",
        "|--------|------|-----|-----|-----|-------|",
    ]
    
    overall_metric_stats = run_data.get("overall_metric_stats", {})
    
    if not overall_metric_stats:
        lines.append("| *No metrics available* | - | - | - | - | - |")
    else:
        for metric_name, stats in sorted(overall_metric_stats.items()):
            # Skip comment fields (keys starting with _)
            if metric_name.startswith("_"):
                continue
            
            # Skip if stats is not a dictionary
            if not isinstance(stats, dict):
                continue
            
            mean_of_means = stats.get("mean_of_means")
            std_of_means = stats.get("std_of_means")
            min_of_means = stats.get("min_of_means")
            max_of_means = stats.get("max_of_means")
            num_cases = stats.get("num_cases", 0)
            
            mean_str = f"{mean_of_means:.2f}" if mean_of_means is not None else "N/A"
            std_str = f"{std_of_means:.2f}" if std_of_means is not None else "N/A"
            min_str = f"{min_of_means:.2f}" if min_of_means is not None else "N/A"
            max_str = f"{max_of_means:.2f}" if max_of_means is not None else "N/A"
            
            lines.append(
                f"| {metric_name} | {mean_str} | {std_str} | {min_str} | {max_str} | {num_cases} |"
            )
    
    return "\n".join(lines) + "\n"


def render_suite_flags_table(run_data: dict[str, Any]) -> str:
    """
    Render the suite-level flags table.

    Args:
        run_data: Run artifact dictionary

    Returns:
        Markdown-formatted flags table
    """
    lines = [
        "## Suite-Level Flags\n",
        "| Flag | True Count | False Count | Total | Proportion |",
        "|------|------------|-------------|-------|------------|",
    ]
    
    overall_flag_stats = run_data.get("overall_flag_stats", {})
    
    if not overall_flag_stats:
        lines.append("| *No flags available* | - | - | - | - |")
    else:
        for flag_name, stats in sorted(overall_flag_stats.items()):
            # Skip comment fields (keys starting with _)
            if flag_name.startswith("_"):
                continue
            
            # Skip if stats is not a dictionary
            if not isinstance(stats, dict):
                continue
            
            true_count = stats.get("true_count", 0)
            false_count = stats.get("false_count", 0)
            total_count = stats.get("total_count", 0)
            true_proportion = stats.get("true_proportion", 0.0)
            
            lines.append(
                f"| {flag_name} | {true_count} | {false_count} | {total_count} | {true_proportion:.1%} |"
            )
    
    return "\n".join(lines) + "\n"


def render_test_case_table(
    run_data: dict[str, Any], unstable_by_case: dict[str, list[str]], weak_by_case: dict[str, list[str]]
) -> str:
    """
    Render the per-test-case summary table with annotations.

    Args:
        run_data: Run artifact dictionary
        unstable_by_case: Map of test case ID to unstable metrics
        weak_by_case: Map of test case ID to weak metrics

    Returns:
        Markdown-formatted test case table
    """
    lines = [
        "## Per-Test-Case Summary\n",
        "| Case ID | Status | Samples | Annotations |",
        "|---------|--------|---------|-------------|",
    ]
    
    test_case_results = run_data.get("test_case_results", [])
    
    if not test_case_results:
        lines.append("| *No test cases* | - | - | - |")
    else:
        for tc in test_case_results:
            test_case_id = tc.get("test_case_id", "unknown")
            status = tc.get("status", "unknown")
            num_samples = tc.get("num_samples", 0)
            
            # Build annotations
            annotations = []
            if test_case_id in unstable_by_case:
                metrics = ", ".join(unstable_by_case[test_case_id])
                annotations.append(f"⚠️ **UNSTABLE** ({metrics})")
            if test_case_id in weak_by_case:
                metrics = ", ".join(weak_by_case[test_case_id])
                annotations.append(f"⚠️ **WEAK** ({metrics})")
            
            annotations_str = "; ".join(annotations) if annotations else "-"
            
            lines.append(
                f"| `{test_case_id}` | {status} | {num_samples} | {annotations_str} |"
            )
    
    return "\n".join(lines) + "\n"


def render_qualitative_section(
    qualitative_samples: list[dict[str, Any]], max_text_length: int
) -> str:
    """
    Render the qualitative examples section.

    Args:
        qualitative_samples: List of test case dictionaries to showcase
        max_text_length: Maximum text length for truncation

    Returns:
        Markdown-formatted qualitative section
    """
    lines = [
        "## Qualitative Examples\n",
        "*Showing worst-performing test cases based on mean metric scores.*\n",
    ]
    
    if not qualitative_samples:
        lines.append("*No qualitative samples available.*\n")
        return "\n".join(lines)
    
    for idx, tc in enumerate(qualitative_samples, 1):
        test_case_id = tc.get("test_case_id", "unknown")
        test_case_input = tc.get("test_case_input", "")
        samples = tc.get("samples", [])
        per_metric_stats = tc.get("per_metric_stats", {})
        
        lines.append(f"### Example {idx}: `{test_case_id}`\n")
        
        # Show metric scores for this case
        if per_metric_stats:
            lines.append("**Metrics for this case:**\n")
            for metric_name, stats in sorted(per_metric_stats.items()):
                # Skip comment fields (keys starting with _)
                if metric_name.startswith("_"):
                    continue
                
                # Skip if stats is not a dictionary
                if not isinstance(stats, dict):
                    continue
                
                mean = stats.get("mean")
                std = stats.get("std")
                mean_str = f"{mean:.2f}" if mean is not None else "N/A"
                std_str = f"{std:.2f}" if std is not None else "N/A"
                lines.append(f"- {metric_name}: mean={mean_str}, std={std_str}")
            lines.append("")
        
        # Show input
        lines.append("**Input:**\n")
        truncated_input = truncate_text(test_case_input, max_text_length)
        lines.append(f"```\n{truncated_input}\n```\n")
        
        # Show up to N sample outputs
        if samples:
            lines.append("**Sample Outputs:**\n")
            for sample_idx, sample in enumerate(samples[:3], 1):  # Show max 3 samples
                generator_output = sample.get("generator_output", "")
                truncated_output = truncate_text(generator_output, max_text_length)
                
                # Show judge metrics for this sample
                judge_metrics = sample.get("judge_metrics", {})
                metrics_summary = []
                for metric_name, metric_data in sorted(judge_metrics.items()):
                    # Skip comment fields
                    if metric_name.startswith("_"):
                        continue
                    
                    # Skip if metric_data is not a dictionary
                    if not isinstance(metric_data, dict):
                        continue
                    
                    score = metric_data.get("score")
                    if score is not None:
                        metrics_summary.append(f"{metric_name}={score:.1f}")
                
                metrics_str = ", ".join(metrics_summary) if metrics_summary else "no scores"
                
                lines.append(f"*Sample {sample_idx} ({metrics_str}):*\n")
                lines.append(f"```\n{truncated_output}\n```\n")
                
                # Show rationales
                if judge_metrics:
                    lines.append("*Judge rationales:*")
                    for metric_name, metric_data in sorted(judge_metrics.items()):
                        # Skip comment fields
                        if metric_name.startswith("_"):
                            continue
                        
                        # Skip if metric_data is not a dictionary
                        if not isinstance(metric_data, dict):
                            continue
                        
                        rationale = metric_data.get("rationale", "")
                        if rationale:
                            truncated_rationale = truncate_text(rationale, 200)
                            lines.append(f"- **{metric_name}**: {truncated_rationale}")
                    lines.append("")
        else:
            lines.append("*No samples available for this case.*\n")
        
        lines.append("---\n")
    
    return "\n".join(lines)


def render_markdown_report(run_data: dict[str, Any], config: ReportConfig) -> str:
    """
    Render a complete Markdown report from run data.

    Args:
        run_data: Run artifact dictionary
        config: Report configuration

    Returns:
        Complete Markdown report as string
    """
    # Identify unstable and weak metrics
    test_case_results = run_data.get("test_case_results", [])
    unstable_by_case = identify_unstable_metrics(test_case_results, config.std_threshold)
    weak_by_case = identify_weak_metrics(test_case_results, config.weak_score_threshold)
    
    # Select qualitative samples
    qualitative_samples = select_qualitative_samples(
        run_data, config.qualitative_sample_count
    )
    
    # Render sections
    sections = [
        render_metadata_section(run_data),
        render_suite_metrics_table(run_data),
        render_suite_flags_table(run_data),
        render_test_case_table(run_data, unstable_by_case, weak_by_case),
        render_qualitative_section(qualitative_samples, config.max_text_length),
    ]
    
    return "\n".join(sections)


def convert_markdown_to_html(markdown_text: str) -> str | None:
    """
    Convert Markdown to HTML.

    Uses markdown library if available, otherwise returns None.

    Args:
        markdown_text: Markdown text to convert

    Returns:
        HTML string, or None if converter unavailable
    """
    try:
        import markdown
        
        html = markdown.markdown(
            markdown_text,
            extensions=["tables", "fenced_code", "nl2br"]
        )
        
        # Wrap in basic HTML template
        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Run Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', Courier, monospace;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        hr {{
            border: none;
            border-top: 2px solid #eee;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
{html}
</body>
</html>"""
        
        return html_doc
    except ImportError:
        logger.warning(
            "markdown library not available. Install with: pip install markdown"
        )
        return None


def render_run_report(
    run_dir: Path,
    std_threshold: float = 1.0,
    weak_score_threshold: float = 3.0,
    qualitative_sample_count: int = 3,
    max_text_length: int = 500,
    generate_html: bool = False,
    output_name: str = "report.md",
    html_output_name: str = "report.html",
) -> Path:
    """
    Generate a Markdown (and optional HTML) report for a dataset evaluation run.

    Args:
        run_dir: Path to run directory containing dataset_evaluation.json
        std_threshold: Std dev threshold for marking metrics as unstable
        weak_score_threshold: Mean score threshold for marking metrics as weak
        qualitative_sample_count: Number of worst-case examples to include
        max_text_length: Maximum text length for truncation
        generate_html: Whether to generate HTML alongside Markdown
        output_name: Filename for Markdown report
        html_output_name: Filename for HTML report

    Returns:
        Path to generated Markdown report file

    Raises:
        FileNotFoundError: If run directory or artifact not found
        ValueError: If artifact is invalid
    """
    # Validate run directory
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    if not run_dir.is_dir():
        raise ValueError(f"Path is not a directory: {run_dir}")
    
    # Load run artifact
    logger.info(f"Loading run artifact from {run_dir}")
    run_data = load_run_artifact(run_dir)
    
    # Create config
    config = ReportConfig(
        std_threshold=std_threshold,
        weak_score_threshold=weak_score_threshold,
        qualitative_sample_count=qualitative_sample_count,
        max_text_length=max_text_length,
        generate_html=generate_html,
    )
    
    # Render Markdown report
    logger.info("Rendering Markdown report")
    markdown_report = render_markdown_report(run_data, config)
    
    # Write Markdown file
    markdown_path = run_dir / output_name
    markdown_path.write_text(markdown_report, encoding="utf-8")
    logger.info(f"Markdown report written to {markdown_path}")
    
    # Optionally generate HTML
    if generate_html:
        html_content = convert_markdown_to_html(markdown_report)
        if html_content:
            html_path = run_dir / html_output_name
            html_path.write_text(html_content, encoding="utf-8")
            logger.info(f"HTML report written to {html_path}")
        else:
            logger.warning("HTML generation skipped (converter unavailable)")
    
    return markdown_path
