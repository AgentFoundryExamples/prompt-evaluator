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
Comparison logic for evaluating performance differences between runs.

This module provides functions to compare baseline and candidate evaluation runs,
compute deltas for metrics and flags, and detect regressions based on thresholds.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prompt_evaluator.models import (
    ComparisonResult,
    FlagDelta,
    MetricDelta,
    TestCaseComparison,
)


# Constants for tie detection
TIE_DETECTION_EPSILON = 0.01  # Delta below this threshold is considered a tie


def load_run_artifact(artifact_path: Path) -> dict[str, Any]:
    """
    Load and validate a run artifact JSON file.

    Args:
        artifact_path: Path to the run artifact JSON file

    Returns:
        Parsed JSON dictionary

    Raises:
        FileNotFoundError: If artifact file doesn't exist
        ValueError: If artifact file is not valid JSON or missing required fields
    """
    if not artifact_path.exists():
        raise FileNotFoundError(f"Run artifact not found: {artifact_path}")

    try:
        with open(artifact_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in artifact {artifact_path}: {e}") from e

    # Validate required fields
    required_fields = ["run_id"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(
            f"Artifact {artifact_path} missing required fields: {', '.join(missing_fields)}"
        )

    return data


def compute_metric_delta(
    metric_name: str,
    baseline_stats: dict[str, Any],
    candidate_stats: dict[str, Any],
    threshold: float,
) -> MetricDelta:
    """
    Compute delta for a single metric between baseline and candidate.

    Args:
        metric_name: Name of the metric
        baseline_stats: Baseline overall_metric_stats for this metric
        candidate_stats: Candidate overall_metric_stats for this metric
        threshold: Absolute threshold for regression detection (negative delta)

    Returns:
        MetricDelta object with comparison results
    """
    baseline_mean = baseline_stats.get("mean_of_means")
    candidate_mean = candidate_stats.get("mean_of_means")

    # Compute delta and percent change
    if baseline_mean is not None and candidate_mean is not None:
        delta = candidate_mean - baseline_mean
        if baseline_mean != 0:
            percent_change = (delta / abs(baseline_mean)) * 100.0
        else:
            percent_change = None if delta == 0 else float("inf") if delta > 0 else float("-inf")
    else:
        delta = None
        percent_change = None

    # Detect regression: negative delta exceeding threshold
    # Uses strict inequality (>) so delta exactly equal to threshold is NOT a regression
    # This is intentional per requirements: "Delta exactly equal to threshold remains unchanged"
    is_regression = False
    if delta is not None and delta < 0 and abs(delta) > threshold:
        is_regression = True

    return MetricDelta(
        metric_name=metric_name,
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        delta=delta,
        percent_change=percent_change,
        is_regression=is_regression,
        threshold_used=threshold,
    )


def compute_flag_delta(
    flag_name: str,
    baseline_stats: dict[str, Any],
    candidate_stats: dict[str, Any],
    threshold: float,
) -> FlagDelta:
    """
    Compute delta for a single flag between baseline and candidate.

    Args:
        flag_name: Name of the flag
        baseline_stats: Baseline overall_flag_stats for this flag
        candidate_stats: Candidate overall_flag_stats for this flag
        threshold: Absolute threshold for regression detection (positive delta)

    Returns:
        FlagDelta object with comparison results
    """
    baseline_proportion = baseline_stats.get("true_proportion", 0.0)
    candidate_proportion = candidate_stats.get("true_proportion", 0.0)

    delta = candidate_proportion - baseline_proportion

    # Compute percent change in proportions
    if baseline_proportion != 0:
        percent_change = (delta / baseline_proportion) * 100.0
    else:
        percent_change = None if delta == 0 else float("inf") if delta > 0 else float("-inf")

    # Detect regression: positive delta exceeding threshold (more flags = worse)
    # Uses strict inequality (>) so delta exactly equal to threshold is NOT a regression
    is_regression = False
    if delta > 0 and delta > threshold:
        is_regression = True

    return FlagDelta(
        flag_name=flag_name,
        baseline_proportion=baseline_proportion,
        candidate_proportion=candidate_proportion,
        delta=delta,
        percent_change=percent_change,
        is_regression=is_regression,
        threshold_used=threshold,
    )


def compute_test_case_comparisons(
    baseline_data: dict[str, Any],
    candidate_data: dict[str, Any],
    metric_threshold: float,
) -> tuple[list[TestCaseComparison], dict[str, int]]:
    """
    Compute per-test-case comparisons between baseline and candidate.
    
    Args:
        baseline_data: Baseline run artifact
        candidate_data: Candidate run artifact
        metric_threshold: Threshold for regression detection
        
    Returns:
        Tuple of (test_case_comparisons, win_loss_tie_stats)
    """
    test_case_comparisons: list[TestCaseComparison] = []
    win_count = 0
    loss_count = 0
    tie_count = 0
    
    # Get test case results from both runs
    baseline_test_cases = baseline_data.get("test_case_results", [])
    candidate_test_cases = candidate_data.get("test_case_results", [])
    
    # Create lookup maps by test case ID
    baseline_map = {tc["test_case_id"]: tc for tc in baseline_test_cases}
    candidate_map = {tc["test_case_id"]: tc for tc in candidate_test_cases}
    
    # Get all test case IDs (union of both runs)
    all_test_case_ids = sorted(set(baseline_map.keys()) | set(candidate_map.keys()))
    
    for test_case_id in all_test_case_ids:
        baseline_tc = baseline_map.get(test_case_id)
        candidate_tc = candidate_map.get(test_case_id)
        
        # Skip if either is missing
        if not baseline_tc or not candidate_tc:
            continue
            
        # Get per_metric_stats for this test case
        baseline_metrics = baseline_tc.get("per_metric_stats", {})
        candidate_metrics = candidate_tc.get("per_metric_stats", {})
        
        # Compute per-metric deltas for this test case
        per_metric_deltas: dict[str, float | None] = {}
        metric_scores_baseline: list[float] = []
        metric_scores_candidate: list[float] = []
        
        # Get all metrics (union)
        all_metrics = set(baseline_metrics.keys()) | set(candidate_metrics.keys())
        
        for metric_name in sorted(all_metrics):
            baseline_mean = baseline_metrics.get(metric_name, {}).get("mean")
            candidate_mean = candidate_metrics.get(metric_name, {}).get("mean")
            
            if baseline_mean is not None and candidate_mean is not None:
                delta = candidate_mean - baseline_mean
                per_metric_deltas[metric_name] = delta
                metric_scores_baseline.append(baseline_mean)
                metric_scores_candidate.append(candidate_mean)
            else:
                per_metric_deltas[metric_name] = None
        
        # Compute overall means (average across all metrics)
        baseline_overall_mean = (
            sum(metric_scores_baseline) / len(metric_scores_baseline)
            if metric_scores_baseline
            else None
        )
        candidate_overall_mean = (
            sum(metric_scores_candidate) / len(metric_scores_candidate)
            if metric_scores_candidate
            else None
        )
        
        # Determine winner
        # Candidate wins if overall mean is higher and no regressions exceed threshold
        # Tie if overall means are equal (or very close)
        # Baseline wins otherwise
        is_regression = False
        winner = "tie"
        
        if baseline_overall_mean is not None and candidate_overall_mean is not None:
            overall_delta = candidate_overall_mean - baseline_overall_mean
            
            # Check for regressions (any metric drops more than threshold)
            for delta in per_metric_deltas.values():
                if delta is not None and delta < 0 and abs(delta) > metric_threshold:
                    is_regression = True
                    break
            
            # Determine winner based on overall delta
            # Use TIE_DETECTION_EPSILON for tie detection
            if abs(overall_delta) < TIE_DETECTION_EPSILON:
                winner = "tie"
                tie_count += 1
            elif overall_delta > 0:
                winner = "candidate"
                win_count += 1
            else:
                winner = "baseline"
                loss_count += 1
        
        test_case_comparison = TestCaseComparison(
            test_case_id=test_case_id,
            baseline_mean=baseline_overall_mean,
            candidate_mean=candidate_overall_mean,
            per_metric_deltas=per_metric_deltas,
            winner=winner,
            is_regression=is_regression,
        )
        
        test_case_comparisons.append(test_case_comparison)
    
    win_loss_tie_stats = {
        "candidate_wins": win_count,
        "baseline_wins": loss_count,
        "ties": tie_count,
        "total": len(test_case_comparisons),
    }
    
    return test_case_comparisons, win_loss_tie_stats


def compare_runs(
    baseline_artifact: Path,
    candidate_artifact: Path,
    metric_threshold: float = 0.1,
    flag_threshold: float = 0.05,
    include_test_case_comparisons: bool = True,
) -> ComparisonResult:
    """
    Compare baseline and candidate evaluation runs.

    Args:
        baseline_artifact: Path to baseline run artifact JSON
        candidate_artifact: Path to candidate run artifact JSON
        metric_threshold: Absolute threshold for metric regression (default: 0.1)
                         A regression is flagged if candidate_mean < baseline_mean - threshold
        flag_threshold: Absolute threshold for flag regression (default: 0.05)
                       A regression is flagged if:
                       candidate_proportion > baseline_proportion + threshold
        include_test_case_comparisons: Whether to compute per-test-case comparisons
                                      (default: True)

    Returns:
        ComparisonResult with all deltas and regression flags

    Raises:
        FileNotFoundError: If artifact files don't exist
        ValueError: If artifacts are invalid or incompatible
    """
    # Load artifacts
    baseline_data = load_run_artifact(baseline_artifact)
    candidate_data = load_run_artifact(candidate_artifact)

    # Extract metadata
    baseline_run_id = baseline_data["run_id"]
    candidate_run_id = candidate_data["run_id"]
    baseline_prompt_version = baseline_data.get("prompt_version_id")
    candidate_prompt_version = candidate_data.get("prompt_version_id")

    # Get overall statistics
    baseline_metric_stats = baseline_data.get("overall_metric_stats", {})
    candidate_metric_stats = candidate_data.get("overall_metric_stats", {})
    baseline_flag_stats = baseline_data.get("overall_flag_stats", {})
    candidate_flag_stats = candidate_data.get("overall_flag_stats", {})

    # Compute metric deltas for all metrics present in either run
    all_metric_names = set(baseline_metric_stats.keys()) | set(candidate_metric_stats.keys())
    metric_deltas = []

    for metric_name in sorted(all_metric_names):
        baseline_stats = baseline_metric_stats.get(metric_name, {})
        candidate_stats = candidate_metric_stats.get(metric_name, {})
        delta = compute_metric_delta(metric_name, baseline_stats, candidate_stats, metric_threshold)
        metric_deltas.append(delta)

    # Compute flag deltas for all flags present in either run
    all_flag_names = set(baseline_flag_stats.keys()) | set(candidate_flag_stats.keys())
    flag_deltas: list[FlagDelta] = []

    for flag_name in sorted(all_flag_names):
        baseline_stats = baseline_flag_stats.get(flag_name, {})
        candidate_stats = candidate_flag_stats.get(flag_name, {})
        flag_delta = compute_flag_delta(flag_name, baseline_stats, candidate_stats, flag_threshold)
        flag_deltas.append(flag_delta)

    # Count regressions
    regression_count = sum(1 for d in metric_deltas if d.is_regression)
    regression_count += sum(1 for d in flag_deltas if d.is_regression)
    has_regressions = regression_count > 0

    # Compute per-test-case comparisons if requested and data available
    test_case_comparisons: list[TestCaseComparison] = []
    win_loss_tie_stats: dict[str, int] = {}
    
    if include_test_case_comparisons:
        try:
            test_case_comparisons, win_loss_tie_stats = compute_test_case_comparisons(
                baseline_data, candidate_data, metric_threshold
            )
        except Exception as e:
            # If per-test-case comparison fails, continue without it
            # This maintains backward compatibility with runs that don't have test_case_results
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to compute test case comparisons: {e}")
            pass

    # Create comparison result
    result = ComparisonResult(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        baseline_prompt_version=baseline_prompt_version,
        candidate_prompt_version=candidate_prompt_version,
        metric_deltas=metric_deltas,
        flag_deltas=flag_deltas,
        has_regressions=has_regressions,
        regression_count=regression_count,
        comparison_timestamp=datetime.now(timezone.utc),
        thresholds_config={
            "metric_threshold": metric_threshold,
            "flag_threshold": flag_threshold,
        },
        test_case_comparisons=test_case_comparisons,
        win_loss_tie_stats=win_loss_tie_stats,
    )

    return result
