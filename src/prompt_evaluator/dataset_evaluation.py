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
Dataset evaluation orchestration.

This module provides functionality for evaluating multiple test cases from
a dataset, generating multiple samples per test case, and computing both
per-case and overall aggregate statistics.
"""

import json
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prompt_evaluator.models import (
    DatasetEvaluationRun,
    GeneratorConfig,
    JudgeConfig,
    Rubric,
    Sample,
    TestCase,
    TestCaseResult,
)
from prompt_evaluator.provider import OpenAIProvider, generate_completion, judge_completion


def compute_per_case_statistics(
    samples: list[Sample], rubric: Rubric | None = None
) -> tuple[dict[str, dict[str, float | int | None]], dict[str, dict[str, int | float]]]:
    """
    Compute per-case statistics from samples for a single test case.

    Args:
        samples: List of Sample objects for a single test case
        rubric: Optional Rubric object defining metrics and flags

    Returns:
        Tuple of (per_metric_stats, per_flag_stats) dictionaries
    """
    per_metric_stats: dict[str, dict[str, float | int | None]] = {}
    per_flag_stats: dict[str, dict[str, int | float]] = {}

    if not rubric:
        return per_metric_stats, per_flag_stats

    # Filter out invalid samples
    valid_samples = [s for s in samples if s.status != "judge_invalid_response"]

    # Compute per-metric statistics
    for metric in rubric.metrics:
        metric_name = metric.name
        metric_scores = [
            s.judge_metrics[metric_name]["score"]
            for s in valid_samples
            if s.status == "completed"
            and metric_name in s.judge_metrics
            and "score" in s.judge_metrics[metric_name]
        ]

        if metric_scores:
            # Compute mean, std, min, max, count
            mean = sum(metric_scores) / len(metric_scores)
            # Compute standard deviation
            if len(metric_scores) > 1:
                variance = sum((x - mean) ** 2 for x in metric_scores) / len(metric_scores)
                std = variance**0.5
            else:
                std = 0.0

            per_metric_stats[metric_name] = {
                "mean": mean,
                "std": std,
                "min": min(metric_scores),
                "max": max(metric_scores),
                "count": len(metric_scores),
            }
        else:
            per_metric_stats[metric_name] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "count": 0,
            }

    # Compute per-flag statistics
    for flag in rubric.flags:
        flag_name = flag.name
        flag_values = [
            s.judge_flags[flag_name]
            for s in valid_samples
            if s.status == "completed" and flag_name in s.judge_flags
        ]

        true_count = sum(1 for v in flag_values if v)
        total_count = len(flag_values)

        per_flag_stats[flag_name] = {
            "true_count": true_count,
            "false_count": total_count - true_count,
            "total_count": total_count,
            "true_proportion": true_count / total_count if total_count > 0 else 0.0,
        }

    return per_metric_stats, per_flag_stats


def compute_overall_statistics(
    test_case_results: list[TestCaseResult], rubric: Rubric | None = None
) -> tuple[dict[str, dict[str, float | int | None]], dict[str, dict[str, int | float]]]:
    """
    Compute overall statistics from all test case results.

    Computes:
    - Mean of per-case means for each metric
    - Overall flag rates across all samples

    Args:
        test_case_results: List of TestCaseResult objects
        rubric: Optional Rubric object defining metrics and flags

    Returns:
        Tuple of (overall_metric_stats, overall_flag_stats) dictionaries
    """
    overall_metric_stats: dict[str, dict[str, float | int | None]] = {}
    overall_flag_stats: dict[str, dict[str, int | float]] = {}

    if not rubric:
        return overall_metric_stats, overall_flag_stats

    # Compute overall metric statistics (mean of per-case means)
    for metric in rubric.metrics:
        metric_name = metric.name
        # Collect per-case means for this metric
        per_case_means: list[float | int] = [
            tc.per_metric_stats[metric_name]["mean"]  # type: ignore[misc]
            for tc in test_case_results
            if tc.status == "completed"
            and metric_name in tc.per_metric_stats
            and tc.per_metric_stats[metric_name]["mean"] is not None
        ]

        if per_case_means:
            overall_metric_stats[metric_name] = {
                "mean_of_means": sum(per_case_means) / len(per_case_means),
                "min_of_means": min(per_case_means),
                "max_of_means": max(per_case_means),
                "num_cases": len(per_case_means),
            }
        else:
            overall_metric_stats[metric_name] = {
                "mean_of_means": None,
                "min_of_means": None,
                "max_of_means": None,
                "num_cases": 0,
            }

    # Compute overall flag statistics (aggregate across all samples)
    for flag in rubric.flags:
        flag_name = flag.name
        # Aggregate flag counts across all test cases
        total_true: int = 0
        total_false: int = 0
        total_count: int = 0

        for tc in test_case_results:
            if tc.status == "completed" and flag_name in tc.per_flag_stats:
                total_true += int(tc.per_flag_stats[flag_name]["true_count"])
                total_false += int(tc.per_flag_stats[flag_name]["false_count"])
                total_count += int(tc.per_flag_stats[flag_name]["total_count"])

        overall_flag_stats[flag_name] = {
            "true_count": total_true,
            "false_count": total_false,
            "total_count": total_count,
            "true_proportion": total_true / total_count if total_count > 0 else 0.0,
        }

    return overall_metric_stats, overall_flag_stats


def evaluate_dataset(
    provider: OpenAIProvider,
    test_cases: list[TestCase],
    dataset_metadata: dict[str, Any],
    system_prompt: str,
    num_samples_per_case: int,
    generator_config: GeneratorConfig,
    judge_config: JudgeConfig,
    judge_system_prompt: str,
    rubric: Rubric | None,
    rubric_metadata: dict[str, Any],
    output_dir: Path,
    progress_callback: Callable[..., None] | None = None,
) -> DatasetEvaluationRun:
    """
    Evaluate a dataset of test cases with multiple samples per case.

    This function orchestrates the complete evaluation process:
    1. Iterates through test cases serially
    2. Generates multiple samples per test case
    3. Judges each sample
    4. Computes per-case statistics
    5. Computes overall suite statistics
    6. Streams results to disk

    Args:
        provider: OpenAI provider instance
        test_cases: List of TestCase objects to evaluate
        dataset_metadata: Metadata about the dataset (path, hash, count)
        system_prompt: System prompt for the generator
        num_samples_per_case: Number of samples to generate per test case
        generator_config: Configuration for the generator model
        judge_config: Configuration for the judge model
        judge_system_prompt: System prompt for the judge
        rubric: Optional Rubric object for evaluation
        rubric_metadata: Metadata about the rubric
        output_dir: Directory to write results
        progress_callback: Optional callback for progress updates (e.g., typer.echo)

    Returns:
        DatasetEvaluationRun object with complete results
    """
    # Create run metadata
    run_id = str(uuid.uuid4())
    timestamp_start = datetime.now(timezone.utc)

    # Create output directory
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize evaluation run
    evaluation_run = DatasetEvaluationRun(
        run_id=run_id,
        dataset_path=dataset_metadata["path"],
        dataset_hash=dataset_metadata["hash"],
        dataset_count=dataset_metadata["count"],
        num_samples_per_case=num_samples_per_case,
        generator_config=generator_config,
        judge_config=judge_config,
        rubric_metadata=rubric_metadata,
        timestamp_start=timestamp_start,
    )

    # Iterate through test cases
    for idx, test_case in enumerate(test_cases, start=1):
        if progress_callback:
            progress_callback(
                f"Evaluating test case {idx}/{len(test_cases)}: {test_case.id}...",
                err=True,
            )

        tc_timestamp_start = datetime.now(timezone.utc)

        # Initialize test case result
        test_case_result = TestCaseResult(
            test_case_id=test_case.id,
            test_case_input=test_case.input,
            test_case_metadata={
                "description": test_case.description,
                "task": test_case.task,
                "expected_constraints": test_case.expected_constraints,
                "reference": test_case.reference,
                **test_case.metadata,
            },
            num_samples=num_samples_per_case,
            timestamp_start=tc_timestamp_start,
        )

        # Generate and judge samples for this test case
        samples: list[Sample] = []
        num_successful = 0

        for sample_idx in range(num_samples_per_case):
            sample_id = f"{run_id}-{test_case.id}-sample-{sample_idx + 1}"

            try:
                # Generate completion
                response_text, metadata = generate_completion(
                    provider=provider,
                    system_prompt=system_prompt,
                    user_prompt=test_case.input,
                    model=generator_config.model_name,
                    temperature=generator_config.temperature,
                    max_completion_tokens=generator_config.max_completion_tokens,
                    seed=generator_config.seed,
                )

                # Judge the output
                judge_result = judge_completion(
                    provider=provider,
                    input_text=test_case.input,
                    generator_output=response_text,
                    judge_config=judge_config,
                    judge_system_prompt=judge_system_prompt,
                    task_description=test_case.task,
                    rubric=rubric,
                )

                # Create sample with judge results
                sample = Sample(
                    sample_id=sample_id,
                    input_text=test_case.input,
                    generator_output=response_text,
                    judge_score=judge_result.get("judge_score"),
                    judge_rationale=judge_result.get("judge_rationale"),
                    judge_raw_response=judge_result.get("judge_raw_response"),
                    status=judge_result["status"],
                    task_description=test_case.task,
                    judge_metrics=judge_result.get("judge_metrics", {}),
                    judge_flags=judge_result.get("judge_flags", {}),
                    judge_overall_comment=judge_result.get("judge_overall_comment"),
                )

                samples.append(sample)

                if sample.status == "completed":
                    num_successful += 1

            except Exception as e:
                # Create a sample with error status
                sample = Sample(
                    sample_id=sample_id,
                    input_text=test_case.input,
                    generator_output="",
                    status="generation_error",
                    task_description=test_case.task,
                )
                samples.append(sample)

                if progress_callback:
                    progress_callback(f"  Error in sample {sample_idx + 1}: {e}", err=True)

        # Store samples in test case result
        test_case_result.samples = samples

        # Compute per-case statistics
        per_metric_stats, per_flag_stats = compute_per_case_statistics(samples, rubric)
        test_case_result.per_metric_stats = per_metric_stats
        test_case_result.per_flag_stats = per_flag_stats

        # Update test case result status
        tc_timestamp_end = datetime.now(timezone.utc)
        test_case_result.timestamp_end = tc_timestamp_end

        if num_successful == num_samples_per_case:
            test_case_result.status = "completed"
        elif num_successful > 0:
            test_case_result.status = "partial"
        else:
            test_case_result.status = "failed"
            test_case_result.error_message = "All samples failed"

        # Add to evaluation run
        evaluation_run.test_case_results.append(test_case_result)

        # Stream intermediate result to disk
        intermediate_file = run_dir / f"test_case_{test_case.id}.json"
        try:
            intermediate_file.write_text(
                json.dumps(test_case_result.to_dict(), indent=2), encoding="utf-8"
            )
        except Exception as e:
            if progress_callback:
                progress_callback(f"  Warning: Failed to write intermediate result: {e}", err=True)

        if progress_callback:
            progress_callback(
                f"  Completed {num_successful}/{num_samples_per_case} samples successfully",
                err=True,
            )

    # Compute overall statistics
    overall_metric_stats, overall_flag_stats = compute_overall_statistics(
        evaluation_run.test_case_results, rubric
    )
    evaluation_run.overall_metric_stats = overall_metric_stats
    evaluation_run.overall_flag_stats = overall_flag_stats

    # Update run status
    timestamp_end = datetime.now(timezone.utc)
    evaluation_run.timestamp_end = timestamp_end

    num_completed = sum(1 for tc in evaluation_run.test_case_results if tc.status == "completed")
    if num_completed == len(test_cases):
        evaluation_run.status = "completed"
    elif num_completed > 0:
        evaluation_run.status = "partial"
    else:
        evaluation_run.status = "failed"

    # Write final consolidated results
    final_file = run_dir / "dataset_evaluation.json"
    final_file.write_text(json.dumps(evaluation_run.to_dict(), indent=2), encoding="utf-8")

    return evaluation_run
