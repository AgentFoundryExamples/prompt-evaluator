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
Prompt Evaluator - A tool for evaluating and comparing prompts across LLM providers.

This package provides functionality for:
- Managing prompt templates and configurations
- Integrating with multiple LLM providers
- Running systematic prompt evaluations
- Analyzing and comparing results
"""

from prompt_evaluator.config import (
    DefaultGeneratorConfig,
    DefaultJudgeConfig,
    DefaultsConfig,
    PromptEvaluatorConfig,
    load_prompt_evaluator_config,
    locate_config_file,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DefaultGeneratorConfig",
    "DefaultJudgeConfig",
    "DefaultsConfig",
    "PromptEvaluatorConfig",
    "load_prompt_evaluator_config",
    "locate_config_file",
]
