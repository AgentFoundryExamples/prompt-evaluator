# Dependency Graph

Multi-language intra-repository dependency analysis.

Supports Python, JavaScript/TypeScript, C/C++, Rust, Go, Java, C#, Swift, HTML/CSS, and SQL.

Includes classification of external dependencies as stdlib vs third-party.

## Statistics

- **Total files**: 29
- **Intra-repo dependencies**: 49
- **External stdlib dependencies**: 26
- **External third-party dependencies**: 14

## External Dependencies

### Standard Library / Core Modules

Total: 26 unique modules

- `abc.ABC`
- `abc.abstractmethod`
- `collections.Counter`
- `collections.abc.Callable`
- `concurrent.futures`
- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime.datetime`
- `datetime.timezone`
- `enum.Enum`
- `hashlib`
- `html`
- `json`
- `logging`
- `os`
- `pathlib.Path`
- `re`
- `sys`
- `threading`
- ... and 6 more (see JSON for full list)

### Third-Party Packages

Total: 14 unique packages

- `anthropic.Anthropic`
- `anthropic.AnthropicError`
- `markdown`
- `openai.OpenAI`
- `openai.OpenAIError`
- `pydantic.BaseModel`
- `pydantic.Field`
- `pydantic.ValidationError`
- `pydantic.field_validator`
- `pytest`
- `tomli`
- `typer`
- `typer.testing.CliRunner`
- `yaml`

## Most Depended Upon Files (Intra-Repo)

- `src/prompt_evaluator/models.py` (14 dependents)
- `src/prompt_evaluator/cli.py` (9 dependents)
- `src/prompt_evaluator/provider.py` (7 dependents)
- `src/prompt_evaluator/config.py` (6 dependents)
- `src/prompt_evaluator/dataset_evaluation.py` (3 dependents)
- `src/prompt_evaluator/reporting/run_report.py` (3 dependents)
- `src/prompt_evaluator/__init__.py` (2 dependents)
- `src/prompt_evaluator/comparison.py` (2 dependents)
- `src/prompt_evaluator/reporting/compare_report.py` (2 dependents)
- `src/prompt_evaluator/reporting/__init__.py` (1 dependents)

## Files with Most Dependencies (Intra-Repo)

- `src/prompt_evaluator/cli.py` (7 dependencies)
- `tests/test_basic.py` (5 dependencies)
- `tests/test_mock_provider_integration.py` (4 dependencies)
- `tests/test_prompt_version_metadata.py` (3 dependencies)
- `src/prompt_evaluator/dataset_evaluation.py` (2 dependencies)
- `src/prompt_evaluator/reporting/__init__.py` (2 dependencies)
- `tests/test_aggregation.py` (2 dependencies)
- `tests/test_config_models.py` (2 dependencies)
- `tests/test_dataset_evaluation.py` (2 dependencies)
- `tests/test_dataset_loader.py` (2 dependencies)
