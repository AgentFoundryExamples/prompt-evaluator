# Dependency Graph

Multi-language intra-repository dependency analysis.

Supports Python, JavaScript/TypeScript, C/C++, Rust, Go, Java, C#, Swift, HTML/CSS, and SQL.

Includes classification of external dependencies as stdlib vs third-party.

## Statistics

- **Total files**: 22
- **Intra-repo dependencies**: 38
- **External stdlib dependencies**: 23
- **External third-party dependencies**: 11

## External Dependencies

### Standard Library / Core Modules

Total: 23 unique modules

- `abc.ABC`
- `abc.abstractmethod`
- `collections.Counter`
- `collections.abc.Callable`
- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime.datetime`
- `datetime.timezone`
- `enum.Enum`
- `hashlib`
- `json`
- `logging`
- `os`
- `pathlib.Path`
- `re`
- `sys`
- `time`
- `tomllib`
- `typing.Any`
- ... and 3 more (see JSON for full list)

### Third-Party Packages

Total: 11 unique packages

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

- `src/prompt_evaluator/models.py` (13 dependents)
- `src/prompt_evaluator/cli.py` (8 dependents)
- `src/prompt_evaluator/config.py` (6 dependents)
- `src/prompt_evaluator/provider.py` (5 dependents)
- `src/prompt_evaluator/__init__.py` (2 dependents)
- `src/prompt_evaluator/dataset_evaluation.py` (2 dependents)
- `src/prompt_evaluator/comparison.py` (2 dependents)

## Files with Most Dependencies (Intra-Repo)

- `src/prompt_evaluator/cli.py` (6 dependencies)
- `tests/test_basic.py` (5 dependencies)
- `tests/test_prompt_version_metadata.py` (3 dependencies)
- `src/prompt_evaluator/dataset_evaluation.py` (2 dependencies)
- `tests/test_aggregation.py` (2 dependencies)
- `tests/test_config_models.py` (2 dependencies)
- `tests/test_dataset_evaluation.py` (2 dependencies)
- `tests/test_dataset_loader.py` (2 dependencies)
- `tests/test_judge_models.py` (2 dependencies)
- `tests/test_rubric_cli.py` (2 dependencies)
