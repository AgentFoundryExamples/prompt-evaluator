# Dependency Graph

Multi-language intra-repository dependency analysis.

Supports Python, JavaScript/TypeScript, C/C++, Rust, Go, Java, C#, Swift, HTML/CSS, and SQL.

Includes classification of external dependencies as stdlib vs third-party.

## Statistics

- **Total files**: 14
- **Intra-repo dependencies**: 23
- **External stdlib dependencies**: 22
- **External third-party dependencies**: 10

## External Dependencies

### Standard Library / Core Modules

Total: 22 unique modules

- `abc.ABC`
- `abc.abstractmethod`
- `collections.Counter`
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
- `unittest.mock.MagicMock`
- ... and 2 more (see JSON for full list)

### Third-Party Packages

Total: 10 unique packages

- `openai.OpenAI`
- `openai.OpenAIError`
- `pydantic.BaseModel`
- `pydantic.Field`
- `pydantic.field_validator`
- `pytest`
- `tomli`
- `typer`
- `typer.testing.CliRunner`
- `yaml`

## Most Depended Upon Files (Intra-Repo)

- `src/prompt_evaluator/models.py` (8 dependents)
- `src/prompt_evaluator/config.py` (5 dependents)
- `src/prompt_evaluator/cli.py` (5 dependents)
- `src/prompt_evaluator/provider.py` (3 dependents)
- `src/prompt_evaluator/__init__.py` (2 dependents)

## Files with Most Dependencies (Intra-Repo)

- `tests/test_basic.py` (5 dependencies)
- `src/prompt_evaluator/cli.py` (4 dependencies)
- `tests/test_aggregation.py` (2 dependencies)
- `tests/test_config_models.py` (2 dependencies)
- `tests/test_judge_models.py` (2 dependencies)
- `tests/test_rubric_cli.py` (2 dependencies)
- `tests/test_rubric_models.py` (2 dependencies)
- `src/prompt_evaluator/config.py` (1 dependencies)
- `src/prompt_evaluator/provider.py` (1 dependencies)
- `tests/test_evaluate_single_cli.py` (1 dependencies)
