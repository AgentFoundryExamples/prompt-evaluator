# Dependency Graph

Multi-language intra-repository dependency analysis.

Supports Python, JavaScript/TypeScript, C/C++, Rust, Go, Java, C#, Swift, HTML/CSS, and SQL.

Includes classification of external dependencies as stdlib vs third-party.

## Statistics

- **Total files**: 9
- **Intra-repo dependencies**: 13
- **External stdlib dependencies**: 18
- **External third-party dependencies**: 10

## External Dependencies

### Standard Library / Core Modules

Total: 18 unique modules

- `abc.ABC`
- `abc.abstractmethod`
- `dataclasses.asdict`
- `dataclasses.dataclass`
- `datetime.datetime`
- `datetime.timezone`
- `enum.Enum`
- `json`
- `logging`
- `os`
- `pathlib.Path`
- `re`
- `sys`
- `time`
- `tomllib`
- `typing.Any`
- `unittest.mock.patch`
- `uuid`

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

- `src/prompt_evaluator/models.py` (4 dependents)
- `src/prompt_evaluator/config.py` (3 dependents)
- `src/prompt_evaluator/provider.py` (2 dependents)
- `src/prompt_evaluator/__init__.py` (2 dependents)
- `src/prompt_evaluator/cli.py` (2 dependents)

## Files with Most Dependencies (Intra-Repo)

- `tests/test_basic.py` (5 dependencies)
- `src/prompt_evaluator/cli.py` (4 dependencies)
- `tests/test_config_models.py` (2 dependencies)
- `src/prompt_evaluator/provider.py` (1 dependencies)
- `tests/test_generate_cli.py` (1 dependencies)
