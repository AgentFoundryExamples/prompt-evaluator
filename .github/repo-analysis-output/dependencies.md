# Dependency Graph

Multi-language intra-repository dependency analysis.

Supports Python, JavaScript/TypeScript, C/C++, Rust, Go, Java, C#, Swift, HTML/CSS, and SQL.

Includes classification of external dependencies as stdlib vs third-party.

## Statistics

- **Total files**: 8
- **Intra-repo dependencies**: 9
- **External stdlib dependencies**: 14
- **External third-party dependencies**: 9

## External Dependencies

### Standard Library / Core Modules

Total: 14 unique modules

- `abc.ABC`
- `abc.abstractmethod`
- `dataclasses.asdict`
- `dataclasses.dataclass`
- `datetime.datetime`
- `datetime.timezone`
- `enum.Enum`
- `logging`
- `os`
- `pathlib.Path`
- `re`
- `time`
- `tomllib`
- `typing.Any`

### Third-Party Packages

Total: 9 unique packages

- `openai.OpenAI`
- `openai.OpenAIError`
- `pydantic.BaseModel`
- `pydantic.Field`
- `pydantic.field_validator`
- `pytest`
- `tomli`
- `typer`
- `yaml`

## Most Depended Upon Files (Intra-Repo)

- `src/prompt_evaluator/models.py` (3 dependents)
- `src/prompt_evaluator/__init__.py` (2 dependents)
- `src/prompt_evaluator/config.py` (2 dependents)
- `src/prompt_evaluator/cli.py` (1 dependents)
- `src/prompt_evaluator/provider.py` (1 dependents)

## Files with Most Dependencies (Intra-Repo)

- `tests/test_basic.py` (5 dependencies)
- `tests/test_config_models.py` (2 dependencies)
- `src/prompt_evaluator/cli.py` (1 dependencies)
- `src/prompt_evaluator/provider.py` (1 dependencies)
