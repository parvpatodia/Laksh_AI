# Code Style Rules

## Python
- Line length: 100 chars max
- Type hints on all function signatures
- Google-style docstrings on public functions
- Imports ordered: stdlib, third-party, local (blank line between groups)
- Use pathlib.Path over os.path
- Use f-strings, never .format() or %
- Prefer dataclass or pydantic.BaseModel over raw dicts for config

## Naming
- snake_case: functions, variables, modules
- PascalCase: classes
- UPPER_SNAKE: constants
- Descriptive names. No single-letter vars outside loops/lambdas

## Error handling
- Catch specific exceptions, never bare `except:`
- Log errors with context (file path, shapes, config values)
- Fail fast: validate inputs at function entry
