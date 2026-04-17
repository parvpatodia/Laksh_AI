# Testing Rules

## Conventions
- Test files mirror source: src/pkg/x.py -> tests/test_x.py
- Fixtures in conftest.py
- Test names describe behavior: test_forward_pass_output_shape

## What to test
- Model forward pass with known input shape -> correct output shape
- Data transforms: sample input -> correct dtype/shape/range
- Config loading: valid YAML parses, invalid raises clear error
- Loss functions: known input -> expected output value
- Do NOT test training loops end-to-end in unit tests

## Running
- Run all: `pytest -x`
- With prints: `pytest -x -s`
- Single test: `pytest tests/test_x.py::test_name`
