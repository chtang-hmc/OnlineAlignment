# Tests

This directory contains unit tests for the online-alignment package.

## Running Tests

Install test dependencies:

```bash
pip install -e ".[dev]"
```

Run all tests:

```bash
pytest
```

Run tests for a specific module:

```bash
pytest tests/core/cost/
```

Run with coverage:

```bash
pytest --cov=core --cov-report=html
```

## Test Structure

Tests are organized to mirror the package structure:

- `tests/core/cost/` - Tests for cost metrics
  - `test_cosine.py` - Cosine distance tests
  - `test_euclidean.py` - Euclidean distance tests
  - `test_cost_metric.py` - Base CostMetric class tests
  - `test_registry.py` - Cost metric registry tests

## Writing Tests

Follow pytest conventions:

- Test files should start with `test_`
- Test classes should start with `Test`
- Test functions should start with `test_`
- Use fixtures for common test data
