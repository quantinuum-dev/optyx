# optyx

Compiler for photonics technologies

# Development

## Format the code

```
black optyx/
```

## Lint the code

```
pflake8 optyx/
pylint optyx/
```

## Test the code

```
pytest .
```

## Test the code and produce coverage statistics

```
coverage run -m pytest .
coverage report --fail-under=95 --show-missing
```

## Build the documentation

```
sphinx-build docs docs/_build/html
```
