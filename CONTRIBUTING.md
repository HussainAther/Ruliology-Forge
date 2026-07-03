# Contributing to Ruliology Forge

Thanks for your interest in contributing.

## Local setup

```bash
git clone https://github.com/HussainAther/ruliology-forge.git
cd ruliology-forge
pip install -e '.[dev]'
pytest
```

## Contribution ideas

Good first contributions include tests, examples, documentation, plotting improvements, and additional perturbation operators.

## Pull request checklist

Before opening a pull request:

- Run `pytest`
- Keep dependencies lightweight
- Add or update tests for code changes
- Add docstrings for public functions
- Keep biological claims careful and clearly labeled as research motivation unless supported by data
