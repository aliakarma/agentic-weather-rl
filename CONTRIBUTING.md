# Contributing

Thank you for your interest in contributing to this project!

## Getting Started

```bash
git clone https://github.com/aliakarma/agentic-weather-rl.git
cd agentic-weather-rl
pip install -e ".[dev]"
```

## How to Contribute

- **Bug reports** — open a GitHub Issue with a minimal reproducible example
- **Feature requests** — open a GitHub Issue describing the use case
- **Pull requests** — fork the repo, make changes on a feature branch, then open a PR against `main`

## Code Style

- Follow PEP 8
- Keep docstrings on all public functions and classes
- All new algorithms should live under `src/algorithms/`
- All new models should live under `src/models/`

## Running Tests

```bash
pytest tests/
```

## Reporting Results

If you reproduce or extend the paper results, please include:
- Seed values used
- Hardware spec
- Full console output or a linked notebook

We welcome reproduction reports as GitHub Issues even if results match exactly — it helps build confidence in the codebase.
