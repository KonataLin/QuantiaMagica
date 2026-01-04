# Contributing to QuantiaMagica

Thank you for your interest in contributing to QuantiaMagica!

## How to Contribute

### Reporting Issues

1. Check existing issues first
2. Use the issue template
3. Include minimal reproducible example
4. Specify your Python version and OS

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Add tests for new functionality
5. Run tests: `pytest tests/`
6. Update documentation if needed
7. Submit PR with clear description

### Code Style

- Follow PEP 8
- Use type annotations
- Write NumPy-style docstrings
- Keep functions focused and small

### Commit Messages

```
type: short description

Longer description if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Development Setup

```bash
git clone https://github.com/yourusername/QuantiaMagica.git
cd QuantiaMagica
pip install -e ".[dev]"
pytest tests/
```

## Questions?

Open a discussion or issue on GitHub.
