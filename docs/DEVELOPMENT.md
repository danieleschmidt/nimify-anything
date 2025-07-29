# Development Guide

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nimify-anything.git
   cd nimify-anything
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. Install in development mode:
   ```bash
   pip install -e .[dev]
   ```

## Development Workflow

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_core.py
```

### Code Quality
```bash
# Format code
black src tests

# Lint code
ruff check src tests

# Type checking
mypy src
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Project Structure

```
nimify-anything/
├── src/nimify/           # Main package
│   ├── __init__.py
│   ├── core.py           # Core classes
│   └── cli.py            # CLI interface
├── tests/                # Test files
├── docs/                 # Documentation
└── pyproject.toml        # Project configuration
```

## Building and Testing

### Local Development
```bash
# Install in editable mode
pip install -e .

# Test CLI
nimify --version
```

### Docker Development
```bash
# Build development image
docker build -t nimify-dev .

# Run tests in container
docker run --rm nimify-dev pytest
```

## Release Process

1. Update version in `pyproject.toml`
2. Create release notes
3. Tag release: `git tag v1.0.0`
4. Push: `git push origin main --tags`
5. GitHub Actions will build and publish to PyPI