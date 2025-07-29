.PHONY: install test lint format clean docs help

# Default target
help:
	@echo "Available targets:"
	@echo "  install    Install package in development mode"
	@echo "  test       Run tests with coverage"
	@echo "  lint       Run linting checks"
	@echo "  format     Format code with black"
	@echo "  clean      Clean build artifacts"
	@echo "  docs       Build documentation"

install:
	pip install -e .[dev]

test:
	pytest --cov=src --cov-report=term-missing --cov-report=html

lint:
	ruff check src tests
	mypy src
	black --check src tests

format:
	black src tests
	ruff check --fix src tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

docs:
	@echo "Documentation build target - implement with chosen doc system"