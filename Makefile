.PHONY: install test lint format clean docs help security build docker-build docker-up docker-down setup dev-setup pre-commit

# Default target
help:
	@echo "Available targets:"
	@echo "  install       Install package in development mode"
	@echo "  dev-setup     Complete development environment setup"
	@echo "  test          Run tests with coverage"
	@echo "  test-fast     Run tests without coverage"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with black and ruff"
	@echo "  security      Run security scans"
	@echo "  pre-commit    Install and run pre-commit hooks"
	@echo "  clean         Clean build artifacts"
	@echo "  build         Build Python package"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-up     Start development services"
	@echo "  docker-down   Stop development services" 
	@echo "  docs          Build documentation"

install:
	pip install -e .[dev]

dev-setup:
	@./scripts/setup-dev.sh

test:
	pytest --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml

test-fast:
	pytest -x -v

lint:
	ruff check src tests
	mypy src
	black --check src tests

format:
	black src tests
	ruff check --fix src tests

security:
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json
	detect-secrets scan --baseline .secrets.baseline

pre-commit:
	pre-commit install --install-hooks
	pre-commit run --all-files

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf bandit-report.json
	rm -rf safety-report.json
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:
	python -m build

docker-build:
	docker build -t nimify-anything:dev .

docker-up:
	docker-compose --profile dev up -d

docker-down:
	docker-compose down -v

docs:
	@echo "Documentation build target - implement with Sphinx"
	@echo "Run: pip install -e .[docs] && sphinx-build -b html docs docs/_build"

# Quality assurance target - runs all checks
qa: lint security test
	@echo "All quality assurance checks passed!"

# CI target - for continuous integration
ci: install lint security test build
	@echo "CI pipeline completed successfully!"