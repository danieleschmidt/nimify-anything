# GitHub Actions Workflows

This document outlines the recommended GitHub Actions workflows for Nimify Anything.

## Required Workflows

### 1. Continuous Integration (`.github/workflows/ci.yml`)

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Run tests
      run: pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### 2. Code Quality (`.github/workflows/quality.yml`)

```yaml
name: Code Quality

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Install dependencies
      run: pip install -e .[dev]
    
    - name: Run ruff
      run: ruff check src tests
    
    - name: Run mypy
      run: mypy src
    
    - name: Check formatting
      run: black --check src tests
```

### 3. Security Scanning (`.github/workflows/security.yml`)

```yaml
name: Security

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run Bandit Security Scan
      uses: securecodewarrior/github-action-bandit@v1
      with:
        path: src/
    
    - name: Run Safety Check
      run: |
        pip install safety
        safety check
```

### 4. Release (`.github/workflows/release.yml`)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Build package
      run: |
        pip install build
        python -m build
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

## Setup Instructions

1. Create `.github/workflows/` directory in your repository
2. Add the workflow files above
3. Configure required secrets in repository settings:
   - `PYPI_API_TOKEN` for package publishing
   - `CODECOV_TOKEN` for coverage reporting (optional)

## Branch Protection

Configure branch protection rules for `main`:
- Require status checks to pass
- Require branches to be up to date
- Require review from code owners
- Dismiss stale reviews when new commits are pushed