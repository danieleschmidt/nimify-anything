# GitHub Actions Setup Guide

## Overview
This guide provides the GitHub Actions workflows that need to be manually created due to permission restrictions. The workflows implement a complete CI/CD pipeline for the nimify-anything project.

## Required Workflows

### 1. CI Pipeline (`.github/workflows/ci.yml`)

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

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
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Run linting
      run: |
        ruff check src tests
        black --check src tests
        mypy src
    
    - name: Run security checks
      run: |
        bandit -r src/ -f json -o bandit-report.json
        safety check --json --output safety-report.json
        detect-secrets scan --baseline .secrets.baseline
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml --cov-report=term-missing
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

### 2. Security Scanning (`.github/workflows/security.yml`)

```yaml
name: Security Scan

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6 AM UTC

permissions:
  contents: read
  security-events: write
  actions: read

jobs:
  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[security]
    
    - name: Run Bandit security scan
      run: |
        bandit -r src/ -f sarif -o bandit-results.sarif
    
    - name: Upload Bandit results to GitHub Security
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: bandit-results.sarif
    
    - name: Run Safety check
      run: |
        safety check --json --output safety-report.json
    
    - name: Check for secrets
      run: |
        detect-secrets scan --baseline .secrets.baseline --force-use-all-plugins
    
    - name: Upload security artifacts
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-reports
        path: |
          bandit-results.sarif
          safety-report.json
```

### 3. Build and Publish (`.github/workflows/build-and-publish.yml`)

```yaml
name: Build and Publish

on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      publish:
        description: 'Publish to PyPI'
        required: true
        default: 'false'
        type: choice
        options:
        - 'true'
        - 'false'

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Check package
      run: twine check dist/*
    
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/
    
    - name: Publish to Test PyPI
      if: github.event_name == 'workflow_dispatch' && github.event.inputs.publish == 'true'
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.TEST_PYPI_API_TOKEN }}
      run: |
        twine upload --repository testpypi dist/*
    
    - name: Publish to PyPI
      if: github.event_name == 'release'
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: |
        twine upload dist/*
```

### 4. Docker Build (`.github/workflows/docker.yml`)

```yaml
name: Docker Build

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
    - name: Checkout
      uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      if: github.event_name != 'pull_request'
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

## Setup Instructions

### Prerequisites
1. Repository admin access
2. GitHub Actions enabled for the repository
3. Required secrets configured (see below)

### Required Secrets
Configure these in GitHub Settings > Secrets and variables > Actions:

- `PYPI_API_TOKEN`: PyPI API token for publishing packages
- `TEST_PYPI_API_TOKEN`: Test PyPI API token for testing
- `CODECOV_TOKEN`: Codecov token for coverage reporting (optional)

### Installation Steps

1. **Create workflow directory:**
   ```bash
   mkdir -p .github/workflows
   ```

2. **Copy workflow files:**
   Copy each workflow YAML content above into corresponding files in `.github/workflows/`

3. **Configure branch protection:**
   - Go to Settings > Branches
   - Add branch protection rule for `main`
   - Require status checks to pass before merging
   - Require CI workflow to pass

4. **Test the workflows:**
   - Create a test PR to verify CI workflow
   - Check security scan results in Security tab
   - Verify Docker builds are working

## Workflow Features

### CI Pipeline
- Multi-version Python testing (3.10, 3.11, 3.12)
- Dependency caching for faster builds
- Code quality checks (linting, formatting, type checking)
- Security scanning (Bandit, Safety, secrets detection)
- Test coverage reporting with Codecov integration

### Security Scanning
- Automated SARIF-compliant vulnerability scanning
- Weekly scheduled security scans
- GitHub Security tab integration
- Artifact upload for security reports

### Build & Publish
- Automated PyPI publishing on releases
- Manual publishing with workflow dispatch
- Test PyPI support for pre-release testing
- Package validation before publishing

### Docker Build
- Multi-platform container builds (AMD64, ARM64)
- GitHub Container Registry integration
- Build caching for performance
- Semantic versioning support

## Maintenance

### Regular Updates
- Update action versions quarterly
- Review security scan results weekly
- Monitor build performance and optimize caching
- Update Python versions as new releases become available

### Troubleshooting
- Check workflow logs in Actions tab
- Verify secrets are correctly configured
- Ensure branch protection rules don't conflict
- Review security scan false positives

---

**Note**: These workflows implement enterprise-grade CI/CD practices with comprehensive security scanning, multi-platform builds, and automated publishing capabilities.