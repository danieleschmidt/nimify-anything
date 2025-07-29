# Production GitHub Actions Workflows

**Note**: These workflow files should be manually created in `.github/workflows/` directory due to repository permissions.

## 1. Comprehensive CI/CD Pipeline

**File**: `.github/workflows/ci.yml`

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM

env:
  PYTHON_VERSION: "3.10"
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev,test]
        
    - name: Lint with ruff
      run: ruff check src tests --format=github
      
    - name: Type check with mypy
      run: mypy src --show-error-codes
      
    - name: Check formatting with black
      run: black --check src tests
      
    - name: Test with pytest
      run: |
        pytest --cov=src --cov-report=xml --cov-report=term-missing --junitxml=junit.xml
        
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.python-version }}
        path: junit.xml
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: false
        verbose: true

  security:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install security tools
      run: |
        python -m pip install --upgrade pip
        pip install bandit[toml] safety detect-secrets
        
    - name: Security scan with bandit
      run: |
        bandit -r src/ -f json -o bandit-report.json || true
        bandit -r src/ -f txt
      
    - name: Dependency security check
      run: |
        safety check --json --output safety-report.json || true
        safety check
        
    - name: Secrets scan
      run: |
        detect-secrets scan --baseline .secrets.baseline
        
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'
        
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    permissions:
      contents: read
      packages: write
      
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Log in to Container Registry
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
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
          
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        platforms: linux/amd64,linux/arm64

  performance-test:
    needs: [test]
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev,test]
        
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --benchmark-json=benchmark.json
        
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
        comment-on-alert: true
        alert-threshold: '200%'
        fail-on-alert: true

  package-test:
    needs: [test, security]
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install build tools
      run: |
        python -m pip install --upgrade pip
        pip install build twine
        
    - name: Build package
      run: python -m build
      
    - name: Check package
      run: twine check dist/*
      
    - name: Test package installation
      run: |
        pip install dist/*.whl
        python -c "import nimify; print(nimify.__version__)"
```

## 2. Advanced Security Scanning

**File**: `.github/workflows/security-scan.yml`

```yaml
name: Security Scan

on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday at 6 AM
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  security-events: write

jobs:
  codeql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    
    strategy:
      fail-fast: false
      matrix:
        language: ['python']
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: ${{ matrix.language }}
        queries: security-and-quality
    
    - name: Autobuild
      uses: github/codeql-action/autobuild@v2
    
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
      with:
        category: "/language:${{ matrix.language }}"

  dependency-review:
    name: Dependency Review
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    
    steps:
    - name: Checkout Repository
      uses: actions/checkout@v4
    
    - name: Dependency Review
      uses: actions/dependency-review-action@v3
      with:
        fail-on-severity: moderate
        allow-dependencies-licenses: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC

  container-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Build image
      run: docker build -t nimify-test:latest .
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'nimify-test:latest'
        format: 'sarif'
        output: 'trivy-container-results.sarif'
    
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-container-results.sarif'
    
    - name: Run Grype vulnerability scanner
      uses: anchore/scan-action@v3
      id: grype
      with:
        image: "nimify-test:latest"
        fail-build: false
        severity-cutoff: medium
    
    - name: Upload Grype scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: ${{ steps.grype.outputs.sarif }}

  secrets-scan:
    name: Secrets Scan
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout Repository
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: TruffleHog OSS
      uses: trufflesecurity/trufflehog@main
      with:
        path: ./
        base: main
        head: HEAD
        extra_args: --debug --only-verified

  supply-chain:
    name: Supply Chain Security
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout Repository
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Install cyclonedx-bom
      run: pip install cyclonedx-bom
    
    - name: Generate SBOM
      run: |
        cyclonedx-py -o sbom.json .
        cyclonedx-py -o sbom.xml -f xml .
    
    - name: Upload SBOM
      uses: actions/upload-artifact@v3
      with:
        name: sbom
        path: |
          sbom.json
          sbom.xml
    
    - name: SLSA Provenance
      uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.7.0
      with:
        base64-subjects: ${{ hashFiles('sbom.json') }}
        
  license-scan:
    name: License Compliance
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout Repository
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Install pip-licenses
      run: pip install pip-licenses
    
    - name: Install dependencies
      run: pip install -e .
    
    - name: Check licenses
      run: |
        pip-licenses --format=json --output-file=licenses.json
        pip-licenses --format=markdown --output-file=licenses.md
        
        # Check for non-approved licenses
        pip-licenses --fail-on 'GPL-3.0;GPL-2.0;AGPL-3.0;LGPL-3.0'
    
    - name: Upload license report
      uses: actions/upload-artifact@v3
      with:
        name: license-report
        path: |
          licenses.json
          licenses.md

  notify:
    name: Security Notifications
    runs-on: ubuntu-latest
    needs: [codeql, container-scan, secrets-scan, supply-chain, license-scan]
    if: failure() && github.event_name == 'schedule'
    
    steps:
    - name: Notify security team
      uses: 8398a7/action-slack@v3
      with:
        status: failure
        channel: '#security-alerts'
        text: 'Security scan failed for ${{ github.repository }}'
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

## 3. Release Automation

**File**: `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: write
  packages: write
  id-token: write

jobs:
  release:
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
        
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
        
    - name: Build package
      run: python -m build
      
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
        
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Extract version
      id: version
      run: echo "VERSION=${GITHUB_REF#refs/tags/}" >> $GITHUB_OUTPUT
        
    - name: Build and push release image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          ghcr.io/${{ github.repository }}:${{ steps.version.outputs.VERSION }}
          ghcr.io/${{ github.repository }}:latest
        platforms: linux/amd64,linux/arm64
        cache-from: type=gha
        cache-to: type=gha,mode=max
        
    - name: Generate release notes
      id: release_notes
      run: |
        # Extract changes from CHANGELOG.md for this version
        if [ -f CHANGELOG.md ]; then
          # Get changes between current and previous version
          awk '/^## / { if (found) exit; if ($0 ~ /'"${GITHUB_REF#refs/tags/}"'/) found=1; next } found' CHANGELOG.md > release_notes.md
        else
          echo "No changelog found. See commit history for changes." > release_notes.md
        fi
        
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: dist/*
        body_path: release_notes.md
        generate_release_notes: true
        draft: false
        prerelease: ${{ contains(github.ref, 'alpha') || contains(github.ref, 'beta') || contains(github.ref, 'rc') }}
        
    - name: Update Docker Hub description
      uses: peter-evans/dockerhub-description@v3
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
        repository: nimify/anything
        readme-filepath: ./README.md
      continue-on-error: true
```

## Setup Instructions

1. **Create Workflow Files**: Manually create the above files in `.github/workflows/` directory
2. **Configure Repository Secrets**:
   - `PYPI_API_TOKEN` for PyPI publishing
   - `CODECOV_TOKEN` for coverage reporting (optional)
   - `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` for Docker Hub (optional)
   - `SLACK_WEBHOOK_URL` for security notifications (optional)

3. **Enable Branch Protection**: Configure branch protection rules in repository settings
4. **Configure Environments**: Set up staging and production environments with approval requirements

These workflows provide enterprise-grade CI/CD with comprehensive security scanning, automated releases, and performance monitoring.