# CI/CD Workflow Setup Guide

This document provides comprehensive CI/CD workflow templates and setup instructions for the Nimify project.

## GitHub Actions Workflows

### 1. Main CI/CD Pipeline

Create `.github/workflows/ci-cd.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.10"
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    
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
        
    - name: Lint with ruff
      run: ruff check src tests
      
    - name: Type check with mypy
      run: mypy src
      
    - name: Test with pytest
      run: |
        pytest --cov=src --cov-report=xml --cov-report=term-missing
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety
        
    - name: Security scan with bandit
      run: bandit -r src/ -f json -o bandit-report.json
      
    - name: Dependency security check
      run: safety check --json --output safety-report.json
      
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
      
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    if: github.ref == 'refs/heads/develop'
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to staging
      run: |
        # Add staging deployment logic here
        echo "Deploying to staging environment"
        
  deploy-production:
    if: github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to production
      run: |
        # Add production deployment logic here
        echo "Deploying to production environment"
```

### 2. Security Scanning Workflow

Create `.github/workflows/security.yml`:

```yaml
name: Security Scan

on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM
  push:
    branches: [main]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
        
    - name: Container scan
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'ghcr.io/${{ github.repository }}:main'
        format: 'sarif'
        output: 'container-scan.sarif'
```

### 3. Release Workflow

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      packages: write
      
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
      
    - name: Upload to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
      
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: dist/*
        generate_release_notes: true
        body: |
          ## Changes
          
          See the [changelog](CHANGELOG.md) for detailed changes.
          
          ## Installation
          
          ```bash
          pip install nimify-anything==${{ github.ref_name }}
          ```
```

## GitLab CI Configuration

Create `.gitlab-ci.yml`:

```yaml
stages:
  - test
  - security
  - build
  - deploy

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

cache:
  paths:
    - .cache/pip/
    - .venv/

before_script:
  - python -m venv .venv
  - source .venv/bin/activate
  - pip install --upgrade pip
  - pip install -e .[dev,test]

test:
  stage: test
  image: python:3.10
  script:
    - ruff check src tests
    - mypy src
    - pytest --cov=src --cov-report=xml --cov-report=term
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

security:
  stage: security
  image: python:3.10
  script:
    - pip install bandit safety
    - bandit -r src/ -f json -o bandit-report.json
    - safety check --json --output safety-report.json
  artifacts:
    reports:
      sast: bandit-report.json
    paths:
      - safety-report.json

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main
    - develop

deploy-staging:
  stage: deploy
  image: alpine/helm:latest
  script:
    - helm upgrade --install nimify-staging ./helm/nimify
        --set image.tag=$CI_COMMIT_SHA
        --set environment=staging
  environment:
    name: staging
    url: https://nimify-staging.example.com
  only:
    - develop

deploy-production:
  stage: deploy
  image: alpine/helm:latest
  script:
    - helm upgrade --install nimify-production ./helm/nimify
        --set image.tag=$CI_COMMIT_SHA
        --set environment=production
  environment:
    name: production
    url: https://nimify.example.com
  when: manual
  only:
    - main
```

## Azure DevOps Pipeline

Create `azure-pipelines.yml`:

```yaml
trigger:
  branches:
    include:
      - main
      - develop

pool:
  vmImage: 'ubuntu-latest'

variables:
  pythonVersion: '3.10'
  containerRegistry: 'nimifyregistry'
  imageRepository: 'nimify-anything'

stages:
- stage: Test
  displayName: 'Test Stage'
  jobs:
  - job: TestJob
    displayName: 'Run Tests'
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '$(pythonVersion)'
      displayName: 'Use Python $(pythonVersion)'
      
    - script: |
        python -m pip install --upgrade pip
        pip install -e .[dev,test]
      displayName: 'Install dependencies'
      
    - script: |
        ruff check src tests
        mypy src
      displayName: 'Lint and type check'
      
    - script: |
        pytest --cov=src --cov-report=xml --cov-report=html
      displayName: 'Run tests'
      
    - task: PublishTestResults@2
      inputs:
        testResultsFiles: '**/test-*.xml'
        testRunTitle: 'Publish test results'
        
    - task: PublishCodeCoverageResults@1
      inputs:
        codeCoverageTool: Cobertura
        summaryFileLocation: '$(System.DefaultWorkingDirectory)/**/coverage.xml'

- stage: Security
  displayName: 'Security Scanning'
  dependsOn: Test
  jobs:
  - job: SecurityJob
    displayName: 'Security Scan'
    steps:
    - script: |
        pip install bandit safety
        bandit -r src/ -f json -o bandit-report.json
        safety check
      displayName: 'Security scanning'

- stage: Build
  displayName: 'Build and Push'
  dependsOn: [Test, Security]
  jobs:
  - job: BuildJob
    displayName: 'Build Docker Image'
    steps:
    - task: Docker@2
      displayName: 'Build and push Docker image'
      inputs:
        command: 'buildAndPush'
        repository: '$(imageRepository)'
        dockerfile: 'Dockerfile'
        containerRegistry: '$(containerRegistry)'
        tags: |
          $(Build.BuildId)
          latest
```

## Required Secrets and Variables

### GitHub Secrets
- `PYPI_API_TOKEN`: PyPI API token for package publishing
- `CODECOV_TOKEN`: Codecov token for coverage reporting
- Custom deployment secrets as needed

### GitLab Variables
- `CI_REGISTRY`: Container registry URL
- `CI_REGISTRY_USER`: Registry username
- `CI_REGISTRY_PASSWORD`: Registry password

### Azure DevOps Variables
- Service connections for container registries
- Variable groups for environment-specific settings

## Environment Setup

1. **Development Environment Protection**
   - No restrictions, auto-deploy from develop branch

2. **Staging Environment Protection**
   - Require pull request reviews
   - Auto-deploy from develop branch after tests pass

3. **Production Environment Protection**
   - Require manual approval
   - Deploy only from main branch
   - Require all status checks to pass

## Integration with External Services

### Monitoring Integration
- Prometheus metrics endpoint: `/metrics`
- Health check endpoint: `/health`
- Readiness probe: `/ready`

### Notification Webhooks
- Slack notifications for deployment status
- Email notifications for security alerts
- Discord integration for team updates

## Rollback Procedures

### Automatic Rollback Triggers
- Health check failures after deployment
- Error rate above 5% for 2 minutes
- P95 latency above 1 second for 5 minutes

### Manual Rollback Commands
```bash
# Kubernetes rollback
kubectl rollout undo deployment/nimify-service

# Helm rollback
helm rollback nimify-production

# Docker rollback
docker service update --rollback nimify-service
```