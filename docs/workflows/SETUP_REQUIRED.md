# Manual Setup Required - GitHub Workflow Configuration

Due to GitHub App permission limitations, the following workflows need to be manually created from the templates provided in this repository.

## Required Permissions

The GitHub App needs the following permissions to automatically create workflows:
- **Contents**: Write (to create workflow files)
- **Actions**: Write (to manage workflow settings)
- **Pull Requests**: Write (for PR workflows)
- **Packages**: Write (for container registry access)

## Manual Setup Instructions

### 1. Create Workflow Directory

If it doesn't exist, create the `.github/workflows/` directory:

```bash
mkdir -p .github/workflows
```

### 2. Copy Workflow Templates

Copy the templates from `docs/workflows/templates/` to `.github/workflows/`:

```bash
# Main CI workflow
cp docs/workflows/templates/ci-comprehensive.yml .github/workflows/ci.yml

# Release automation
cp docs/workflows/templates/release-automation.yml .github/workflows/release.yml

# Performance monitoring
cp docs/workflows/templates/performance-monitoring.yml .github/workflows/performance.yml

# Security scanning
cp docs/workflows/templates/security-scan.yml .github/workflows/security.yml
```

### 3. Configure Secrets

Add the following secrets to your GitHub repository:

#### Required Secrets
- `NGC_API_KEY`: NVIDIA NGC API key for NIM access
- `DOCKER_USERNAME`: Docker registry username
- `DOCKER_PASSWORD`: Docker registry password/token

#### Optional Secrets
- `CODECOV_TOKEN`: For code coverage reporting
- `SONAR_TOKEN`: For SonarCloud integration
- `SLACK_WEBHOOK`: For Slack notifications
- `DISCORD_WEBHOOK`: For Discord notifications

#### AWS Secrets (if using AWS)
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_REGION`: AWS region

#### Google Cloud Secrets (if using GCP)
- `GCP_SA_KEY`: Google Cloud service account key (JSON)
- `GCP_PROJECT_ID`: Google Cloud project ID

### 4. Configure Branch Protection

Set up branch protection rules for `main` branch:

1. Go to Settings → Branches
2. Add branch protection rule for `main`
3. Enable:
   - Require status checks to pass before merging
   - Require branches to be up to date before merging
   - Required status checks:
     - `build`
     - `test`
     - `security-scan`
     - `lint`

### 5. Environment Configuration

Create environments for different deployment stages:

#### Development Environment
- Name: `development`
- Required reviewers: None
- Deployment branches: Any branch

#### Staging Environment
- Name: `staging`
- Required reviewers: 1
- Deployment branches: `main`, `develop`

#### Production Environment
- Name: `production`
- Required reviewers: 2
- Deployment branches: `main` only
- Wait timer: 10 minutes

## Workflow Overview

### CI Workflow (`ci.yml`)

Triggers:
- Push to any branch
- Pull requests to `main`

Jobs:
1. **Lint & Format**: Code quality checks
2. **Test**: Unit and integration tests
3. **Security**: Security scanning and vulnerability checks
4. **Build**: Container image building
5. **Deploy**: Deployment to staging (if main branch)

### Release Workflow (`release.yml`)

Triggers:
- Push to `main` branch (after CI passes)
- Manual dispatch

Jobs:
1. **Semantic Release**: Automated versioning and changelog
2. **Build & Push**: Production container images
3. **Deploy**: Production deployment
4. **Notify**: Success/failure notifications

### Performance Monitoring (`performance.yml`)

Triggers:
- Schedule: Daily at 2 AM UTC
- Manual dispatch
- After releases

Jobs:
1. **Benchmark**: Performance regression testing
2. **Load Test**: Stress testing scenarios
3. **Report**: Performance metrics and trends

### Security Scanning (`security.yml`)

Triggers:
- Schedule: Weekly
- Push to `main`
- Manual dispatch

Jobs:
1. **Dependency Scan**: Check for vulnerable dependencies
2. **Container Scan**: Scan built images
3. **Code Scan**: Static analysis security testing
4. **SBOM**: Generate software bill of materials

## Customization Guide

### Modifying CI Pipeline

Edit `.github/workflows/ci.yml`:

```yaml
# Add custom test commands
- name: Custom Tests
  run: |
    make custom-test
    pytest tests/custom/

# Add deployment steps
- name: Deploy to Custom Environment
  if: github.ref == 'refs/heads/main'
  run: |
    ./scripts/deploy-custom.sh
```

### Adding Notifications

Add notification steps:

```yaml
- name: Slack Notification
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    text: "CI pipeline failed for ${{ github.sha }}"
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### Custom Environments

Add environment-specific configurations:

```yaml
deploy-staging:
  environment: staging
  steps:
    - name: Deploy to Staging
      run: ./scripts/deploy.sh staging
      env:
        ENVIRONMENT: staging
        API_URL: https://api-staging.example.com
```

## Validation

After setup, validate the workflows:

### 1. Check Workflow Syntax

```bash
# Use GitHub CLI to validate
gh workflow list

# Check specific workflow
gh workflow view ci.yml
```

### 2. Test Workflows

1. Create a test branch
2. Make a small change
3. Create pull request
4. Verify all checks pass

### 3. Monitor Workflow Runs

1. Go to Actions tab in GitHub
2. Check recent workflow runs
3. Verify all jobs complete successfully

## Troubleshooting

### Common Issues

#### Workflow Not Triggering
- Check branch names match trigger conditions
- Verify workflow file syntax is valid
- Ensure proper indentation (YAML is whitespace-sensitive)

#### Permission Errors
- Check repository secrets are set
- Verify GitHub App permissions
- Ensure service account has required access

#### Build Failures
- Check Docker registry access
- Verify all required secrets are available
- Review build logs for specific errors

#### Test Failures
- Ensure test dependencies are installed
- Check for environment-specific issues
- Verify database/service connectivity

### Getting Help

1. **GitHub Actions Documentation**: https://docs.github.com/en/actions
2. **Community Forum**: https://github.community/
3. **Issue Tracker**: Create issue in this repository
4. **Discord Community**: Join our development Discord

## Maintenance

### Regular Tasks

1. **Update Dependencies**: Keep action versions current
2. **Review Logs**: Check for warnings or deprecated features
3. **Security Updates**: Update base images and tools
4. **Performance**: Monitor workflow execution times

### Monitoring Workflow Health

Set up monitoring for:
- Workflow success rates
- Execution duration trends
- Resource usage patterns
- Cost optimization opportunities

## Next Steps

After manual setup:

1. Test all workflows with sample changes
2. Configure monitoring and alerting
3. Set up deployment environments
4. Document team-specific procedures
5. Train team members on workflow usage

This manual setup is required due to GitHub API limitations, but provides the same functionality as automated setup would.