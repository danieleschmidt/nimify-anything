# Operational Runbooks

This directory contains operational runbooks for managing Nimify Anything in production environments.

## Available Runbooks

### Incident Response
- [Service Outage](./service-outage.md) - Steps for handling service outages
- [Performance Degradation](./performance-degradation.md) - Troubleshooting performance issues  
- [Security Incident](./security-incident.md) - Security incident response procedures

### Maintenance
- [Deployment Procedures](./deployment.md) - Safe deployment practices
- [Backup and Restore](./backup-restore.md) - Data backup and recovery procedures
- [Scaling Operations](./scaling.md) - Horizontal and vertical scaling procedures

### Monitoring and Alerting
- [Alert Response](./alert-response.md) - How to respond to specific alerts
- [Metrics Troubleshooting](./metrics-troubleshooting.md) - Debugging monitoring issues
- [Log Analysis](./log-analysis.md) - Log investigation procedures

### Troubleshooting
- [Common Issues](./common-issues.md) - Frequently encountered problems and solutions
- [GPU Issues](./gpu-troubleshooting.md) - NVIDIA GPU and CUDA troubleshooting
- [Kubernetes Issues](./k8s-troubleshooting.md) - Kubernetes-specific problems

## Runbook Structure

Each runbook follows a standard structure:

```markdown
# Title

## Overview
Brief description of the scenario or issue

## Symptoms
How to identify this issue

## Immediate Actions
Critical first steps to take

## Investigation
How to gather more information

## Resolution
Step-by-step resolution procedures

## Prevention
How to prevent this issue in the future

## Escalation
When and how to escalate
```

## Emergency Contacts

### Primary On-Call
- **Engineering**: [oncall-engineering@company.com]
- **Operations**: [oncall-ops@company.com]
- **Security**: [security-incidents@company.com]

### Escalation
- **Engineering Manager**: [eng-manager@company.com]
- **VP Engineering**: [vp-eng@company.com]
- **CTO**: [cto@company.com]

## Communication Channels

### Status Updates
- **Status Page**: [status.company.com]
- **Slack**: #incidents
- **PagerDuty**: [pagerduty-service-url]

### Internal Communication
- **Engineering**: #engineering
- **Operations**: #ops  
- **Customer Success**: #customer-success

## Severity Levels

### SEV-1 (Critical)
- Complete service outage
- Data loss or corruption
- Security breach
- **Response Time**: 15 minutes
- **Resolution Target**: 1 hour

### SEV-2 (High)
- Significant functionality impaired
- Performance severely degraded
- Some users affected
- **Response Time**: 30 minutes
- **Resolution Target**: 4 hours

### SEV-3 (Medium)
- Minor functionality issues
- Limited user impact
- Workarounds available
- **Response Time**: 2 hours
- **Resolution Target**: 1 business day

### SEV-4 (Low)
- Cosmetic issues
- Documentation problems
- Feature requests
- **Response Time**: 1 business day
- **Resolution Target**: 1 week

## Tools and Resources

### Monitoring
- **Prometheus**: [prometheus-url]
- **Grafana**: [grafana-url]  
- **AlertManager**: [alertmanager-url]

### Logs
- **Elasticsearch**: [elastic-url]
- **Kibana**: [kibana-url]
- **Fluentd**: [fluentd-config]

### Infrastructure
- **Kubernetes Dashboard**: [k8s-dashboard-url]
- **AWS Console**: [aws-console-url]
- **Docker Registry**: [registry-url]

### Documentation
- **Architecture Docs**: [/docs/ARCHITECTURE.md]
- **API Docs**: [api-docs-url]
- **Deployment Guide**: [/docs/deployment/]

## Quick Reference

### Common Commands

```bash
# Check service health
kubectl get pods -n nimify-system
kubectl describe pod <pod-name> -n nimify-system

# View logs
kubectl logs -f <pod-name> -n nimify-system
kubectl logs -f deployment/nimify-api -n nimify-system

# Scale deployment
kubectl scale deployment nimify-api --replicas=3 -n nimify-system

# Check metrics
curl http://nimify-api:9090/metrics
curl http://prometheus:9090/api/v1/query?query=up

# Test inference endpoint
curl -X POST http://nimify-api:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [1, 2, 3]}'
```

### Key Metrics to Monitor

```promql
# Service availability
up{job="nimify-api"}

# Request rate
rate(http_requests_total[5m])

# Error rate  
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Response time
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# GPU utilization
nvidia_gpu_utilization_percent

# Memory usage
container_memory_usage_bytes / container_spec_memory_limit_bytes
```

### Alert Conditions

```yaml
# High error rate
expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
for: 2m

# High response time
expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
for: 5m

# Service down
expr: up{job="nimify-api"} == 0
for: 1m

# High memory usage
expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
for: 5m
```

## Training and Knowledge Transfer

### New Team Member Onboarding
1. Review architecture documentation
2. Complete local development setup
3. Shadow incident response (observer mode)
4. Practice runbook procedures in staging
5. Complete on-call certification

### Regular Training
- Monthly incident response drills
- Quarterly disaster recovery tests
- Annual security incident simulations
- Ongoing monitoring and alerting training

## Continuous Improvement

### Post-Incident Reviews
- Conduct blameless post-mortems
- Document lessons learned
- Update runbooks based on findings
- Share knowledge across teams

### Runbook Maintenance
- Review runbooks quarterly
- Update based on system changes
- Test procedures in staging environments
- Gather feedback from on-call engineers

---

**Last Updated**: $(date)
**Next Review**: $(date -d "+3 months")
**Maintained By**: Platform Engineering Team