# nimify-anything Operations Runbook

## Emergency Procedures

### Service Outage
1. Check service status: `kubectl get pods -l app=nimify-anything`
2. Check recent deployments: `kubectl rollout history deployment/nimify-anything`
3. Rollback if needed: `kubectl rollout undo deployment/nimify-anything`
4. Check logs: `kubectl logs -l app=nimify-anything --tail=100`

### High Latency
1. Check current metrics in Grafana
2. Scale up replicas: `kubectl scale deployment nimify-anything --replicas=10`
3. Check resource utilization
4. Review recent changes

### High Error Rate
1. Check application logs
2. Review recent deployments
3. Check external dependencies
4. Consider rolling back

## Maintenance Procedures

### Scaling
```bash
# Scale up
kubectl scale deployment nimify-anything --replicas=10

# Scale down  
kubectl scale deployment nimify-anything --replicas=3
```

### Updates
```bash
# Rolling update
kubectl set image deployment/nimify-anything nimify-anything=new-image:tag

# Monitor rollout
kubectl rollout status deployment/nimify-anything
```

### Backup
- Model artifacts: Stored in container registry
- Configuration: Version controlled in Git
- Monitoring data: Prometheus retention policy

## Monitoring Alerts

### Critical Alerts
- Service Down: Immediate response required
- High Error Rate: Response within 5 minutes
- Certificate Expiry: Response within 24 hours

### Warning Alerts  
- High Latency: Response within 15 minutes
- Resource Utilization: Response within 30 minutes
- Disk Space: Response within 1 hour

## Contact Information

- On-call Engineer: PagerDuty rotation
- Platform Team: Slack #nim-platform
- Security Team: security@company.com
