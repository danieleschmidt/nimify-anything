# Operational Runbooks

This document provides comprehensive runbooks for operating, monitoring, and troubleshooting Nimify services in production environments.

## Table of Contents

- [Emergency Response](#emergency-response)
- [Service Management](#service-management)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Performance Troubleshooting](#performance-troubleshooting)
- [Security Incident Response](#security-incident-response)
- [Disaster Recovery](#disaster-recovery)
- [Maintenance Procedures](#maintenance-procedures)

## Emergency Response

### Severity Classification

#### P0 - Critical (Response: Immediate)
- Complete service outage affecting all users
- Data corruption or loss
- Security breach with active exploitation
- GPU/hardware failures causing service unavailability

#### P1 - High (Response: 1 hour)
- Partial service degradation affecting >50% of users
- Significant performance degradation (>5x latency increase)
- Authentication/authorization failures
- Critical vulnerability requiring immediate patching

#### P2 - Medium (Response: 4 hours)
- Minor service degradation affecting <25% of users
- Performance issues affecting specific endpoints
- Non-critical security vulnerabilities
- Configuration issues with workarounds available

#### P3 - Low (Response: Next business day)
- Documentation issues
- Minor UI/UX problems
- Enhancement requests
- Non-urgent maintenance tasks

### Emergency Contacts

#### On-Call Rotation
- **Primary On-Call**: Check PagerDuty/Opsgenie for current assignment
- **Secondary On-Call**: Backup for primary escalation
- **Engineering Manager**: For resource allocation decisions
- **Security Team**: For security-related incidents

#### Communication Channels
- **Incident Channel**: `#incident-response` in Slack/Teams
- **Status Page**: `https://status.nimify.dev`
- **Emergency Hotline**: Internal emergency number
- **Customer Communication**: Support team notification process

### Initial Response Procedure

1. **Acknowledge Incident** (within 5 minutes)
   ```bash
   # Join incident bridge
   # Update status page
   # Notify relevant teams
   ```

2. **Assess Impact** (within 10 minutes)
   ```bash
   # Check service health dashboard
   kubectl get pods -n nimify-production
   kubectl top nodes
   curl -s https://api.nimify.dev/health | jq
   ```

3. **Implement Immediate Mitigation** (within 30 minutes)
   ```bash
   # Scale services if needed
   kubectl scale deployment nimify-service --replicas=10 -n nimify-production
   
   # Enable circuit breakers
   kubectl patch configmap nimify-config -n nimify-production --patch '{"data":{"circuit_breaker":"enabled"}}'
   
   # Rollback if recent deployment
   kubectl rollout undo deployment/nimify-service -n nimify-production
   ```

## Service Management

### Health Checks and Monitoring

#### Service Health Endpoints
```bash
# Application health
curl -s https://api.nimify.dev/health
# Expected: {"status": "healthy", "timestamp": "...", "version": "..."}

# Readiness probe
curl -s https://api.nimify.dev/ready
# Expected: {"status": "ready", "dependencies": {"database": "ok", "gpu": "ok"}}

# Liveness probe
curl -s https://api.nimify.dev/alive
# Expected: {"status": "alive", "uptime": 3600}

# Metrics endpoint
curl -s https://api.nimify.dev/metrics
# Expected: Prometheus metrics format
```

#### GPU Health Monitoring
```bash
# Check GPU utilization
nvidia-smi
nvidia-smi dmon -s pucvmet -d 5

# Check GPU memory
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv

# GPU health in Kubernetes
kubectl exec -it nimify-pod-xxx -- nvidia-smi

# TensorRT engine validation
kubectl exec -it nimify-pod-xxx -- /opt/tensorrt/bin/trtexec --loadEngine=model.trt --shapes=input:1x3x224x224
```

### Deployment Procedures

#### Blue-Green Deployment
```bash
# 1. Deploy to blue environment
helm upgrade nimify-blue ./helm/nimify \
  --set environment=blue \
  --set image.tag=v1.2.3 \
  --namespace nimify-production

# 2. Validate blue environment
kubectl get pods -l environment=blue -n nimify-production
curl -s https://blue.api.nimify.dev/health

# 3. Run smoke tests
./scripts/smoke-tests.sh blue.api.nimify.dev

# 4. Switch traffic (gradually)
kubectl patch service nimify-service -p '{"spec":{"selector":{"environment":"blue"}}}'

# 5. Monitor for 30 minutes
kubectl logs -f deployment/nimify-blue -n nimify-production

# 6. Decommission green if successful
kubectl scale deployment nimify-green --replicas=0 -n nimify-production
```

#### Canary Deployment
```bash
# 1. Deploy canary with 5% traffic
helm upgrade nimify-canary ./helm/nimify \
  --set environment=canary \
  --set image.tag=v1.2.3 \
  --set replicaCount=1 \
  --set trafficSplit=0.05 \
  --namespace nimify-production

# 2. Monitor error rates and latency
kubectl exec -it prometheus-pod -- promtool query instant \
  'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])'

# 3. Gradually increase traffic
# 5% -> 25% -> 50% -> 100%
helm upgrade nimify-canary ./helm/nimify \
  --set trafficSplit=0.25 \
  --namespace nimify-production

# 4. Full rollout if metrics are good
kubectl patch service nimify-service -p '{"spec":{"selector":{"environment":"canary"}}}'
```

### Scaling Operations

#### Horizontal Pod Autoscaling
```bash
# Check current HPA status
kubectl get hpa nimify-hpa -n nimify-production

# Manual scaling (emergency)
kubectl scale deployment nimify-service --replicas=20 -n nimify-production

# Update HPA limits
kubectl patch hpa nimify-hpa -p '{"spec":{"maxReplicas":50}}'

# Check scaling events
kubectl describe hpa nimify-hpa -n nimify-production
```

#### Vertical Scaling (Resource Adjustment)
```bash
# Update resource requests/limits
kubectl patch deployment nimify-service -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "nimify",
          "resources": {
            "requests": {"cpu": "2", "memory": "8Gi", "nvidia.com/gpu": "1"},
            "limits": {"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": "1"}
          }
        }]
      }
    }
  }
}'

# Restart pods to apply changes
kubectl rollout restart deployment/nimify-service -n nimify-production
```

## Monitoring and Alerting

### Key Metrics to Monitor

#### Application Metrics
- **Request Rate**: Requests per second
- **Error Rate**: 4xx/5xx response percentage
- **Response Time**: P50, P95, P99 latencies
- **Queue Depth**: Pending inference requests
- **Batch Size**: Average inference batch size
- **GPU Utilization**: GPU compute and memory usage

#### Infrastructure Metrics
- **CPU Utilization**: Node and pod CPU usage
- **Memory Usage**: Available memory and OOM events
- **Disk I/O**: Read/write operations and latency
- **Network**: Bandwidth utilization and packet loss
- **Container Health**: Pod restart count and status

### Alert Response Procedures

#### High Error Rate (>5% for 5 minutes)
```bash
# 1. Check recent deployments
kubectl rollout history deployment/nimify-service -n nimify-production

# 2. Examine error logs
kubectl logs -l app=nimify --since=10m -n nimify-production | grep -i error

# 3. Check downstream dependencies
curl -s https://api.nimify.dev/health | jq '.dependencies'

# 4. Consider rollback if recent deployment
kubectl rollout undo deployment/nimify-service -n nimify-production
```

#### High Latency (P95 >1000ms for 10 minutes)
```bash
# 1. Check GPU utilization
kubectl exec -it nimify-pod-xxx -- nvidia-smi

# 2. Examine batch processing
kubectl logs -l app=nimify --since=15m | grep -i "batch\|latency"

# 3. Check for memory pressure
kubectl top pods -n nimify-production --sort-by=memory

# 4. Scale up if needed
kubectl scale deployment nimify-service --replicas=$(($(kubectl get deployment nimify-service -o jsonpath='{.spec.replicas}') * 2))
```

#### GPU Out of Memory
```bash
# 1. Check GPU memory usage
kubectl exec -it nimify-pod-xxx -- nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 2. Reduce batch size temporarily
kubectl patch configmap nimify-config -p '{"data":{"max_batch_size":"16"}}'

# 3. Restart affected pods
kubectl delete pod -l app=nimify,gpu-memory-error=true

# 4. Consider model optimization
# Check if TensorRT optimization can reduce memory usage
```

### Dashboard Queries

#### Prometheus Queries
```promql
# Request rate
rate(nim_request_count_total[5m])

# Error rate
rate(nim_request_count_total{status=~"5.."}[5m]) / rate(nim_request_count_total[5m]) * 100

# P95 latency
histogram_quantile(0.95, rate(nim_request_duration_seconds_bucket[5m]))

# GPU utilization
nvidia_smi_utilization_gpu_ratio * 100

# Queue depth
nim_request_queue_size

# Memory usage
container_memory_usage_bytes{pod=~"nimify-.*"} / container_spec_memory_limit_bytes{pod=~"nimify-.*"} * 100
```

## Performance Troubleshooting

### Common Performance Issues

#### High Inference Latency
**Symptoms**: P95 latency >1000ms, slow response times
**Investigation Steps**:
```bash
# 1. Check GPU utilization
kubectl exec -it nimify-pod-xxx -- nvidia-smi

# 2. Analyze batch processing efficiency
kubectl logs -l app=nimify --since=30m | grep -E "batch_size|processing_time"

# 3. Check for CPU bottlenecks
kubectl top pods -n nimify-production --sort-by=cpu

# 4. Examine TensorRT engine optimization
kubectl exec -it nimify-pod-xxx -- cat /app/models/config.pbtxt
```

**Resolution**:
```bash
# Optimize batch size
kubectl patch configmap nimify-config -p '{"data":{"preferred_batch_sizes":"[8,16,32]"}}'

# Enable dynamic batching
kubectl patch configmap nimify-config -p '{"data":{"dynamic_batching":"true"}}'

# Scale horizontally if CPU-bound
kubectl scale deployment nimify-service --replicas=6
```

#### Memory Leaks
**Symptoms**: Gradually increasing memory usage, OOM kills
**Investigation Steps**:
```bash
# 1. Monitor memory usage over time
kubectl top pods -n nimify-production --sort-by=memory
kubectl describe pod nimify-pod-xxx | grep -A 10 "Memory"

# 2. Check for memory leaks in application logs
kubectl logs -l app=nimify --since=1h | grep -i "memory\|oom\|allocation"

# 3. Analyze heap dumps (if available)
kubectl exec -it nimify-pod-xxx -- python -c "import gc; print(len(gc.get_objects()))"
```

**Resolution**:
```bash
# Implement memory limits
kubectl patch deployment nimify-service -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "nimify",
          "resources": {"limits": {"memory": "8Gi"}}
        }]
      }
    }
  }
}'

# Enable garbage collection tuning
kubectl patch configmap nimify-config -p '{"data":{"python_gc_threshold":"700,10,10"}}'

# Restart pods with memory issues
kubectl delete pod -l app=nimify,memory-usage=high
```

### Performance Optimization

#### Model Optimization
```bash
# TensorRT engine optimization
kubectl exec -it nimify-pod-xxx -- trtexec \
  --onnx=model.onnx \
  --saveEngine=model_fp16.trt \
  --fp16 \
  --minShapes=input:1x3x224x224 \
  --optShapes=input:16x3x224x224 \
  --maxShapes=input:32x3x224x224

# Update model configuration
kubectl patch configmap model-config -p '{
  "data": {
    "optimization_level": "O3",
    "enable_fp16": "true",
    "max_workspace_size": "4294967296"
  }
}'
```

#### Batch Processing Optimization
```bash
# Configure optimal batch sizes based on GPU memory
kubectl patch configmap nimify-config -p '{
  "data": {
    "max_batch_size": "32",
    "preferred_batch_sizes": "[1,2,4,8,16,32]",
    "max_queue_delay_microseconds": "100000"
  }
}'

# Enable request batching
kubectl patch configmap nimify-config -p '{"data":{"dynamic_batching_enabled":"true"}}'
```

## Security Incident Response

### Security Alert Classification

#### Critical Security Incidents
- Active exploitation of vulnerabilities
- Unauthorized access to production systems
- Data breach or exfiltration
- Ransomware or malware infection
- Compromise of service accounts or credentials

#### High Priority Security Events
- Suspicious authentication patterns
- Privilege escalation attempts
- Unusual network traffic patterns
- Failed security scans
- Critical vulnerability disclosure

### Incident Response Procedures

#### Immediate Response (0-30 minutes)
```bash
# 1. Isolate affected systems
kubectl patch networkpolicy nimify-network-policy -p '{
  "spec": {
    "podSelector": {"matchLabels": {"security-incident": "true"}},
    "policyTypes": ["Ingress", "Egress"],
    "ingress": [],
    "egress": []
  }
}'

# 2. Preserve evidence
kubectl get events --sort-by='.firstTimestamp' -n nimify-production > incident-events.log
kubectl logs -l app=nimify --since=2h > incident-logs.log

# 3. Notify security team
# Send alert to security channel
# Update incident tracking system
```

#### Investigation Phase (30 minutes - 4 hours)
```bash
# 1. Analyze logs for IOCs (Indicators of Compromise)
kubectl logs -l app=nimify --since=24h | grep -E "(failed|error|unauthorized|suspicious)"

# 2. Check for lateral movement
kubectl get pods -o wide | grep -E "(running|pending)"
kubectl describe nodes | grep -A 5 "Conditions"

# 3. Validate integrity of running containers
kubectl exec -it nimify-pod-xxx -- sha256sum /app/main.py
kubectl exec -it nimify-pod-xxx -- find /app -type f -name "*.py" -exec sha256sum {} \;

# 4. Review recent configuration changes
kubectl get events --field-selector type=Normal --sort-by='.firstTimestamp'
git log --since="1 day ago" --oneline
```

#### Containment and Remediation
```bash
# 1. Patch vulnerabilities immediately
kubectl set image deployment/nimify-service nimify=nimify:security-patch-v1.2.4

# 2. Rotate compromised credentials
kubectl delete secret nimify-secrets
kubectl create secret generic nimify-secrets --from-env-file=new-secrets.env

# 3. Update security policies
kubectl apply -f security/updated-network-policies.yaml
kubectl apply -f security/updated-rbac.yaml

# 4. Scan for persistence mechanisms
kubectl get cronjobs,jobs -A
kubectl get serviceaccounts -A | grep -v default
```

### Post-Incident Activities

#### Evidence Collection
```bash
# 1. Container forensics
kubectl cp nimify-pod-xxx:/var/log/app.log ./evidence/app-logs-$(date +%Y%m%d-%H%M%S).log
kubectl exec -it nimify-pod-xxx -- ps aux > ./evidence/process-list.txt

# 2. Network traffic analysis
kubectl logs -l app=istio-proxy --since=24h > ./evidence/network-logs.log

# 3. System state capture
kubectl get pods,services,configmaps,secrets -o yaml > ./evidence/k8s-state.yaml
```

#### Root Cause Analysis
1. **Timeline Construction**: Create detailed timeline of events
2. **Attack Vector Analysis**: Determine how the incident occurred
3. **Impact Assessment**: Evaluate scope and severity of compromise
4. **Control Failures**: Identify which security controls failed
5. **Lessons Learned**: Document improvements for future prevention

## Disaster Recovery

### Backup Procedures

#### Database Backups
```bash
# Automated daily backups
kubectl create cronjob nimify-db-backup \
  --image=postgres:13 \
  --schedule="0 2 * * *" \
  --restart=OnFailure \
  -- /bin/bash -c "pg_dump -h postgres-service nimify_db | gzip > /backups/nimify-$(date +%Y%m%d).sql.gz"

# Manual backup
kubectl exec -it postgres-pod -- pg_dump nimify_db > nimify-backup-$(date +%Y%m%d).sql
```

#### Configuration Backups
```bash
# Backup Kubernetes configurations
kubectl get all,configmaps,secrets -o yaml > k8s-backup-$(date +%Y%m%d).yaml

# Backup Helm values
helm get values nimify > helm-values-backup-$(date +%Y%m%d).yaml

# Backup certificates and secrets
kubectl get secrets -o yaml > secrets-backup-$(date +%Y%m%d).yaml
```

#### Model and Data Backups
```bash
# Backup trained models
kubectl exec -it nimify-pod-xxx -- tar -czf /tmp/models-backup.tar.gz /app/models/
kubectl cp nimify-pod-xxx:/tmp/models-backup.tar.gz ./backups/models-$(date +%Y%m%d).tar.gz

# Backup inference data (if applicable)
kubectl exec -it nimify-pod-xxx -- tar -czf /tmp/data-backup.tar.gz /app/data/
```

### Recovery Procedures

#### Service Recovery
```bash
# 1. Restore from healthy backup
kubectl apply -f k8s-backup-YYYYMMDD.yaml

# 2. Restore database
kubectl exec -it postgres-pod -- psql -c "DROP DATABASE IF EXISTS nimify_db;"
kubectl exec -it postgres-pod -- psql -c "CREATE DATABASE nimify_db;"
kubectl exec -i postgres-pod -- psql nimify_db < nimify-backup-YYYYMMDD.sql

# 3. Restore models
kubectl cp ./backups/models-YYYYMMDD.tar.gz nimify-pod-xxx:/tmp/
kubectl exec -it nimify-pod-xxx -- tar -xzf /tmp/models-backup.tar.gz -C /

# 4. Validate service health
kubectl rollout status deployment/nimify-service
curl -s https://api.nimify.dev/health
```

#### Multi-Region Failover
```bash
# 1. Activate secondary region
kubectl config use-context nimify-west
helm upgrade nimify-west ./helm/nimify --set primaryRegion=true

# 2. Update DNS routing
# Update load balancer configuration to point to west region

# 3. Verify failover
curl -s https://api.nimify.dev/health
# Should return healthy status from west region

# 4. Monitor application performance
kubectl logs -f deployment/nimify-service -n nimify-production
```

### Business Continuity

#### RTO/RPO Targets
- **Recovery Time Objective (RTO)**: 4 hours for complete service restoration
- **Recovery Point Objective (RPO)**: 1 hour maximum data loss
- **Availability Target**: 99.9% uptime (8.77 hours downtime/year)

#### Communication Plan
1. **Internal Stakeholders**: Engineering, Product, Executive teams
2. **External Communication**: Customer notifications, status page updates
3. **Regulatory Reporting**: Compliance team notification for required disclosures
4. **Media Relations**: PR team coordination for significant outages

## Maintenance Procedures

### Scheduled Maintenance

#### Monthly Maintenance Window
**Schedule**: First Sunday of each month, 2-6 AM UTC
**Duration**: 4 hours maximum

**Pre-maintenance Checklist**:
```bash
# 1. Backup current state
./scripts/backup-production.sh

# 2. Verify rollback procedures
helm history nimify
kubectl rollout history deployment/nimify-service

# 3. Prepare maintenance runbook
# Review planned changes and rollback procedures

# 4. Notify stakeholders
# Send maintenance notification 48 hours in advance
```

**Maintenance Procedure**:
```bash
# 1. Enable maintenance mode
kubectl patch configmap nimify-config -p '{"data":{"maintenance_mode":"true"}}'

# 2. Drain traffic gradually
kubectl scale deployment nimify-service --replicas=1

# 3. Perform maintenance tasks
# - OS updates
# - Security patches
# - Database maintenance
# - Certificate renewals

# 4. Validate changes
./scripts/smoke-tests.sh
kubectl get pods -n nimify-production

# 5. Restore service
kubectl patch configmap nimify-config -p '{"data":{"maintenance_mode":"false"}}'
kubectl scale deployment nimify-service --replicas=3

# 6. Monitor for issues
kubectl logs -f deployment/nimify-service --since=30m
```

#### Emergency Maintenance
**Trigger Conditions**:
- Critical security vulnerabilities (CVSS >9.0)
- System stability issues
- Data corruption risks
- Regulatory compliance requirements

**Emergency Procedure**:
```bash
# 1. Immediate notification
# Send emergency maintenance alert

# 2. Implement emergency patch
kubectl set image deployment/nimify-service nimify=nimify:emergency-patch

# 3. Monitor rollout
kubectl rollout status deployment/nimify-service

# 4. Validate fix
./scripts/security-validation.sh
curl -s https://api.nimify.dev/health

# 5. Post-emergency review
# Document incident and improve procedures
```

### Certificate Management
```bash
# Check certificate expiration
kubectl get secrets -o json | jq -r '.items[] | select(.type == "kubernetes.io/tls") | .metadata.name + ": " + .data."tls.crt"' | while IFS=: read name cert; do
  echo "Certificate: $name"
  echo "$cert" | base64 -d | openssl x509 -noout -enddate
done

# Renew certificates (Let's Encrypt)
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: nimify-tls
spec:
  secretName: nimify-tls-secret
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - api.nimify.dev
  - dashboard.nimify.dev
EOF

# Verify certificate renewal
kubectl describe certificate nimify-tls
```

### Database Maintenance
```bash
# Vacuum and analyze PostgreSQL
kubectl exec -it postgres-pod -- psql nimify_db -c "VACUUM ANALYZE;"

# Check database size and growth
kubectl exec -it postgres-pod -- psql nimify_db -c "
  SELECT schemaname,tablename,attname,n_distinct,correlation 
  FROM pg_stats WHERE schemaname = 'public' ORDER BY n_distinct DESC;
"

# Archive old logs
kubectl exec -it postgres-pod -- psql nimify_db -c "
  DELETE FROM access_logs WHERE created_at < NOW() - INTERVAL '90 days';
"
```

---

## Support and Escalation

### Contact Information
- **Primary On-Call**: Check current rotation in PagerDuty
- **Engineering Manager**: eng-manager@nimify.dev
- **Security Team**: security@nimify.dev
- **Customer Support**: support@nimify.dev

### Escalation Matrix
1. **L1 Support**: Initial triage and basic troubleshooting
2. **L2 Engineering**: Complex technical issues and service recovery
3. **L3 Architect**: System design issues and major incidents
4. **Management**: Resource allocation and executive decisions

### Documentation Updates
This runbook should be reviewed and updated:
- After each major incident
- Monthly during maintenance windows
- When new services or features are deployed
- Annually for comprehensive review

For questions or improvements to this runbook, please create an issue in the repository or contact the Platform Engineering team.