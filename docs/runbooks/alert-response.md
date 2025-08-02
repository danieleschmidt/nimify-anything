# Alert Response Runbook

## Overview

This runbook provides specific response procedures for each alert configured in the Nimify Anything monitoring system.

## Alert Categories

### 🔴 Critical Alerts (SEV-1)
- **ServiceDown**: Complete service unavailability
- **HighErrorRate**: Error rate > 5% for > 2 minutes
- **DatabaseDown**: Database connectivity issues
- **GPUUnavailable**: All GPUs offline

### 🟡 Warning Alerts (SEV-2)  
- **HighLatency**: P95 latency > 1 second
- **HighMemoryUsage**: Memory usage > 90%
- **HighDiskUsage**: Disk usage > 85%
- **ModelLoadFailure**: Model loading issues

### 🟢 Info Alerts (SEV-3)
- **PodRestart**: Pod restart frequency high
- **SlowQueries**: Database query performance
- **CacheHitRatelow**: Cache performance degraded

## Critical Alert Responses

### ServiceDown

**Alert**: `up{job="nimify-api"} == 0`

```bash
# Immediate Assessment
kubectl get pods -n nimify-system
kubectl get nodes
kubectl get events -n nimify-system --sort-by='.lastTimestamp' | tail -10

# Quick Fixes
# 1. Check if pods are running
kubectl get pods -n nimify-system -l app=nimify-api

# 2. If no pods running, check deployment
kubectl describe deployment nimify-api -n nimify-system

# 3. Check recent changes
kubectl rollout history deployment/nimify-api -n nimify-system

# 4. Rollback if recent deployment
kubectl rollout undo deployment/nimify-api -n nimify-system

# 5. Scale up if scaled down
kubectl scale deployment nimify-api --replicas=3 -n nimify-system
```

**Escalation**: Engineering Manager after 15 minutes

### HighErrorRate

**Alert**: `rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05`

```bash
# Investigation
# 1. Check specific error types
kubectl logs -f deployment/nimify-api -n nimify-system | grep ERROR

# 2. Check recent deployments
kubectl rollout history deployment/nimify-api -n nimify-system

# 3. Check dependencies
kubectl get pods -n triton-system
kubectl logs triton-server -n triton-system | tail -50

# 4. Check resource usage
kubectl top pods -n nimify-system

# Quick Fixes
# 1. Rollback recent deployment
kubectl rollout undo deployment/nimify-api -n nimify-system

# 2. Restart pods if memory leak
kubectl rollout restart deployment/nimify-api -n nimify-system

# 3. Scale up if resource constrained
kubectl scale deployment nimify-api --replicas=5 -n nimify-system

# 4. Check and restart dependencies
kubectl rollout restart deployment/triton-server -n triton-system
```

**Escalation**: Engineering team lead after 10 minutes

### DatabaseDown

**Alert**: `up{job="postgres"} == 0`

```bash
# Investigation
kubectl get pods -n database
kubectl describe pod postgres-0 -n database
kubectl logs postgres-0 -n database

# Check storage
kubectl get pvc -n database
kubectl describe pvc postgres-storage -n database

# Check if StatefulSet issue
kubectl get statefulset -n database
kubectl describe statefulset postgres -n database

# Quick Fixes
# 1. Restart database pod
kubectl delete pod postgres-0 -n database

# 2. Check and fix storage issues
kubectl describe pvc postgres-storage -n database

# 3. If PVC issue, check storage class
kubectl get storageclass
kubectl describe storageclass <storage-class-name>

# 4. Emergency: Use database backup
# (Follow backup-restore runbook)
```

**Escalation**: Database administrator immediately

### GPUUnavailable

**Alert**: `nvidia_gpu_count == 0`

```bash
# Investigation
# 1. Check GPU availability on nodes
kubectl describe nodes | grep nvidia.com/gpu

# 2. Check NVIDIA device plugin
kubectl get pods -n kube-system | grep nvidia-device-plugin
kubectl logs nvidia-device-plugin-daemonset -n kube-system

# 3. Check driver status
kubectl exec -it <any-pod> -- nvidia-smi

# 4. Check NVIDIA runtime
kubectl describe nodes | grep -A 10 "Container Runtime"

# Quick Fixes
# 1. Restart NVIDIA device plugin
kubectl delete pods -n kube-system -l name=nvidia-device-plugin-ds

# 2. Restart kubelet on nodes (if accessible)
# ssh to node and run: sudo systemctl restart kubelet

# 3. Check and restart containerd/docker
# ssh to node and run: sudo systemctl restart containerd

# 4. Fallback to CPU-only mode
kubectl patch deployment nimify-api -n nimify-system -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"nimify-api","env":[{"name":"CUDA_VISIBLE_DEVICES","value":""}]}]}}}}'
```

**Escalation**: Infrastructure team immediately

## Warning Alert Responses

### HighLatency

**Alert**: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1`

```bash
# Investigation
# 1. Check current request patterns
kubectl logs -f deployment/nimify-api -n nimify-system | grep -E "(WARN|ERROR)"

# 2. Check resource usage
kubectl top pods -n nimify-system
kubectl top nodes

# 3. Check database performance
kubectl exec -it postgres-0 -n database -- \
  psql -U nimify -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# 4. Check model inference performance
kubectl logs triton-server -n triton-system | grep -i "inference time"

# Mitigation
# 1. Scale up API pods
kubectl scale deployment nimify-api --replicas=5 -n nimify-system

# 2. Enable request caching
kubectl patch configmap nimify-config -n nimify-system --patch \
  '{"data":{"ENABLE_CACHE":"true","CACHE_TTL":"300"}}'

# 3. Optimize batch sizes
kubectl exec -it triton-server -n triton-system -- \
  curl -X POST localhost:8000/v2/repository/models/<model>/load \
  -d '{"parameters":{"max_batch_size":32}}'

# 4. Monitor for improvement
watch "curl -s http://prometheus:9090/api/v1/query?query=histogram_quantile\\(0.95,rate\\(http_request_duration_seconds_bucket[5m]\\)\\) | jq '.data.result[0].value[1]'"
```

### HighMemoryUsage

**Alert**: `container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9`

```bash
# Investigation
# 1. Identify memory-heavy pods
kubectl top pods -n nimify-system --sort-by=memory

# 2. Check for memory leaks
kubectl exec -it <high-memory-pod> -n nimify-system -- \
  cat /proc/meminfo

# 3. Check application metrics
curl http://nimify-api:9090/metrics | grep -E "(memory|heap)"

# 4. Check for large models in memory
kubectl exec -it triton-server -n triton-system -- \
  curl localhost:8000/v2/models/ | jq '.[] | select(.state=="READY")'

# Mitigation
# 1. Restart high-memory pods
kubectl delete pod <high-memory-pod> -n nimify-system

# 2. Increase memory limits temporarily
kubectl patch deployment nimify-api -n nimify-system -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"nimify-api","resources":{"limits":{"memory":"4Gi"}}}]}}}}'

# 3. Enable memory optimization
kubectl patch configmap nimify-config -n nimify-system --patch \
  '{"data":{"MEMORY_OPTIMIZATION":"true","GC_THRESHOLD":"0.8"}}'

# 4. Unload unused models
kubectl exec -it triton-server -n triton-system -- \
  curl -X POST localhost:8000/v2/repository/models/<unused-model>/unload
```

### HighDiskUsage

**Alert**: `(node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes > 0.85`

```bash
# Investigation
# 1. Check disk usage by path
kubectl exec -it <any-pod> -n nimify-system -- df -h

# 2. Find large directories
kubectl exec -it <any-pod> -n nimify-system -- \
  du -sh /cache/* /models/* /logs/* | sort -hr | head -20

# 3. Check log sizes
kubectl exec -it <any-pod> -n nimify-system -- \
  find /var/log -name "*.log" -exec ls -lh {} \; | sort -k5 -hr

# 4. Check temp files
kubectl exec -it <any-pod> -n nimify-system -- \
  find /tmp -type f -size +100M -exec ls -lh {} \;

# Mitigation
# 1. Clean old logs
kubectl exec -it <any-pod> -n nimify-system -- \
  find /var/log -name "*.log" -mtime +7 -delete

# 2. Clean cache files
kubectl exec -it <any-pod> -n nimify-system -- \
  find /cache -type f -mtime +3 -delete

# 3. Clean old models
kubectl exec -it <any-pod> -n nimify-system -- \
  find /models -name "*.bak" -o -name "*.tmp" -delete

# 4. Emergency: Expand storage
kubectl patch pvc <pvc-name> -n nimify-system -p \
  '{"spec":{"resources":{"requests":{"storage":"200Gi"}}}}'
```

## Info Alert Responses

### PodRestart

**Alert**: `rate(kube_pod_container_status_restarts_total[15m]) > 0.1`

```bash
# Investigation
# 1. Check restart reasons
kubectl describe pod <restarting-pod> -n nimify-system | grep -A 10 "Last State"

# 2. Check exit codes
kubectl get pods -n nimify-system -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.containerStatuses[*].lastState.terminated.exitCode}{"\n"}{end}'

# 3. Check resource constraints
kubectl describe pod <restarting-pod> -n nimify-system | grep -A 5 "Resource"

# 4. Check recent config changes
kubectl diff configmap nimify-config -n nimify-system

# Actions
# 1. If OOMKilled, increase memory
kubectl patch deployment <deployment> -n nimify-system -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"<container>","resources":{"limits":{"memory":"2Gi"}}}]}}}}'

# 2. If liveness probe failing, adjust probe
kubectl patch deployment <deployment> -n nimify-system -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"<container>","livenessProbe":{"initialDelaySeconds":60}}]}}}}'

# 3. If configuration issue, fix config
kubectl edit configmap nimify-config -n nimify-system
```

### ModelLoadFailure

**Alert**: `triton_model_load_failures_total > 0`

```bash
# Investigation
# 1. Check Triton logs
kubectl logs triton-server -n triton-system | grep -i "load.*fail\|error.*model"

# 2. Check model repository
kubectl exec -it triton-server -n triton-system -- ls -la /models/

# 3. Check model configuration
kubectl exec -it triton-server -n triton-system -- \
  cat /models/<model-name>/config.pbtxt

# 4. Check model format and size
kubectl exec -it triton-server -n triton-system -- \
  file /models/<model-name>/1/model.*

# Actions
# 1. Reload specific model
kubectl exec -it triton-server -n triton-system -- \
  curl -X POST localhost:8000/v2/repository/models/<model-name>/load

# 2. Check and fix model permissions
kubectl exec -it triton-server -n triton-system -- \
  chmod 644 /models/<model-name>/1/model.*

# 3. Validate model configuration
kubectl exec -it triton-server -n triton-system -- \
  /opt/tritonserver/bin/tritonserver --model-repository=/models --strict-model-config --exit-on-error

# 4. Restart Triton if multiple failures
kubectl rollout restart deployment/triton-server -n triton-system
```

## Alert Acknowledgment Process

### Using AlertManager API

```bash
# Acknowledge alert
curl -X POST http://alertmanager:9093/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d '[{
    "labels": {
      "alertname": "ServiceDown",
      "instance": "nimify-api:8000"
    },
    "annotations": {
      "comment": "Investigating - scaling up replicas"
    },
    "generatorURL": "http://prometheus:9090/graph"
  }]'

# Silence alert for 1 hour
curl -X POST http://alertmanager:9093/api/v1/silences \
  -H "Content-Type: application/json" \
  -d '{
    "matchers": [
      {
        "name": "alertname",
        "value": "ServiceDown",
        "isRegex": false
      }
    ],
    "startsAt": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'",
    "endsAt": "'$(date -u -d '+1 hour' +%Y-%m-%dT%H:%M:%S.%3NZ)'",
    "createdBy": "oncall-engineer",
    "comment": "Planned maintenance - applying fixes"
  }'
```

### Communication Templates

#### Slack Update Template
```
🚨 **ALERT**: <AlertName>
📊 **Severity**: <SEV-Level>
⏰ **Started**: <Timestamp>
🔍 **Status**: Investigating
👤 **Assigned**: @<engineer>

**Actions taken**:
- [ ] Initial assessment completed
- [ ] Mitigation steps in progress
- [ ] Root cause investigation ongoing

**ETA for resolution**: <estimated-time>
```

#### Status Page Update Template
```
**Investigating** - We are currently investigating elevated error rates affecting our inference API. Users may experience increased response times or failures when making inference requests. We are actively working to resolve this issue.

**Update** - We have identified the root cause as a dependency issue and are implementing a fix. Service should be restored within 30 minutes.

**Resolved** - The issue has been resolved. All services are operating normally. We will publish a post-mortem within 24 hours.
```

## Monitoring Tools Quick Reference

### Prometheus Queries

```promql
# Service health
up{job="nimify-api"}

# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Latency percentiles
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Memory usage
container_memory_usage_bytes{pod=~"nimify-.*"} / 1024/1024/1024

# CPU usage
rate(container_cpu_usage_seconds_total{pod=~"nimify-.*"}[5m])

# GPU utilization
nvidia_gpu_utilization_percent

# Disk space
(node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes
```

### Grafana Dashboard Links

- **Overview**: [grafana.company.com/d/nimify-overview]
- **API Metrics**: [grafana.company.com/d/nimify-api]
- **Infrastructure**: [grafana.company.com/d/nimify-infra]
- **Triton Server**: [grafana.company.com/d/triton-metrics]

---

**Remember**: Always acknowledge alerts promptly and communicate status updates regularly, even if investigation is ongoing.