# Service Outage Runbook

## Overview

This runbook covers the response procedures for complete or partial Nimify Anything service outages.

## Symptoms

- HTTP 5xx errors or connection failures
- Inference endpoints returning errors
- Health check endpoints failing
- Zero successful requests in monitoring dashboards
- User reports of service unavailability

## Immediate Actions (First 5 minutes)

### 1. Confirm the Outage

```bash
# Check service status
curl -I http://nimify-api:8000/health
kubectl get pods -n nimify-system

# Verify monitoring is working
curl http://prometheus:9090/-/healthy
```

### 2. Assess Scope

```bash
# Check all service instances
kubectl get pods -n nimify-system -o wide
kubectl get svc -n nimify-system

# Check upstream dependencies
kubectl get pods -n triton-system
kubectl get pods -n monitoring
```

### 3. Communicate Status

1. **Update status page** immediately
2. **Post in #incidents** Slack channel
3. **Page primary on-call** if not already alerted
4. **Start incident bridge** if SEV-1

## Investigation Phase (5-15 minutes)

### Check Infrastructure

```bash
# Kubernetes cluster health
kubectl get nodes
kubectl describe nodes | grep -E "Ready|Conditions"

# Check resource usage
kubectl top nodes
kubectl top pods -n nimify-system

# Check events
kubectl get events -n nimify-system --sort-by='.lastTimestamp'
```

### Check Application Health

```bash
# Pod status and logs
kubectl describe pod <failing-pod> -n nimify-system
kubectl logs -f <failing-pod> -n nimify-system --previous

# Check recent deployments
kubectl rollout history deployment/nimify-api -n nimify-system
kubectl describe deployment nimify-api -n nimify-system
```

### Check Dependencies

```bash
# Database connectivity
kubectl exec -it <api-pod> -n nimify-system -- \
  psql -h postgres -U nimify -c "SELECT 1;"

# NVIDIA Triton Server
kubectl get pods -n triton-system
kubectl logs -f triton-server -n triton-system

# Storage/PVC issues
kubectl get pv,pvc -n nimify-system
kubectl describe pvc model-storage -n nimify-system
```

### Check Network and Ingress

```bash
# Ingress controller
kubectl get ingress -n nimify-system
kubectl describe ingress nimify-ingress -n nimify-system

# Service endpoints
kubectl get endpoints -n nimify-system
kubectl describe svc nimify-api -n nimify-system

# Network policies
kubectl get networkpolicies -n nimify-system
```

## Common Resolution Scenarios

### Scenario 1: Pod Crash Loop

**Symptoms**: Pods restarting continuously

```bash
# Check crash reason
kubectl describe pod <pod-name> -n nimify-system
kubectl logs <pod-name> -n nimify-system --previous

# Common fixes:
# 1. Resource limits
kubectl patch deployment nimify-api -n nimify-system -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"nimify-api","resources":{"limits":{"memory":"2Gi"}}}]}}}}'

# 2. Configuration issues
kubectl get configmap nimify-config -n nimify-system -o yaml
kubectl edit configmap nimify-config -n nimify-system

# 3. Rollback to previous version
kubectl rollout undo deployment/nimify-api -n nimify-system
```

### Scenario 2: Resource Exhaustion

**Symptoms**: High CPU/memory usage, OOMKilled pods

```bash
# Check resource usage
kubectl top pods -n nimify-system
kubectl describe nodes | grep -A 5 "Allocated resources"

# Immediate mitigation:
# 1. Scale down non-critical services
kubectl scale deployment nimify-worker --replicas=1 -n nimify-system

# 2. Add more nodes (if autoscaling available)
kubectl get nodes -l node.kubernetes.io/instance-type

# 3. Increase resource limits temporarily
kubectl patch deployment nimify-api -n nimify-system -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"nimify-api","resources":{"limits":{"cpu":"2","memory":"4Gi"}}}]}}}}'
```

### Scenario 3: Storage Issues

**Symptoms**: PVC mounting failures, disk space errors

```bash
# Check storage
kubectl get pv,pvc -n nimify-system
kubectl describe pvc model-storage -n nimify-system

# Check disk space
kubectl exec -it <pod-name> -n nimify-system -- df -h

# Solutions:
# 1. Clean up old models/cache
kubectl exec -it <pod-name> -n nimify-system -- \
  find /cache -type f -mtime +7 -delete

# 2. Expand PVC (if supported)
kubectl patch pvc model-storage -n nimify-system -p \
  '{"spec":{"resources":{"requests":{"storage":"200Gi"}}}}'

# 3. Create new PVC and migrate
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage-new
  namespace: nimify-system
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: 500Gi
EOF
```

### Scenario 4: Network/Ingress Issues

**Symptoms**: External traffic can't reach service

```bash
# Check ingress
kubectl get ingress -n nimify-system
kubectl describe ingress nimify-ingress -n nimify-system

# Check ingress controller
kubectl get pods -n ingress-nginx
kubectl logs -f nginx-ingress-controller -n ingress-nginx

# Test internal connectivity
kubectl run test-pod --image=curlimages/curl --rm -it -- \
  curl http://nimify-api.nimify-system.svc.cluster.local:8000/health

# Solutions:
# 1. Restart ingress controller
kubectl rollout restart deployment nginx-ingress-controller -n ingress-nginx

# 2. Check DNS resolution
kubectl run test-pod --image=busybox --rm -it -- \
  nslookup nimify-api.nimify-system.svc.cluster.local

# 3. Verify certificates (if using TLS)
kubectl get secrets -n nimify-system | grep tls
kubectl describe secret nimify-tls -n nimify-system
```

### Scenario 5: Dependency Failures

**Symptoms**: Service running but all requests failing

```bash
# Check NVIDIA Triton Server
kubectl get pods -n triton-system
kubectl logs triton-server -n triton-system
kubectl exec -it triton-server -n triton-system -- \
  curl localhost:8000/v2/health/ready

# Check GPU resources
kubectl describe nodes | grep nvidia.com/gpu
kubectl get pods -o yaml | grep -A 10 -B 10 "nvidia.com/gpu"

# Check model loading
kubectl exec -it triton-server -n triton-system -- \
  curl localhost:8000/v2/models/

# Solutions:
# 1. Restart Triton server
kubectl rollout restart deployment triton-server -n triton-system

# 2. Check model repository
kubectl exec -it triton-server -n triton-system -- \
  ls -la /models/

# 3. Reload models
kubectl exec -it triton-server -n triton-system -- \
  curl -X POST localhost:8000/v2/repository/models/<model-name>/load
```

## Recovery Verification

### 1. Health Checks

```bash
# Service health
curl http://nimify-api:8000/health
curl http://nimify-api:8000/v1/models

# End-to-end test
curl -X POST http://nimify-api:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [1, 2, 3]}'
```

### 2. Monitoring Verification

```bash
# Check metrics are flowing
curl http://nimify-api:9090/metrics | head -20

# Verify Prometheus targets
curl http://prometheus:9090/api/v1/targets | jq '.data.activeTargets'

# Check key metrics
curl "http://prometheus:9090/api/v1/query?query=up{job=\"nimify-api\"}"
```

### 3. Load Testing

```bash
# Light load test
for i in {1..10}; do
  curl -X POST http://nimify-api:8000/v1/predict \
    -H "Content-Type: application/json" \
    -d '{"input": [1, 2, 3]}' &
done
wait
```

## Post-Incident Actions

### 1. Update Status

- Update status page to "Operational"
- Post resolution in #incidents
- Cancel any active pages

### 2. Immediate Documentation

```bash
# Capture current state
kubectl get all -n nimify-system > incident-$(date +%Y%m%d-%H%M)-state.txt
kubectl describe pods -n nimify-system > incident-$(date +%Y%m%d-%H%M)-pods.txt
kubectl logs deployment/nimify-api -n nimify-system > incident-$(date +%Y%m%d-%H%M)-logs.txt
```

### 3. Schedule Post-Mortem

- Create post-mortem document
- Schedule review meeting within 3 business days
- Assign action items for prevention

## Prevention Strategies

### Monitoring Improvements

```yaml
# Add more comprehensive health checks
- alert: NimifyAPIDown
  expr: up{job="nimify-api"} == 0
  for: 30s
  
- alert: NimifyHighErrorRate  
  expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.01
  for: 2m

- alert: NimifyHighLatency
  expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
  for: 5m
```

### Infrastructure Hardening

```yaml
# Resource quotas
apiVersion: v1
kind: ResourceQuota
metadata:
  name: nimify-quota
  namespace: nimify-system
spec:
  hard:
    requests.cpu: "4"
    requests.memory: 8Gi
    limits.cpu: "8"
    limits.memory: 16Gi

# Pod disruption budgets
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: nimify-api-pdb
  namespace: nimify-system
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: nimify-api
```

### Automated Recovery

```yaml
# Liveness and readiness probes
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 30
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
  failureThreshold: 2
```

## Escalation Criteria

**Escalate to Engineering Manager if:**
- Outage duration > 30 minutes
- Multiple failed resolution attempts
- Potential data loss identified
- Customer impact significant

**Escalate to VP Engineering if:**
- Outage duration > 2 hours  
- Security implications
- Major customer escalations
- Need for external resources

**Contact CEO/Legal if:**
- Data breach suspected
- SLA violations with penalties
- Regulatory implications
- Public relations impact

---

**Remember**: In a crisis, communication is as important as technical resolution. Keep stakeholders informed with regular updates even if there's no progress to report.