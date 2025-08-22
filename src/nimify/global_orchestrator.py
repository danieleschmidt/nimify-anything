"""
Global Production Orchestrator for Nimify Anything

This module implements global production deployment and orchestration capabilities
with autonomous scaling, multi-region support, and advanced operational intelligence.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger(__name__)


class DeploymentRegion(Enum):
    """Supported deployment regions."""
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    AP_NORTHEAST_1 = "ap-northeast-1"


class DeploymentStatus(Enum):
    """Deployment status states."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    SCALING = "scaling"
    UPDATING = "updating"
    DRAINING = "draining"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class GlobalDeploymentConfig:
    """Configuration for global deployment."""
    
    service_name: str
    model_path: str
    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Regional configuration
    primary_region: DeploymentRegion = DeploymentRegion.US_EAST_1
    regions: List[DeploymentRegion] = field(default_factory=lambda: [
        DeploymentRegion.US_EAST_1,
        DeploymentRegion.US_WEST_2,
        DeploymentRegion.EU_WEST_1
    ])
    
    # Scaling configuration
    min_replicas_per_region: int = 2
    max_replicas_per_region: int = 20
    target_cpu_utilization: int = 70
    target_gpu_utilization: int = 80
    target_latency_p95: int = 100  # milliseconds
    
    # Traffic management
    traffic_splitting: Dict[str, float] = field(default_factory=lambda: {})
    canary_deployment_enabled: bool = True
    blue_green_deployment: bool = False
    
    # Performance configuration
    enable_auto_scaling: bool = True
    enable_predictive_scaling: bool = True
    enable_spot_instances: bool = False
    
    # Monitoring and observability
    enable_distributed_tracing: bool = True
    enable_structured_logging: bool = True
    metrics_collection_interval: int = 30  # seconds
    
    # Security configuration
    enable_network_policies: bool = True
    enable_pod_security_policies: bool = True
    enable_secrets_encryption: bool = True
    
    # Compliance
    data_residency_requirements: Dict[str, List[str]] = field(default_factory=dict)
    compliance_frameworks: List[str] = field(default_factory=lambda: ["SOC2", "GDPR"])


@dataclass
class RegionalDeployment:
    """Represents a deployment in a specific region."""
    
    deployment_id: str
    region: DeploymentRegion
    status: DeploymentStatus = DeploymentStatus.PENDING
    replicas: int = 2
    target_replicas: int = 2
    
    # Performance metrics
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    requests_per_second: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    error_rate: float = 0.0
    
    # Operational data
    last_update: float = field(default_factory=time.time)
    health_score: float = 1.0
    deployment_time: float = 0.0
    
    # Traffic data
    traffic_percentage: float = 0.0
    active_connections: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deployment_id": self.deployment_id,
            "region": self.region.value,
            "status": self.status.value,
            "replicas": self.replicas,
            "target_replicas": self.target_replicas,
            "cpu_utilization": self.cpu_utilization,
            "gpu_utilization": self.gpu_utilization,
            "memory_utilization": self.memory_utilization,
            "requests_per_second": self.requests_per_second,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99,
            "error_rate": self.error_rate,
            "health_score": self.health_score,
            "traffic_percentage": self.traffic_percentage,
            "active_connections": self.active_connections,
            "last_update": self.last_update
        }


class TrafficManager:
    """Manages traffic routing and load balancing."""
    
    def __init__(self):
        self.routing_rules: List[Dict[str, Any]] = []
        self.health_checks: Dict[str, bool] = {}
    
    async def calculate_optimal_routing(
        self, 
        deployments: List[RegionalDeployment],
        traffic_requirements: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate optimal traffic routing."""
        
        # Health-based filtering
        healthy_deployments = [
            d for d in deployments 
            if d.status == DeploymentStatus.ACTIVE and d.health_score > 0.8
        ]
        
        if not healthy_deployments:
            return {}
        
        # Performance-based weighting
        weights = {}
        total_weight = 0.0
        
        for deployment in healthy_deployments:
            # Calculate weight based on performance metrics
            latency_score = max(0.1, 1.0 - (deployment.latency_p95 / 200))
            utilization_score = max(0.1, 1.0 - deployment.cpu_utilization / 100)
            health_score = deployment.health_score
            
            # Combine scores
            weight = (latency_score * 0.4 + utilization_score * 0.3 + health_score * 0.3)
            weights[deployment.region.value] = weight
            total_weight += weight
        
        # Normalize to percentages
        routing = {}
        if total_weight > 0:
            for region, weight in weights.items():
                routing[region] = (weight / total_weight) * 100
        
        return routing
    
    async def apply_canary_deployment(
        self,
        stable_deployments: List[RegionalDeployment],
        canary_deployment: RegionalDeployment,
        canary_percentage: float = 10.0
    ) -> Dict[str, float]:
        """Apply canary deployment traffic split."""
        
        if not stable_deployments:
            return {canary_deployment.region.value: 100.0}
        
        # Calculate stable traffic distribution
        stable_routing = await self.calculate_optimal_routing(
            stable_deployments, 
            {"strategy": "performance_weighted"}
        )
        
        # Apply canary split
        canary_routing = {canary_deployment.region.value: canary_percentage}
        
        # Adjust stable routing
        stable_factor = (100.0 - canary_percentage) / 100.0
        for region, percentage in stable_routing.items():
            canary_routing[region] = percentage * stable_factor
        
        return canary_routing
    
    def generate_traffic_config(
        self, 
        routing: Dict[str, float],
        service_name: str
    ) -> Dict[str, Any]:
        """Generate traffic management configuration."""
        
        return {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {
                "name": f"{service_name}-traffic-split",
                "namespace": "nim-services"
            },
            "spec": {
                "hosts": [f"{service_name}.nim-services.svc.cluster.local"],
                "http": [{
                    "match": [{"headers": {"user-agent": {"regex": ".*"}}}],
                    "route": [
                        {
                            "destination": {
                                "host": f"{service_name}.nim-services.svc.cluster.local",
                                "subset": region.replace("-", "_")
                            },
                            "weight": int(percentage)
                        }
                        for region, percentage in routing.items()
                    ]
                }]
            }
        }


class AutoScalingEngine:
    """Advanced auto-scaling engine with predictive capabilities."""
    
    def __init__(self):
        self.scaling_history: List[Dict[str, Any]] = []
        self.prediction_model = None
    
    async def calculate_scaling_decision(
        self,
        deployment: RegionalDeployment,
        config: GlobalDeploymentConfig,
        historical_metrics: List[Dict[str, Any]]
    ) -> Tuple[int, str]:
        """Calculate scaling decision based on metrics and predictions."""
        
        current_replicas = deployment.replicas
        target_replicas = current_replicas
        reason = "no_scaling_needed"
        
        # Reactive scaling based on current metrics
        cpu_scale_factor = self._calculate_cpu_scale_factor(
            deployment.cpu_utilization, 
            config.target_cpu_utilization
        )
        
        gpu_scale_factor = self._calculate_gpu_scale_factor(
            deployment.gpu_utilization, 
            config.target_gpu_utilization
        )
        
        latency_scale_factor = self._calculate_latency_scale_factor(
            deployment.latency_p95, 
            config.target_latency_p95
        )
        
        # Use the most aggressive scaling factor
        scale_factors = [cpu_scale_factor, gpu_scale_factor, latency_scale_factor]
        max_scale_factor = max(scale_factors)
        min_scale_factor = min(scale_factors)
        
        if max_scale_factor > 1.2:  # Scale up
            target_replicas = min(
                config.max_replicas_per_region,
                int(current_replicas * max_scale_factor)
            )
            reason = f"scale_up_due_to_{'cpu' if cpu_scale_factor == max_scale_factor else 'gpu' if gpu_scale_factor == max_scale_factor else 'latency'}"
        
        elif min_scale_factor < 0.8 and all(f < 1.0 for f in scale_factors):  # Scale down
            target_replicas = max(
                config.min_replicas_per_region,
                int(current_replicas * min_scale_factor)
            )
            reason = "scale_down_due_to_low_utilization"
        
        # Predictive scaling (if enabled)
        if config.enable_predictive_scaling and historical_metrics:
            predicted_demand = await self._predict_demand(historical_metrics)
            if predicted_demand > current_replicas * 1.5:
                target_replicas = min(
                    config.max_replicas_per_region,
                    int(predicted_demand)
                )
                reason = "predictive_scale_up"
        
        return target_replicas, reason
    
    def _calculate_cpu_scale_factor(self, current_cpu: float, target_cpu: float) -> float:
        """Calculate CPU-based scale factor."""
        if current_cpu == 0:
            return 1.0
        return current_cpu / target_cpu
    
    def _calculate_gpu_scale_factor(self, current_gpu: float, target_gpu: float) -> float:
        """Calculate GPU-based scale factor."""
        if current_gpu == 0:
            return 1.0
        return current_gpu / target_gpu
    
    def _calculate_latency_scale_factor(self, current_latency: float, target_latency: float) -> float:
        """Calculate latency-based scale factor."""
        if current_latency == 0:
            return 1.0
        # If latency is too high, we need more replicas (scale up)
        if current_latency > target_latency:
            return current_latency / target_latency
        else:
            # If latency is much lower, we might scale down
            return max(0.7, target_latency / current_latency)
    
    async def _predict_demand(self, historical_metrics: List[Dict[str, Any]]) -> float:
        """Predict future demand based on historical patterns."""
        if len(historical_metrics) < 10:
            return 0.0
        
        # Simple trend analysis (in production, use more sophisticated ML models)
        recent_rps = [m.get("requests_per_second", 0) for m in historical_metrics[-10:]]
        
        if len(recent_rps) >= 2:
            trend = sum(recent_rps[-3:]) / 3 - sum(recent_rps[-6:-3]) / 3
            current_avg = sum(recent_rps[-3:]) / 3
            
            # Predict next period
            predicted_rps = current_avg + trend
            
            # Convert to required replicas (assuming each replica can handle 50 RPS)
            required_replicas = max(2, predicted_rps / 50)
            return required_replicas
        
        return 0.0


class GlobalOrchestrator:
    """Main global orchestrator for Nimify deployments."""
    
    def __init__(self, config: GlobalDeploymentConfig):
        self.config = config
        self.regional_deployments: Dict[str, RegionalDeployment] = {}
        self.traffic_manager = TrafficManager()
        self.autoscaling_engine = AutoScalingEngine()
        self.deployment_history: List[Dict[str, Any]] = []
        
        # Operational intelligence
        self.performance_history: List[Dict[str, Any]] = []
        self.anomaly_detected: bool = False
        self.last_optimization: float = 0.0
        
    async def deploy_globally(self) -> Dict[str, Any]:
        """Deploy service globally across all configured regions."""
        logger.info(f"Starting global deployment for {self.config.service_name}")
        
        deployment_results = {
            "deployment_id": self.config.deployment_id,
            "service_name": self.config.service_name,
            "total_regions": len(self.config.regions),
            "successful_deployments": 0,
            "failed_deployments": 0,
            "deployment_start_time": time.time(),
            "regional_results": {}
        }
        
        # Deploy to each region
        deployment_tasks = []
        for region in self.config.regions:
            task = self._deploy_to_region(region)
            deployment_tasks.append(task)
        
        # Execute deployments in parallel
        regional_results = await asyncio.gather(*deployment_tasks, return_exceptions=True)
        
        # Process results
        for region, result in zip(self.config.regions, regional_results):
            region_name = region.value
            
            if isinstance(result, Exception):
                logger.error(f"Deployment failed for {region_name}: {result}")
                deployment_results["failed_deployments"] += 1
                deployment_results["regional_results"][region_name] = {
                    "status": "failed",
                    "error": str(result)
                }
            else:
                logger.info(f"Deployment successful for {region_name}")
                deployment_results["successful_deployments"] += 1
                deployment_results["regional_results"][region_name] = result
                
                # Store regional deployment
                self.regional_deployments[region_name] = result["deployment"]
        
        deployment_results["deployment_end_time"] = time.time()
        deployment_results["deployment_duration"] = (
            deployment_results["deployment_end_time"] - 
            deployment_results["deployment_start_time"]
        )
        
        # Configure initial traffic routing
        if deployment_results["successful_deployments"] > 0:
            await self._configure_global_traffic()
        
        # Store deployment history
        self.deployment_history.append(deployment_results)
        
        return deployment_results
    
    async def _deploy_to_region(self, region: DeploymentRegion) -> Dict[str, Any]:
        """Deploy service to a specific region."""
        logger.info(f"Deploying to region {region.value}")
        
        # Simulate deployment process
        deployment_start = time.time()
        
        # Create regional deployment
        regional_deployment = RegionalDeployment(
            deployment_id=f"{self.config.deployment_id}-{region.value}",
            region=region,
            replicas=self.config.min_replicas_per_region,
            target_replicas=self.config.min_replicas_per_region
        )
        
        # Simulate deployment steps
        await asyncio.sleep(0.5)  # Simulate deployment time
        regional_deployment.status = DeploymentStatus.DEPLOYING
        
        # Generate deployment manifests
        manifests = self._generate_deployment_manifests(region, regional_deployment)
        
        await asyncio.sleep(1.0)  # Simulate deployment completion
        regional_deployment.status = DeploymentStatus.ACTIVE
        regional_deployment.deployment_time = time.time() - deployment_start
        
        # Initialize with realistic metrics
        regional_deployment.cpu_utilization = 45.0 + (hash(region.value) % 20)
        regional_deployment.gpu_utilization = 60.0 + (hash(region.value) % 25)
        regional_deployment.memory_utilization = 55.0 + (hash(region.value) % 30)
        regional_deployment.requests_per_second = 25.0 + (hash(region.value) % 40)
        regional_deployment.latency_p95 = 80.0 + (hash(region.value) % 30)
        regional_deployment.health_score = 0.9 + (hash(region.value) % 10) / 100
        
        return {
            "region": region.value,
            "status": "success",
            "deployment": regional_deployment,
            "manifests": manifests,
            "deployment_time": regional_deployment.deployment_time
        }
    
    def _generate_deployment_manifests(
        self, 
        region: DeploymentRegion, 
        deployment: RegionalDeployment
    ) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifests."""
        
        return {
            "deployment": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": f"{self.config.service_name}-{region.value}",
                    "namespace": "nim-services",
                    "labels": {
                        "app": self.config.service_name,
                        "region": region.value,
                        "deployment-id": deployment.deployment_id
                    }
                },
                "spec": {
                    "replicas": deployment.target_replicas,
                    "selector": {
                        "matchLabels": {
                            "app": self.config.service_name,
                            "region": region.value
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app": self.config.service_name,
                                "region": region.value
                            }
                        },
                        "spec": {
                            "containers": [{
                                "name": self.config.service_name,
                                "image": f"{self.config.service_name}:latest",
                                "ports": [{"containerPort": 8000}],
                                "resources": {
                                    "requests": {
                                        "cpu": "500m",
                                        "memory": "1Gi",
                                        "nvidia.com/gpu": "1"
                                    },
                                    "limits": {
                                        "cpu": "2",
                                        "memory": "4Gi",
                                        "nvidia.com/gpu": "1"
                                    }
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": 8000
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/ready",
                                        "port": 8000
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5
                                }
                            }]
                        }
                    }
                }
            },
            "service": {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": f"{self.config.service_name}-{region.value}",
                    "namespace": "nim-services"
                },
                "spec": {
                    "selector": {
                        "app": self.config.service_name,
                        "region": region.value
                    },
                    "ports": [{
                        "port": 80,
                        "targetPort": 8000
                    }],
                    "type": "LoadBalancer"
                }
            },
            "hpa": {
                "apiVersion": "autoscaling/v2",
                "kind": "HorizontalPodAutoscaler",
                "metadata": {
                    "name": f"{self.config.service_name}-{region.value}-hpa",
                    "namespace": "nim-services"
                },
                "spec": {
                    "scaleTargetRef": {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "name": f"{self.config.service_name}-{region.value}"
                    },
                    "minReplicas": self.config.min_replicas_per_region,
                    "maxReplicas": self.config.max_replicas_per_region,
                    "metrics": [{
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.config.target_cpu_utilization
                            }
                        }
                    }]
                }
            }
        }
    
    async def _configure_global_traffic(self) -> None:
        """Configure global traffic routing."""
        logger.info("Configuring global traffic routing")
        
        active_deployments = [
            d for d in self.regional_deployments.values() 
            if d.status == DeploymentStatus.ACTIVE
        ]
        
        if not active_deployments:
            logger.warning("No active deployments for traffic configuration")
            return
        
        # Calculate optimal routing
        routing = await self.traffic_manager.calculate_optimal_routing(
            active_deployments,
            {"strategy": "performance_weighted"}
        )
        
        # Generate traffic configuration
        traffic_config = self.traffic_manager.generate_traffic_config(
            routing,
            self.config.service_name
        )
        
        logger.info(f"Traffic routing configured: {routing}")
    
    async def monitor_and_optimize(self) -> Dict[str, Any]:
        """Continuous monitoring and optimization."""
        logger.info("Starting continuous monitoring and optimization")
        
        optimization_results = {
            "timestamp": time.time(),
            "monitored_regions": len(self.regional_deployments),
            "scaling_actions": 0,
            "traffic_adjustments": 0,
            "anomalies_detected": 0,
            "performance_improvements": []
        }
        
        # Update metrics for all deployments
        await self._update_deployment_metrics()
        
        # Check for scaling needs
        for region_name, deployment in self.regional_deployments.items():
            if deployment.status == DeploymentStatus.ACTIVE:
                target_replicas, reason = await self.autoscaling_engine.calculate_scaling_decision(
                    deployment,
                    self.config,
                    self.performance_history[-10:]  # Last 10 data points
                )
                
                if target_replicas != deployment.target_replicas:
                    await self._scale_deployment(deployment, target_replicas, reason)
                    optimization_results["scaling_actions"] += 1
        
        # Detect anomalies
        anomalies = await self._detect_anomalies()
        if anomalies:
            optimization_results["anomalies_detected"] = len(anomalies)
            await self._handle_anomalies(anomalies)
        
        # Optimize traffic routing
        traffic_optimized = await self._optimize_traffic_routing()
        if traffic_optimized:
            optimization_results["traffic_adjustments"] = 1
        
        # Store performance data
        current_metrics = {
            "timestamp": time.time(),
            "regional_metrics": {
                region: deployment.to_dict()
                for region, deployment in self.regional_deployments.items()
            }
        }
        self.performance_history.append(current_metrics)
        
        # Keep only recent history (last 100 entries)
        self.performance_history = self.performance_history[-100:]
        
        return optimization_results
    
    async def _update_deployment_metrics(self) -> None:
        """Update metrics for all deployments."""
        import random
        
        for deployment in self.regional_deployments.values():
            if deployment.status == DeploymentStatus.ACTIVE:
                # Simulate metric updates with some randomness
                deployment.cpu_utilization = max(10, min(95, 
                    deployment.cpu_utilization + random.uniform(-5, 5)
                ))
                deployment.gpu_utilization = max(20, min(90, 
                    deployment.gpu_utilization + random.uniform(-3, 3)
                ))
                deployment.memory_utilization = max(30, min(85, 
                    deployment.memory_utilization + random.uniform(-2, 2)
                ))
                deployment.requests_per_second = max(1, 
                    deployment.requests_per_second + random.uniform(-5, 10)
                )
                deployment.latency_p95 = max(20, min(200, 
                    deployment.latency_p95 + random.uniform(-10, 10)
                ))
                deployment.error_rate = max(0, min(0.05, 
                    deployment.error_rate + random.uniform(-0.001, 0.002)
                ))
                
                deployment.last_update = time.time()
    
    async def _scale_deployment(
        self, 
        deployment: RegionalDeployment, 
        target_replicas: int, 
        reason: str
    ) -> None:
        """Scale a regional deployment."""
        logger.info(f"Scaling {deployment.region.value} from {deployment.replicas} to {target_replicas} ({reason})")
        
        deployment.status = DeploymentStatus.SCALING
        deployment.target_replicas = target_replicas
        
        # Simulate scaling time
        await asyncio.sleep(0.2)
        
        deployment.replicas = target_replicas
        deployment.status = DeploymentStatus.ACTIVE
    
    async def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies."""
        anomalies = []
        
        for region_name, deployment in self.regional_deployments.items():
            # High latency anomaly
            if deployment.latency_p95 > 150:
                anomalies.append({
                    "type": "high_latency",
                    "region": region_name,
                    "severity": "high" if deployment.latency_p95 > 200 else "medium",
                    "value": deployment.latency_p95,
                    "threshold": 150
                })
            
            # High error rate anomaly
            if deployment.error_rate > 0.01:
                anomalies.append({
                    "type": "high_error_rate",
                    "region": region_name,
                    "severity": "critical" if deployment.error_rate > 0.05 else "high",
                    "value": deployment.error_rate,
                    "threshold": 0.01
                })
            
            # Resource utilization anomaly
            if deployment.cpu_utilization > 90 or deployment.gpu_utilization > 95:
                anomalies.append({
                    "type": "high_resource_utilization",
                    "region": region_name,
                    "severity": "high",
                    "cpu_util": deployment.cpu_utilization,
                    "gpu_util": deployment.gpu_utilization
                })
        
        return anomalies
    
    async def _handle_anomalies(self, anomalies: List[Dict[str, Any]]) -> None:
        """Handle detected anomalies."""
        for anomaly in anomalies:
            logger.warning(f"Anomaly detected: {anomaly}")
            
            if anomaly["type"] == "high_latency":
                # Scale up to handle latency issues
                region = anomaly["region"]
                deployment = self.regional_deployments[region]
                if deployment.replicas < self.config.max_replicas_per_region:
                    await self._scale_deployment(
                        deployment, 
                        min(deployment.replicas + 2, self.config.max_replicas_per_region),
                        "anomaly_high_latency"
                    )
            
            elif anomaly["type"] == "high_error_rate":
                # Consider draining traffic from problematic region
                region = anomaly["region"]
                logger.critical(f"High error rate in {region}, consider traffic adjustment")
    
    async def _optimize_traffic_routing(self) -> bool:
        """Optimize global traffic routing based on current performance."""
        active_deployments = [
            d for d in self.regional_deployments.values() 
            if d.status == DeploymentStatus.ACTIVE
        ]
        
        if len(active_deployments) <= 1:
            return False
        
        # Calculate new optimal routing
        new_routing = await self.traffic_manager.calculate_optimal_routing(
            active_deployments,
            {"strategy": "performance_weighted"}
        )
        
        logger.info(f"Optimized traffic routing: {new_routing}")
        return True
    
    def generate_operational_report(self) -> Dict[str, Any]:
        """Generate comprehensive operational report."""
        current_time = time.time()
        
        # Calculate aggregate metrics
        total_replicas = sum(d.replicas for d in self.regional_deployments.values())
        total_rps = sum(d.requests_per_second for d in self.regional_deployments.values())
        avg_latency = sum(d.latency_p95 for d in self.regional_deployments.values()) / len(self.regional_deployments) if self.regional_deployments else 0
        avg_error_rate = sum(d.error_rate for d in self.regional_deployments.values()) / len(self.regional_deployments) if self.regional_deployments else 0
        
        # Health assessment
        healthy_regions = sum(1 for d in self.regional_deployments.values() if d.health_score > 0.8)
        
        return {
            "report_timestamp": current_time,
            "service_name": self.config.service_name,
            "deployment_id": self.config.deployment_id,
            
            # Global metrics
            "global_metrics": {
                "total_regions": len(self.regional_deployments),
                "healthy_regions": healthy_regions,
                "total_replicas": total_replicas,
                "total_requests_per_second": total_rps,
                "average_latency_p95": avg_latency,
                "average_error_rate": avg_error_rate,
                "global_health_score": healthy_regions / len(self.regional_deployments) if self.regional_deployments else 0
            },
            
            # Regional breakdown
            "regional_status": {
                region: deployment.to_dict()
                for region, deployment in self.regional_deployments.items()
            },
            
            # Performance trends
            "performance_trends": self._analyze_performance_trends(),
            
            # Operational insights
            "insights": {
                "scaling_recommendations": self._generate_scaling_recommendations(),
                "cost_optimization_opportunities": self._identify_cost_optimizations(),
                "reliability_assessment": self._assess_reliability(),
                "performance_optimization_suggestions": self._suggest_performance_optimizations()
            },
            
            # Recent activities
            "recent_activities": {
                "deployments": len(self.deployment_history),
                "last_optimization": self.last_optimization,
                "anomalies_detected": self.anomaly_detected
            }
        }
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends from historical data."""
        if len(self.performance_history) < 5:
            return {"status": "insufficient_data"}
        
        # Simple trend analysis
        recent_metrics = self.performance_history[-5:]
        
        # Calculate trends
        latency_trend = "stable"
        rps_trend = "stable"
        error_trend = "stable"
        
        if len(recent_metrics) >= 2:
            # Compare first and last periods
            first_period = recent_metrics[0]["regional_metrics"]
            last_period = recent_metrics[-1]["regional_metrics"]
            
            if first_period and last_period:
                # Calculate average changes
                latency_changes = []
                rps_changes = []
                error_changes = []
                
                for region in first_period.keys():
                    if region in last_period:
                        lat_change = last_period[region]["latency_p95"] - first_period[region]["latency_p95"]
                        rps_change = last_period[region]["requests_per_second"] - first_period[region]["requests_per_second"]
                        err_change = last_period[region]["error_rate"] - first_period[region]["error_rate"]
                        
                        latency_changes.append(lat_change)
                        rps_changes.append(rps_change)
                        error_changes.append(err_change)
                
                if latency_changes:
                    avg_lat_change = sum(latency_changes) / len(latency_changes)
                    if avg_lat_change > 5:
                        latency_trend = "increasing"
                    elif avg_lat_change < -5:
                        latency_trend = "decreasing"
                
                if rps_changes:
                    avg_rps_change = sum(rps_changes) / len(rps_changes)
                    if avg_rps_change > 2:
                        rps_trend = "increasing"
                    elif avg_rps_change < -2:
                        rps_trend = "decreasing"
                
                if error_changes:
                    avg_err_change = sum(error_changes) / len(error_changes)
                    if avg_err_change > 0.001:
                        error_trend = "increasing"
                    elif avg_err_change < -0.001:
                        error_trend = "decreasing"
        
        return {
            "latency_trend": latency_trend,
            "traffic_trend": rps_trend,
            "error_rate_trend": error_trend,
            "data_points_analyzed": len(recent_metrics)
        }
    
    def _generate_scaling_recommendations(self) -> List[Dict[str, Any]]:
        """Generate scaling recommendations."""
        recommendations = []
        
        for region_name, deployment in self.regional_deployments.items():
            if deployment.cpu_utilization > 80:
                recommendations.append({
                    "region": region_name,
                    "action": "scale_up",
                    "reason": "high_cpu_utilization",
                    "current_replicas": deployment.replicas,
                    "recommended_replicas": min(deployment.replicas + 2, self.config.max_replicas_per_region),
                    "priority": "high"
                })
            elif deployment.cpu_utilization < 30 and deployment.replicas > self.config.min_replicas_per_region:
                recommendations.append({
                    "region": region_name,
                    "action": "scale_down",
                    "reason": "low_cpu_utilization",
                    "current_replicas": deployment.replicas,
                    "recommended_replicas": max(deployment.replicas - 1, self.config.min_replicas_per_region),
                    "priority": "medium"
                })
        
        return recommendations
    
    def _identify_cost_optimizations(self) -> List[Dict[str, Any]]:
        """Identify cost optimization opportunities."""
        optimizations = []
        
        # Identify over-provisioned regions
        for region_name, deployment in self.regional_deployments.items():
            if (deployment.cpu_utilization < 40 and 
                deployment.gpu_utilization < 50 and 
                deployment.replicas > self.config.min_replicas_per_region):
                
                optimizations.append({
                    "type": "right_sizing",
                    "region": region_name,
                    "description": "Region appears over-provisioned",
                    "potential_savings": f"{(deployment.replicas - self.config.min_replicas_per_region) * 100}$/month",
                    "action": "Consider reducing replica count"
                })
        
        # Spot instance recommendations
        if not self.config.enable_spot_instances:
            optimizations.append({
                "type": "spot_instances",
                "description": "Spot instances not enabled",
                "potential_savings": "20-60% cost reduction",
                "action": "Enable spot instances for non-critical workloads"
            })
        
        return optimizations
    
    def _assess_reliability(self) -> Dict[str, Any]:
        """Assess system reliability."""
        healthy_deployments = sum(1 for d in self.regional_deployments.values() if d.health_score > 0.8)
        total_deployments = len(self.regional_deployments)
        
        reliability_score = healthy_deployments / total_deployments if total_deployments > 0 else 0
        
        return {
            "overall_reliability_score": reliability_score,
            "healthy_regions": healthy_deployments,
            "total_regions": total_deployments,
            "reliability_grade": "A" if reliability_score > 0.9 else "B" if reliability_score > 0.8 else "C" if reliability_score > 0.7 else "D",
            "single_point_failures": total_deployments == 1,
            "multi_region_resilience": total_deployments >= 2
        }
    
    def _suggest_performance_optimizations(self) -> List[str]:
        """Suggest performance optimizations."""
        suggestions = []
        
        # Check average latency
        avg_latency = sum(d.latency_p95 for d in self.regional_deployments.values()) / len(self.regional_deployments) if self.regional_deployments else 0
        
        if avg_latency > 100:
            suggestions.append("Consider implementing advanced caching strategies")
            suggestions.append("Optimize model inference pipeline")
        
        # Check error rates
        high_error_regions = [d for d in self.regional_deployments.values() if d.error_rate > 0.01]
        if high_error_regions:
            suggestions.append("Investigate and resolve high error rates in affected regions")
            suggestions.append("Implement circuit breaker patterns")
        
        # Check resource utilization
        underutilized_regions = [d for d in self.regional_deployments.values() if d.cpu_utilization < 40]
        if underutilized_regions:
            suggestions.append("Optimize resource allocation in underutilized regions")
        
        return suggestions


# Example usage and integration functions
async def deploy_nimify_service_globally():
    """Example of deploying a Nimify service globally."""
    
    # Create global deployment configuration
    config = GlobalDeploymentConfig(
        service_name="sentiment-analyzer",
        model_path="/models/sentiment-bert.onnx",
        regions=[
            DeploymentRegion.US_EAST_1,
            DeploymentRegion.US_WEST_2,
            DeploymentRegion.EU_WEST_1,
            DeploymentRegion.AP_SOUTHEAST_1
        ],
        min_replicas_per_region=3,
        max_replicas_per_region=15,
        target_cpu_utilization=75,
        enable_predictive_scaling=True,
        enable_distributed_tracing=True
    )
    
    # Create orchestrator
    orchestrator = GlobalOrchestrator(config)
    
    # Deploy globally
    deployment_result = await orchestrator.deploy_globally()
    
    print(f"Global deployment completed:")
    print(f"  Service: {config.service_name}")
    print(f"  Regions: {deployment_result['successful_deployments']}/{deployment_result['total_regions']}")
    print(f"  Duration: {deployment_result['deployment_duration']:.2f}s")
    
    # Start monitoring and optimization
    for i in range(5):  # Monitor for 5 cycles
        print(f"\nMonitoring cycle {i + 1}...")
        optimization_result = await orchestrator.monitor_and_optimize()
        print(f"  Scaling actions: {optimization_result['scaling_actions']}")
        print(f"  Anomalies: {optimization_result['anomalies_detected']}")
        
        await asyncio.sleep(1)  # Wait between cycles
    
    # Generate operational report
    report = orchestrator.generate_operational_report()
    print(f"\nOperational Report:")
    print(f"  Global Health: {report['global_metrics']['global_health_score']:.2%}")
    print(f"  Total RPS: {report['global_metrics']['total_requests_per_second']:.1f}")
    print(f"  Avg Latency: {report['global_metrics']['average_latency_p95']:.1f}ms")
    print(f"  Reliability Grade: {report['insights']['reliability_assessment']['reliability_grade']}")
    
    return orchestrator, report


if __name__ == "__main__":
    # Run global deployment example
    asyncio.run(deploy_nimify_service_globally())