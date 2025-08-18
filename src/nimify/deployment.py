"""Advanced deployment configuration and automation."""

import json
import logging
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Complete deployment configuration."""
    service_name: str
    image_name: str
    image_tag: str = "latest"
    replicas: int = 3
    namespace: str = "default"
    
    # Resource limits
    cpu_request: str = "100m"
    cpu_limit: str = "1000m"
    memory_request: str = "256Mi"
    memory_limit: str = "2Gi"
    gpu_limit: int = 1
    
    # Networking
    service_port: int = 8000
    target_port: int = 8000
    service_type: str = "LoadBalancer"
    
    # Health checks
    liveness_probe_path: str = "/health"
    readiness_probe_path: str = "/health"
    probe_timeout: int = 30
    
    # Scaling
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    
    # Security
    enable_network_policies: bool = True
    enable_pod_security_policy: bool = True
    service_account: str | None = None
    
    # Monitoring
    enable_prometheus: bool = True
    metrics_port: int = 9090
    enable_tracing: bool = True
    
    # Storage
    persistent_volume_size: str = "10Gi"
    storage_class: str = "fast-ssd"
    
    # Environment
    environment_variables: dict[str, str] = None
    secrets: dict[str, str] = None
    
    def __post_init__(self):
        if self.environment_variables is None:
            self.environment_variables = {}
        if self.secrets is None:
            self.secrets = {}


class KubernetesManifestGenerator:
    """Generates comprehensive Kubernetes manifests."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def generate_namespace(self) -> dict[str, Any]:
        """Generate namespace manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.config.namespace,
                "labels": {
                    "name": self.config.namespace,
                    "managed-by": "nimify"
                }
            }
        }
    
    def generate_deployment(self) -> dict[str, Any]:
        """Generate deployment manifest with advanced features."""
        container_env = []
        
        # Add environment variables
        for key, value in self.config.environment_variables.items():
            container_env.append({"name": key, "value": value})
        
        # Add secrets as environment variables
        for key, secret_ref in self.config.secrets.items():
            container_env.append({
                "name": key,
                "valueFrom": {
                    "secretKeyRef": {
                        "name": f"{self.config.service_name}-secrets",
                        "key": secret_ref
                    }
                }
            })
        
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.config.service_name,
                "namespace": self.config.namespace,
                "labels": {
                    "app": self.config.service_name,
                    "version": self.config.image_tag,
                    "managed-by": "nimify"
                }
            },
            "spec": {
                "replicas": self.config.replicas,
                "strategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {
                        "maxSurge": 1,
                        "maxUnavailable": 0
                    }
                },
                "selector": {
                    "matchLabels": {
                        "app": self.config.service_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.config.service_name,
                            "version": self.config.image_tag
                        },
                        "annotations": {
                            "prometheus.io/scrape": "true" if self.config.enable_prometheus else "false",
                            "prometheus.io/port": str(self.config.metrics_port),
                            "prometheus.io/path": "/metrics"
                        }
                    },
                    "spec": {
                        "serviceAccountName": self.config.service_account or f"{self.config.service_name}-sa",
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 2000
                        },
                        "containers": [{
                            "name": self.config.service_name,
                            "image": f"{self.config.image_name}:{self.config.image_tag}",
                            "imagePullPolicy": "Always",
                            "ports": [
                                {
                                    "containerPort": self.config.target_port,
                                    "name": "http"
                                },
                                {
                                    "containerPort": self.config.metrics_port,
                                    "name": "metrics"
                                }
                            ],
                            "env": container_env,
                            "resources": {
                                "requests": {
                                    "cpu": self.config.cpu_request,
                                    "memory": self.config.memory_request,
                                    "nvidia.com/gpu": str(self.config.gpu_limit)
                                },
                                "limits": {
                                    "cpu": self.config.cpu_limit,
                                    "memory": self.config.memory_limit,
                                    "nvidia.com/gpu": str(self.config.gpu_limit)
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": self.config.liveness_probe_path,
                                    "port": self.config.target_port
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                                "timeoutSeconds": self.config.probe_timeout
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": self.config.readiness_probe_path,
                                    "port": self.config.target_port
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5,
                                "timeoutSeconds": self.config.probe_timeout
                            },
                            "volumeMounts": [{
                                "name": "model-storage",
                                "mountPath": "/models"
                            }]
                        }],
                        "volumes": [{
                            "name": "model-storage",
                            "persistentVolumeClaim": {
                                "claimName": f"{self.config.service_name}-pvc"
                            }
                        }],
                        "nodeSelector": {
                            "accelerator": "nvidia-tesla-gpu"
                        },
                        "tolerations": [{
                            "key": "nvidia.com/gpu",
                            "operator": "Exists",
                            "effect": "NoSchedule"
                        }]
                    }
                }
            }
        }
    
    def generate_service(self) -> dict[str, Any]:
        """Generate service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": self.config.service_name,
                "namespace": self.config.namespace,
                "labels": {
                    "app": self.config.service_name,
                    "managed-by": "nimify"
                }
            },
            "spec": {
                "type": self.config.service_type,
                "selector": {
                    "app": self.config.service_name
                },
                "ports": [
                    {
                        "name": "http",
                        "port": self.config.service_port,
                        "targetPort": self.config.target_port,
                        "protocol": "TCP"
                    },
                    {
                        "name": "metrics",
                        "port": self.config.metrics_port,
                        "targetPort": self.config.metrics_port,
                        "protocol": "TCP"
                    }
                ]
            }
        }
    
    def generate_hpa(self) -> dict[str, Any]:
        """Generate Horizontal Pod Autoscaler."""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.config.service_name}-hpa",
                "namespace": self.config.namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": self.config.service_name
                },
                "minReplicas": self.config.min_replicas,
                "maxReplicas": self.config.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.config.target_cpu_utilization
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.config.target_memory_utilization
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "nvidia.com/gpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ],
                "behavior": {
                    "scaleUp": {
                        "stabilizationWindowSeconds": 60,
                        "policies": [{
                            "type": "Percent",
                            "value": 100,
                            "periodSeconds": 15
                        }]
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [{
                            "type": "Percent",
                            "value": 50,
                            "periodSeconds": 60
                        }]
                    }
                }
            }
        }
    
    def generate_pvc(self) -> dict[str, Any]:
        """Generate Persistent Volume Claim."""
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": f"{self.config.service_name}-pvc",
                "namespace": self.config.namespace
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": self.config.storage_class,
                "resources": {
                    "requests": {
                        "storage": self.config.persistent_volume_size
                    }
                }
            }
        }
    
    def generate_service_account(self) -> dict[str, Any]:
        """Generate Service Account with RBAC."""
        return {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": f"{self.config.service_name}-sa",
                "namespace": self.config.namespace
            }
        }
    
    def generate_network_policy(self) -> dict[str, Any]:
        """Generate Network Policy for security."""
        if not self.config.enable_network_policies:
            return {}
            
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"{self.config.service_name}-netpol",
                "namespace": self.config.namespace
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app": self.config.service_name
                    }
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [{
                    "from": [
                        {"namespaceSelector": {"matchLabels": {"name": "ingress-nginx"}}},
                        {"namespaceSelector": {"matchLabels": {"name": "monitoring"}}}
                    ],
                    "ports": [
                        {"protocol": "TCP", "port": self.config.target_port},
                        {"protocol": "TCP", "port": self.config.metrics_port}
                    ]
                }],
                "egress": [{
                    "to": [],
                    "ports": [
                        {"protocol": "TCP", "port": 443},  # HTTPS
                        {"protocol": "TCP", "port": 53},   # DNS
                        {"protocol": "UDP", "port": 53}    # DNS
                    ]
                }]
            }
        }
    
    def generate_pod_disruption_budget(self) -> dict[str, Any]:
        """Generate Pod Disruption Budget for availability."""
        return {
            "apiVersion": "policy/v1",
            "kind": "PodDisruptionBudget",
            "metadata": {
                "name": f"{self.config.service_name}-pdb",
                "namespace": self.config.namespace
            },
            "spec": {
                "minAvailable": max(1, self.config.min_replicas // 2),
                "selector": {
                    "matchLabels": {
                        "app": self.config.service_name
                    }
                }
            }
        }
    
    def generate_all_manifests(self) -> dict[str, dict[str, Any]]:
        """Generate all Kubernetes manifests."""
        manifests = {
            "namespace": self.generate_namespace(),
            "service_account": self.generate_service_account(),
            "pvc": self.generate_pvc(),
            "deployment": self.generate_deployment(),
            "service": self.generate_service(),
            "hpa": self.generate_hpa(),
            "pdb": self.generate_pod_disruption_budget()
        }
        
        if self.config.enable_network_policies:
            manifests["network_policy"] = self.generate_network_policy()
        
        return manifests


class HelmChartGenerator:
    """Advanced Helm chart generation."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def generate_chart_yaml(self) -> dict[str, Any]:
        """Generate Chart.yaml."""
        return {
            "apiVersion": "v2",
            "name": self.config.service_name,
            "description": f"Helm chart for {self.config.service_name} NIM service",
            "type": "application",
            "version": "1.0.0",
            "appVersion": self.config.image_tag,
            "keywords": ["ai", "ml", "nvidia", "nim", "inference"],
            "maintainers": [{
                "name": "Nimify",
                "email": "support@nimify.ai"
            }],
            "dependencies": [
                {
                    "name": "prometheus",
                    "version": "15.x.x",
                    "repository": "https://prometheus-community.github.io/helm-charts",
                    "condition": "prometheus.enabled"
                }
            ]
        }
    
    def generate_values_yaml(self) -> dict[str, Any]:
        """Generate comprehensive values.yaml."""
        return {
            "replicaCount": self.config.replicas,
            "image": {
                "repository": self.config.image_name,
                "tag": self.config.image_tag,
                "pullPolicy": "Always"
            },
            "service": {
                "type": self.config.service_type,
                "port": self.config.service_port,
                "targetPort": self.config.target_port,
                "annotations": {}
            },
            "ingress": {
                "enabled": False,
                "className": "nginx",
                "annotations": {
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true"
                },
                "hosts": [],
                "tls": []
            },
            "resources": {
                "requests": {
                    "cpu": self.config.cpu_request,
                    "memory": self.config.memory_request,
                    "nvidia.com/gpu": self.config.gpu_limit
                },
                "limits": {
                    "cpu": self.config.cpu_limit,
                    "memory": self.config.memory_limit,
                    "nvidia.com/gpu": self.config.gpu_limit
                }
            },
            "autoscaling": {
                "enabled": True,
                "minReplicas": self.config.min_replicas,
                "maxReplicas": self.config.max_replicas,
                "targetCPUUtilizationPercentage": self.config.target_cpu_utilization,
                "targetMemoryUtilizationPercentage": self.config.target_memory_utilization
            },
            "nodeSelector": {
                "accelerator": "nvidia-tesla-gpu"
            },
            "tolerations": [{
                "key": "nvidia.com/gpu",
                "operator": "Exists",
                "effect": "NoSchedule"
            }],
            "affinity": {
                "podAntiAffinity": {
                    "preferredDuringSchedulingIgnoredDuringExecution": [{
                        "weight": 100,
                        "podAffinityTerm": {
                            "labelSelector": {
                                "matchExpressions": [{
                                    "key": "app",
                                    "operator": "In",
                                    "values": [self.config.service_name]
                                }]
                            },
                            "topologyKey": "kubernetes.io/hostname"
                        }
                    }]
                }
            },
            "persistence": {
                "enabled": True,
                "size": self.config.persistent_volume_size,
                "storageClass": self.config.storage_class,
                "accessMode": "ReadWriteOnce"
            },
            "monitoring": {
                "prometheus": {
                    "enabled": self.config.enable_prometheus,
                    "port": self.config.metrics_port
                },
                "tracing": {
                    "enabled": self.config.enable_tracing
                }
            },
            "security": {
                "networkPolicies": {
                    "enabled": self.config.enable_network_policies
                },
                "podSecurityPolicy": {
                    "enabled": self.config.enable_pod_security_policy
                },
                "serviceAccount": {
                    "create": True,
                    "name": self.config.service_account or ""
                }
            },
            "environment": self.config.environment_variables,
            "secrets": self.config.secrets
        }
    
    def save_chart(self, output_dir: Path) -> Path:
        """Save complete Helm chart to directory."""
        chart_dir = output_dir / self.config.service_name
        chart_dir.mkdir(parents=True, exist_ok=True)
        
        # Create chart structure
        (chart_dir / "templates").mkdir(exist_ok=True)
        (chart_dir / "charts").mkdir(exist_ok=True)
        
        # Generate Chart.yaml
        with open(chart_dir / "Chart.yaml", 'w') as f:
            yaml.dump(self.generate_chart_yaml(), f, default_flow_style=False)
        
        # Generate values.yaml
        with open(chart_dir / "values.yaml", 'w') as f:
            yaml.dump(self.generate_values_yaml(), f, default_flow_style=False)
        
        # Generate Kubernetes manifests as templates
        manifest_generator = KubernetesManifestGenerator(self.config)
        manifests = manifest_generator.generate_all_manifests()
        
        for name, manifest in manifests.items():
            if manifest:  # Skip empty manifests
                template_path = chart_dir / "templates" / f"{name}.yaml"
                with open(template_path, 'w') as f:
                    yaml.dump(manifest, f, default_flow_style=False)
        
        logger.info(f"Generated Helm chart at {chart_dir}")
        return chart_dir


class DeploymentOrchestrator:
    """Orchestrates deployment process."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    async def deploy_to_kubernetes(self, kubeconfig_path: str | None = None) -> bool:
        """Deploy to Kubernetes cluster."""
        try:
            # Generate manifests
            generator = KubernetesManifestGenerator(self.config)
            manifests = generator.generate_all_manifests()
            
            # Create temporary files for manifests
            with tempfile.TemporaryDirectory() as temp_dir:
                manifest_files = []
                
                for name, manifest in manifests.items():
                    if manifest:
                        file_path = Path(temp_dir) / f"{name}.yaml"
                        with open(file_path, 'w') as f:
                            yaml.dump(manifest, f)
                        manifest_files.append(file_path)
                
                # Apply manifests
                for file_path in manifest_files:
                    cmd = ["kubectl", "apply", "-f", str(file_path)]
                    if kubeconfig_path:
                        cmd.extend(["--kubeconfig", kubeconfig_path])
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.error(f"Failed to apply {file_path}: {result.stderr}")
                        return False
                    else:
                        logger.info(f"Applied {file_path}: {result.stdout}")
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def generate_deployment_package(self, output_dir: Path) -> Path:
        """Generate complete deployment package."""
        package_dir = output_dir / f"{self.config.service_name}-deployment"
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate Helm chart
        helm_generator = HelmChartGenerator(self.config)
        helm_generator.save_chart(package_dir / "helm")
        
        # Generate raw Kubernetes manifests
        k8s_dir = package_dir / "kubernetes"
        k8s_dir.mkdir(exist_ok=True)
        
        manifest_generator = KubernetesManifestGenerator(self.config)
        manifests = manifest_generator.generate_all_manifests()
        
        for name, manifest in manifests.items():
            if manifest:
                with open(k8s_dir / f"{name}.yaml", 'w') as f:
                    yaml.dump(manifest, f)
        
        # Generate deployment scripts
        self._generate_deployment_scripts(package_dir)
        
        # Generate configuration
        config_dict = asdict(self.config)
        with open(package_dir / "deployment-config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Generated deployment package at {package_dir}")
        return package_dir
    
    def _generate_deployment_scripts(self, package_dir: Path):
        """Generate deployment and management scripts."""
        scripts_dir = package_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Deploy script
        deploy_script = f"""#!/bin/bash
set -e

echo "Deploying {self.config.service_name}..."

# Apply Kubernetes manifests
kubectl apply -f kubernetes/

# Install/upgrade Helm chart
helm upgrade --install {self.config.service_name} helm/{self.config.service_name}/ \\
    --namespace {self.config.namespace} \\
    --create-namespace \\
    --wait

echo "Deployment complete!"
"""
        
        with open(scripts_dir / "deploy.sh", 'w') as f:
            f.write(deploy_script)
        
        # Status script
        status_script = f"""#!/bin/bash

echo "Checking status of {self.config.service_name}..."

kubectl get pods -n {self.config.namespace} -l app={self.config.service_name}
kubectl get svc -n {self.config.namespace} -l app={self.config.service_name}
kubectl get hpa -n {self.config.namespace} -l app={self.config.service_name}

echo "Service endpoints:"
kubectl get svc {self.config.service_name} -n {self.config.namespace} -o jsonpath='{{.status.loadBalancer.ingress[0].ip}}'
"""
        
        with open(scripts_dir / "status.sh", 'w') as f:
            f.write(status_script)
        
        # Make scripts executable
        for script in scripts_dir.glob("*.sh"):
            script.chmod(0o755)