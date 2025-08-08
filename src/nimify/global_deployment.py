"""Global-first deployment with multi-region support and internationalization."""

import json
import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class Region(Enum):
    """Supported deployment regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2" 
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"
    CANADA = "ca-central-1"
    AUSTRALIA = "ap-southeast-2"
    BRAZIL = "sa-east-1"
    INDIA = "ap-south-1"


class ComplianceStandard(Enum):
    """Supported compliance standards."""
    GDPR = "gdpr"
    CCPA = "ccpa" 
    PDPA = "pdpa"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"


@dataclass
class RegionConfig:
    """Configuration for a specific region."""
    region: Region
    enabled: bool = True
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    data_residency_required: bool = False
    local_regulations: Dict[str, Any] = field(default_factory=dict)
    edge_locations: List[str] = field(default_factory=list)
    cost_optimization_tier: str = "standard"  # economy, standard, premium
    
    # Resource configuration
    min_replicas: int = 2
    max_replicas: int = 20
    instance_type: str = "standard"
    storage_class: str = "standard"
    
    # Network configuration
    vpc_config: Optional[Dict[str, Any]] = None
    security_groups: List[str] = field(default_factory=list)
    load_balancer_type: str = "application"


@dataclass
class I18nConfig:
    """Internationalization configuration."""
    default_language: str = "en"
    supported_languages: List[str] = field(default_factory=lambda: ["en", "es", "fr", "de", "ja", "zh-CN", "pt", "ru"])
    locale_detection: str = "header"  # header, geo, user_preference
    fallback_behavior: str = "default"  # default, error, closest_match
    
    # Translation configuration
    auto_translation: bool = False
    translation_service: str = "none"  # none, google, azure, aws
    quality_threshold: float = 0.9
    
    # Content configuration
    date_formats: Dict[str, str] = field(default_factory=lambda: {
        "en": "%Y-%m-%d",
        "de": "%d.%m.%Y", 
        "ja": "%Y年%m月%d日",
        "zh-CN": "%Y年%m月%d日"
    })
    
    currency_formats: Dict[str, str] = field(default_factory=lambda: {
        "en": "USD",
        "eu": "EUR",
        "jp": "JPY",
        "cn": "CNY"
    })


class GlobalDeploymentManager:
    """Manages global multi-region deployments."""
    
    def __init__(self):
        self.regions: Dict[Region, RegionConfig] = {}
        self.i18n_config = I18nConfig()
        self.deployment_strategy = "blue_green"  # blue_green, rolling, canary
        self.traffic_routing = "latency_based"  # latency_based, geo_based, weighted
        
        # Initialize default regions
        self._initialize_default_regions()
    
    def _initialize_default_regions(self):
        """Initialize default regional configurations."""
        # North America
        self.regions[Region.US_EAST] = RegionConfig(
            region=Region.US_EAST,
            compliance_standards=[ComplianceStandard.SOC2, ComplianceStandard.CCPA],
            edge_locations=["us-east-1a", "us-east-1b", "us-east-1c"],
            min_replicas=3,
            max_replicas=50
        )
        
        self.regions[Region.US_WEST] = RegionConfig(
            region=Region.US_WEST,
            compliance_standards=[ComplianceStandard.SOC2, ComplianceStandard.CCPA],
            edge_locations=["us-west-2a", "us-west-2b"],
            min_replicas=2,
            max_replicas=30
        )
        
        # Europe
        self.regions[Region.EU_WEST] = RegionConfig(
            region=Region.EU_WEST,
            compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.ISO27001],
            data_residency_required=True,
            edge_locations=["eu-west-1a", "eu-west-1b", "eu-west-1c"],
            min_replicas=3,
            max_replicas=40
        )
        
        self.regions[Region.EU_CENTRAL] = RegionConfig(
            region=Region.EU_CENTRAL,
            compliance_standards=[ComplianceStandard.GDPR],
            data_residency_required=True,
            edge_locations=["eu-central-1a", "eu-central-1b"],
            min_replicas=2,
            max_replicas=20
        )
        
        # Asia Pacific
        self.regions[Region.ASIA_PACIFIC] = RegionConfig(
            region=Region.ASIA_PACIFIC,
            compliance_standards=[ComplianceStandard.PDPA],
            edge_locations=["ap-southeast-1a", "ap-southeast-1b"],
            min_replicas=2,
            max_replicas=25
        )
        
        self.regions[Region.ASIA_NORTHEAST] = RegionConfig(
            region=Region.ASIA_NORTHEAST,
            edge_locations=["ap-northeast-1a", "ap-northeast-1c"],
            min_replicas=2,
            max_replicas=30
        )
    
    def generate_global_deployment_manifests(self, service_name: str) -> Dict[str, Any]:
        """Generate deployment manifests for all enabled regions."""
        manifests = {
            "global_config": self._generate_global_config(service_name),
            "regions": {},
            "traffic_management": self._generate_traffic_management_config(service_name),
            "compliance": self._generate_compliance_config(),
            "monitoring": self._generate_global_monitoring_config(service_name)
        }
        
        for region, config in self.regions.items():
            if config.enabled:
                manifests["regions"][region.value] = self._generate_regional_manifests(
                    service_name, config
                )
        
        return manifests
    
    def _generate_global_config(self, service_name: str) -> Dict[str, Any]:
        """Generate global configuration."""
        return {
            "service_name": service_name,
            "deployment_strategy": self.deployment_strategy,
            "traffic_routing": self.traffic_routing,
            "global_load_balancer": {
                "type": "anycast",
                "health_check": {
                    "path": "/health",
                    "interval_seconds": 30,
                    "timeout_seconds": 10,
                    "failure_threshold": 3
                },
                "failover": {
                    "enabled": True,
                    "strategy": "automatic",
                    "fallback_regions": ["us-east-1", "eu-west-1"]
                }
            },
            "cdn": {
                "enabled": True,
                "provider": "cloudflare",  # cloudflare, cloudfront, azure_cdn
                "cache_policies": {
                    "api_responses": {
                        "ttl_seconds": 300,
                        "edge_cache": True,
                        "browser_cache": False
                    },
                    "static_assets": {
                        "ttl_seconds": 86400,
                        "edge_cache": True,
                        "browser_cache": True
                    }
                }
            },
            "security": {
                "waf_enabled": True,
                "ddos_protection": True,
                "rate_limiting": {
                    "global_limit": 10000,
                    "per_ip_limit": 100,
                    "burst_limit": 200
                },
                "ssl_configuration": {
                    "minimum_tls_version": "1.2",
                    "cipher_suites": ["ECDHE-RSA-AES256-GCM-SHA384", "ECDHE-RSA-AES128-GCM-SHA256"],
                    "hsts_enabled": True,
                    "certificate_type": "wildcard"
                }
            }
        }
    
    def _generate_regional_manifests(self, service_name: str, region_config: RegionConfig) -> Dict[str, Any]:
        """Generate manifests for a specific region."""
        manifests = {
            "deployment": self._generate_regional_deployment(service_name, region_config),
            "service": self._generate_regional_service(service_name, region_config),
            # "ingress": self._generate_regional_ingress(service_name, region_config),  # TODO: Implement ingress
            "hpa": self._generate_regional_hpa(service_name, region_config),
            "network_policy": self._generate_network_policy(service_name, region_config),
            "pod_disruption_budget": self._generate_pdb(service_name, region_config)
        }
        
        # Add compliance-specific manifests
        if ComplianceStandard.GDPR in region_config.compliance_standards:
            manifests["gdpr_config"] = self._generate_gdpr_config(service_name)
        
        if ComplianceStandard.HIPAA in region_config.compliance_standards:
            manifests["hipaa_config"] = self._generate_hipaa_config(service_name)
        
        return manifests
    
    def _generate_regional_deployment(self, service_name: str, config: RegionConfig) -> Dict[str, Any]:
        """Generate regional deployment manifest."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{service_name}-{config.region.value}",
                "namespace": "nim-global",
                "labels": {
                    "app": service_name,
                    "region": config.region.value,
                    "compliance": ",".join([c.value for c in config.compliance_standards])
                },
                "annotations": {
                    "deployment.kubernetes.io/revision": "1",
                    "nimify.ai/region": config.region.value,
                    "nimify.ai/data-residency": str(config.data_residency_required).lower()
                }
            },
            "spec": {
                "replicas": config.min_replicas,
                "strategy": {
                    "type": "RollingUpdate" if self.deployment_strategy == "rolling" else "Recreate",
                    "rollingUpdate": {
                        "maxSurge": "25%",
                        "maxUnavailable": "10%"
                    } if self.deployment_strategy == "rolling" else None
                },
                "selector": {
                    "matchLabels": {
                        "app": service_name,
                        "region": config.region.value
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": service_name,
                            "region": config.region.value,
                            "version": "v1"
                        },
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "9090",
                            "prometheus.io/path": "/metrics"
                        }
                    },
                    "spec": {
                        "affinity": {
                            "podAntiAffinity": {
                                "preferredDuringSchedulingIgnoredDuringExecution": [{
                                    "weight": 100,
                                    "podAffinityTerm": {
                                        "labelSelector": {
                                            "matchExpressions": [{
                                                "key": "app",
                                                "operator": "In",
                                                "values": [service_name]
                                            }]
                                        },
                                        "topologyKey": "kubernetes.io/hostname"
                                    }
                                }]
                            },
                            "nodeAffinity": {
                                "requiredDuringSchedulingIgnoredDuringExecution": {
                                    "nodeSelectorTerms": [{
                                        "matchExpressions": [{
                                            "key": "kubernetes.io/arch",
                                            "operator": "In",
                                            "values": ["amd64"]
                                        }, {
                                            "key": "node.kubernetes.io/instance-type",
                                            "operator": "In",
                                            "values": ["gpu-optimized"]
                                        }]
                                    }]
                                }
                            }
                        },
                        "containers": [{
                            "name": service_name,
                            "image": f"{service_name}:latest",
                            "ports": [
                                {"containerPort": 8000, "name": "http"},
                                {"containerPort": 9090, "name": "metrics"}
                            ],
                            "env": [
                                {"name": "REGION", "value": config.region.value},
                                {"name": "COMPLIANCE_STANDARDS", "value": ",".join([c.value for c in config.compliance_standards])},
                                {"name": "DATA_RESIDENCY", "value": str(config.data_residency_required)},
                                {"name": "I18N_DEFAULT_LANG", "value": self.i18n_config.default_language},
                                {"name": "I18N_SUPPORTED_LANGS", "value": ",".join(self.i18n_config.supported_languages)}
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "500m",
                                    "memory": "1Gi",
                                    "nvidia.com/gpu": "1"
                                },
                                "limits": {
                                    "cpu": "2000m", 
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
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                                "failureThreshold": 3
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5,
                                "timeoutSeconds": 3,
                                "failureThreshold": 2
                            },
                            "securityContext": {
                                "runAsNonRoot": True,
                                "runAsUser": 1000,
                                "readOnlyRootFilesystem": True,
                                "allowPrivilegeEscalation": False,
                                "capabilities": {
                                    "drop": ["ALL"]
                                }
                            }
                        }],
                        "securityContext": {
                            "fsGroup": 2000,
                            "seccompProfile": {
                                "type": "RuntimeDefault"
                            }
                        },
                        "serviceAccountName": f"{service_name}-sa",
                        "imagePullSecrets": [{"name": "registry-secret"}]
                    }
                }
            }
        }
    
    def _generate_regional_service(self, service_name: str, config: RegionConfig) -> Dict[str, Any]:
        """Generate regional service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{service_name}-{config.region.value}",
                "namespace": "nim-global",
                "labels": {
                    "app": service_name,
                    "region": config.region.value
                },
                "annotations": {
                    "service.beta.kubernetes.io/aws-load-balancer-type": config.load_balancer_type,
                    "service.beta.kubernetes.io/aws-load-balancer-ssl-cert": "arn:aws:acm:region:account:certificate/cert-id",
                    "service.beta.kubernetes.io/aws-load-balancer-backend-protocol": "http"
                }
            },
            "spec": {
                "type": "LoadBalancer",
                "selector": {
                    "app": service_name,
                    "region": config.region.value
                },
                "ports": [
                    {
                        "name": "http",
                        "port": 443,
                        "targetPort": 8000,
                        "protocol": "TCP"
                    },
                    {
                        "name": "metrics",
                        "port": 9090,
                        "targetPort": 9090,
                        "protocol": "TCP"
                    }
                ]
            }
        }
    
    def _generate_regional_hpa(self, service_name: str, config: RegionConfig) -> Dict[str, Any]:
        """Generate Horizontal Pod Autoscaler for the region."""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{service_name}-{config.region.value}-hpa",
                "namespace": "nim-global"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{service_name}-{config.region.value}"
                },
                "minReplicas": config.min_replicas,
                "maxReplicas": config.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization", 
                                "averageUtilization": 80
                            }
                        }
                    },
                    {
                        "type": "Pods",
                        "pods": {
                            "metric": {
                                "name": "nim_request_duration_seconds_p95"
                            },
                            "target": {
                                "type": "AverageValue",
                                "averageValue": "200m"  # 200ms
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
                            "periodSeconds": 60
                        }]
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [{
                            "type": "Percent",
                            "value": 10,
                            "periodSeconds": 60
                        }]
                    }
                }
            }
        }
    
    def _generate_traffic_management_config(self, service_name: str) -> Dict[str, Any]:
        """Generate global traffic management configuration."""
        return {
            "dns": {
                "provider": "route53",  # route53, cloudflare, google_dns
                "zone": "nimify.ai",
                "records": [
                    {
                        "name": f"{service_name}.api.nimify.ai",
                        "type": "A",
                        "routing_policy": self.traffic_routing,
                        "health_check": True,
                        "regions": {
                            region.value: f"{service_name}-{region.value}.elb.amazonaws.com"
                            for region in self.regions.keys()
                        }
                    }
                ]
            },
            "load_balancing": {
                "strategy": self.traffic_routing,
                "health_checks": {
                    "interval": 30,
                    "timeout": 10,
                    "healthy_threshold": 2,
                    "unhealthy_threshold": 3
                },
                "routing_rules": [
                    {
                        "condition": "geo.country == 'US'",
                        "target_regions": ["us-east-1", "us-west-2"],
                        "weight_distribution": {"us-east-1": 70, "us-west-2": 30}
                    },
                    {
                        "condition": "geo.continent == 'Europe'",
                        "target_regions": ["eu-west-1", "eu-central-1"],
                        "weight_distribution": {"eu-west-1": 60, "eu-central-1": 40}
                    },
                    {
                        "condition": "geo.continent == 'Asia'",
                        "target_regions": ["ap-southeast-1", "ap-northeast-1"],
                        "weight_distribution": {"ap-southeast-1": 50, "ap-northeast-1": 50}
                    }
                ]
            },
            "failover": {
                "enabled": True,
                "detection_threshold": 3,
                "recovery_threshold": 2,
                "failover_chains": {
                    "us-east-1": ["us-west-2", "eu-west-1"],
                    "eu-west-1": ["eu-central-1", "us-east-1"],
                    "ap-southeast-1": ["ap-northeast-1", "us-west-2"]
                }
            }
        }
    
    def _generate_compliance_config(self) -> Dict[str, Any]:
        """Generate compliance configuration."""
        return {
            "gdpr": {
                "data_protection": {
                    "encryption_at_rest": True,
                    "encryption_in_transit": True,
                    "data_anonymization": True,
                    "retention_period_days": 365,
                    "right_to_deletion": True,
                    "data_portability": True
                },
                "consent_management": {
                    "explicit_consent": True,
                    "consent_tracking": True,
                    "withdrawal_mechanism": True
                },
                "breach_notification": {
                    "enabled": True,
                    "notification_time_hours": 72,
                    "contact_email": "dpo@nimify.ai"
                }
            },
            "ccpa": {
                "data_protection": {
                    "encryption_at_rest": True,
                    "encryption_in_transit": True,
                    "data_minimization": True
                },
                "consumer_rights": {
                    "right_to_know": True,
                    "right_to_delete": True,
                    "right_to_opt_out": True,
                    "non_discrimination": True
                }
            },
            "soc2": {
                "security": {
                    "access_controls": True,
                    "vulnerability_management": True,
                    "incident_response": True
                },
                "availability": {
                    "uptime_monitoring": True,
                    "disaster_recovery": True,
                    "backup_procedures": True
                },
                "confidentiality": {
                    "data_classification": True,
                    "encryption": True,
                    "access_logging": True
                }
            }
        }
    
    def _generate_global_monitoring_config(self, service_name: str) -> Dict[str, Any]:
        """Generate global monitoring configuration."""
        return {
            "prometheus": {
                "global_aggregation": True,
                "federation": {
                    "enabled": True,
                    "scrape_interval": "30s",
                    "regions": [region.value for region in self.regions.keys()]
                },
                "external_labels": {
                    "service": service_name,
                    "environment": "production",
                    "cluster": "global"
                }
            },
            "grafana": {
                "global_dashboards": True,
                "regional_dashboards": True,
                "alerting": {
                    "enabled": True,
                    "notification_channels": [
                        "slack", "pagerduty", "email"
                    ]
                }
            },
            "distributed_tracing": {
                "enabled": True,
                "sampling_rate": 0.01,  # 1% sampling
                "backend": "jaeger"
            },
            "log_aggregation": {
                "enabled": True,
                "backend": "elasticsearch",
                "retention_days": 30,
                "structured_logging": True
            }
        }
    
    def _generate_gdpr_config(self, service_name: str) -> Dict[str, Any]:
        """Generate GDPR-specific configuration."""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{service_name}-gdpr-config",
                "namespace": "nim-global"
            },
            "data": {
                "data_retention_days": "365",
                "anonymization_enabled": "true",
                "consent_required": "true",
                "data_export_format": "json",
                "deletion_verification": "true"
            }
        }
    
    def _generate_hipaa_config(self, service_name: str) -> Dict[str, Any]:
        """Generate HIPAA-specific configuration."""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap", 
            "metadata": {
                "name": f"{service_name}-hipaa-config",
                "namespace": "nim-global"
            },
            "data": {
                "encryption_required": "true",
                "audit_logging": "true",
                "access_controls": "strict",
                "data_backup_encrypted": "true",
                "incident_response_plan": "enabled"
            }
        }
    
    def _generate_network_policy(self, service_name: str, config: RegionConfig) -> Dict[str, Any]:
        """Generate network policy for the region."""
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"{service_name}-{config.region.value}-netpol",
                "namespace": "nim-global"
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app": service_name,
                        "region": config.region.value
                    }
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {"namespaceSelector": {"matchLabels": {"name": "ingress-nginx"}}},
                            {"namespaceSelector": {"matchLabels": {"name": "monitoring"}}}
                        ],
                        "ports": [
                            {"protocol": "TCP", "port": 8000},
                            {"protocol": "TCP", "port": 9090}
                        ]
                    }
                ],
                "egress": [
                    {
                        "to": [],
                        "ports": [
                            {"protocol": "TCP", "port": 443},
                            {"protocol": "TCP", "port": 53},
                            {"protocol": "UDP", "port": 53}
                        ]
                    }
                ]
            }
        }
    
    def _generate_pdb(self, service_name: str, config: RegionConfig) -> Dict[str, Any]:
        """Generate Pod Disruption Budget."""
        return {
            "apiVersion": "policy/v1",
            "kind": "PodDisruptionBudget",
            "metadata": {
                "name": f"{service_name}-{config.region.value}-pdb",
                "namespace": "nim-global"
            },
            "spec": {
                "minAvailable": max(1, config.min_replicas // 2),
                "selector": {
                    "matchLabels": {
                        "app": service_name,
                        "region": config.region.value
                    }
                }
            }
        }
    
    def save_global_deployment(self, service_name: str, output_dir: Path):
        """Save complete global deployment configuration."""
        manifests = self.generate_global_deployment_manifests(service_name)
        
        # Create directory structure
        global_dir = output_dir / f"{service_name}-global-deployment"
        global_dir.mkdir(parents=True, exist_ok=True)
        
        # Save global configuration
        with open(global_dir / "global-config.json", 'w') as f:
            json.dump(manifests["global_config"], f, indent=2)
        
        # Save regional manifests
        for region, regional_manifests in manifests["regions"].items():
            region_dir = global_dir / "regions" / region
            region_dir.mkdir(parents=True, exist_ok=True)
            
            for manifest_type, manifest in regional_manifests.items():
                with open(region_dir / f"{manifest_type}.yaml", 'w') as f:
                    f.write("# YAML conversion would be applied here\\n")
                    f.write(f"# {manifest_type.upper()} for {region}\\n")
                    f.write(json.dumps(manifest, indent=2))
        
        # Save traffic management
        with open(global_dir / "traffic-management.json", 'w') as f:
            json.dump(manifests["traffic_management"], f, indent=2)
        
        # Save compliance configuration
        with open(global_dir / "compliance.json", 'w') as f:
            json.dump(manifests["compliance"], f, indent=2)
        
        # Save monitoring configuration
        with open(global_dir / "monitoring.json", 'w') as f:
            json.dump(manifests["monitoring"], f, indent=2)
        
        # Generate deployment scripts
        self._generate_deployment_scripts(service_name, global_dir)
        
        logger.info(f"Saved global deployment configuration to {global_dir}")
        return global_dir
    
    def _generate_deployment_scripts(self, service_name: str, output_dir: Path):
        """Generate deployment scripts for global deployment."""
        scripts_dir = output_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Global deployment script
        deploy_script = f"""#!/bin/bash
set -e

echo "🌍 Deploying {service_name} globally..."

# Deploy to all regions in parallel
regions=("{' '.join([r.value for r in self.regions.keys() if self.regions[r].enabled])}")

for region in "${{regions[@]}}"; do
    echo "📍 Deploying to $region..."
    
    # Set regional context
    kubectl config use-context "$region"
    
    # Apply regional manifests
    kubectl apply -f "regions/$region/"
    
    # Wait for deployment to be ready
    kubectl rollout status deployment/{service_name}-$region -n nim-global
    
    echo "✅ $region deployment complete"
done

echo "🎉 Global deployment complete!"
echo "🔗 Service available at: https://{service_name}.api.nimify.ai"
"""
        
        with open(scripts_dir / "deploy-global.sh", 'w') as f:
            f.write(deploy_script)
        
        # Regional deployment script
        region_deploy_script = f"""#!/bin/bash
set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <region>"
    echo "Available regions: {', '.join([r.value for r in self.regions.keys()])}"
    exit 1
fi

REGION=$1
echo "📍 Deploying {service_name} to $REGION..."

# Validate region
if [ ! -d "regions/$REGION" ]; then
    echo "❌ Region $REGION not found"
    exit 1
fi

# Set regional context
kubectl config use-context "$REGION"

# Apply regional manifests  
kubectl apply -f "regions/$REGION/"

# Wait for deployment
kubectl rollout status deployment/{service_name}-$REGION -n nim-global

echo "✅ Deployment to $REGION complete"
"""
        
        with open(scripts_dir / "deploy-region.sh", 'w') as f:
            f.write(region_deploy_script)
        
        # Make scripts executable
        for script in scripts_dir.glob("*.sh"):
            script.chmod(0o755)


# Global deployment manager instance
global_deployment_manager = GlobalDeploymentManager()