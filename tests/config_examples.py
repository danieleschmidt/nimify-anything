"""Configuration examples and templates for testing."""

from typing import Dict, Any, List
from pathlib import Path


class TestConfigExamples:
    """Examples of test configurations for different scenarios."""
    
    @staticmethod
    def minimal_model_config() -> Dict[str, Any]:
        """Minimal model configuration for basic tests."""
        return {
            "name": "minimal-test-model",
            "platform": "onnxruntime_onnx",
            "max_batch_size": 1,
            "input": [{
                "name": "input",
                "data_type": "TYPE_FP32",
                "dims": [1, 3, 224, 224]
            }],
            "output": [{
                "name": "output",
                "data_type": "TYPE_FP32", 
                "dims": [1, 1000]
            }]
        }
    
    @staticmethod
    def dynamic_batching_config() -> Dict[str, Any]:
        """Configuration with dynamic batching enabled."""
        return {
            "name": "dynamic-batch-model",
            "platform": "onnxruntime_onnx",
            "max_batch_size": 16,
            "dynamic_batching": {
                "preferred_batch_size": [1, 4, 8, 16],
                "max_queue_delay_microseconds": 1000,
                "preserve_ordering": True
            },
            "input": [{
                "name": "input",
                "data_type": "TYPE_FP32",
                "dims": [-1, 3, 224, 224]  # Dynamic batch dimension
            }],
            "output": [{
                "name": "output",
                "data_type": "TYPE_FP32",
                "dims": [-1, 1000]  # Dynamic batch dimension
            }]
        }
    
    @staticmethod
    def tensorrt_optimized_config() -> Dict[str, Any]:
        """Configuration for TensorRT optimized models."""
        return {
            "name": "tensorrt-optimized-model",
            "platform": "tensorrt_plan",
            "max_batch_size": 32,
            "dynamic_batching": {
                "preferred_batch_size": [1, 4, 8, 16, 32]
            },
            "optimization": {
                "cuda": {
                    "graphs": True,
                    "busy_wait_events": True
                }
            },
            "input": [{
                "name": "input",
                "data_type": "TYPE_FP32",
                "dims": [-1, 3, 224, 224]
            }],
            "output": [{
                "name": "output", 
                "data_type": "TYPE_FP32",
                "dims": [-1, 1000]
            }],
            "instance_group": [{
                "count": 2,
                "kind": "KIND_GPU",
                "gpus": [0]
            }]
        }
    
    @staticmethod
    def multi_model_ensemble_config() -> Dict[str, Any]:
        """Configuration for ensemble of multiple models."""
        return {
            "name": "multi-model-ensemble",
            "platform": "ensemble",
            "ensemble_scheduling": {
                "step": [
                    {
                        "model_name": "preprocessing",
                        "model_version": -1,
                        "input_map": {
                            "raw_input": "INPUT"
                        },
                        "output_map": {
                            "processed_input": "preprocessed_data"
                        }
                    },
                    {
                        "model_name": "inference",
                        "model_version": -1,
                        "input_map": {
                            "input": "preprocessed_data"
                        },
                        "output_map": {
                            "predictions": "raw_predictions"
                        }
                    },
                    {
                        "model_name": "postprocessing",
                        "model_version": -1,
                        "input_map": {
                            "predictions": "raw_predictions"
                        },
                        "output_map": {
                            "final_output": "OUTPUT"
                        }
                    }
                ]
            },
            "input": [{
                "name": "INPUT",
                "data_type": "TYPE_FP32",
                "dims": [-1, 3, 224, 224]
            }],
            "output": [{
                "name": "OUTPUT",
                "data_type": "TYPE_FP32",
                "dims": [-1, 10]
            }]
        }


class TestEnvironmentConfigs:
    """Environment-specific test configurations."""
    
    @staticmethod
    def development_env() -> Dict[str, str]:
        """Development environment configuration."""
        return {
            "NIMIFY_ENV": "development",
            "NIMIFY_LOG_LEVEL": "DEBUG",
            "NIMIFY_MODEL_CACHE": "/tmp/nimify_test_cache",
            "CUDA_VISIBLE_DEVICES": "0",
            "TRITON_SERVER_URL": "localhost:8000",
            "PROMETHEUS_PORT": "9090",
            "ENABLE_METRICS": "true",
            "DEBUG_MODE": "true"
        }
    
    @staticmethod
    def ci_env() -> Dict[str, str]:
        """CI/CD environment configuration."""
        return {
            "NIMIFY_ENV": "ci",
            "NIMIFY_LOG_LEVEL": "INFO",
            "NIMIFY_MODEL_CACHE": "/tmp/ci_cache",
            "CUDA_VISIBLE_DEVICES": "",  # No GPU in CI
            "TRITON_SERVER_URL": "mock://localhost:8000",
            "ENABLE_METRICS": "false",
            "PARALLEL_JOBS": "4",
            "TEST_TIMEOUT": "300"
        }
    
    @staticmethod
    def integration_test_env() -> Dict[str, str]:
        """Integration test environment configuration."""
        return {
            "NIMIFY_ENV": "integration",
            "NIMIFY_LOG_LEVEL": "INFO",
            "NIMIFY_MODEL_CACHE": "/tmp/integration_cache",
            "CUDA_VISIBLE_DEVICES": "0",
            "TRITON_SERVER_URL": "localhost:8000",
            "K8S_NAMESPACE": "nimify-integration-test",
            "DOCKER_REGISTRY": "localhost:5000",
            "ENABLE_CLEANUP": "true",
            "TEST_TIMEOUT": "600"
        }


class MockDataExamples:
    """Examples of mock data for testing."""
    
    @staticmethod
    def sample_inference_request() -> Dict[str, Any]:
        """Sample inference request payload."""
        return {
            "inputs": [{
                "name": "input",
                "shape": [1, 3, 224, 224],
                "datatype": "FP32",
                "data": [[[[0.5] * 224] * 224] * 3]  # Mock image data
            }]
        }
    
    @staticmethod
    def sample_inference_response() -> Dict[str, Any]:
        """Sample inference response payload."""
        return {
            "outputs": [{
                "name": "output",
                "shape": [1, 1000],
                "datatype": "FP32",
                "data": [[0.001] * 1000]  # Mock predictions
            }],
            "model_name": "test-model",
            "model_version": "1",
            "id": "test-request-123"
        }
    
    @staticmethod
    def sample_openapi_spec() -> Dict[str, Any]:
        """Sample OpenAPI specification for NIM service."""
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "Test Model NIM API",
                "version": "1.0.0",
                "description": "Generated NIM API for test model"
            },
            "paths": {
                "/v1/predict": {
                    "post": {
                        "summary": "Run inference",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "input": {
                                                "type": "array",
                                                "items": {"type": "number"}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Successful prediction",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "predictions": {
                                                    "type": "array",
                                                    "items": {"type": "number"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/health": {
                    "get": {
                        "summary": "Health check",
                        "responses": {
                            "200": {
                                "description": "Service is healthy"
                            }
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def sample_helm_values() -> Dict[str, Any]:
        """Sample Helm chart values for testing."""
        return {
            "replicaCount": 1,
            "image": {
                "repository": "localhost:5000/test-model",
                "tag": "test",
                "pullPolicy": "IfNotPresent"
            },
            "service": {
                "type": "ClusterIP",
                "port": 80,
                "targetPort": 8000
            },
            "resources": {
                "limits": {
                    "cpu": "1000m",
                    "memory": "2Gi"
                },
                "requests": {
                    "cpu": "500m",
                    "memory": "1Gi"
                }
            },
            "autoscaling": {
                "enabled": False,
                "minReplicas": 1,
                "maxReplicas": 10,
                "targetCPUUtilizationPercentage": 80
            },
            "monitoring": {
                "enabled": True,
                "port": 9090
            }
        }


class TestScenarios:
    """Predefined test scenarios for different use cases."""
    
    @staticmethod
    def basic_model_creation_scenario() -> Dict[str, Any]:
        """Basic model creation test scenario."""
        return {
            "name": "basic_model_creation",
            "description": "Test basic ONNX model wrapping",
            "steps": [
                {"action": "create_model_file", "params": {"type": "onnx", "size": 1024}},
                {"action": "create_config", "params": {"name": "basic-model"}},
                {"action": "wrap_model", "params": {}},
                {"action": "validate_output", "params": {"expect_openapi": True}}
            ],
            "expected_artifacts": [
                "openapi.json",
                "triton_config.pbtxt",
                "Dockerfile"
            ]
        }
    
    @staticmethod
    def container_build_scenario() -> Dict[str, Any]:
        """Container building test scenario."""
        return {
            "name": "container_build",
            "description": "Test container image building",
            "prerequisites": ["docker_available"],
            "steps": [
                {"action": "create_model_file", "params": {"type": "onnx"}},
                {"action": "wrap_model", "params": {}},
                {"action": "build_container", "params": {"tag": "test:latest"}},
                {"action": "verify_container", "params": {"expect_running": True}}
            ],
            "cleanup": [
                {"action": "remove_container", "params": {}},
                {"action": "remove_image", "params": {}}
            ]
        }
    
    @staticmethod
    def kubernetes_deployment_scenario() -> Dict[str, Any]:
        """Kubernetes deployment test scenario."""
        return {
            "name": "kubernetes_deployment",
            "description": "Test Kubernetes deployment workflow",
            "prerequisites": ["kubernetes_available", "docker_available"],
            "steps": [
                {"action": "create_namespace", "params": {"name": "test-nimify"}},
                {"action": "build_and_push_image", "params": {}},
                {"action": "generate_helm_chart", "params": {}},
                {"action": "deploy_to_k8s", "params": {}},
                {"action": "wait_for_ready", "params": {"timeout": 300}},
                {"action": "test_inference", "params": {}}
            ],
            "cleanup": [
                {"action": "delete_deployment", "params": {}},
                {"action": "delete_namespace", "params": {}}
            ]
        }
    
    @staticmethod
    def performance_test_scenario() -> Dict[str, Any]:
        """Performance testing scenario."""
        return {
            "name": "performance_test",
            "description": "Test model serving performance",
            "prerequisites": ["model_deployed"],
            "steps": [
                {"action": "warmup_requests", "params": {"count": 10}},
                {"action": "measure_latency", "params": {"requests": 100}},
                {"action": "measure_throughput", "params": {"duration": 60}},
                {"action": "stress_test", "params": {"concurrent": 50}}
            ],
            "assertions": [
                {"metric": "p99_latency", "threshold": 100, "unit": "ms"},
                {"metric": "throughput", "threshold": 1000, "unit": "rps"},
                {"metric": "error_rate", "threshold": 0.01, "unit": "percent"}
            ]
        }