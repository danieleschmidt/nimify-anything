"""
Advanced Quality Gates Testing Framework

Comprehensive testing framework that validates system quality across multiple
dimensions including performance, security, reliability, and AI-driven insights.
"""

import asyncio
import pytest
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nimify.core import ModelConfig, Nimifier, NIMService
from nimify.api import app
from nimify.validation import ServiceNameValidator, ValidationError


class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str, threshold: float, weight: float = 1.0):
        self.name = name
        self.threshold = threshold
        self.weight = weight
        self.results = []
    
    async def evaluate(self, system_context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate quality gate."""
        raise NotImplementedError
    
    def passed(self) -> bool:
        """Check if quality gate passed."""
        if not self.results:
            return False
        return all(result.get("passed", False) for result in self.results)
    
    def score(self) -> float:
        """Get quality score."""
        if not self.results:
            return 0.0
        scores = [result.get("score", 0.0) for result in self.results]
        return np.mean(scores)


class PerformanceQualityGate(QualityGate):
    """Performance quality gate."""
    
    async def evaluate(self, system_context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate performance characteristics."""
        result = {
            "gate_name": self.name,
            "timestamp": time.time(),
            "metrics": {}
        }
        
        try:
            # Simulate performance testing
            latency_ms = await self._measure_latency(system_context)
            throughput_rps = await self._measure_throughput(system_context)
            memory_usage_mb = await self._measure_memory_usage(system_context)
            cpu_utilization = await self._measure_cpu_usage(system_context)
            
            result["metrics"] = {
                "latency_ms": latency_ms,
                "throughput_rps": throughput_rps,
                "memory_usage_mb": memory_usage_mb,
                "cpu_utilization": cpu_utilization
            }
            
            # Performance scoring
            latency_score = max(0, 1 - (latency_ms - 10) / 90) if latency_ms > 10 else 1.0
            throughput_score = min(1.0, throughput_rps / 100) if throughput_rps > 0 else 0.0
            memory_score = max(0, 1 - (memory_usage_mb - 500) / 1500) if memory_usage_mb > 500 else 1.0
            cpu_score = max(0, 1 - cpu_utilization / 100) if cpu_utilization > 0 else 1.0
            
            overall_score = (latency_score + throughput_score + memory_score + cpu_score) / 4
            
            result["score"] = overall_score
            result["passed"] = overall_score >= self.threshold
            result["details"] = {
                "latency_score": latency_score,
                "throughput_score": throughput_score,
                "memory_score": memory_score,
                "cpu_score": cpu_score
            }
            
        except Exception as e:
            result["error"] = str(e)
            result["passed"] = False
            result["score"] = 0.0
        
        self.results.append(result)
        return result
    
    async def _measure_latency(self, context: Dict[str, Any]) -> float:
        """Measure average response latency."""
        # Simulate latency measurement
        base_latency = 25.0  # Base latency in ms
        load_factor = context.get("load_factor", 1.0)
        optimization_factor = context.get("optimization_level", 1.0)
        
        simulated_latency = base_latency * load_factor / optimization_factor
        return simulated_latency + np.random.normal(0, 2)  # Add realistic noise
    
    async def _measure_throughput(self, context: Dict[str, Any]) -> float:
        """Measure requests per second throughput."""
        base_throughput = 80.0
        batch_size = context.get("batch_size", 32)
        parallel_workers = context.get("parallel_workers", 4)
        
        simulated_throughput = base_throughput * (batch_size / 32) * (parallel_workers / 4)
        return max(1, simulated_throughput + np.random.normal(0, 5))
    
    async def _measure_memory_usage(self, context: Dict[str, Any]) -> float:
        """Measure memory usage in MB."""
        base_memory = 800.0
        model_complexity = context.get("model_complexity", 1.0)
        
        simulated_memory = base_memory * model_complexity
        return simulated_memory + np.random.normal(0, 50)
    
    async def _measure_cpu_usage(self, context: Dict[str, Any]) -> float:
        """Measure CPU utilization percentage."""
        base_cpu = 40.0
        load_factor = context.get("load_factor", 1.0)
        
        simulated_cpu = base_cpu * load_factor
        return min(100, max(0, simulated_cpu + np.random.normal(0, 5)))


class SecurityQualityGate(QualityGate):
    """Security quality gate."""
    
    async def evaluate(self, system_context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate security characteristics."""
        result = {
            "gate_name": self.name,
            "timestamp": time.time(),
            "security_checks": {}
        }
        
        try:
            # Security assessments
            vulnerability_scan = await self._vulnerability_scan(system_context)
            authentication_check = await self._authentication_check(system_context)
            authorization_check = await self._authorization_check(system_context)
            data_encryption_check = await self._data_encryption_check(system_context)
            input_validation_check = await self._input_validation_check(system_context)
            
            result["security_checks"] = {
                "vulnerability_scan": vulnerability_scan,
                "authentication": authentication_check,
                "authorization": authorization_check,
                "data_encryption": data_encryption_check,
                "input_validation": input_validation_check
            }
            
            # Calculate security score
            checks = [vulnerability_scan, authentication_check, authorization_check, 
                     data_encryption_check, input_validation_check]
            security_score = np.mean([check["score"] for check in checks])
            
            result["score"] = security_score
            result["passed"] = security_score >= self.threshold
            result["vulnerabilities_found"] = sum(1 for check in checks if check["vulnerabilities"] > 0)
            
        except Exception as e:
            result["error"] = str(e)
            result["passed"] = False
            result["score"] = 0.0
        
        self.results.append(result)
        return result
    
    async def _vulnerability_scan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate vulnerability scanning."""
        # Mock vulnerability scan results
        vulnerabilities = np.random.poisson(0.5)  # Average 0.5 vulnerabilities
        severity_levels = np.random.choice(["low", "medium", "high", "critical"], 
                                         size=vulnerabilities, 
                                         p=[0.4, 0.3, 0.2, 0.1])
        
        score = max(0, 1 - vulnerabilities * 0.2)  # Reduce score for each vulnerability
        
        return {
            "vulnerabilities": vulnerabilities,
            "severity_distribution": dict(zip(*np.unique(severity_levels, return_counts=True))) if vulnerabilities > 0 else {},
            "score": score,
            "scan_duration_ms": np.random.uniform(1000, 3000)
        }
    
    async def _authentication_check(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check authentication mechanisms."""
        has_auth = context.get("authentication_enabled", True)
        multi_factor = context.get("multi_factor_auth", False)
        token_expiry = context.get("token_expiry_hours", 24)
        
        score = 0.0
        if has_auth:
            score += 0.6
        if multi_factor:
            score += 0.3
        if token_expiry <= 8:  # Shorter token expiry is more secure
            score += 0.1
        
        return {
            "authentication_enabled": has_auth,
            "multi_factor_enabled": multi_factor,
            "token_expiry_hours": token_expiry,
            "score": min(1.0, score),
            "vulnerabilities": 0 if score >= 0.8 else 1
        }
    
    async def _authorization_check(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check authorization and access control."""
        rbac_enabled = context.get("rbac_enabled", True)
        api_key_required = context.get("api_key_required", True)
        rate_limiting = context.get("rate_limiting_enabled", True)
        
        score = 0.0
        if rbac_enabled:
            score += 0.4
        if api_key_required:
            score += 0.3
        if rate_limiting:
            score += 0.3
        
        return {
            "rbac_enabled": rbac_enabled,
            "api_key_required": api_key_required,
            "rate_limiting_enabled": rate_limiting,
            "score": score,
            "vulnerabilities": 0 if score >= 0.7 else 1
        }
    
    async def _data_encryption_check(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check data encryption."""
        tls_enabled = context.get("tls_enabled", True)
        data_at_rest_encrypted = context.get("data_encryption_at_rest", True)
        key_rotation = context.get("key_rotation_enabled", False)
        
        score = 0.0
        if tls_enabled:
            score += 0.5
        if data_at_rest_encrypted:
            score += 0.3
        if key_rotation:
            score += 0.2
        
        return {
            "tls_enabled": tls_enabled,
            "data_at_rest_encrypted": data_at_rest_encrypted,
            "key_rotation_enabled": key_rotation,
            "score": score,
            "vulnerabilities": 0 if score >= 0.8 else 1
        }
    
    async def _input_validation_check(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check input validation."""
        input_sanitization = context.get("input_sanitization", True)
        sql_injection_protection = context.get("sql_injection_protection", True)
        xss_protection = context.get("xss_protection", True)
        
        score = 0.0
        if input_sanitization:
            score += 0.4
        if sql_injection_protection:
            score += 0.3
        if xss_protection:
            score += 0.3
        
        return {
            "input_sanitization": input_sanitization,
            "sql_injection_protection": sql_injection_protection,
            "xss_protection": xss_protection,
            "score": score,
            "vulnerabilities": 0 if score >= 0.9 else 1
        }


class ReliabilityQualityGate(QualityGate):
    """Reliability and stability quality gate."""
    
    async def evaluate(self, system_context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate system reliability."""
        result = {
            "gate_name": self.name,
            "timestamp": time.time(),
            "reliability_metrics": {}
        }
        
        try:
            # Reliability assessments
            error_rate = await self._measure_error_rate(system_context)
            uptime_percentage = await self._measure_uptime(system_context)
            recovery_time = await self._measure_recovery_time(system_context)
            fault_tolerance = await self._assess_fault_tolerance(system_context)
            
            result["reliability_metrics"] = {
                "error_rate": error_rate,
                "uptime_percentage": uptime_percentage,
                "recovery_time_seconds": recovery_time,
                "fault_tolerance_score": fault_tolerance
            }
            
            # Calculate reliability score
            error_score = max(0, 1 - error_rate / 0.05)  # Target <5% error rate
            uptime_score = uptime_percentage / 100
            recovery_score = max(0, 1 - recovery_time / 60)  # Target <60s recovery
            
            reliability_score = (error_score + uptime_score + recovery_score + fault_tolerance) / 4
            
            result["score"] = reliability_score
            result["passed"] = reliability_score >= self.threshold
            
        except Exception as e:
            result["error"] = str(e)
            result["passed"] = False
            result["score"] = 0.0
        
        self.results.append(result)
        return result
    
    async def _measure_error_rate(self, context: Dict[str, Any]) -> float:
        """Measure system error rate."""
        base_error_rate = 0.01  # 1% base error rate
        load_factor = context.get("load_factor", 1.0)
        stability_measures = context.get("stability_measures", 1.0)
        
        simulated_error_rate = base_error_rate * load_factor / stability_measures
        return max(0, simulated_error_rate + np.random.normal(0, 0.005))
    
    async def _measure_uptime(self, context: Dict[str, Any]) -> float:
        """Measure system uptime percentage."""
        base_uptime = 99.5  # 99.5% base uptime
        redundancy = context.get("redundancy_level", 1.0)
        
        simulated_uptime = base_uptime + (redundancy - 1) * 0.3
        return min(100, max(90, simulated_uptime + np.random.normal(0, 0.2)))
    
    async def _measure_recovery_time(self, context: Dict[str, Any]) -> float:
        """Measure average recovery time from failures."""
        base_recovery = 30.0  # 30 second base recovery
        automation_level = context.get("automation_level", 1.0)
        
        simulated_recovery = base_recovery / automation_level
        return max(1, simulated_recovery + np.random.normal(0, 5))
    
    async def _assess_fault_tolerance(self, context: Dict[str, Any]) -> float:
        """Assess fault tolerance capabilities."""
        circuit_breaker = context.get("circuit_breaker_enabled", False)
        retry_mechanism = context.get("retry_mechanism", False)
        graceful_degradation = context.get("graceful_degradation", False)
        health_checks = context.get("health_checks_enabled", False)
        
        tolerance_score = 0.0
        if circuit_breaker:
            tolerance_score += 0.3
        if retry_mechanism:
            tolerance_score += 0.2
        if graceful_degradation:
            tolerance_score += 0.3
        if health_checks:
            tolerance_score += 0.2
        
        return tolerance_score


class AIInsightsQualityGate(QualityGate):
    """AI-driven quality insights gate."""
    
    async def evaluate(self, system_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-driven quality insights."""
        result = {
            "gate_name": self.name,
            "timestamp": time.time(),
            "ai_insights": {}
        }
        
        try:
            # AI-powered analysis
            anomaly_detection = await self._detect_anomalies(system_context)
            pattern_analysis = await self._analyze_patterns(system_context)
            optimization_suggestions = await self._suggest_optimizations(system_context)
            risk_assessment = await self._assess_risks(system_context)
            
            result["ai_insights"] = {
                "anomaly_detection": anomaly_detection,
                "pattern_analysis": pattern_analysis,
                "optimization_suggestions": optimization_suggestions,
                "risk_assessment": risk_assessment
            }
            
            # Calculate AI insights score
            anomaly_score = 1.0 - anomaly_detection["anomaly_count"] * 0.1
            pattern_score = pattern_analysis["confidence_score"]
            optimization_score = len(optimization_suggestions) * 0.1
            risk_score = 1.0 - risk_assessment["overall_risk_level"]
            
            ai_score = (anomaly_score + pattern_score + optimization_score + risk_score) / 4
            
            result["score"] = min(1.0, max(0.0, ai_score))
            result["passed"] = result["score"] >= self.threshold
            
        except Exception as e:
            result["error"] = str(e)
            result["passed"] = False
            result["score"] = 0.0
        
        self.results.append(result)
        return result
    
    async def _detect_anomalies(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """AI-powered anomaly detection."""
        # Simulate AI anomaly detection
        anomaly_count = np.random.poisson(1)  # Average 1 anomaly
        
        anomalies = []
        for i in range(anomaly_count):
            anomaly = {
                "type": np.random.choice(["performance", "security", "resource"]),
                "severity": np.random.choice(["low", "medium", "high"], p=[0.6, 0.3, 0.1]),
                "confidence": np.random.uniform(0.7, 0.95),
                "description": f"Detected anomaly {i+1} in system behavior",
                "timestamp": time.time() - np.random.uniform(0, 3600)
            }
            anomalies.append(anomaly)
        
        return {
            "anomaly_count": anomaly_count,
            "anomalies": anomalies,
            "detection_confidence": np.mean([a["confidence"] for a in anomalies]) if anomalies else 1.0
        }
    
    async def _analyze_patterns(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """AI pattern analysis."""
        # Simulate pattern recognition
        patterns = [
            {
                "pattern_type": "traffic_spike",
                "frequency": "hourly",
                "confidence": 0.85,
                "impact": "medium"
            },
            {
                "pattern_type": "resource_usage_cycle",
                "frequency": "daily",
                "confidence": 0.92,
                "impact": "low"
            },
            {
                "pattern_type": "error_clustering",
                "frequency": "irregular",
                "confidence": 0.78,
                "impact": "high"
            }
        ]
        
        # Randomly select patterns
        selected_patterns = np.random.choice(patterns, size=np.random.randint(1, 4), replace=False).tolist()
        
        return {
            "patterns_detected": len(selected_patterns),
            "patterns": selected_patterns,
            "confidence_score": np.mean([p["confidence"] for p in selected_patterns])
        }
    
    async def _suggest_optimizations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """AI-generated optimization suggestions."""
        optimization_suggestions = [
            {
                "category": "performance",
                "suggestion": "Increase batch size to 64 for better GPU utilization",
                "expected_improvement": "15-20% throughput increase",
                "confidence": 0.87,
                "implementation_effort": "low"
            },
            {
                "category": "resource",
                "suggestion": "Enable memory pooling to reduce allocation overhead",
                "expected_improvement": "10-15% memory efficiency",
                "confidence": 0.82,
                "implementation_effort": "medium"
            },
            {
                "category": "reliability",
                "suggestion": "Implement circuit breaker pattern for external dependencies",
                "expected_improvement": "Improved fault tolerance",
                "confidence": 0.90,
                "implementation_effort": "medium"
            },
            {
                "category": "security",
                "suggestion": "Enable request rate limiting per client",
                "expected_improvement": "Better DDoS protection",
                "confidence": 0.95,
                "implementation_effort": "low"
            }
        ]
        
        # Return random subset
        num_suggestions = np.random.randint(2, len(optimization_suggestions) + 1)
        return np.random.choice(optimization_suggestions, size=num_suggestions, replace=False).tolist()
    
    async def _assess_risks(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """AI-powered risk assessment."""
        risk_categories = {
            "performance_degradation": np.random.uniform(0.1, 0.4),
            "security_breach": np.random.uniform(0.05, 0.2),
            "system_failure": np.random.uniform(0.02, 0.15),
            "data_loss": np.random.uniform(0.01, 0.1),
            "compliance_violation": np.random.uniform(0.03, 0.12)
        }
        
        overall_risk = np.mean(list(risk_categories.values()))
        
        return {
            "risk_categories": risk_categories,
            "overall_risk_level": overall_risk,
            "risk_trend": np.random.choice(["increasing", "stable", "decreasing"], p=[0.2, 0.6, 0.2]),
            "mitigation_priority": max(risk_categories, key=risk_categories.get)
        }


class AdvancedQualityGateFramework:
    """Advanced quality gate testing framework."""
    
    def __init__(self):
        self.quality_gates: List[QualityGate] = []
        self.evaluation_results: List[Dict[str, Any]] = []
        self.overall_score: float = 0.0
        self.passed: bool = False
        
    def add_quality_gate(self, gate: QualityGate) -> None:
        """Add a quality gate to the framework."""
        self.quality_gates.append(gate)
    
    async def evaluate_all_gates(self, system_context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all quality gates."""
        print("🔍 Running Advanced Quality Gate Framework...")
        
        gate_results = []
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for gate in self.quality_gates:
            print(f"  🎯 Evaluating {gate.name}...")
            result = await gate.evaluate(system_context)
            gate_results.append(result)
            
            # Accumulate weighted scores
            total_weighted_score += result.get("score", 0.0) * gate.weight
            total_weight += gate.weight
        
        # Calculate overall metrics
        self.overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        self.passed = all(gate.passed() for gate in self.quality_gates)
        
        evaluation_summary = {
            "framework_version": "1.0",
            "evaluation_timestamp": time.time(),
            "total_gates_evaluated": len(self.quality_gates),
            "gates_passed": sum(1 for gate in self.quality_gates if gate.passed()),
            "overall_score": self.overall_score,
            "overall_passed": self.passed,
            "gate_results": gate_results,
            "system_context": system_context,
            "recommendations": self._generate_recommendations(gate_results)
        }
        
        self.evaluation_results.append(evaluation_summary)
        return evaluation_summary
    
    def _generate_recommendations(self, gate_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate improvement recommendations."""
        recommendations = []
        
        for result in gate_results:
            if not result.get("passed", False):
                gate_name = result.get("gate_name", "Unknown")
                score = result.get("score", 0.0)
                
                recommendation = {
                    "priority": "high" if score < 0.5 else "medium",
                    "gate": gate_name,
                    "current_score": score,
                    "target_score": max(0.8, score + 0.2),
                    "improvement_areas": self._identify_improvement_areas(result)
                }
                recommendations.append(recommendation)
        
        return recommendations
    
    def _identify_improvement_areas(self, gate_result: Dict[str, Any]) -> List[str]:
        """Identify specific areas for improvement."""
        areas = []
        
        if "performance" in gate_result.get("gate_name", "").lower():
            if gate_result.get("metrics", {}).get("latency_ms", 0) > 50:
                areas.append("Reduce response latency")
            if gate_result.get("metrics", {}).get("throughput_rps", 0) < 100:
                areas.append("Increase throughput capacity")
            if gate_result.get("metrics", {}).get("memory_usage_mb", 0) > 1000:
                areas.append("Optimize memory usage")
        
        elif "security" in gate_result.get("gate_name", "").lower():
            if gate_result.get("vulnerabilities_found", 0) > 0:
                areas.append("Address security vulnerabilities")
            if not gate_result.get("security_checks", {}).get("authentication", {}).get("multi_factor_enabled", False):
                areas.append("Implement multi-factor authentication")
        
        elif "reliability" in gate_result.get("gate_name", "").lower():
            if gate_result.get("reliability_metrics", {}).get("error_rate", 0) > 0.02:
                areas.append("Reduce error rate")
            if gate_result.get("reliability_metrics", {}).get("uptime_percentage", 0) < 99.0:
                areas.append("Improve system uptime")
        
        return areas if areas else ["Review gate-specific metrics for improvement opportunities"]
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality assessment report."""
        if not self.evaluation_results:
            return {"error": "No evaluation results available"}
        
        latest_evaluation = self.evaluation_results[-1]
        
        report = {
            "executive_summary": {
                "overall_quality_score": self.overall_score,
                "quality_grade": self._calculate_quality_grade(self.overall_score),
                "gates_evaluated": len(self.quality_gates),
                "gates_passed": sum(1 for gate in self.quality_gates if gate.passed()),
                "critical_issues": self._identify_critical_issues(latest_evaluation),
                "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest_evaluation["evaluation_timestamp"]))
            },
            "detailed_gate_analysis": self._analyze_gate_performance(latest_evaluation),
            "trend_analysis": self._analyze_trends(),
            "improvement_roadmap": self._create_improvement_roadmap(latest_evaluation),
            "compliance_status": self._assess_compliance_status(latest_evaluation),
            "risk_matrix": self._create_risk_matrix(latest_evaluation),
            "recommendations": latest_evaluation.get("recommendations", [])
        }
        
        return report
    
    def _calculate_quality_grade(self, score: float) -> str:
        """Calculate quality grade based on score."""
        if score >= 0.95:
            return "A+"
        elif score >= 0.90:
            return "A"
        elif score >= 0.85:
            return "A-"
        elif score >= 0.80:
            return "B+"
        elif score >= 0.75:
            return "B"
        elif score >= 0.70:
            return "B-"
        elif score >= 0.65:
            return "C+"
        elif score >= 0.60:
            return "C"
        else:
            return "F"
    
    def _identify_critical_issues(self, evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify critical quality issues."""
        critical_issues = []
        
        for result in evaluation.get("gate_results", []):
            if result.get("score", 1.0) < 0.5:
                issue = {
                    "gate": result.get("gate_name", "Unknown"),
                    "severity": "critical",
                    "score": result.get("score", 0.0),
                    "description": f"Quality gate {result.get('gate_name', 'Unknown')} failed with low score"
                }
                critical_issues.append(issue)
        
        return critical_issues
    
    def _analyze_gate_performance(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual gate performance."""
        gate_analysis = {}
        
        for result in evaluation.get("gate_results", []):
            gate_name = result.get("gate_name", "Unknown")
            gate_analysis[gate_name] = {
                "score": result.get("score", 0.0),
                "passed": result.get("passed", False),
                "performance_tier": self._categorize_performance(result.get("score", 0.0)),
                "key_metrics": self._extract_key_metrics(result),
                "improvement_potential": 1.0 - result.get("score", 1.0)
            }
        
        return gate_analysis
    
    def _categorize_performance(self, score: float) -> str:
        """Categorize performance level."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.7:
            return "acceptable"
        elif score >= 0.6:
            return "needs_improvement"
        else:
            return "critical"
    
    def _extract_key_metrics(self, gate_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from gate result."""
        key_metrics = {}
        
        if "metrics" in gate_result:
            key_metrics.update(gate_result["metrics"])
        
        if "security_checks" in gate_result:
            key_metrics["security_score"] = gate_result.get("score", 0.0)
            key_metrics["vulnerabilities"] = gate_result.get("vulnerabilities_found", 0)
        
        if "reliability_metrics" in gate_result:
            key_metrics.update(gate_result["reliability_metrics"])
        
        return key_metrics
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze quality trends over time."""
        if len(self.evaluation_results) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        scores = [result["overall_score"] for result in self.evaluation_results]
        timestamps = [result["evaluation_timestamp"] for result in self.evaluation_results]
        
        # Simple trend analysis
        score_trend = "stable"
        if len(scores) >= 2:
            recent_change = scores[-1] - scores[-2]
            if recent_change > 0.05:
                score_trend = "improving"
            elif recent_change < -0.05:
                score_trend = "declining"
        
        return {
            "overall_trend": score_trend,
            "score_history": scores,
            "evaluation_timestamps": timestamps,
            "average_score": np.mean(scores),
            "score_volatility": np.std(scores)
        }
    
    def _create_improvement_roadmap(self, evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create improvement roadmap."""
        roadmap = []
        
        recommendations = evaluation.get("recommendations", [])
        
        # Sort by priority
        high_priority = [r for r in recommendations if r.get("priority") == "high"]
        medium_priority = [r for r in recommendations if r.get("priority") == "medium"]
        
        # Create roadmap phases
        if high_priority:
            roadmap.append({
                "phase": "immediate",
                "duration": "1-2 weeks",
                "focus": "Critical quality issues",
                "items": high_priority[:3]  # Top 3 high priority items
            })
        
        if medium_priority:
            roadmap.append({
                "phase": "short-term",
                "duration": "1-2 months",
                "focus": "Quality improvements",
                "items": medium_priority[:5]  # Top 5 medium priority items
            })
        
        roadmap.append({
            "phase": "long-term",
            "duration": "3-6 months", 
            "focus": "Continuous improvement and optimization",
            "items": [
                {"description": "Implement automated quality monitoring"},
                {"description": "Enhance AI-driven insights"},
                {"description": "Optimize system architecture"}
            ]
        })
        
        return roadmap
    
    def _assess_compliance_status(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compliance with quality standards."""
        compliance_requirements = {
            "performance_sla": {"threshold": 0.8, "current": 0.0},
            "security_standards": {"threshold": 0.9, "current": 0.0},
            "reliability_targets": {"threshold": 0.85, "current": 0.0},
            "ai_governance": {"threshold": 0.7, "current": 0.0}
        }
        
        # Extract compliance scores from gate results
        for result in evaluation.get("gate_results", []):
            gate_name = result.get("gate_name", "").lower()
            score = result.get("score", 0.0)
            
            if "performance" in gate_name:
                compliance_requirements["performance_sla"]["current"] = score
            elif "security" in gate_name:
                compliance_requirements["security_standards"]["current"] = score
            elif "reliability" in gate_name:
                compliance_requirements["reliability_targets"]["current"] = score
            elif "ai" in gate_name:
                compliance_requirements["ai_governance"]["current"] = score
        
        # Calculate compliance status
        compliant_areas = 0
        total_areas = len(compliance_requirements)
        
        for req_name, req_data in compliance_requirements.items():
            if req_data["current"] >= req_data["threshold"]:
                compliant_areas += 1
        
        return {
            "overall_compliance": compliant_areas / total_areas,
            "compliant_areas": compliant_areas,
            "total_areas": total_areas,
            "detailed_compliance": compliance_requirements,
            "compliance_grade": "Pass" if compliant_areas == total_areas else "Conditional" if compliant_areas >= total_areas * 0.8 else "Fail"
        }
    
    def _create_risk_matrix(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Create quality risk matrix."""
        risk_matrix = {
            "high_impact_high_probability": [],
            "high_impact_low_probability": [],
            "low_impact_high_probability": [],
            "low_impact_low_probability": []
        }
        
        for result in evaluation.get("gate_results", []):
            gate_name = result.get("gate_name", "Unknown")
            score = result.get("score", 1.0)
            
            # Risk assessment based on score
            probability = 1.0 - score  # Lower score = higher probability of issues
            
            # Impact assessment based on gate type
            impact = 0.5  # Default medium impact
            if "security" in gate_name.lower():
                impact = 0.9  # High impact
            elif "performance" in gate_name.lower():
                impact = 0.7  # Medium-high impact
            elif "reliability" in gate_name.lower():
                impact = 0.8  # High impact
            
            # Categorize risk
            if impact > 0.7 and probability > 0.3:
                risk_matrix["high_impact_high_probability"].append({
                    "gate": gate_name,
                    "impact": impact,
                    "probability": probability,
                    "risk_score": impact * probability
                })
            elif impact > 0.7 and probability <= 0.3:
                risk_matrix["high_impact_low_probability"].append({
                    "gate": gate_name,
                    "impact": impact,
                    "probability": probability,
                    "risk_score": impact * probability
                })
            elif impact <= 0.7 and probability > 0.3:
                risk_matrix["low_impact_high_probability"].append({
                    "gate": gate_name,
                    "impact": impact,
                    "probability": probability,
                    "risk_score": impact * probability
                })
            else:
                risk_matrix["low_impact_low_probability"].append({
                    "gate": gate_name,
                    "impact": impact,
                    "probability": probability,
                    "risk_score": impact * probability
                })
        
        return risk_matrix


class TestAdvancedQualityGates:
    """Test suite for advanced quality gates."""
    
    @pytest.fixture
    def quality_framework(self):
        """Create quality gate framework."""
        framework = AdvancedQualityGateFramework()
        
        # Add quality gates
        framework.add_quality_gate(PerformanceQualityGate("Performance Gate", 0.8, 1.0))
        framework.add_quality_gate(SecurityQualityGate("Security Gate", 0.85, 1.2))
        framework.add_quality_gate(ReliabilityQualityGate("Reliability Gate", 0.75, 0.9))
        framework.add_quality_gate(AIInsightsQualityGate("AI Insights Gate", 0.7, 0.8))
        
        return framework
    
    @pytest.fixture
    def system_context(self):
        """Create system context for testing."""
        return {
            "load_factor": 1.2,
            "optimization_level": 1.5,
            "batch_size": 64,
            "parallel_workers": 8,
            "model_complexity": 1.3,
            "authentication_enabled": True,
            "multi_factor_auth": True,
            "token_expiry_hours": 4,
            "rbac_enabled": True,
            "api_key_required": True,
            "rate_limiting_enabled": True,
            "tls_enabled": True,
            "data_encryption_at_rest": True,
            "key_rotation_enabled": False,
            "input_sanitization": True,
            "sql_injection_protection": True,
            "xss_protection": True,
            "stability_measures": 1.2,
            "redundancy_level": 2.0,
            "automation_level": 1.5,
            "circuit_breaker_enabled": True,
            "retry_mechanism": True,
            "graceful_degradation": True,
            "health_checks_enabled": True
        }
    
    @pytest.mark.asyncio
    async def test_performance_quality_gate(self, system_context):
        """Test performance quality gate."""
        gate = PerformanceQualityGate("Test Performance Gate", 0.7)
        result = await gate.evaluate(system_context)
        
        assert "gate_name" in result
        assert "metrics" in result
        assert "score" in result
        assert "passed" in result
        
        assert result["metrics"]["latency_ms"] > 0
        assert result["metrics"]["throughput_rps"] > 0
        assert result["metrics"]["memory_usage_mb"] > 0
        assert result["metrics"]["cpu_utilization"] >= 0
        
        assert 0.0 <= result["score"] <= 1.0
        assert isinstance(result["passed"], bool)
    
    @pytest.mark.asyncio
    async def test_security_quality_gate(self, system_context):
        """Test security quality gate."""
        gate = SecurityQualityGate("Test Security Gate", 0.8)
        result = await gate.evaluate(system_context)
        
        assert "gate_name" in result
        assert "security_checks" in result
        assert "score" in result
        assert "passed" in result
        
        security_checks = result["security_checks"]
        assert "vulnerability_scan" in security_checks
        assert "authentication" in security_checks
        assert "authorization" in security_checks
        assert "data_encryption" in security_checks
        assert "input_validation" in security_checks
        
        assert 0.0 <= result["score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_reliability_quality_gate(self, system_context):
        """Test reliability quality gate."""
        gate = ReliabilityQualityGate("Test Reliability Gate", 0.75)
        result = await gate.evaluate(system_context)
        
        assert "gate_name" in result
        assert "reliability_metrics" in result
        assert "score" in result
        assert "passed" in result
        
        metrics = result["reliability_metrics"]
        assert "error_rate" in metrics
        assert "uptime_percentage" in metrics
        assert "recovery_time_seconds" in metrics
        assert "fault_tolerance_score" in metrics
        
        assert metrics["error_rate"] >= 0
        assert 90 <= metrics["uptime_percentage"] <= 100
        assert metrics["recovery_time_seconds"] > 0
        assert 0.0 <= metrics["fault_tolerance_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_ai_insights_quality_gate(self, system_context):
        """Test AI insights quality gate."""
        gate = AIInsightsQualityGate("Test AI Gate", 0.6)
        result = await gate.evaluate(system_context)
        
        assert "gate_name" in result
        assert "ai_insights" in result
        assert "score" in result
        assert "passed" in result
        
        insights = result["ai_insights"]
        assert "anomaly_detection" in insights
        assert "pattern_analysis" in insights
        assert "optimization_suggestions" in insights
        assert "risk_assessment" in insights
        
        anomalies = insights["anomaly_detection"]
        assert "anomaly_count" in anomalies
        assert "detection_confidence" in anomalies
        
        patterns = insights["pattern_analysis"]
        assert "patterns_detected" in patterns
        assert "confidence_score" in patterns
    
    @pytest.mark.asyncio
    async def test_quality_framework_evaluation(self, quality_framework, system_context):
        """Test complete quality framework evaluation."""
        evaluation_result = await quality_framework.evaluate_all_gates(system_context)
        
        assert "framework_version" in evaluation_result
        assert "evaluation_timestamp" in evaluation_result
        assert "total_gates_evaluated" in evaluation_result
        assert "gates_passed" in evaluation_result
        assert "overall_score" in evaluation_result
        assert "overall_passed" in evaluation_result
        assert "gate_results" in evaluation_result
        assert "recommendations" in evaluation_result
        
        assert evaluation_result["total_gates_evaluated"] == 4
        assert 0.0 <= evaluation_result["overall_score"] <= 1.0
        assert len(evaluation_result["gate_results"]) == 4
    
    @pytest.mark.asyncio
    async def test_comprehensive_report_generation(self, quality_framework, system_context):
        """Test comprehensive report generation."""
        # Run evaluation first
        await quality_framework.evaluate_all_gates(system_context)
        
        # Generate comprehensive report
        report = quality_framework.generate_comprehensive_report()
        
        assert "executive_summary" in report
        assert "detailed_gate_analysis" in report
        assert "improvement_roadmap" in report
        assert "compliance_status" in report
        assert "risk_matrix" in report
        assert "recommendations" in report
        
        executive_summary = report["executive_summary"]
        assert "overall_quality_score" in executive_summary
        assert "quality_grade" in executive_summary
        assert "gates_evaluated" in executive_summary
        assert "gates_passed" in executive_summary
        
        # Test quality grade calculation
        grade = executive_summary["quality_grade"]
        assert grade in ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "F"]
    
    def test_quality_gate_scoring(self):
        """Test quality gate scoring mechanisms."""
        gate = PerformanceQualityGate("Test Gate", 0.8)
        
        # Add mock results
        gate.results = [
            {"score": 0.9, "passed": True},
            {"score": 0.7, "passed": False},
            {"score": 0.85, "passed": True}
        ]
        
        assert gate.score() == 0.8166666666666667  # (0.9 + 0.7 + 0.85) / 3
        assert not gate.passed()  # Not all results passed
        
        # Test with all passing results
        gate.results = [
            {"score": 0.9, "passed": True},
            {"score": 0.85, "passed": True}
        ]
        
        assert gate.passed()  # All results passed
    
    def test_improvement_recommendations(self, quality_framework):
        """Test improvement recommendation generation."""
        # Mock some failing gate results
        mock_gate_results = [
            {
                "gate_name": "Performance Gate",
                "score": 0.6,
                "passed": False,
                "metrics": {"latency_ms": 80, "throughput_rps": 50}
            },
            {
                "gate_name": "Security Gate", 
                "score": 0.4,
                "passed": False,
                "vulnerabilities_found": 3,
                "security_checks": {"authentication": {"multi_factor_enabled": False}}
            }
        ]
        
        recommendations = quality_framework._generate_recommendations(mock_gate_results)
        
        assert len(recommendations) == 2
        assert all("priority" in rec for rec in recommendations)
        assert all("improvement_areas" in rec for rec in recommendations)
        
        # Check that high priority is assigned to low scores
        security_rec = next(rec for rec in recommendations if rec["gate"] == "Security Gate")
        assert security_rec["priority"] == "high"  # Score < 0.5


# Integration test
@pytest.mark.asyncio
async def test_full_quality_pipeline():
    """Test complete quality assessment pipeline."""
    print("\n🚀 Running Full Quality Assessment Pipeline...")
    
    # Create framework
    framework = AdvancedQualityGateFramework()
    
    # Add all quality gates
    framework.add_quality_gate(PerformanceQualityGate("Performance Assessment", 0.8, 1.0))
    framework.add_quality_gate(SecurityQualityGate("Security Assessment", 0.85, 1.2))
    framework.add_quality_gate(ReliabilityQualityGate("Reliability Assessment", 0.75, 0.9))
    framework.add_quality_gate(AIInsightsQualityGate("AI Insights Assessment", 0.7, 0.8))
    
    # System context with realistic values
    system_context = {
        "load_factor": 1.0,
        "optimization_level": 1.2,
        "batch_size": 32,
        "parallel_workers": 4,
        "model_complexity": 1.0,
        "authentication_enabled": True,
        "multi_factor_auth": False,  # Potential issue
        "token_expiry_hours": 24,    # Potential issue
        "rbac_enabled": True,
        "api_key_required": True,
        "rate_limiting_enabled": True,
        "tls_enabled": True,
        "data_encryption_at_rest": True,
        "key_rotation_enabled": False,  # Potential issue
        "input_sanitization": True,
        "sql_injection_protection": True,
        "xss_protection": True,
        "stability_measures": 1.0,
        "redundancy_level": 1.0,
        "automation_level": 1.0,
        "circuit_breaker_enabled": False,  # Potential issue
        "retry_mechanism": True,
        "graceful_degradation": False,     # Potential issue
        "health_checks_enabled": True
    }
    
    # Run evaluation
    evaluation_result = await framework.evaluate_all_gates(system_context)
    
    # Generate comprehensive report
    comprehensive_report = framework.generate_comprehensive_report()
    
    print(f"✅ Quality Assessment Complete!")
    print(f"   Overall Score: {evaluation_result['overall_score']:.3f}")
    print(f"   Quality Grade: {comprehensive_report['executive_summary']['quality_grade']}")
    print(f"   Gates Passed: {evaluation_result['gates_passed']}/{evaluation_result['total_gates_evaluated']}")
    print(f"   Recommendations: {len(evaluation_result['recommendations'])}")
    
    # Verify results
    assert evaluation_result["total_gates_evaluated"] == 4
    assert 0.0 <= evaluation_result["overall_score"] <= 1.0
    assert "recommendations" in evaluation_result
    assert "executive_summary" in comprehensive_report
    
    return evaluation_result, comprehensive_report


if __name__ == "__main__":
    # Run comprehensive test
    asyncio.run(test_full_quality_pipeline())