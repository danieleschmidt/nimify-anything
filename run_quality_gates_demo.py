"""
Demonstration of Advanced Quality Gates Framework

This script demonstrates the advanced quality gates system without pytest dependency.
"""

import asyncio
import json
import time
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import quality gates components
from tests.test_advanced_quality_gates import (
    AdvancedQualityGateFramework,
    PerformanceQualityGate,
    SecurityQualityGate,
    ReliabilityQualityGate,
    AIInsightsQualityGate
)


async def run_quality_gates_demonstration():
    """Run comprehensive quality gates demonstration."""
    print("🔬 NIMIFY ADVANCED QUALITY GATES DEMONSTRATION")
    print("=" * 60)
    
    # Create framework
    framework = AdvancedQualityGateFramework()
    
    # Add quality gates with different thresholds and weights
    print("🎯 Initializing Quality Gates...")
    framework.add_quality_gate(PerformanceQualityGate("Performance Assessment", 0.8, 1.0))
    framework.add_quality_gate(SecurityQualityGate("Security Assessment", 0.85, 1.2))
    framework.add_quality_gate(ReliabilityQualityGate("Reliability Assessment", 0.75, 0.9))
    framework.add_quality_gate(AIInsightsQualityGate("AI Insights Assessment", 0.7, 0.8))
    
    # System context representing current system state
    system_context = {
        # Performance parameters
        "load_factor": 1.0,
        "optimization_level": 1.2,
        "batch_size": 32,
        "parallel_workers": 4,
        "model_complexity": 1.0,
        
        # Security parameters
        "authentication_enabled": True,
        "multi_factor_auth": False,  # Potential security issue
        "token_expiry_hours": 24,    # Potential security issue
        "rbac_enabled": True,
        "api_key_required": True,
        "rate_limiting_enabled": True,
        "tls_enabled": True,
        "data_encryption_at_rest": True,
        "key_rotation_enabled": False,  # Potential security issue
        "input_sanitization": True,
        "sql_injection_protection": True,
        "xss_protection": True,
        
        # Reliability parameters
        "stability_measures": 1.0,
        "redundancy_level": 1.0,
        "automation_level": 1.0,
        "circuit_breaker_enabled": False,  # Potential reliability issue
        "retry_mechanism": True,
        "graceful_degradation": False,     # Potential reliability issue
        "health_checks_enabled": True
    }
    
    print("\n📊 System Context:")
    for category in ["Performance", "Security", "Reliability"]:
        relevant_params = {k: v for k, v in system_context.items() 
                          if category.lower() in k.lower() or 
                          (category == "Performance" and k in ["load_factor", "optimization_level", "batch_size", "parallel_workers", "model_complexity"]) or
                          (category == "Security" and k in ["authentication_enabled", "multi_factor_auth", "token_expiry_hours", "rbac_enabled", "api_key_required", "rate_limiting_enabled", "tls_enabled", "data_encryption_at_rest", "key_rotation_enabled", "input_sanitization", "sql_injection_protection", "xss_protection"]) or
                          (category == "Reliability" and k in ["stability_measures", "redundancy_level", "automation_level", "circuit_breaker_enabled", "retry_mechanism", "graceful_degradation", "health_checks_enabled"])}
        
        print(f"  {category}: {len(relevant_params)} parameters configured")
    
    # Run comprehensive evaluation
    print("\n🔍 Running Quality Gate Evaluations...")
    evaluation_result = await framework.evaluate_all_gates(system_context)
    
    # Display results
    print("\n📈 EVALUATION RESULTS:")
    print("-" * 40)
    print(f"Overall Quality Score: {evaluation_result['overall_score']:.3f}")
    print(f"Gates Passed: {evaluation_result['gates_passed']}/{evaluation_result['total_gates_evaluated']}")
    print(f"Overall Status: {'✅ PASSED' if evaluation_result['overall_passed'] else '❌ FAILED'}")
    
    print("\n🎯 Individual Gate Results:")
    for result in evaluation_result['gate_results']:
        gate_name = result['gate_name']
        score = result['score']
        passed = result['passed']
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"  {gate_name}: {score:.3f} {status}")
        
        # Show specific metrics for each gate type
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"    • Latency: {metrics.get('latency_ms', 0):.1f}ms")
            print(f"    • Throughput: {metrics.get('throughput_rps', 0):.1f} RPS")
            print(f"    • Memory: {metrics.get('memory_usage_mb', 0):.1f}MB")
            print(f"    • CPU: {metrics.get('cpu_utilization', 0):.1f}%")
        
        elif 'security_checks' in result:
            vuln_count = result.get('vulnerabilities_found', 0)
            print(f"    • Vulnerabilities Found: {vuln_count}")
            print(f"    • Authentication Score: {result['security_checks']['authentication']['score']:.3f}")
            print(f"    • Encryption Score: {result['security_checks']['data_encryption']['score']:.3f}")
        
        elif 'reliability_metrics' in result:
            metrics = result['reliability_metrics']
            print(f"    • Error Rate: {metrics.get('error_rate', 0):.4f}")
            print(f"    • Uptime: {metrics.get('uptime_percentage', 0):.2f}%")
            print(f"    • Recovery Time: {metrics.get('recovery_time_seconds', 0):.1f}s")
        
        elif 'ai_insights' in result:
            insights = result['ai_insights']
            print(f"    • Anomalies Detected: {insights['anomaly_detection']['anomaly_count']}")
            print(f"    • Patterns Found: {insights['pattern_analysis']['patterns_detected']}")
            print(f"    • Optimizations Suggested: {len(insights['optimization_suggestions'])}")
    
    # Generate comprehensive report
    print("\n📋 Generating Comprehensive Report...")
    comprehensive_report = framework.generate_comprehensive_report()
    
    print("\n🎖️  QUALITY ASSESSMENT SUMMARY:")
    print("-" * 40)
    executive_summary = comprehensive_report['executive_summary']
    print(f"Quality Grade: {executive_summary['quality_grade']}")
    print(f"Critical Issues: {len(executive_summary['critical_issues'])}")
    
    if executive_summary['critical_issues']:
        print("\n⚠️  CRITICAL ISSUES:")
        for issue in executive_summary['critical_issues']:
            print(f"  • {issue['gate']}: {issue['description']} (Score: {issue['score']:.3f})")
    
    # Show improvement recommendations
    recommendations = evaluation_result.get('recommendations', [])
    if recommendations:
        print(f"\n💡 IMPROVEMENT RECOMMENDATIONS ({len(recommendations)}):")
        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
            print(f"  {i}. {rec['gate']} (Priority: {rec['priority'].upper()})")
            print(f"     Current: {rec['current_score']:.3f} → Target: {rec['target_score']:.3f}")
            for area in rec['improvement_areas']:
                print(f"     • {area}")
    
    # Show compliance status
    compliance = comprehensive_report['compliance_status']
    print(f"\n🎯 COMPLIANCE STATUS:")
    print(f"Overall Compliance: {compliance['overall_compliance']:.1%}")
    print(f"Compliance Grade: {compliance['compliance_grade']}")
    print(f"Areas Compliant: {compliance['compliant_areas']}/{compliance['total_areas']}")
    
    # Show improvement roadmap
    roadmap = comprehensive_report['improvement_roadmap']
    if roadmap:
        print(f"\n🗺️  IMPROVEMENT ROADMAP:")
        for phase in roadmap:
            print(f"  {phase['phase'].upper()} ({phase['duration']})")
            print(f"    Focus: {phase['focus']}")
            print(f"    Items: {len(phase['items'])} action items")
    
    # Save detailed report
    report_file = Path("quality_assessment_report.json")
    with open(report_file, 'w') as f:
        json.dump({
            "evaluation_result": evaluation_result,
            "comprehensive_report": comprehensive_report
        }, f, indent=2, default=str)
    
    print(f"\n💾 Detailed report saved to: {report_file}")
    print(f"📊 Framework Evaluation Complete!")
    
    return evaluation_result, comprehensive_report


def demonstrate_individual_gates():
    """Demonstrate individual quality gate functionality."""
    print("\n🔬 INDIVIDUAL GATE DEMONSTRATIONS:")
    print("=" * 50)
    
    # Demo context
    demo_context = {
        "load_factor": 1.5,
        "optimization_level": 0.8,
        "batch_size": 64,
        "authentication_enabled": True,
        "multi_factor_auth": False,
        "stability_measures": 1.2,
        "circuit_breaker_enabled": False
    }
    
    async def demo_performance_gate():
        print("\n⚡ Performance Gate Demo:")
        gate = PerformanceQualityGate("Demo Performance", 0.7)
        result = await gate.evaluate(demo_context)
        print(f"   Score: {result['score']:.3f} | Status: {'PASS' if result['passed'] else 'FAIL'}")
        print(f"   Latency: {result['metrics']['latency_ms']:.1f}ms")
        print(f"   Throughput: {result['metrics']['throughput_rps']:.1f} RPS")
    
    async def demo_security_gate():
        print("\n🔒 Security Gate Demo:")
        gate = SecurityQualityGate("Demo Security", 0.8)
        result = await gate.evaluate(demo_context)
        print(f"   Score: {result['score']:.3f} | Status: {'PASS' if result['passed'] else 'FAIL'}")
        print(f"   Vulnerabilities: {result.get('vulnerabilities_found', 0)}")
        print(f"   Auth Score: {result['security_checks']['authentication']['score']:.3f}")
    
    async def demo_reliability_gate():
        print("\n🛡️  Reliability Gate Demo:")
        gate = ReliabilityQualityGate("Demo Reliability", 0.75)
        result = await gate.evaluate(demo_context)
        print(f"   Score: {result['score']:.3f} | Status: {'PASS' if result['passed'] else 'FAIL'}")
        print(f"   Error Rate: {result['reliability_metrics']['error_rate']:.4f}")
        print(f"   Uptime: {result['reliability_metrics']['uptime_percentage']:.2f}%")
    
    async def demo_ai_gate():
        print("\n🤖 AI Insights Gate Demo:")
        gate = AIInsightsQualityGate("Demo AI", 0.6)
        result = await gate.evaluate(demo_context)
        print(f"   Score: {result['score']:.3f} | Status: {'PASS' if result['passed'] else 'FAIL'}")
        insights = result['ai_insights']
        print(f"   Anomalies: {insights['anomaly_detection']['anomaly_count']}")
        print(f"   Patterns: {insights['pattern_analysis']['patterns_detected']}")
        print(f"   Suggestions: {len(insights['optimization_suggestions'])}")
    
    # Run individual demos
    asyncio.run(demo_performance_gate())
    asyncio.run(demo_security_gate())
    asyncio.run(demo_reliability_gate())
    asyncio.run(demo_ai_gate())


if __name__ == "__main__":
    print("🚀 STARTING NIMIFY QUALITY GATES DEMONSTRATION")
    print()
    
    try:
        # Run individual gate demonstrations
        demonstrate_individual_gates()
        
        # Run comprehensive framework demonstration
        asyncio.run(run_quality_gates_demonstration())
        
        print("\n✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()