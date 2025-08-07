#!/usr/bin/env python3
"""Performance benchmarking script for Nimify services."""

import asyncio
import time
import statistics
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from pathlib import Path
import sys
import argparse
import requests
from dataclasses import dataclass, asdict

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    endpoint: str = "http://localhost:8000"
    concurrent_users: int = 10
    requests_per_user: int = 100
    ramp_up_seconds: int = 5
    test_duration_seconds: int = 60
    payload_size: str = "small"  # small, medium, large


@dataclass 
class BenchmarkResult:
    """Benchmark result metrics."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    requests_per_second: float
    errors: List[str]
    test_duration_seconds: float
    throughput_mbps: float = 0.0


class LoadTester:
    """Load testing utility."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = []
        self.errors = []
    
    def generate_payload(self, size: str) -> Dict[str, Any]:
        """Generate test payload of specified size."""
        if size == "small":
            # Small payload: 1x10 input
            return {"input": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]}
        elif size == "medium":
            # Medium payload: 8x32 input (batch of 8, 32 features each)
            return {"input": [[i + j * 0.1 for j in range(32)] for i in range(8)]}
        elif size == "large":
            # Large payload: 32x224 input (batch of 32, 224 features each) 
            return {"input": [[i + j * 0.001 for j in range(224)] for i in range(32)]}
        else:
            return self.generate_payload("small")
    
    def make_request(self, session: requests.Session) -> Dict[str, Any]:
        """Make a single request and measure performance."""
        payload = self.generate_payload(self.config.payload_size)
        start_time = time.time()
        
        try:
            response = session.post(
                f"{self.config.endpoint}/v1/predict",
                json=payload,
                timeout=30
            )
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return {
                "success": response.status_code == 200,
                "response_time_ms": response_time_ms,
                "status_code": response.status_code,
                "response_size": len(response.content) if response.content else 0,
                "error": None if response.status_code == 200 else response.text
            }
            
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return {
                "success": False,
                "response_time_ms": response_time_ms,
                "status_code": 0,
                "response_size": 0,
                "error": str(e)
            }
    
    def run_user_load(self, user_id: int) -> List[Dict[str, Any]]:
        """Run load test for a single user."""
        print(f"Starting user {user_id}")
        
        session = requests.Session()
        user_results = []
        
        # Ramp up delay
        ramp_delay = (self.config.ramp_up_seconds * user_id) / self.config.concurrent_users
        time.sleep(ramp_delay)
        
        start_time = time.time()
        request_count = 0
        
        while (time.time() - start_time) < self.config.test_duration_seconds:
            if request_count >= self.config.requests_per_user:
                break
            
            result = self.make_request(session)
            user_results.append(result)
            request_count += 1
            
            # Small delay to prevent overwhelming
            time.sleep(0.001)
        
        session.close()
        print(f"User {user_id} completed {request_count} requests")
        return user_results
    
    def run_benchmark(self) -> BenchmarkResult:
        """Run the complete benchmark test."""
        print(f"🚀 Starting benchmark test...")
        print(f"   Endpoint: {self.config.endpoint}")
        print(f"   Users: {self.config.concurrent_users}")
        print(f"   Requests per user: {self.config.requests_per_user}")
        print(f"   Test duration: {self.config.test_duration_seconds}s")
        print(f"   Payload size: {self.config.payload_size}")
        
        # Health check first
        try:
            response = requests.get(f"{self.config.endpoint}/health", timeout=10)
            if response.status_code != 200:
                raise Exception(f"Health check failed: {response.status_code}")
            print("✅ Service health check passed")
        except Exception as e:
            print(f"❌ Service health check failed: {e}")
            return BenchmarkResult(0, 0, 1, 0, 0, 0, 0, 0, 0, 0, [str(e)], 0)
        
        # Run concurrent load test
        start_time = time.time()
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.config.concurrent_users) as executor:
            futures = [
                executor.submit(self.run_user_load, user_id) 
                for user_id in range(self.config.concurrent_users)
            ]
            
            for future in as_completed(futures):
                try:
                    user_results = future.result()
                    all_results.extend(user_results)
                except Exception as e:
                    self.errors.append(f"User thread failed: {e}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Calculate metrics
        successful_requests = [r for r in all_results if r["success"]]
        failed_requests = [r for r in all_results if not r["success"]]
        
        if successful_requests:
            response_times = [r["response_time_ms"] for r in successful_requests]
            response_sizes = [r["response_size"] for r in successful_requests]
            
            avg_response_time = statistics.mean(response_times)
            p50_response_time = statistics.median(response_times)
            p95_response_time = self._percentile(response_times, 0.95)
            p99_response_time = self._percentile(response_times, 0.99)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            rps = len(successful_requests) / total_duration
            
            # Calculate throughput in Mbps
            total_bytes = sum(response_sizes)
            throughput_mbps = (total_bytes * 8) / (total_duration * 1_000_000)
            
        else:
            avg_response_time = p50_response_time = p95_response_time = p99_response_time = 0
            min_response_time = max_response_time = rps = throughput_mbps = 0
        
        # Collect error messages
        error_messages = list(set([r["error"] for r in failed_requests if r["error"]]))
        error_messages.extend(self.errors)
        
        return BenchmarkResult(
            total_requests=len(all_results),
            successful_requests=len(successful_requests),
            failed_requests=len(failed_requests),
            avg_response_time_ms=avg_response_time,
            p50_response_time_ms=p50_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            requests_per_second=rps,
            errors=error_messages,
            test_duration_seconds=total_duration,
            throughput_mbps=throughput_mbps
        )
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        
        return sorted_data[index]


class MemoryProfiler:
    """Profile memory usage during benchmarks."""
    
    def __init__(self):
        self.measurements = []
    
    def start_profiling(self):
        """Start memory profiling."""
        try:
            import psutil
            
            def profile_memory():
                process = psutil.Process()
                while True:
                    memory_info = process.memory_info()
                    self.measurements.append({
                        "timestamp": time.time(),
                        "rss_mb": memory_info.rss / (1024 * 1024),
                        "vms_mb": memory_info.vms / (1024 * 1024)
                    })
                    time.sleep(1)
            
            import threading
            self.profile_thread = threading.Thread(target=profile_memory, daemon=True)
            self.profile_thread.start()
            
        except ImportError:
            print("⚠️  psutil not available, skipping memory profiling")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if not self.measurements:
            return {}
        
        rss_values = [m["rss_mb"] for m in self.measurements]
        vms_values = [m["vms_mb"] for m in self.measurements]
        
        return {
            "max_rss_mb": max(rss_values),
            "avg_rss_mb": statistics.mean(rss_values),
            "max_vms_mb": max(vms_values),
            "avg_vms_mb": statistics.mean(vms_values)
        }


def run_stress_test(endpoint: str, duration: int = 300) -> Dict[str, Any]:
    """Run extended stress test."""
    print(f"🔥 Running stress test for {duration} seconds...")
    
    configs = [
        BenchmarkConfig(endpoint=endpoint, concurrent_users=5, test_duration_seconds=duration, payload_size="small"),
        BenchmarkConfig(endpoint=endpoint, concurrent_users=10, test_duration_seconds=duration, payload_size="medium"),
        BenchmarkConfig(endpoint=endpoint, concurrent_users=20, test_duration_seconds=duration, payload_size="large")
    ]
    
    results = {}
    
    for i, config in enumerate(configs, 1):
        print(f"\n--- Stress Test Phase {i}/3 ---")
        
        tester = LoadTester(config)
        result = tester.run_benchmark()
        
        results[f"phase_{i}"] = asdict(result)
        
        # Brief cooldown between phases
        time.sleep(10)
    
    return results


def generate_performance_report(results: Dict[str, Any], output_file: str = None):
    """Generate comprehensive performance report."""
    print("\n" + "="*60)
    print("⚡ PERFORMANCE BENCHMARK RESULTS")
    print("="*60)
    
    if isinstance(results, BenchmarkResult):
        # Single benchmark result
        print_single_result(results)
    elif isinstance(results, dict) and "phase_1" in results:
        # Stress test results
        print_stress_results(results)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results if isinstance(results, dict) else asdict(results), f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")


def print_single_result(result: BenchmarkResult):
    """Print single benchmark result."""
    success_rate = (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0
    
    print(f"📊 REQUEST METRICS")
    print(f"   Total Requests: {result.total_requests}")
    print(f"   Successful: {result.successful_requests}")
    print(f"   Failed: {result.failed_requests}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    print(f"\n⚡ PERFORMANCE METRICS")
    print(f"   Requests/Second: {result.requests_per_second:.2f}")
    print(f"   Throughput: {result.throughput_mbps:.2f} Mbps")
    print(f"   Test Duration: {result.test_duration_seconds:.1f}s")
    
    print(f"\n⏱️  LATENCY METRICS (ms)")
    print(f"   Average: {result.avg_response_time_ms:.2f}")
    print(f"   Median (P50): {result.p50_response_time_ms:.2f}")
    print(f"   95th Percentile: {result.p95_response_time_ms:.2f}")
    print(f"   99th Percentile: {result.p99_response_time_ms:.2f}")
    print(f"   Min: {result.min_response_time_ms:.2f}")
    print(f"   Max: {result.max_response_time_ms:.2f}")
    
    if result.errors:
        print(f"\n❌ ERRORS ({len(result.errors)})")
        for error in result.errors[:5]:  # Show first 5 errors
            print(f"   • {error}")
        if len(result.errors) > 5:
            print(f"   ... and {len(result.errors) - 5} more")
    
    # Performance assessment
    print(f"\n🎯 ASSESSMENT")
    if result.p95_response_time_ms < 100:
        print("   ✅ EXCELLENT: P95 latency < 100ms")
    elif result.p95_response_time_ms < 200:
        print("   ✅ GOOD: P95 latency < 200ms")
    elif result.p95_response_time_ms < 500:
        print("   ⚠️  ACCEPTABLE: P95 latency < 500ms")
    else:
        print("   ❌ POOR: P95 latency > 500ms")
    
    if success_rate >= 99.9:
        print("   ✅ EXCELLENT: Success rate ≥ 99.9%")
    elif success_rate >= 99.0:
        print("   ✅ GOOD: Success rate ≥ 99.0%")
    elif success_rate >= 95.0:
        print("   ⚠️  ACCEPTABLE: Success rate ≥ 95.0%")
    else:
        print("   ❌ POOR: Success rate < 95.0%")


def print_stress_results(results: Dict[str, Any]):
    """Print stress test results."""
    print("🔥 STRESS TEST SUMMARY")
    
    for phase, data in results.items():
        if phase.startswith("phase_"):
            print(f"\n--- {phase.upper()} ---")
            result = BenchmarkResult(**data)
            print(f"   RPS: {result.requests_per_second:.2f}")
            print(f"   P95 Latency: {result.p95_response_time_ms:.2f}ms")
            print(f"   Success Rate: {(result.successful_requests/result.total_requests*100):.1f}%")


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description="Performance benchmark for Nimify services")
    parser.add_argument("--endpoint", default="http://localhost:8000", help="Service endpoint")
    parser.add_argument("--users", type=int, default=10, help="Concurrent users")
    parser.add_argument("--requests", type=int, default=100, help="Requests per user")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--payload", choices=["small", "medium", "large"], default="small", help="Payload size")
    parser.add_argument("--stress", action="store_true", help="Run extended stress test")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    if args.stress:
        results = run_stress_test(args.endpoint, args.duration)
    else:
        config = BenchmarkConfig(
            endpoint=args.endpoint,
            concurrent_users=args.users,
            requests_per_user=args.requests,
            test_duration_seconds=args.duration,
            payload_size=args.payload
        )
        
        # Start memory profiling
        profiler = MemoryProfiler()
        profiler.start_profiling()
        
        tester = LoadTester(config)
        results = tester.run_benchmark()
        
        # Add memory stats to results
        memory_stats = profiler.get_memory_stats()
        if memory_stats:
            print(f"\n🧠 MEMORY USAGE")
            print(f"   Max RSS: {memory_stats['max_rss_mb']:.1f} MB")
            print(f"   Avg RSS: {memory_stats['avg_rss_mb']:.1f} MB")
    
    generate_performance_report(results, args.output)
    
    # Exit with appropriate code
    if isinstance(results, BenchmarkResult):
        if results.failed_requests > 0 or results.p95_response_time_ms > 500:
            sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()