"""Performance benchmarks for Nimify Anything."""

import pytest
import time
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from nimify.core import ModelConfig, Nimifier


class TestModelLoadingPerformance:
    """Benchmark model loading operations."""
    
    @pytest.mark.performance
    def test_model_loading_speed(self, benchmark, mock_model_file):
        """Benchmark model loading time."""
        
        def load_model():
            config = ModelConfig(name="benchmark-model")
            nimifier = Nimifier(config)
            return nimifier.analyze_model(mock_model_file)
        
        result = benchmark(load_model)
        
        # Performance assertions (adjust based on requirements)
        assert result.stats['mean'] < 1.0  # <1s mean loading time
        assert result.stats['stddev'] < 0.2  # Low variance
    
    @pytest.mark.performance
    def test_config_generation_speed(self, benchmark, sample_model_config):
        """Benchmark configuration generation."""
        
        def generate_config():
            nimifier = Nimifier(sample_model_config)
            return nimifier.generate_triton_config()
        
        result = benchmark(generate_config)
        assert result.stats['mean'] < 0.1  # <100ms for config generation


class TestConcurrencyPerformance:
    """Test performance under concurrent load."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_concurrent_model_analysis(self, mock_model_file):
        """Test performance with concurrent model analysis."""
        
        def analyze_model():
            config = ModelConfig(name=f"concurrent-{time.time()}")
            nimifier = Nimifier(config)
            return nimifier.analyze_model(mock_model_file)
        
        # Test with multiple concurrent operations
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(analyze_model) for _ in range(10)]
            results = [future.result() for future in futures]
        
        duration = time.time() - start_time
        
        # Should complete 10 operations in reasonable time
        assert duration < 5.0  # <5s for 10 concurrent operations
        assert len(results) == 10
    
    @pytest.mark.performance
    async def test_async_operations_performance(self):
        """Test async operation performance."""
        
        async def async_operation():
            # Simulate async model operation
            await asyncio.sleep(0.01)  # 10ms operation
            return "completed"
        
        start_time = time.time()
        
        # Run 100 concurrent async operations
        tasks = [async_operation() for _ in range(100)]
        results = await asyncio.gather(*tasks)
        
        duration = time.time() - start_time
        
        # Should be much faster than sequential (100 * 10ms = 1s)
        assert duration < 0.5  # <500ms for 100 concurrent ops
        assert len(results) == 100


class TestMemoryPerformance:
    """Test memory usage and efficiency."""
    
    @pytest.mark.performance
    def test_memory_usage_model_loading(self, mock_model_file):
        """Test memory usage during model loading."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Load multiple models
        configs = [
            ModelConfig(name=f"memory-test-{i}")
            for i in range(5)
        ]
        
        nimifiers = []
        for config in configs:
            nimifier = Nimifier(config)
            nimifier.analyze_model(mock_model_file)
            nimifiers.append(nimifier)
        
        peak_memory = process.memory_info().rss
        memory_increase = (peak_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable
        assert memory_increase < 100  # <100MB increase
        
        # Clean up
        del nimifiers
    
    @pytest.mark.performance
    def test_memory_leak_detection(self, mock_model_file):
        """Test for memory leaks in repeated operations."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        def create_and_destroy_nimifier():
            config = ModelConfig(name="leak-test")
            nimifier = Nimifier(config)
            nimifier.analyze_model(mock_model_file)
            del nimifier
            gc.collect()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss
        
        # Perform operations multiple times
        for _ in range(10):
            create_and_destroy_nimifier()
        
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - baseline_memory) / 1024 / 1024  # MB
        
        # Should not have significant memory leaks
        assert memory_increase < 10  # <10MB increase after cleanup


class TestScalabilityBenchmarks:
    """Test scalability characteristics."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_linear_scaling_characteristics(self, mock_model_file):
        """Test how performance scales with load."""
        
        def measure_throughput(num_operations):
            """Measure throughput for given number of operations."""
            start_time = time.time()
            
            for i in range(num_operations):
                config = ModelConfig(name=f"scale-test-{i}")
                nimifier = Nimifier(config)
                nimifier.analyze_model(mock_model_file)
            
            duration = time.time() - start_time
            return num_operations / duration  # operations per second
        
        # Test different scales
        scales = [1, 5, 10]
        throughputs = []
        
        for scale in scales:
            throughput = measure_throughput(scale)
            throughputs.append(throughput)
        
        # Throughput should remain relatively stable
        # (indicating good scaling characteristics)
        throughput_variance = max(throughputs) / min(throughputs)
        assert throughput_variance < 2.0  # <2x variance across scales


class TestResourceUtilization:
    """Test resource utilization efficiency."""
    
    @pytest.mark.performance
    def test_cpu_utilization_efficiency(self, mock_model_file):
        """Test CPU utilization during operations."""
        import psutil
        import threading
        import time
        
        cpu_usage_samples = []
        monitoring = True
        
        def monitor_cpu():
            while monitoring:
                cpu_usage_samples.append(psutil.cpu_percent(interval=0.1))
        
        # Start CPU monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Perform CPU-intensive operations
        for i in range(5):
            config = ModelConfig(name=f"cpu-test-{i}")
            nimifier = Nimifier(config)
            nimifier.analyze_model(mock_model_file)
        
        # Stop monitoring
        monitoring = False
        monitor_thread.join()
        
        if cpu_usage_samples:
            avg_cpu_usage = sum(cpu_usage_samples) / len(cpu_usage_samples)
            
            # Should utilize CPU efficiently but not excessively
            assert 10 < avg_cpu_usage < 80  # Between 10% and 80%


class TestRegressionBenchmarks:
    """Regression tests for performance."""
    
    @pytest.mark.performance
    def test_baseline_model_creation_time(self, benchmark, mock_model_file):
        """Baseline benchmark for model creation time."""
        
        def create_model_service():
            config = ModelConfig(
                name="baseline-test",
                max_batch_size=16,
                dynamic_batching=True
            )
            nimifier = Nimifier(config)
            return nimifier.wrap_model(
                mock_model_file,
                input_schema={"input": "float32[?,3,224,224]"},
                output_schema={"output": "float32[?,1000]"}
            )
        
        result = benchmark(create_model_service)
        
        # Store baseline for comparison in future runs
        # These values should be adjusted based on actual baseline measurements
        assert result.stats['mean'] < 2.0  # <2s for complete model service creation
        assert result.stats['min'] < 1.5   # <1.5s best case
    
    @pytest.mark.performance
    def test_configuration_generation_regression(self, benchmark, sample_model_config):
        """Regression test for configuration generation performance."""
        
        def generate_all_configs():
            nimifier = Nimifier(sample_model_config)
            return {
                'triton': nimifier.generate_triton_config(),
                'dockerfile': nimifier.generate_dockerfile(),
                'helm': nimifier.generate_helm_chart(),
                'openapi': nimifier.generate_openapi_spec()
            }
        
        result = benchmark(generate_all_configs)
        
        # Should generate all configurations quickly
        assert result.stats['mean'] < 0.5  # <500ms for all configs