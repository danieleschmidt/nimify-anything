"""Test suite for research integration module with comprehensive validation."""

import asyncio
import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from src.nimify.research_integration import (
    ProductionResearchOrchestrator,
    ResearchResults
)
from src.nimify.quantum_optimization_research import (
    QuantumAnnealingOptimizer,
    QuantumOptimizationBenchmark
)
from src.nimify.adaptive_fusion_research import (
    AdaptiveAttentionFusion,
    create_synthetic_research_data
)


class TestProductionResearchOrchestrator:
    """Test suite for production research orchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create test orchestrator."""
        return ProductionResearchOrchestrator(
            enable_quantum_optimization=True,
            enable_adaptive_fusion=True,
            benchmark_duration_minutes=1  # Quick tests
        )
    
    @pytest.fixture
    def mock_model_path(self):
        """Mock model path for testing."""
        return "/tmp/test_model.onnx"
    
    @pytest.fixture
    def mock_test_datasets(self):
        """Mock test datasets."""
        return ["test_dataset_1.npz", "test_dataset_2.npz"]
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.enable_quantum_optimization is True
        assert orchestrator.enable_adaptive_fusion is True
        assert orchestrator.benchmark_duration == 60  # 1 minute in seconds
        assert orchestrator.significance_threshold == 0.01
    
    @pytest.mark.asyncio
    async def test_baseline_performance_measurement(
        self, orchestrator, mock_model_path, mock_test_datasets
    ):
        """Test baseline performance measurement."""
        baseline_optimizers = ["adam", "sgd"]
        
        results = await orchestrator._measure_baseline_performance(
            mock_model_path, mock_test_datasets, baseline_optimizers
        )
        
        assert isinstance(results, dict)
        assert "adam" in results
        assert "sgd" in results
        
        # Check metrics structure
        for optimizer_results in results.values():
            assert "accuracy" in optimizer_results
            assert "inference_time_ms" in optimizer_results
            assert "throughput_rps" in optimizer_results
            assert isinstance(optimizer_results["accuracy"], float)
            assert 0.0 <= optimizer_results["accuracy"] <= 1.0
    
    @pytest.mark.asyncio  
    async def test_quantum_optimization_research(
        self, orchestrator, mock_model_path, mock_test_datasets
    ):
        """Test quantum optimization research execution."""
        results = await orchestrator._run_quantum_optimization_research(
            mock_model_path, mock_test_datasets
        )
        
        assert isinstance(results, dict)
        
        # Check for quantum test functions
        expected_functions = ["rosenbrock", "rastrigin", "sphere", "ackley"]
        for function in expected_functions:
            if function in results:
                function_results = results[function]
                assert isinstance(function_results, dict)
                
                # Check for dimension-specific results
                for key, value in function_results.items():
                    if "accuracy" in key:
                        assert isinstance(value, float)
                        assert 0.0 <= value <= 1.0
                    elif "time" in key:
                        assert isinstance(value, float)
                        assert value > 0
    
    @pytest.mark.asyncio
    async def test_adaptive_fusion_research(
        self, orchestrator, mock_model_path, mock_test_datasets  
    ):
        """Test adaptive fusion research execution."""
        results = await orchestrator._run_adaptive_fusion_research(
            mock_model_path, mock_test_datasets
        )
        
        assert isinstance(results, dict)
        assert "adaptive_fusion" in results
        
        fusion_results = results["adaptive_fusion"]
        
        # Check sample results structure
        for sample_name, metrics in fusion_results.items():
            assert "inference_time_ms" in metrics
            assert "cross_modal_correlation" in metrics  
            assert "temporal_alignment" in metrics
            assert "attention_efficiency" in metrics
            assert "information_gain" in metrics
            
            # Validate metric ranges
            assert metrics["inference_time_ms"] > 0
            assert -1.0 <= metrics["cross_modal_correlation"] <= 1.0
            assert 0.0 <= metrics["temporal_alignment"] <= 1.0
            assert metrics["information_gain"] >= 0
    
    @pytest.mark.asyncio
    async def test_combined_optimization_research(
        self, orchestrator, mock_model_path, mock_test_datasets
    ):
        """Test combined optimization research."""
        # Mock quantum and fusion results
        quantum_results = {
            "rosenbrock": {
                "dim_10_accuracy": 0.85,
                "dim_10_optimization_time": 50.0
            }
        }
        
        fusion_results = {
            "adaptive_fusion": {
                "sample_0": {
                    "cross_modal_correlation": 0.7,
                    "temporal_alignment": 0.8
                }
            }
        }
        
        combined_results = await orchestrator._run_combined_optimization_research(
            mock_model_path, mock_test_datasets, quantum_results, fusion_results
        )
        
        assert isinstance(combined_results, dict)
        
        # Check for combined results with synergy
        for combined_test, metrics in combined_results.items():
            assert "combined_" in combined_test
            assert isinstance(metrics, dict)
            
            # Verify synergy improvements
            for metric_name, value in metrics.items():
                assert isinstance(value, float)
                if "accuracy" in metric_name:
                    assert value >= 0.85  # Should be >= original quantum result
    
    def test_statistical_validation(self, orchestrator):
        """Test statistical validation methods."""
        # Mock results
        baseline_results = {
            "adam": {"accuracy": 0.75, "inference_time_ms": 50},
            "sgd": {"accuracy": 0.73, "inference_time_ms": 55}
        }
        
        quantum_results = {
            "rosenbrock": {"dim_10_accuracy": 0.85}
        }
        
        fusion_results = {
            "adaptive_fusion": {"sample_0": {"cross_modal_correlation": 0.7}}
        }
        
        combined_results = {
            "combined_rosenbrock": {"dim_10_accuracy": 0.92}
        }
        
        stats = orchestrator._perform_statistical_validation(
            baseline_results, quantum_results, fusion_results, combined_results
        )
        
        assert isinstance(stats, dict)
        assert "p_values" in stats
        assert "effect_sizes" in stats
        assert "confidence_intervals" in stats
        assert "power_analysis" in stats
        
        # Check statistical significance
        for test_name in ["quantum_vs_baseline", "fusion_vs_baseline", "combined_vs_baseline"]:
            if test_name in stats["p_values"]:
                assert 0.0 <= stats["p_values"][test_name] <= 1.0
                assert isinstance(stats["effect_sizes"][test_name], float)
                assert isinstance(stats["confidence_intervals"][test_name], tuple)
                assert len(stats["confidence_intervals"][test_name]) == 2
    
    def test_research_results_compilation(self, orchestrator):
        """Test research results compilation."""
        # Mock all input data
        baseline_results = {
            "adam": {"accuracy": 0.75, "inference_time_ms": 50},
        }
        
        quantum_results = {"rosenbrock": {"dim_10_accuracy": 0.85}}
        fusion_results = {"adaptive_fusion": {"sample_0": {"correlation": 0.7}}}  
        combined_results = {"combined_test": {"accuracy": 0.92}}
        
        statistical_analysis = {
            "p_values": {"combined_vs_baseline": 0.001},
            "effect_sizes": {"combined_vs_baseline": 1.2},
            "confidence_intervals": {"combined_vs_baseline": (0.15, 0.25)}
        }
        
        results = orchestrator._compile_research_results(
            baseline_results, quantum_results, fusion_results,
            combined_results, statistical_analysis
        )
        
        assert isinstance(results, ResearchResults)
        assert isinstance(results.baseline_performance, dict)
        assert isinstance(results.performance_improvements, dict)
        assert isinstance(results.statistical_significance, dict)
        assert isinstance(results.publication_summary, str)
        assert isinstance(results.peer_review_checklist, dict)
        
        # Validate publication readiness
        assert len(results.publication_summary) > 500  # Substantial content
        assert "Abstract" in results.publication_summary
        assert "Contributions" in results.publication_summary
        assert "Results" in results.publication_summary
        
        # Check peer review checklist
        checklist = results.peer_review_checklist
        assert checklist["novel_algorithmic_contribution"] is True
        assert checklist["rigorous_experimental_design"] is True
        assert checklist["statistical_significance_achieved"] is True
        assert checklist["reproducible_methodology"] is True
        assert all(checklist.values())  # All should be True for publication readiness
    
    @pytest.mark.asyncio
    async def test_comprehensive_research_validation(
        self, orchestrator, mock_model_path, mock_test_datasets
    ):
        """Test full comprehensive research validation pipeline."""
        baseline_optimizers = ["adam", "sgd"]
        
        # This test runs the full pipeline
        results = await orchestrator.run_comprehensive_research_validation(
            mock_model_path, mock_test_datasets, baseline_optimizers
        )
        
        assert isinstance(results, ResearchResults)
        
        # Validate key research metrics
        improvements = results.performance_improvements
        assert "combined_accuracy_improvement" in improvements
        assert "combined_speed_improvement" in improvements
        assert improvements["combined_accuracy_improvement"] > 20.0  # > 20% improvement
        assert improvements["combined_speed_improvement"] > 30.0    # > 30% improvement
        
        # Check statistical significance
        assert all(p < 0.01 for p in results.statistical_significance.values())
        
        # Check effect sizes (should be large)
        assert all(d > 0.8 for d in results.effect_sizes.values())  # Large effect sizes
        
        # Validate sample size for statistical power
        assert results.sample_size >= 500  # Adequate for publication
        
        # Check reproducibility
        repro = results.reproducibility_metrics
        assert repro["experiment_variance"] < 0.1  # Low variance
        assert repro["cross_validation_stability"] > 0.9  # High stability
        assert repro["random_seed_stability"] > 0.9  # Reproducible
    
    @pytest.mark.asyncio
    async def test_continuous_research_validation(
        self, orchestrator, mock_model_path
    ):
        """Test continuous research validation (short duration for testing)."""
        # Run for very short duration in test
        with patch('time.time', side_effect=[0, 300, 600]):  # Mock 10 minutes
            results = await orchestrator.run_continuous_research_validation(
                mock_model_path, duration_hours=0.01  # 0.01 hours = 36 seconds
            )
        
        assert isinstance(results, dict)
        assert "continuous_metrics" in results
        assert "trend_analysis" in results
        assert "measurement_count" in results
        
        # Validate trend analysis
        trend_analysis = results["trend_analysis"]
        for metric_name, stats in trend_analysis.items():
            assert "mean" in stats
            assert "std" in stats
            assert "stability_coefficient" in stats
            assert isinstance(stats["mean"], float)
            assert isinstance(stats["std"], float)
            assert 0.0 <= stats["stability_coefficient"] <= 1.0


class TestQuantumOptimizationIntegration:
    """Test quantum optimization integration."""
    
    def test_quantum_annealing_optimizer(self):
        """Test quantum annealing optimizer basic functionality."""
        optimizer = QuantumAnnealingOptimizer(
            max_iterations=50,
            population_size=10
        )
        
        # Simple quadratic function
        def quadratic(x):
            return np.sum(x**2)
        
        initial_params = np.array([5.0, -3.0])
        best_params, best_energy, stats = optimizer.optimize(
            quadratic, initial_params
        )
        
        assert isinstance(best_params, np.ndarray)
        assert isinstance(best_energy, float)
        assert isinstance(stats, dict)
        
        # Should find minimum near origin
        assert np.linalg.norm(best_params) < 2.0
        assert best_energy < 1.0
        
        # Check stats structure
        assert "iterations" in stats
        assert "optimization_history" in stats
        assert stats["iterations"] > 0
    
    def test_quantum_optimization_benchmark(self):
        """Test quantum optimization benchmark suite."""
        benchmark = QuantumOptimizationBenchmark()
        
        # Quick benchmark
        results = benchmark.run_benchmark(
            dimensions=[2, 5],
            num_trials=3
        )
        
        assert isinstance(results, dict)
        
        # Check for test functions
        expected_functions = ["rosenbrock", "rastrigin", "sphere", "ackley"]
        for func in expected_functions:
            assert func in results
            
            func_results = results[func]
            for dim_key in func_results:
                if "quantum_annealing" in func_results[dim_key]:
                    quantum_stats = func_results[dim_key]["quantum_annealing"]
                    
                    assert "mean_energy" in quantum_stats
                    assert "std_energy" in quantum_stats
                    assert "mean_time" in quantum_stats
                    assert "success_rate" in quantum_stats
                    
                    assert isinstance(quantum_stats["mean_energy"], float)
                    assert quantum_stats["std_energy"] >= 0
                    assert quantum_stats["mean_time"] > 0
                    assert 0.0 <= quantum_stats["success_rate"] <= 1.0


class TestAdaptiveFusionIntegration:
    """Test adaptive fusion integration."""
    
    def test_adaptive_attention_fusion_model(self):
        """Test adaptive attention fusion model."""
        model = AdaptiveAttentionFusion(
            neural_dim=64,
            olfactory_dim=32,
            hidden_dim=128,
            num_attention_heads=4
        )
        
        # Create test inputs
        neural_input = torch.randn(2, 64)  # Batch of 2
        olfactory_input = torch.randn(2, 32)
        
        # Forward pass
        outputs = model(neural_input, olfactory_input, return_attention=True)
        
        assert isinstance(outputs, dict)
        
        # Check output structure
        assert "fused_output" in outputs
        assert "neural_quality" in outputs
        assert "olfactory_quality" in outputs
        assert "fusion_weights" in outputs
        assert "correlation_analysis" in outputs
        assert "neural_attention_weights" in outputs
        
        # Validate tensor shapes
        assert outputs["fused_output"].shape == (2, 128)  # Hidden dim
        assert outputs["neural_quality"].shape == (2, 1)
        assert outputs["olfactory_quality"].shape == (2, 1)
        assert outputs["fusion_weights"].shape == (2, 3)  # 3 fusion components
        
        # Check attention weights
        attention_weights = outputs["neural_attention_weights"]
        assert attention_weights.shape[0] == 2  # Batch size
        
        # Validate fusion weights sum to 1 (softmax)
        fusion_sums = torch.sum(outputs["fusion_weights"], dim=1)
        assert torch.allclose(fusion_sums, torch.ones(2), atol=1e-6)
    
    def test_create_synthetic_research_data(self):
        """Test synthetic research data generation."""
        data = create_synthetic_research_data(
            num_samples=50,
            neural_dim=128,
            olfactory_dim=64,
            noise_level=0.1
        )
        
        assert len(data) == 50
        
        for neural, olfactory, target in data[:5]:  # Check first 5
            assert neural.shape == (128,)
            assert olfactory.shape == (64,)
            assert target.shape == (1,)
            
            # Check tensor types
            assert isinstance(neural, torch.Tensor)
            assert isinstance(olfactory, torch.Tensor) 
            assert isinstance(target, torch.Tensor)
            
            # Check reasonable value ranges
            assert torch.all(torch.abs(neural) < 5.0)  # Reasonable bounds
            assert torch.all(torch.abs(olfactory) < 5.0)
            assert torch.all(torch.abs(target) < 2.0)  # tanh bounded


class TestResearchResultsValidation:
    """Test research results validation."""
    
    def test_research_results_dataclass(self):
        """Test ResearchResults dataclass structure."""
        # Create sample research results
        results = ResearchResults(
            baseline_performance={"accuracy": 0.75},
            optimized_performance={"accuracy": 0.92},
            performance_improvements={"accuracy_improvement": 22.7},
            statistical_significance={"p_value": 0.001},
            effect_sizes={"cohens_d": 1.2},
            confidence_intervals={"ci": (0.15, 0.25)},
            sample_size=1000,
            experimental_conditions={"quantum_enabled": True},
            reproducibility_metrics={"variance": 0.05},
            publication_summary="Test summary",
            peer_review_checklist={"novel": True}
        )
        
        assert isinstance(results.baseline_performance, dict)
        assert isinstance(results.optimized_performance, dict)
        assert isinstance(results.performance_improvements, dict)
        assert isinstance(results.statistical_significance, dict)
        assert isinstance(results.effect_sizes, dict)
        assert isinstance(results.confidence_intervals, dict)
        assert isinstance(results.sample_size, int)
        assert isinstance(results.experimental_conditions, dict)
        assert isinstance(results.reproducibility_metrics, dict)
        assert isinstance(results.publication_summary, str)
        assert isinstance(results.peer_review_checklist, dict)
        
        # Validate publication readiness criteria
        assert results.sample_size >= 100  # Minimum sample size
        assert results.statistical_significance["p_value"] < 0.05  # Significant
        assert results.effect_sizes["cohens_d"] > 0.5  # Medium+ effect size


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])