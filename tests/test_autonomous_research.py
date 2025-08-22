"""
Comprehensive tests for autonomous research capabilities.

Tests the autonomous research agent, quantum neural optimizer, and neural
architecture search components for correctness, performance, and reliability.
"""

import asyncio
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nimify.autonomous_research_agent import (
    AutonomousResearchAgent,
    ResearchHypothesis,
    AdaptiveInferencePipelineResearch
)
from nimify.quantum_neural_optimizer import (
    QuantumNeuralOptimizer,
    QuantumState,
    HadamardGate,
    RotationGate,
    EntanglementGate,
    OptimizationParameter,
    QuantumOptimizationObjective
)
from nimify.neural_architecture_search import (
    GeneticNeuralArchitectureSearch,
    NIMServiceEvaluator,
    NeuralArchitecture,
    ArchitectureType,
    ArchitectureGene
)


class TestAutonomousResearchAgent:
    """Test suite for autonomous research agent."""
    
    @pytest.fixture
    def research_agent(self):
        """Create test research agent."""
        return AutonomousResearchAgent(
            model_path="/test/model.onnx",
            service_config={"name": "test-service", "max_batch_size": 32}
        )
    
    @pytest.fixture
    def sample_hypothesis(self):
        """Create sample research hypothesis."""
        return ResearchHypothesis(
            hypothesis_id="test_001",
            title="Test Optimization Hypothesis",
            description="Test description",
            methodology="test_methodology",
            success_criteria={"improvement": 10.0}
        )
    
    def test_research_agent_initialization(self, research_agent):
        """Test research agent initializes correctly."""
        assert research_agent.model_path == "/test/model.onnx"
        assert research_agent.service_config["name"] == "test-service"
        assert len(research_agent.active_hypotheses) == 0
        assert len(research_agent.validated_optimizations) == 0
    
    @pytest.mark.asyncio
    async def test_discover_research_opportunities(self, research_agent):
        """Test research opportunity discovery."""
        # Mock baseline metrics collection
        with patch.object(research_agent, '_collect_baseline_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "latency_p95": 100,  # High latency triggers hypothesis
                "memory_usage": 2000,  # High memory usage triggers hypothesis
                "scaling_efficiency": 0.6  # Low efficiency triggers hypothesis
            }
            
            hypotheses = await research_agent.discover_research_opportunities()
            
            assert len(hypotheses) >= 1
            assert any("adaptive_inference" in h.hypothesis_id for h in hypotheses)
            assert all(isinstance(h, ResearchHypothesis) for h in hypotheses)
    
    @pytest.mark.asyncio
    async def test_execute_research_cycle(self, research_agent, sample_hypothesis):
        """Test complete research cycle execution."""
        # Mock methodology execution
        mock_methodology = AsyncMock()
        mock_methodology.design_experiment.return_value = {"experiment_type": "test"}
        mock_methodology.run_experiment.return_value = {"results": "test_results"}
        mock_methodology.analyze_results.return_value = (True, 0.9, {"analysis": "test"})
        
        with patch.object(research_agent, '_select_methodology', return_value=mock_methodology):
            with patch.object(research_agent, '_save_research_data') as mock_save:
                result = await research_agent.execute_research_cycle(sample_hypothesis)
                
                assert "hypothesis" in result
                assert result["validation_status"] is True
                assert result["confidence_score"] == 0.9
                mock_save.assert_called_once()
    
    @pytest.mark.asyncio 
    async def test_generate_research_report(self, research_agent):
        """Test research report generation."""
        # Add some test data
        research_agent.active_hypotheses = [
            ResearchHypothesis("test1", "Test 1", "desc", "method", {}, status="validated"),
            ResearchHypothesis("test2", "Test 2", "desc", "method", {}, status="failed")
        ]
        research_agent.validated_optimizations = [
            {"optimization_id": "opt1", "title": "Optimization 1", "confidence_score": 0.9}
        ]
        
        report = await research_agent.generate_research_report()
        
        assert "research_summary" in report
        assert report["research_summary"]["total_hypotheses"] == 2
        assert report["research_summary"]["validated_hypotheses"] == 1
        assert report["research_summary"]["failed_hypotheses"] == 1
        assert len(report["validated_optimizations"]) == 1


class TestAdaptiveInferencePipelineResearch:
    """Test suite for adaptive inference pipeline research."""
    
    @pytest.fixture
    def research_methodology(self):
        """Create test research methodology."""
        return AdaptiveInferencePipelineResearch()
    
    @pytest.fixture
    def test_hypothesis(self):
        """Create test hypothesis."""
        return ResearchHypothesis(
            hypothesis_id="adaptive_001",
            title="Adaptive Pipeline Test",
            description="Test adaptive pipeline optimization",
            methodology="comparative_experimental_analysis",
            success_criteria={"throughput_improvement": 20.0}
        )
    
    @pytest.mark.asyncio
    async def test_design_experiment(self, research_methodology, test_hypothesis):
        """Test experiment design."""
        experiment_config = await research_methodology.design_experiment(test_hypothesis)
        
        assert "experiment_type" in experiment_config
        assert experiment_config["experiment_type"] == "adaptive_pipeline"
        assert "baseline_config" in experiment_config
        assert "experimental_configs" in experiment_config
        assert len(experiment_config["experimental_configs"]) >= 1
        assert "metrics_to_collect" in experiment_config
    
    @pytest.mark.asyncio
    async def test_run_experiment(self, research_methodology):
        """Test experiment execution."""
        experiment_config = {
            "baseline_config": {"batch_size": 32, "optimization_level": "standard"},
            "experimental_configs": [
                {"name": "test_config", "batch_size": 64, "optimization_level": "aggressive"}
            ],
            "test_scenarios": [
                {"load": "medium", "concurrent_requests": 50, "duration": 300}
            ]
        }
        
        results = await research_methodology.run_experiment(experiment_config)
        
        assert "experiment_id" in results
        assert "baseline_results" in results
        assert "experimental_results" in results
        assert "comparative_analysis" in results
    
    @pytest.mark.asyncio
    async def test_analyze_results(self, research_methodology):
        """Test results analysis."""
        mock_results = {
            "comparative_analysis": {
                "test_config": {
                    "throughput_improvement": 25.0,
                    "latency_improvement": 15.0,
                    "efficiency_improvement": 20.0,
                    "overall_score": 22.0
                }
            }
        }
        
        validated, confidence, analysis = await research_methodology.analyze_results(mock_results)
        
        assert isinstance(validated, bool)
        assert 0.0 <= confidence <= 1.0
        assert "overall_improvement_percent" in analysis
        assert "recommendation" in analysis


class TestQuantumNeuralOptimizer:
    """Test suite for quantum neural optimizer."""
    
    @pytest.fixture
    def optimization_parameters(self):
        """Create test optimization parameters."""
        return [
            OptimizationParameter("param1", 1.0, 0.1, 5.0),
            OptimizationParameter("param2", 32, 16, 128),
            OptimizationParameter("param3", 0.5, 0.1, 1.0)
        ]
    
    @pytest.fixture
    def quantum_objective(self, optimization_parameters):
        """Create test quantum optimization objective."""
        return QuantumOptimizationObjective(optimization_parameters)
    
    @pytest.fixture
    def quantum_optimizer(self, quantum_objective):
        """Create test quantum optimizer."""
        return QuantumNeuralOptimizer(quantum_objective, n_qubits=4)
    
    def test_quantum_state_initialization(self):
        """Test quantum state initialization."""
        amplitudes = np.array([1, 0, 0, 0], dtype=complex)
        phase_angles = np.zeros(4)
        entanglement_matrix = np.zeros((2, 2))
        
        state = QuantumState(
            amplitudes=amplitudes,
            phase_angles=phase_angles,
            entanglement_matrix=entanglement_matrix,
            energy_level=1.0,
            coherence_time=100.0
        )
        
        # Check normalization
        assert np.allclose(np.linalg.norm(state.amplitudes), 1.0)
    
    def test_quantum_state_measurement(self):
        """Test quantum state measurement."""
        # Equal superposition state
        amplitudes = np.ones(4, dtype=complex) / 2.0
        state = QuantumState(
            amplitudes=amplitudes,
            phase_angles=np.zeros(4),
            entanglement_matrix=np.zeros((2, 2)),
            energy_level=1.0,
            coherence_time=100.0
        )
        
        # Measure multiple times
        measurements = [state.measure() for _ in range(100)]
        
        # All measurements should be valid indices
        assert all(0 <= m < 4 for m in measurements)
        
        # Should get roughly equal distribution (with some randomness)
        unique_measurements = set(measurements)
        assert len(unique_measurements) > 1  # Should get multiple different outcomes
    
    def test_hadamard_gate(self):
        """Test Hadamard gate application."""
        # Start with |00⟩ state
        initial_state = QuantumState(
            amplitudes=np.array([1, 0, 0, 0], dtype=complex),
            phase_angles=np.zeros(4),
            entanglement_matrix=np.zeros((2, 2)),
            energy_level=1.0,
            coherence_time=100.0
        )
        
        hadamard = HadamardGate()
        superposition_state = hadamard.apply(initial_state)
        
        # Should be in superposition (all amplitudes non-zero)
        assert np.all(np.abs(superposition_state.amplitudes) > 0)
        assert np.allclose(np.linalg.norm(superposition_state.amplitudes), 1.0)
    
    def test_rotation_gate(self):
        """Test rotation gate application."""
        initial_state = QuantumState(
            amplitudes=np.array([1, 0], dtype=complex),
            phase_angles=np.zeros(2),
            entanglement_matrix=np.zeros((1, 1)),
            energy_level=1.0,
            coherence_time=100.0
        )
        
        rotation = RotationGate('x', np.pi/2)
        rotated_state = rotation.apply(initial_state)
        
        # State should be modified
        assert not np.allclose(rotated_state.amplitudes, initial_state.amplitudes)
        assert np.allclose(np.linalg.norm(rotated_state.amplitudes), 1.0)
    
    def test_optimization_objective_evaluation(self, quantum_objective):
        """Test objective function evaluation."""
        test_params = {
            "param1": 2.5,
            "param2": 64.0,
            "param3": 0.8
        }
        
        score = quantum_objective.evaluate(test_params)
        
        assert isinstance(score, float)
        assert score > 0  # Should be positive for reasonable parameters
    
    def test_quantum_optimizer_initialization(self, quantum_optimizer):
        """Test quantum optimizer initialization."""
        assert quantum_optimizer.n_qubits == 4
        assert quantum_optimizer.n_states == 16
        assert quantum_optimizer.quantum_state is not None
        assert quantum_optimizer.best_solution is None
        assert quantum_optimizer.best_score == float('-inf')
    
    @pytest.mark.asyncio
    async def test_quantum_optimization_run(self, quantum_optimizer):
        """Test quantum optimization execution."""
        # Run short optimization
        results = await quantum_optimizer.quantum_optimize(max_iterations=5)
        
        assert "best_parameters" in results
        assert "best_score" in results
        assert "optimization_time" in results
        assert "total_iterations" in results
        assert "quantum_efficiency" in results
        
        assert results["best_parameters"] is not None
        assert results["best_score"] > float('-inf')
        assert results["optimization_time"] > 0
    
    def test_quantum_report_generation(self, quantum_optimizer):
        """Test quantum optimization report."""
        # Set some test data
        quantum_optimizer.best_solution = {"param1": 2.0}
        quantum_optimizer.best_score = 0.85
        
        report = quantum_optimizer.generate_quantum_report()
        
        assert "optimization_summary" in report
        assert "quantum_metrics" in report
        assert "classical_comparison" in report
        assert "implementation_recommendations" in report


class TestNeuralArchitectureSearch:
    """Test suite for neural architecture search."""
    
    @pytest.fixture
    def target_metrics(self):
        """Create target metrics for evaluation."""
        return {
            "throughput": 150,
            "latency": 30,
            "accuracy": 0.9,
            "memory_efficiency": 5.0,
            "gpu_utilization": 0.8
        }
    
    @pytest.fixture
    def evaluator(self, target_metrics):
        """Create test evaluator."""
        return NIMServiceEvaluator(target_metrics)
    
    @pytest.fixture
    def nas_instance(self, evaluator):
        """Create NAS instance."""
        return GeneticNeuralArchitectureSearch(
            evaluator=evaluator,
            population_size=10,  # Small for testing
            elite_ratio=0.3,
            mutation_rate=0.2,
            crossover_rate=0.8
        )
    
    def test_architecture_gene_creation(self):
        """Test architecture gene creation."""
        gene = ArchitectureGene("int", 64, 32, 128)
        
        assert gene.gene_type == "int"
        assert gene.value == 64
        assert gene.min_value == 32
        assert gene.max_value == 128
    
    def test_neural_architecture_creation(self):
        """Test neural architecture creation."""
        genes = {
            "layers": ArchitectureGene("int", 12, 6, 24),
            "batch_size": ArchitectureGene("int", 32, 16, 128)
        }
        
        arch = NeuralArchitecture(
            architecture_id="test_arch",
            architecture_type=ArchitectureType.TRANSFORMER,
            genes=genes
        )
        
        assert arch.architecture_type == ArchitectureType.TRANSFORMER
        assert len(arch.genes) == 2
        assert arch.fitness_score == 0.0
        assert arch.generation == 0
    
    def test_architecture_mutation(self):
        """Test architecture mutation."""
        genes = {
            "param1": ArchitectureGene("int", 10, 1, 20),
            "param2": ArchitectureGene("float", 0.5, 0.1, 1.0)
        }
        
        arch = NeuralArchitecture(
            architecture_id="test",
            architecture_type=ArchitectureType.CNN,
            genes=genes
        )
        
        mutated = arch.mutate(mutation_rate=1.0)  # Force mutation
        
        assert mutated.architecture_id != arch.architecture_id
        assert mutated.generation == arch.generation + 1
        assert mutated.parent_ids == [arch.architecture_id]
    
    def test_architecture_crossover(self):
        """Test architecture crossover."""
        genes1 = {
            "param1": ArchitectureGene("int", 10, 1, 20),
            "param2": ArchitectureGene("float", 0.5, 0.1, 1.0)
        }
        genes2 = {
            "param1": ArchitectureGene("int", 15, 1, 20),
            "param2": ArchitectureGene("float", 0.8, 0.1, 1.0)
        }
        
        arch1 = NeuralArchitecture("arch1", ArchitectureType.CNN, genes1)
        arch2 = NeuralArchitecture("arch2", ArchitectureType.CNN, genes2)
        
        child1, child2 = arch1.crossover(arch2)
        
        assert child1.generation > arch1.generation
        assert child2.generation > arch2.generation
        assert len(child1.parent_ids) == 2
        assert len(child2.parent_ids) == 2
    
    @pytest.mark.asyncio
    async def test_architecture_evaluation(self, evaluator):
        """Test architecture evaluation."""
        genes = {
            "num_layers": ArchitectureGene("int", 12, 6, 24),
            "batch_size": ArchitectureGene("int", 32, 16, 128),
            "use_quantization": ArchitectureGene("bool", True)
        }
        
        arch = NeuralArchitecture(
            architecture_id="test_eval",
            architecture_type=ArchitectureType.TRANSFORMER,
            genes=genes
        )
        
        metrics = await evaluator.evaluate(arch)
        
        assert "throughput" in metrics
        assert "latency" in metrics
        assert "accuracy" in metrics
        assert "memory_efficiency" in metrics
        
        # All metrics should be positive numbers
        assert all(isinstance(v, (int, float)) and v > 0 for v in metrics.values())
    
    def test_resource_estimation(self, evaluator):
        """Test resource requirement estimation."""
        genes = {
            "num_layers": ArchitectureGene("int", 12, 6, 24),
            "hidden_size": ArchitectureGene("int", 768, 512, 1536)
        }
        
        arch = NeuralArchitecture(
            architecture_id="resource_test",
            architecture_type=ArchitectureType.TRANSFORMER,
            genes=genes
        )
        
        resources = evaluator.estimate_resources(arch)
        
        assert "memory_mb" in resources
        assert "compute_flops" in resources
        assert "gpu_memory_mb" in resources
        
        # Resources should be positive
        assert all(v > 0 for v in resources.values())
    
    def test_nas_population_initialization(self, nas_instance):
        """Test NAS population initialization."""
        nas_instance.initialize_population(ArchitectureType.TRANSFORMER)
        
        assert len(nas_instance.population) == 10
        assert all(arch.architecture_type == ArchitectureType.TRANSFORMER for arch in nas_instance.population)
        assert all(len(arch.genes) > 0 for arch in nas_instance.population)
    
    @pytest.mark.asyncio
    async def test_nas_evolution_short(self, nas_instance):
        """Test short NAS evolution."""
        nas_instance.initialize_population(ArchitectureType.TRANSFORMER)
        
        # Run very short evolution
        best_arch = await nas_instance.evolve(num_generations=2)
        
        assert best_arch is not None
        assert best_arch.fitness_score > 0
        assert len(nas_instance.search_history) > 0
        assert nas_instance.best_architecture == best_arch
    
    def test_nas_report_generation(self, nas_instance):
        """Test NAS report generation."""
        # Initialize with dummy data
        nas_instance.best_architecture = NeuralArchitecture(
            architecture_id="best_test",
            architecture_type=ArchitectureType.TRANSFORMER,
            genes={"test": ArchitectureGene("int", 1, 1, 10)}
        )
        nas_instance.best_fitness = 0.95
        nas_instance.generation = 5
        
        # Add some search history
        nas_instance.search_history = [
            {"generation": i, "best_fitness": 0.8 + i*0.03, "avg_fitness": 0.7 + i*0.02}
            for i in range(5)
        ]
        
        nas_instance.population = [
            NeuralArchitecture(f"arch_{i}", ArchitectureType.TRANSFORMER, {"test": ArchitectureGene("int", i, 1, 10)})
            for i in range(10)
        ]
        
        report = nas_instance.generate_search_report()
        
        assert "search_summary" in report
        assert "best_architecture" in report
        assert "search_statistics" in report
        assert "optimization_insights" in report
        assert "implementation_recommendations" in report


class TestIntegrationScenarios:
    """Integration tests for research system components."""
    
    @pytest.mark.asyncio
    async def test_full_research_pipeline(self):
        """Test complete research pipeline integration."""
        # Create research agent
        agent = AutonomousResearchAgent(
            model_path="/test/model.onnx",
            service_config={"name": "test-service"}
        )
        
        # Mock baseline metrics to trigger research
        with patch.object(agent, '_collect_baseline_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "latency_p95": 80,
                "memory_usage": 1200,
                "scaling_efficiency": 0.75
            }
            
            # Discover opportunities
            hypotheses = await agent.discover_research_opportunities()
            
            assert len(hypotheses) >= 1
            
            # Mock research execution for integration test
            for hypothesis in hypotheses[:1]:  # Test one hypothesis
                # Mock the methodology
                mock_methodology = AsyncMock()
                mock_methodology.design_experiment.return_value = {"experiment_type": "integration_test"}
                mock_methodology.run_experiment.return_value = {"test_results": "success"}
                mock_methodology.analyze_results.return_value = (True, 0.88, {"improvement": 15.0})
                
                with patch.object(agent, '_select_methodology', return_value=mock_methodology):
                    with patch.object(agent, '_save_research_data'):
                        result = await agent.execute_research_cycle(hypothesis)
                        
                        assert result["validation_status"] is True
                        assert hypothesis.status == "validated"
                        break
            
            # Generate final report
            report = await agent.generate_research_report()
            assert "research_summary" in report
    
    @pytest.mark.asyncio
    async def test_quantum_nas_integration(self):
        """Test integration between quantum optimizer and NAS."""
        # Create optimization parameters for NAS
        parameters = [
            OptimizationParameter("num_layers", 12, 6, 24),
            OptimizationParameter("batch_size", 32, 16, 128),
            OptimizationParameter("learning_rate", 0.001, 0.0001, 0.01)
        ]
        
        objective = QuantumOptimizationObjective(parameters)
        quantum_opt = QuantumNeuralOptimizer(objective, n_qubits=6)
        
        # Run quantum optimization
        quantum_results = await quantum_opt.quantum_optimize(max_iterations=5)
        
        # Use quantum results to inform NAS
        target_metrics = {
            "throughput": 100,
            "latency": 50,
            "accuracy": 0.85
        }
        
        evaluator = NIMServiceEvaluator(target_metrics)
        nas = GeneticNeuralArchitectureSearch(evaluator, population_size=5)
        
        # Initialize and evolve
        nas.initialize_population(ArchitectureType.TRANSFORMER)
        best_arch = await nas.evolve(num_generations=2)
        
        # Both should produce valid results
        assert quantum_results["best_parameters"] is not None
        assert best_arch.fitness_score > 0
        assert quantum_results["best_score"] > float('-inf')


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])