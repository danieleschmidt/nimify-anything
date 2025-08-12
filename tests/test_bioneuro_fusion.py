"""Comprehensive tests for bioneuro-olfactory fusion system."""

import pytest
import numpy as np
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from src.nimify.core import (
    BioneuroFusion, NeuralConfig, OlfactoryConfig,
    NeuralSignalType, OlfactoryMoleculeType
)
from src.nimify.neural_processor import NeuralSignalProcessor, NeuralFeatures
from src.nimify.olfactory_analyzer import OlfactoryAnalyzer, MolecularFeatures
from src.nimify.fusion_engine import MultiModalFusionEngine, FusionStrategy, FusionResults
from src.nimify.validation import (
    BioneuroDataValidator, BioneuroServiceValidator, BioneuroConfigValidator,
    ValidationError, NeuralDataRequest, OlfactoryDataRequest, FusionRequest,
    DataQualityValidator
)
from src.nimify.error_handling import (
    BioneuroError, NeuralDataError, OlfactoryDataError, FusionProcessingError,
    ErrorCategory, ErrorSeverity, global_error_handler
)
from src.nimify.performance_optimizer import (
    BioneuroOptimizer, AdaptiveCache, ResourceMonitor, ConcurrentProcessor,
    PerformanceMetrics, global_optimizer
)


class TestNeuralSignalProcessor:
    """Test suite for neural signal processing."""
    
    @pytest.fixture
    def neural_config(self):
        """Create test neural configuration."""
        return NeuralConfig(
            signal_type=NeuralSignalType.EEG,
            sampling_rate=1000,
            channels=64,
            time_window=2.0,
            preprocessing_filters=['bandpass', 'notch', 'baseline'],
            artifact_removal=True
        )
    
    @pytest.fixture
    def sample_neural_data(self):
        """Generate sample neural data."""
        # 64 channels, 2 seconds at 1000Hz
        np.random.seed(42)  # For reproducible tests
        return np.random.randn(64, 2000) * 1e-5  # Typical EEG amplitude
    
    def test_neural_processor_initialization(self, neural_config):
        """Test neural processor initialization."""
        processor = NeuralSignalProcessor(neural_config)
        
        assert processor.config == neural_config
        assert len(processor.filters) > 0
        assert 'bandpass' in processor.filters
    
    def test_neural_data_processing(self, neural_config, sample_neural_data):
        """Test complete neural data processing pipeline."""
        processor = NeuralSignalProcessor(neural_config)
        
        result = processor.process(sample_neural_data)
        
        # Check result structure
        assert 'features' in result
        assert 'preprocessed_data' in result
        assert 'timestamps' in result
        assert 'signal_quality' in result
        assert 'metadata' in result
        
        # Check features
        features = result['features']
        assert hasattr(features, 'mean_amplitude')
        assert hasattr(features, 'dominant_frequency')
        assert hasattr(features, 'spectral_power')
        assert hasattr(features, 'phase_locking_value')
        
        # Check signal quality
        quality = result['signal_quality']
        assert 'overall_quality' in quality
        assert 0 <= quality['overall_quality'] <= 1
    
    def test_artifact_removal(self, neural_config, sample_neural_data):
        """Test artifact detection and removal."""
        # Add artificial artifacts
        artifact_data = sample_neural_data.copy()
        artifact_data[10, 500:510] = 0.001  # Large amplitude spike
        
        processor = NeuralSignalProcessor(neural_config)
        result = processor.process(artifact_data)
        
        # Should detect and handle artifacts
        assert result['artifact_mask'] is not None
        assert np.any(result['artifact_mask'])
    
    def test_different_signal_types(self):
        """Test processing different neural signal types."""
        signal_types = [
            NeuralSignalType.EEG,
            NeuralSignalType.fMRI,
            NeuralSignalType.MEG,
            NeuralSignalType.EPHYS
        ]
        
        for signal_type in signal_types:
            config = NeuralConfig(
                signal_type=signal_type,
                sampling_rate=1000 if signal_type != NeuralSignalType.fMRI else 1,
                channels=32
            )
            
            processor = NeuralSignalProcessor(config)
            assert processor.config.signal_type == signal_type
    
    def test_olfactory_response_analysis(self, neural_config, sample_neural_data):
        """Test olfactory response pattern analysis."""
        processor = NeuralSignalProcessor(neural_config)
        
        # Simulate stimulus onset times
        stimulus_times = np.array([0.5, 1.0, 1.5])  # 3 stimuli
        
        response_analysis = processor.analyze_olfactory_response_patterns(
            sample_neural_data,
            stimulus_times,
            pre_stimulus_window=0.2,
            post_stimulus_window=0.8
        )
        
        assert 'individual_responses' in response_analysis
        assert 'aggregated_response' in response_analysis
        assert 'response_consistency' in response_analysis
        assert len(response_analysis['individual_responses']) == len(stimulus_times)


class TestOlfactoryAnalyzer:
    """Test suite for olfactory stimulus analysis."""
    
    @pytest.fixture
    def olfactory_config(self):
        """Create test olfactory configuration."""
        return OlfactoryConfig(
            molecule_types=[OlfactoryMoleculeType.ALDEHYDE, OlfactoryMoleculeType.TERPENE],
            concentration_range=(0.01, 10.0),
            molecular_descriptors=['molecular_weight', 'vapor_pressure', 'polarity'],
            stimulus_duration=3.0
        )
    
    @pytest.fixture
    def sample_molecule_data(self):
        """Sample molecule data for testing."""
        return {
            'name': 'vanillin',
            'molecular_weight': 152.15,
            'vapor_pressure': 0.0133,
            'functional_groups': ['aldehyde', 'phenol', 'methoxy'],
            'odor_character': ['vanilla', 'sweet', 'creamy'],
            'smiles': 'COc1cc(C=O)ccc1O'
        }
    
    def test_olfactory_analyzer_initialization(self, olfactory_config):
        """Test olfactory analyzer initialization."""
        analyzer = OlfactoryAnalyzer(olfactory_config)
        
        assert analyzer.config == olfactory_config
        assert analyzer.molecular_database is not None
        assert len(analyzer.molecular_database) > 0
    
    def test_molecule_analysis(self, olfactory_config, sample_molecule_data):
        """Test complete molecule analysis pipeline."""
        analyzer = OlfactoryAnalyzer(olfactory_config)
        
        result = analyzer.analyze(sample_molecule_data, concentration=1.0)
        
        # Check result structure
        assert 'molecular_features' in result
        assert 'receptor_activation' in result
        assert 'psychophysical_properties' in result
        assert 'neural_predictions' in result
        assert 'temporal_profile' in result
        
        # Check molecular features
        features = result['molecular_features']
        assert hasattr(features, 'molecular_weight')
        assert hasattr(features, 'odor_character')
        assert hasattr(features, 'perceived_intensity')
        
        # Check receptor activation
        activation = result['receptor_activation']
        assert 'activation_strengths' in activation
        assert 'primary_receptors' in activation
        
        # Check neural predictions
        neural_pred = result['neural_predictions']
        assert 'olfactory_bulb' in neural_pred
        assert 'piriform_cortex' in neural_pred
        assert 'orbitofrontal_cortex' in neural_pred
        assert 'limbic_system' in neural_pred
    
    def test_concentration_dependency(self, olfactory_config, sample_molecule_data):
        """Test concentration-dependent responses."""
        analyzer = OlfactoryAnalyzer(olfactory_config)
        
        concentrations = [0.1, 1.0, 5.0]
        results = []
        
        for conc in concentrations:
            result = analyzer.analyze(sample_molecule_data, concentration=conc)
            results.append(result)
        
        # Higher concentration should generally yield higher activation
        activations = [r['receptor_activation']['activation_sum'] for r in results]
        assert activations[1] > activations[0]  # 1.0 > 0.1
        assert activations[2] > activations[1]  # 5.0 > 1.0
    
    def test_mixture_analysis(self, olfactory_config, sample_molecule_data):
        """Test mixture response prediction."""
        analyzer = OlfactoryAnalyzer(olfactory_config)
        
        # Create mixture components
        component1 = (sample_molecule_data, 0.6)  # 60% vanillin
        component2 = ({'name': 'limonene', 'molecular_weight': 136.23}, 0.4)  # 40% limonene
        
        mixture_result = analyzer.predict_mixture_response(
            [component1, component2],
            total_concentration=2.0
        )
        
        assert 'individual_components' in mixture_result
        assert 'mixture_interactions' in mixture_result
        assert 'aggregated_neural_response' in mixture_result
        assert 'emergent_properties' in mixture_result
        
        # Check mixture interactions
        interactions = mixture_result['mixture_interactions']
        assert 'competition_matrix' in interactions
        assert 'synergy_scores' in interactions
        assert 'interaction_type' in interactions
    
    def test_temporal_profile_generation(self, olfactory_config, sample_molecule_data):
        """Test temporal response profile generation."""
        analyzer = OlfactoryAnalyzer(olfactory_config)
        
        result = analyzer.analyze(sample_molecule_data, concentration=1.0)
        temporal_profile = result['temporal_profile']
        
        assert 'time_points' in temporal_profile
        assert 'response_profile' in temporal_profile
        assert 'onset_time' in temporal_profile
        assert 'peak_time' in temporal_profile
        assert 'adaptation_time_constant' in temporal_profile
        
        # Check temporal progression
        time_points = temporal_profile['time_points']
        response_profile = temporal_profile['response_profile']
        
        assert len(time_points) == len(response_profile)
        assert temporal_profile['peak_time'] > temporal_profile['onset_time']


class TestMultiModalFusion:
    """Test suite for multi-modal fusion engine."""
    
    @pytest.fixture
    def sample_neural_features(self):
        """Sample neural features for fusion testing."""
        return {
            'features': Mock(
                mean_amplitude=1e-5,
                std_amplitude=5e-6,
                peak_amplitude=2e-5,
                dominant_frequency=12.0,
                spectral_entropy=0.8,
                phase_locking_value=0.6,
                spectral_power={'alpha': 0.3, 'beta': 0.2, 'gamma': 0.1}
            ),
            'signal_quality': {'overall_quality': 0.85},
            'timestamps': np.arange(0, 2.0, 0.001)
        }
    
    @pytest.fixture
    def sample_olfactory_features(self):
        """Sample olfactory features for fusion testing."""
        return {
            'molecular_features': Mock(
                molecular_weight=152.15,
                perceived_intensity=7.5,
                polarity_index=0.6,
                hydrophobicity=1.2,
                odor_character=['vanilla', 'sweet']
            ),
            'receptor_activation': {
                'activation_sum': 2.5,
                'activation_strengths': np.array([0.8, 0.3, 0.1, 0.2, 0.9, 0.4])
            },
            'psychophysical_properties': {
                'perceived_intensity': 7.5,
                'pleasantness': 8.0
            },
            'temporal_profile': {
                'time_points': np.arange(0, 3.0, 0.01),
                'response_profile': np.exp(-np.arange(0, 3.0, 0.01) / 1.5)
            }
        }
    
    @pytest.mark.parametrize("fusion_strategy", [
        FusionStrategy.EARLY_FUSION,
        FusionStrategy.LATE_FUSION,
        FusionStrategy.ATTENTION_FUSION,
        FusionStrategy.CANONICAL_CORRELATION,
        FusionStrategy.BAYESIAN_FUSION
    ])
    def test_fusion_strategies(self, sample_neural_features, sample_olfactory_features, fusion_strategy):
        """Test different fusion strategies."""
        fusion_engine = MultiModalFusionEngine(fusion_strategy)
        
        result = fusion_engine.fuse(sample_neural_features, sample_olfactory_features)
        
        assert 'fusion_results' in result
        assert isinstance(result['fusion_results'], FusionResults)
        
        fusion_results = result['fusion_results']
        assert fusion_results.fusion_strategy == fusion_strategy
        assert fusion_results.fused_features is not None
        assert fusion_results.fusion_weights is not None
        assert fusion_results.fusion_confidence >= 0
    
    def test_temporal_alignment(self, sample_neural_features, sample_olfactory_features):
        """Test temporal alignment functionality."""
        fusion_engine = MultiModalFusionEngine(FusionStrategy.ATTENTION_FUSION)
        
        # Test with temporal alignment enabled
        result_aligned = fusion_engine.fuse(
            sample_neural_features,
            sample_olfactory_features,
            temporal_alignment=True
        )
        
        # Test with temporal alignment disabled
        result_no_alignment = fusion_engine.fuse(
            sample_neural_features,
            sample_olfactory_features,
            temporal_alignment=False
        )
        
        # Should have different results
        aligned_score = result_aligned['fusion_results'].temporal_alignment_score
        no_align_score = result_no_alignment['fusion_results'].temporal_alignment_score
        
        # Aligned version should have alignment score > 0
        assert aligned_score >= 0
        assert no_align_score == 0
    
    def test_fusion_performance_tracking(self):
        """Test fusion performance tracking."""
        fusion_engine = MultiModalFusionEngine(FusionStrategy.EARLY_FUSION)
        
        # Perform multiple fusions
        for i in range(5):
            neural_features = {'features': Mock(mean_amplitude=i * 1e-6)}
            olfactory_features = {'molecular_features': Mock(molecular_weight=100 + i * 10)}
            
            fusion_engine.fuse(neural_features, olfactory_features)
        
        # Check performance analysis
        performance = fusion_engine.analyze_fusion_performance()
        
        assert 'n_fusions' in performance
        assert performance['n_fusions'] == 5
        assert 'average_confidence' in performance
        assert 'fusion_strategy' in performance


class TestValidation:
    """Test suite for validation components."""
    
    def test_neural_data_validation(self):
        """Test neural data validation."""
        # Valid neural data
        valid_data = {
            'neural_data': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            'timestamps': [0.0, 0.001, 0.002],
            'sampling_rate': 1000
        }
        
        request = NeuralDataRequest(**valid_data)
        assert request.neural_data == valid_data['neural_data']
        assert request.sampling_rate == 1000
        
        # Invalid neural data (inconsistent channel lengths)
        invalid_data = {
            'neural_data': [[1.0, 2.0], [4.0, 5.0, 6.0]],  # Different lengths
            'sampling_rate': 1000
        }
        
        with pytest.raises(ValueError, match="All channels must have same length"):
            NeuralDataRequest(**invalid_data)
    
    def test_olfactory_data_validation(self):
        """Test olfactory data validation."""
        # Valid olfactory data
        valid_data = {
            'molecule_data': {
                'name': 'vanillin',
                'molecular_weight': 152.15,
                'functional_groups': ['aldehyde', 'phenol']
            },
            'concentration': 1.0,
            'stimulus_duration': 3.0
        }
        
        request = OlfactoryDataRequest(**valid_data)
        assert request.concentration == 1.0
        
        # Invalid concentration
        invalid_data = valid_data.copy()
        invalid_data['concentration'] = -1.0  # Negative concentration
        
        with pytest.raises(ValueError):
            OlfactoryDataRequest(**invalid_data)
    
    def test_fusion_request_validation(self):
        """Test fusion request validation."""
        neural_data = {
            'neural_data': [[1.0, 2.0, 3.0]],
            'sampling_rate': 1000
        }
        
        olfactory_data = {
            'molecule_data': {'name': 'vanillin'},
            'concentration': 1.0
        }
        
        valid_fusion_data = {
            'neural_request': neural_data,
            'olfactory_request': olfactory_data,
            'fusion_strategy': 'attention_fusion'
        }
        
        request = FusionRequest(**valid_fusion_data)
        assert request.fusion_strategy == 'attention_fusion'
        
        # Invalid fusion strategy
        invalid_fusion_data = valid_fusion_data.copy()
        invalid_fusion_data['fusion_strategy'] = 'invalid_strategy'
        
        with pytest.raises(ValueError, match="Invalid fusion strategy"):
            FusionRequest(**invalid_fusion_data)
    
    def test_data_quality_assessment(self):
        """Test data quality assessment."""
        # High quality neural data
        good_neural_data = [[1e-5, 2e-5, 3e-5, 1e-5]] * 64  # 64 channels
        quality_metrics = DataQualityValidator.assess_neural_data_quality(
            good_neural_data, sampling_rate=1000
        )
        
        assert quality_metrics['overall_quality'] > 0.5
        assert quality_metrics['quality_grade'] in ['excellent', 'good', 'fair', 'poor']
        
        # High quality olfactory data
        good_molecule_data = {
            'name': 'vanillin',
            'molecular_weight': 152.15,
            'functional_groups': ['aldehyde'],
            'odor_character': ['vanilla']
        }
        
        olfactory_quality = DataQualityValidator.assess_olfactory_data_quality(
            good_molecule_data, concentration=1.0
        )
        
        assert olfactory_quality['overall_quality'] > 0.8
        assert olfactory_quality['quality_grade'] == 'excellent'


class TestErrorHandling:
    """Test suite for error handling system."""
    
    def test_bioneuro_error_creation(self):
        """Test bioneuro-specific error creation."""
        # Neural data error
        neural_error = NeuralDataError(
            "Invalid sampling rate",
            signal_type="EEG",
            sampling_rate=-1000
        )
        
        assert neural_error.category == ErrorCategory.NEURAL_DATA
        assert neural_error.severity == ErrorSeverity.HIGH
        assert neural_error.details['signal_type'] == "EEG"
        assert neural_error.details['sampling_rate'] == -1000
        
        # Olfactory data error
        olfactory_error = OlfactoryDataError(
            "Invalid concentration",
            molecule_name="vanillin",
            concentration=-1.0
        )
        
        assert olfactory_error.category == ErrorCategory.OLFACTORY_DATA
        assert olfactory_error.details['molecule_name'] == "vanillin"
        
        # Fusion processing error
        fusion_error = FusionProcessingError(
            "Incompatible modalities",
            fusion_strategy="attention_fusion",
            modalities=["neural", "olfactory"]
        )
        
        assert fusion_error.category == ErrorCategory.FUSION_PROCESSING
        assert fusion_error.details['fusion_strategy'] == "attention_fusion"
    
    def test_error_recovery_manager(self):
        """Test error recovery management."""
        # Create test error
        test_error = NeuralDataError("Test error")
        
        # Handle error
        error_context = global_error_handler.handle_error(test_error)
        
        assert error_context.category == ErrorCategory.NEURAL_DATA
        assert error_context.severity == ErrorSeverity.HIGH
        assert len(error_context.error_id) > 0
        
        # Check error statistics
        stats = global_error_handler.get_error_statistics()
        assert stats['total_errors'] >= 1
        assert ErrorCategory.NEURAL_DATA.value in stats['by_category']


class TestPerformanceOptimization:
    """Test suite for performance optimization."""
    
    def test_adaptive_cache(self):
        """Test adaptive caching system."""
        cache = AdaptiveCache(max_size=100, ttl_seconds=1)
        
        # Test basic operations
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test cache miss
        assert cache.get("nonexistent") is None
        
        # Test TTL expiration
        time.sleep(1.1)
        assert cache.get("key1") is None  # Should be expired
        
        # Test cache statistics
        stats = cache.get_stats()
        assert 'hit_rate' in stats
        assert 'size' in stats
    
    def test_resource_monitoring(self):
        """Test resource monitoring."""
        monitor = ResourceMonitor(monitor_interval=0.1)
        
        # Test manual metrics collection
        metrics = monitor._collect_metrics()
        assert 'cpu_percent' in metrics
        assert 'memory_percent' in metrics
        assert 'timestamp' in metrics
        
        # Test resource pressure assessment
        pressure = monitor.get_resource_pressure()
        assert 'overall' in pressure
        assert pressure['overall'] in ['low', 'medium', 'high']
        
        # Cleanup
        monitor.stop_monitoring()
    
    def test_concurrent_processor(self):
        """Test concurrent processing."""
        processor = ConcurrentProcessor(max_workers=2)
        
        def simple_task(x):
            return x * 2
        
        # Test batch processing
        data_batch = [1, 2, 3, 4, 5]
        results = processor.submit_batch(simple_task, data_batch)
        
        expected = [2, 4, 6, 8, 10]
        assert results == expected
        
        # Test performance stats
        stats = processor.get_performance_stats()
        assert 'current_load' in stats
        assert 'total_executions' in stats
        
        # Cleanup
        processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_bioneuro_optimizer(self):
        """Test main bioneuro optimizer."""
        optimizer = BioneuroOptimizer()
        
        # Test neural processing optimization
        neural_data = np.random.randn(32, 1000) * 1e-5
        neural_config = {
            'signal_type': 'eeg',
            'sampling_rate': 1000,
            'time_window': 1.0
        }
        
        result, metrics = optimizer.optimize_neural_processing(neural_data, neural_config)
        
        assert result is not None
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.execution_time > 0
        
        # Test optimization statistics
        stats = optimizer.get_optimization_stats()
        assert 'cache_stats' in stats
        assert 'performance_trends' in stats
        assert 'optimization_recommendations' in stats
        
        # Cleanup
        optimizer.cleanup()


class TestIntegration:
    """Integration tests for the complete bioneuro-olfactory fusion system."""
    
    @pytest.fixture
    def bioneuro_fusion(self):
        """Create bioneuro fusion system."""
        neural_config = NeuralConfig(
            signal_type=NeuralSignalType.EEG,
            sampling_rate=1000,
            channels=64,
            time_window=2.0
        )
        
        olfactory_config = OlfactoryConfig(
            molecule_types=[OlfactoryMoleculeType.ALDEHYDE],
            concentration_range=(0.01, 10.0),
            stimulus_duration=3.0
        )
        
        return BioneuroFusion(neural_config, olfactory_config)
    
    def test_end_to_end_fusion_pipeline(self, bioneuro_fusion):
        """Test complete end-to-end fusion pipeline."""
        # Generate test data
        neural_data = np.random.randn(64, 2000) * 1e-5
        timestamps = np.arange(0, 2.0, 0.001)
        
        molecule_data = {
            'name': 'vanillin',
            'molecular_weight': 152.15,
            'functional_groups': ['aldehyde', 'phenol'],
            'odor_character': ['vanilla', 'sweet']
        }
        concentration = 1.0
        
        # Step 1: Process neural data
        neural_result = bioneuro_fusion.process_neural_data(neural_data, timestamps)
        
        assert 'features' in neural_result
        assert 'signal_quality' in neural_result
        
        # Step 2: Analyze olfactory stimulus
        olfactory_result = bioneuro_fusion.analyze_olfactory_stimulus(
            molecule_data, concentration
        )
        
        assert 'molecular_features' in olfactory_result
        assert 'receptor_activation' in olfactory_result
        
        # Step 3: Fuse modalities
        fusion_result = bioneuro_fusion.fuse_modalities(
            neural_result, olfactory_result
        )
        
        assert 'fusion_results' in fusion_result
        assert 'detailed_analysis' in fusion_result
        
        # Verify fusion results
        fusion_data = fusion_result['fusion_results']
        assert hasattr(fusion_data, 'fused_features')
        assert hasattr(fusion_data, 'neural_olfactory_correlation')
        assert hasattr(fusion_data, 'fusion_confidence')
    
    def test_performance_under_load(self, bioneuro_fusion):
        """Test system performance under load."""
        # Generate multiple test cases
        test_cases = []
        for i in range(10):
            neural_data = np.random.randn(32, 1000) * 1e-5
            molecule_data = {'name': f'test_molecule_{i}', 'molecular_weight': 100 + i * 10}
            test_cases.append((neural_data, molecule_data, 1.0 + i * 0.1))
        
        start_time = time.time()
        results = []
        
        for neural_data, molecule_data, concentration in test_cases:
            # Process each case
            neural_result = bioneuro_fusion.process_neural_data(neural_data)
            olfactory_result = bioneuro_fusion.analyze_olfactory_stimulus(
                molecule_data, concentration
            )
            fusion_result = bioneuro_fusion.fuse_modalities(
                neural_result, olfactory_result
            )
            results.append(fusion_result)
        
        total_time = time.time() - start_time
        avg_time_per_case = total_time / len(test_cases)
        
        # Performance assertions
        assert len(results) == len(test_cases)
        assert avg_time_per_case < 5.0  # Should process each case in under 5 seconds
        
        # Verify all results are valid
        for result in results:
            assert 'fusion_results' in result
            assert result['fusion_results'].fusion_confidence >= 0
    
    def test_error_resilience(self, bioneuro_fusion):
        """Test system resilience to various error conditions."""
        # Test with corrupted neural data
        corrupted_neural = np.array([[np.nan, np.inf, 1e10], [0, 0, 0]])
        
        with pytest.raises((ValueError, NeuralDataError)):
            bioneuro_fusion.process_neural_data(corrupted_neural)
        
        # Test with invalid olfactory data
        invalid_molecule = {'name': '', 'molecular_weight': -100}
        
        with pytest.raises((ValueError, OlfactoryDataError)):
            bioneuro_fusion.analyze_olfactory_stimulus(invalid_molecule, -1.0)
        
        # Test with mismatched modalities for fusion
        good_neural = {'features': Mock(), 'signal_quality': {}}
        bad_olfactory = None
        
        with pytest.raises((AttributeError, FusionProcessingError)):
            bioneuro_fusion.fuse_modalities(good_neural, bad_olfactory)
    
    def test_optimization_integration(self):
        """Test integration with performance optimization."""
        # Test with global optimizer
        neural_data = np.random.randn(16, 500) * 1e-5
        neural_config = {
            'signal_type': 'eeg',
            'sampling_rate': 500,
            'channels': 16
        }
        
        # First run (should cache results)
        result1, metrics1 = global_optimizer.optimize_neural_processing(
            neural_data, neural_config
        )
        
        # Second run (should hit cache)
        result2, metrics2 = global_optimizer.optimize_neural_processing(
            neural_data, neural_config
        )
        
        # Second run should be faster due to caching
        assert metrics2.execution_time <= metrics1.execution_time
        
        # Cache hit rate should improve
        cache_stats = global_optimizer.cache.get_stats()
        assert cache_stats['hits'] > 0


# Performance benchmark tests
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.benchmark
    def test_neural_processing_benchmark(self):
        """Benchmark neural processing performance."""
        config = NeuralConfig(
            signal_type=NeuralSignalType.EEG,
            sampling_rate=1000,
            channels=64,
            time_window=5.0
        )
        
        processor = NeuralSignalProcessor(config)
        neural_data = np.random.randn(64, 5000) * 1e-5
        
        # Benchmark processing time
        start_time = time.time()
        result = processor.process(neural_data)
        processing_time = time.time() - start_time
        
        # Performance assertions
        assert processing_time < 2.0  # Should process 5s of 64-channel data in under 2s
        assert result['signal_quality']['overall_quality'] >= 0
    
    @pytest.mark.benchmark
    def test_fusion_engine_benchmark(self):
        """Benchmark fusion engine performance."""
        fusion_engine = MultiModalFusionEngine(FusionStrategy.ATTENTION_FUSION)
        
        # Create large feature sets
        neural_features = {
            'features': Mock(
                mean_amplitude=1e-5,
                spectral_power={'alpha': 0.3, 'beta': 0.2, 'gamma': 0.1}
            ),
            'signal_quality': {'overall_quality': 0.8}
        }
        
        olfactory_features = {
            'molecular_features': Mock(molecular_weight=150.0, perceived_intensity=7.0),
            'receptor_activation': {'activation_sum': 2.5}
        }
        
        # Benchmark fusion time
        start_time = time.time()
        result = fusion_engine.fuse(neural_features, olfactory_features)
        fusion_time = time.time() - start_time
        
        # Performance assertions
        assert fusion_time < 1.0  # Should fuse in under 1 second
        assert result['fusion_results'].fusion_confidence >= 0
    
    @pytest.mark.benchmark
    def test_cache_performance(self):
        """Benchmark cache performance."""
        cache = AdaptiveCache(max_size=10000)
        
        # Benchmark cache operations
        keys = [f"key_{i}" for i in range(1000)]
        values = [f"value_{i}" for i in range(1000)]
        
        # Benchmark set operations
        start_time = time.time()
        for key, value in zip(keys, values):
            cache.set(key, value)
        set_time = time.time() - start_time
        
        # Benchmark get operations
        start_time = time.time()
        for key in keys:
            cache.get(key)
        get_time = time.time() - start_time
        
        # Performance assertions
        assert set_time < 0.5  # Should set 1000 items in under 0.5s
        assert get_time < 0.1  # Should get 1000 items in under 0.1s
        
        cache_stats = cache.get_stats()
        assert cache_stats['hit_rate'] == 1.0  # All gets should hit


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])