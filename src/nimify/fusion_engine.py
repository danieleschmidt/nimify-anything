"""Multi-modal fusion engine for bioneuro-olfactory data integration."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Strategies for multi-modal data fusion."""
    EARLY_FUSION = "early_fusion"           # Concatenate features before processing
    LATE_FUSION = "late_fusion"             # Combine decisions/predictions
    INTERMEDIATE_FUSION = "intermediate"     # Fusion at intermediate processing stages
    ATTENTION_FUSION = "attention"          # Attention-weighted fusion
    CANONICAL_CORRELATION = "canonical"     # Canonical correlation analysis
    BAYESIAN_FUSION = "bayesian"           # Bayesian inference-based fusion


@dataclass
class FusionResults:
    """Container for multi-modal fusion results."""
    
    # Fused representations
    fused_features: np.ndarray
    fusion_weights: np.ndarray
    
    # Cross-modal correlations
    neural_olfactory_correlation: float
    temporal_alignment_score: float
    
    # Prediction outputs
    olfactory_response_prediction: dict[str, float]
    neural_activity_prediction: dict[str, float]
    behavioral_prediction: dict[str, float]
    
    # Quality metrics
    fusion_confidence: float
    prediction_uncertainty: float
    cross_modal_consistency: float
    
    # Metadata
    fusion_strategy: FusionStrategy
    processing_timestamp: np.datetime64


class AttentionMechanism:
    """Attention mechanism for weighting different modalities."""
    
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.attention_weights = None
    
    def compute_attention_weights(
        self, 
        neural_features: np.ndarray, 
        olfactory_features: np.ndarray
    ) -> np.ndarray:
        """Compute attention weights for multi-modal features."""
        
        # Simple attention based on feature variance and correlation
        neural_variance = np.var(neural_features, axis=0) if neural_features.ndim > 1 else np.var(neural_features)
        olfactory_variance = np.var(olfactory_features, axis=0) if olfactory_features.ndim > 1 else np.var(olfactory_features)
        
        # Cross-modal correlation as attention signal
        if neural_features.size > 1 and olfactory_features.size > 1:
            correlation = np.corrcoef(
                neural_features.flatten(), 
                olfactory_features.flatten()
            )[0, 1]
            correlation = np.abs(correlation) if not np.isnan(correlation) else 0.0
        else:
            correlation = 0.5
        
        # Combine variance and correlation to determine attention
        neural_attention = neural_variance * (1 + correlation)
        olfactory_attention = olfactory_variance * (1 + correlation)
        
        # Normalize attention weights
        total_attention = neural_attention + olfactory_attention
        if total_attention > 0:
            self.attention_weights = np.array([
                neural_attention / total_attention,
                olfactory_attention / total_attention
            ])
        else:
            self.attention_weights = np.array([0.5, 0.5])
        
        return self.attention_weights


class TemporalAlignment:
    """Handles temporal alignment between neural and olfactory data."""
    
    @staticmethod
    def align_temporal_signals(
        neural_timestamps: np.ndarray,
        neural_data: np.ndarray,
        olfactory_timestamps: np.ndarray,
        olfactory_data: np.ndarray,
        alignment_window: float = 0.1  # seconds
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Align temporal signals from neural and olfactory modalities.
        
        Returns:
            Aligned neural data, aligned olfactory data, alignment quality score
        """
        
        # Find common time range
        start_time = max(neural_timestamps[0], olfactory_timestamps[0])
        end_time = min(neural_timestamps[-1], olfactory_timestamps[-1])
        
        if start_time >= end_time:
            logger.warning("No temporal overlap between neural and olfactory data")
            return neural_data, olfactory_data, 0.0
        
        # Create common time grid
        common_dt = min(
            np.mean(np.diff(neural_timestamps)),
            np.mean(np.diff(olfactory_timestamps))
        )
        common_timestamps = np.arange(start_time, end_time, common_dt)
        
        # Interpolate both signals to common grid
        aligned_neural = np.interp(common_timestamps, neural_timestamps, neural_data)
        aligned_olfactory = np.interp(common_timestamps, olfactory_timestamps, olfactory_data)
        
        # Compute cross-correlation for alignment quality
        if len(aligned_neural) > 10 and len(aligned_olfactory) > 10:
            correlation = np.corrcoef(aligned_neural, aligned_olfactory)[0, 1]
            alignment_quality = abs(correlation) if not np.isnan(correlation) else 0.0
        else:
            alignment_quality = 0.0
        
        return aligned_neural, aligned_olfactory, alignment_quality
    
    @staticmethod
    def detect_response_onset(
        signal: np.ndarray,
        timestamps: np.ndarray,
        baseline_duration: float = 1.0,
        threshold_factor: float = 2.0
    ) -> float | None:
        """Detect response onset time in a signal."""
        
        if len(signal) < 10:
            return None
        
        # Compute baseline statistics
        baseline_samples = int(baseline_duration * len(signal) / (timestamps[-1] - timestamps[0]))
        baseline_samples = min(baseline_samples, len(signal) // 3)
        
        if baseline_samples < 5:
            return None
        
        baseline_mean = np.mean(signal[:baseline_samples])
        baseline_std = np.std(signal[:baseline_samples])
        
        # Detection threshold
        threshold = baseline_mean + threshold_factor * baseline_std
        
        # Find first crossing
        above_threshold = signal > threshold
        if not np.any(above_threshold):
            return None
        
        onset_idx = np.where(above_threshold)[0][0]
        return timestamps[onset_idx]


class MultiModalFusionEngine:
    """Main engine for fusing neural and olfactory data."""
    
    def __init__(self, fusion_strategy: FusionStrategy = FusionStrategy.ATTENTION_FUSION):
        self.fusion_strategy = fusion_strategy
        self.attention_mechanism = None
        self.fusion_history = []
    
    def fuse(
        self,
        neural_features: dict[str, Any],
        olfactory_features: dict[str, Any],
        temporal_alignment: bool = True
    ) -> dict[str, Any]:
        """
        Main fusion function combining neural and olfactory modalities.
        
        Args:
            neural_features: Processed neural signal features
            olfactory_features: Analyzed olfactory stimulus features
            temporal_alignment: Whether to perform temporal alignment
            
        Returns:
            Comprehensive fusion results
        """
        
        logger.info(f"Fusing modalities using {self.fusion_strategy.value} strategy")
        
        # Extract and prepare features
        neural_vector = self._extract_neural_feature_vector(neural_features)
        olfactory_vector = self._extract_olfactory_feature_vector(olfactory_features)
        
        # Temporal alignment if requested
        alignment_score = 0.0
        if temporal_alignment:
            neural_vector, olfactory_vector, alignment_score = self._perform_temporal_alignment(
                neural_features, olfactory_features, neural_vector, olfactory_vector
            )
        
        # Apply fusion strategy
        fusion_results = self._apply_fusion_strategy(neural_vector, olfactory_vector)
        
        # Cross-modal analysis
        cross_modal_analysis = self._analyze_cross_modal_relationships(
            neural_features, olfactory_features
        )
        
        # Prediction generation
        predictions = self._generate_predictions(
            fusion_results['fused_features'],
            neural_features,
            olfactory_features
        )
        
        # Quality assessment
        quality_metrics = self._assess_fusion_quality(
            neural_vector, olfactory_vector, fusion_results
        )
        
        # Compile comprehensive results
        comprehensive_results = {
            'fusion_results': FusionResults(
                fused_features=fusion_results['fused_features'],
                fusion_weights=fusion_results['fusion_weights'],
                neural_olfactory_correlation=cross_modal_analysis['correlation'],
                temporal_alignment_score=alignment_score,
                olfactory_response_prediction=predictions['olfactory_response'],
                neural_activity_prediction=predictions['neural_activity'],
                behavioral_prediction=predictions['behavioral'],
                fusion_confidence=quality_metrics['confidence'],
                prediction_uncertainty=quality_metrics['uncertainty'],
                cross_modal_consistency=cross_modal_analysis['consistency'],
                fusion_strategy=self.fusion_strategy,
                processing_timestamp=np.datetime64('now')
            ),
            'detailed_analysis': {
                'neural_features_summary': self._summarize_neural_features(neural_features),
                'olfactory_features_summary': self._summarize_olfactory_features(olfactory_features),
                'cross_modal_analysis': cross_modal_analysis,
                'temporal_dynamics': self._analyze_temporal_dynamics(neural_features, olfactory_features),
                'fusion_diagnostics': quality_metrics
            }
        }
        
        # Store in history
        self.fusion_history.append(comprehensive_results)
        
        return comprehensive_results
    
    def _extract_neural_feature_vector(self, neural_features: dict[str, Any]) -> np.ndarray:
        """Extract numerical feature vector from neural features."""
        
        # Get the neural features object
        features = neural_features.get('features')
        if features is None:
            logger.warning("No neural features found")
            return np.array([0.0] * 10)
        
        # Extract key numerical features
        feature_vector = []
        
        # Time domain features
        feature_vector.extend([
            getattr(features, 'mean_amplitude', 0.0),
            getattr(features, 'std_amplitude', 0.0),
            getattr(features, 'peak_amplitude', 0.0)
        ])
        
        # Frequency domain features  
        feature_vector.append(getattr(features, 'dominant_frequency', 0.0))
        feature_vector.append(getattr(features, 'spectral_entropy', 0.0))
        
        # Spectral power features
        spectral_power = getattr(features, 'spectral_power', {})
        for band in ['alpha', 'beta', 'gamma', 'theta', 'delta']:
            feature_vector.append(spectral_power.get(band, 0.0))
        
        # Phase-locking and coherence
        feature_vector.append(getattr(features, 'phase_locking_value', 0.0))
        
        # Signal quality
        signal_quality = neural_features.get('signal_quality', {})
        feature_vector.append(signal_quality.get('overall_quality', 0.0))
        
        return np.array(feature_vector)
    
    def _extract_olfactory_feature_vector(self, olfactory_features: dict[str, Any]) -> np.ndarray:
        """Extract numerical feature vector from olfactory features."""
        
        molecular_features = olfactory_features.get('molecular_features')
        if molecular_features is None:
            logger.warning("No molecular features found")
            return np.array([0.0] * 10)
        
        feature_vector = []
        
        # Molecular properties
        feature_vector.extend([
            getattr(molecular_features, 'molecular_weight', 0.0) / 300.0,  # Normalized
            getattr(molecular_features, 'vapor_pressure', 0.0),
            getattr(molecular_features, 'polarity_index', 0.0),
            getattr(molecular_features, 'hydrophobicity', 0.0),
            getattr(molecular_features, 'surface_area', 0.0) / 1000.0,  # Normalized
            getattr(molecular_features, 'perceived_intensity', 0.0) / 10.0,  # Normalized
            getattr(molecular_features, 'threshold_concentration', 0.0)
        ])
        
        # Receptor activation
        receptor_activation = olfactory_features.get('receptor_activation', {})
        feature_vector.append(receptor_activation.get('activation_sum', 0.0))
        
        # Psychophysical properties
        psychophysical = olfactory_features.get('psychophysical_properties', {})
        feature_vector.extend([
            psychophysical.get('perceived_intensity', 0.0) / 10.0,
            psychophysical.get('pleasantness', 0.0) / 10.0
        ])
        
        return np.array(feature_vector)
    
    def _perform_temporal_alignment(
        self,
        neural_features: dict[str, Any],
        olfactory_features: dict[str, Any],
        neural_vector: np.ndarray,
        olfactory_vector: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Perform temporal alignment between modalities."""
        
        # Get temporal profiles
        neural_timestamps = neural_features.get('timestamps', np.array([]))
        olfactory_temporal = olfactory_features.get('temporal_profile', {})
        olfactory_timestamps = olfactory_temporal.get('time_points', np.array([]))
        olfactory_profile = olfactory_temporal.get('response_profile', np.array([]))
        
        if len(neural_timestamps) == 0 or len(olfactory_timestamps) == 0:
            logger.warning("Insufficient temporal data for alignment")
            return neural_vector, olfactory_vector, 0.0
        
        # Create mock neural temporal profile from features
        neural_profile = np.ones(len(neural_timestamps)) * neural_vector[0] if len(neural_vector) > 0 else np.ones(len(neural_timestamps))
        
        # Align temporal signals
        aligned_neural, aligned_olfactory, alignment_score = TemporalAlignment.align_temporal_signals(
            neural_timestamps, neural_profile,
            olfactory_timestamps, olfactory_profile
        )
        
        # Update feature vectors with temporal alignment information
        if len(aligned_neural) > 0 and len(aligned_olfactory) > 0:
            # Add temporal correlation as feature
            neural_vector = np.append(neural_vector, [np.mean(aligned_neural), np.std(aligned_neural)])
            olfactory_vector = np.append(olfactory_vector, [np.mean(aligned_olfactory), np.std(aligned_olfactory)])
        
        return neural_vector, olfactory_vector, alignment_score
    
    def _apply_fusion_strategy(
        self,
        neural_vector: np.ndarray,
        olfactory_vector: np.ndarray
    ) -> dict[str, Any]:
        """Apply the specified fusion strategy."""
        
        if self.fusion_strategy == FusionStrategy.EARLY_FUSION:
            return self._early_fusion(neural_vector, olfactory_vector)
        elif self.fusion_strategy == FusionStrategy.LATE_FUSION:
            return self._late_fusion(neural_vector, olfactory_vector)
        elif self.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            return self._attention_fusion(neural_vector, olfactory_vector)
        elif self.fusion_strategy == FusionStrategy.CANONICAL_CORRELATION:
            return self._canonical_correlation_fusion(neural_vector, olfactory_vector)
        elif self.fusion_strategy == FusionStrategy.BAYESIAN_FUSION:
            return self._bayesian_fusion(neural_vector, olfactory_vector)
        else:
            # Default to early fusion
            return self._early_fusion(neural_vector, olfactory_vector)
    
    def _early_fusion(
        self,
        neural_vector: np.ndarray,
        olfactory_vector: np.ndarray
    ) -> dict[str, Any]:
        """Early fusion: concatenate features before processing."""
        
        # Normalize feature vectors to same scale
        neural_norm = neural_vector / (np.linalg.norm(neural_vector) + 1e-10)
        olfactory_norm = olfactory_vector / (np.linalg.norm(olfactory_vector) + 1e-10)
        
        # Concatenate normalized features
        fused_features = np.concatenate([neural_norm, olfactory_norm])
        
        # Equal weighting for early fusion
        fusion_weights = np.array([0.5, 0.5])
        
        return {
            'fused_features': fused_features,
            'fusion_weights': fusion_weights,
            'fusion_method': 'concatenation'
        }
    
    def _late_fusion(
        self,
        neural_vector: np.ndarray,
        olfactory_vector: np.ndarray
    ) -> dict[str, Any]:
        """Late fusion: combine at decision level."""
        
        # Process each modality separately then combine
        neural_decision = np.mean(neural_vector) if len(neural_vector) > 0 else 0.0
        olfactory_decision = np.mean(olfactory_vector) if len(olfactory_vector) > 0 else 0.0
        
        # Weight decisions based on modality confidence
        neural_confidence = 1.0 / (1.0 + np.std(neural_vector)) if len(neural_vector) > 0 else 0.5
        olfactory_confidence = 1.0 / (1.0 + np.std(olfactory_vector)) if len(olfactory_vector) > 0 else 0.5
        
        # Normalize weights
        total_confidence = neural_confidence + olfactory_confidence
        if total_confidence > 0:
            neural_weight = neural_confidence / total_confidence
            olfactory_weight = olfactory_confidence / total_confidence
        else:
            neural_weight = olfactory_weight = 0.5
        
        # Fused decision
        fused_decision = neural_weight * neural_decision + olfactory_weight * olfactory_decision
        
        # Create fused feature vector
        fused_features = np.array([
            fused_decision,
            neural_decision,
            olfactory_decision,
            neural_confidence,
            olfactory_confidence
        ])
        
        fusion_weights = np.array([neural_weight, olfactory_weight])
        
        return {
            'fused_features': fused_features,
            'fusion_weights': fusion_weights,
            'fusion_method': 'decision_level'
        }
    
    def _attention_fusion(
        self,
        neural_vector: np.ndarray,
        olfactory_vector: np.ndarray
    ) -> dict[str, Any]:
        """Attention-based fusion with learned weights."""
        
        # Initialize attention mechanism if not exists
        if self.attention_mechanism is None:
            feature_dim = max(len(neural_vector), len(olfactory_vector))
            self.attention_mechanism = AttentionMechanism(feature_dim)
        
        # Compute attention weights
        attention_weights = self.attention_mechanism.compute_attention_weights(
            neural_vector, olfactory_vector
        )
        
        # Pad vectors to same length for weighted combination
        max_len = max(len(neural_vector), len(olfactory_vector))
        neural_padded = np.pad(neural_vector, (0, max_len - len(neural_vector)), mode='constant')
        olfactory_padded = np.pad(olfactory_vector, (0, max_len - len(olfactory_vector)), mode='constant')
        
        # Weighted fusion
        fused_features = (
            attention_weights[0] * neural_padded + 
            attention_weights[1] * olfactory_padded
        )
        
        # Add attention weights as additional features
        fused_features = np.concatenate([fused_features, attention_weights])
        
        return {
            'fused_features': fused_features,
            'fusion_weights': attention_weights,
            'fusion_method': 'attention_weighted'
        }
    
    def _canonical_correlation_fusion(
        self,
        neural_vector: np.ndarray,
        olfactory_vector: np.ndarray
    ) -> dict[str, Any]:
        """Canonical correlation analysis-based fusion."""
        
        # For single vectors, compute correlation and use as weight
        if len(neural_vector) > 1 and len(olfactory_vector) > 1:
            # Ensure same dimensionality
            min_dim = min(len(neural_vector), len(olfactory_vector))
            neural_truncated = neural_vector[:min_dim]
            olfactory_truncated = olfactory_vector[:min_dim]
            
            # Compute correlation
            correlation = np.corrcoef(neural_truncated, olfactory_truncated)[0, 1]
            correlation = abs(correlation) if not np.isnan(correlation) else 0.0
        else:
            correlation = 0.5
        
        # Use correlation to weight modalities
        neural_weight = 0.5 + 0.5 * correlation
        olfactory_weight = 0.5 + 0.5 * (1.0 - correlation)
        
        # Normalize weights
        total_weight = neural_weight + olfactory_weight
        neural_weight /= total_weight
        olfactory_weight /= total_weight
        
        # Weighted combination
        max_len = max(len(neural_vector), len(olfactory_vector))
        neural_padded = np.pad(neural_vector, (0, max_len - len(neural_vector)), mode='constant')
        olfactory_padded = np.pad(olfactory_vector, (0, max_len - len(olfactory_vector)), mode='constant')
        
        fused_features = neural_weight * neural_padded + olfactory_weight * olfactory_padded
        fusion_weights = np.array([neural_weight, olfactory_weight])
        
        return {
            'fused_features': fused_features,
            'fusion_weights': fusion_weights,
            'fusion_method': 'canonical_correlation',
            'correlation': correlation
        }
    
    def _bayesian_fusion(
        self,
        neural_vector: np.ndarray,
        olfactory_vector: np.ndarray
    ) -> dict[str, Any]:
        """Bayesian inference-based fusion."""
        
        # Simple Bayesian approach using means and variances
        neural_mean = np.mean(neural_vector) if len(neural_vector) > 0 else 0.0
        neural_var = np.var(neural_vector) if len(neural_vector) > 0 else 1.0
        
        olfactory_mean = np.mean(olfactory_vector) if len(olfactory_vector) > 0 else 0.0
        olfactory_var = np.var(olfactory_vector) if len(olfactory_vector) > 0 else 1.0
        
        # Bayesian combination (assuming Gaussian distributions)
        if neural_var > 0 and olfactory_var > 0:
            # Precision (inverse variance)
            neural_precision = 1.0 / neural_var
            olfactory_precision = 1.0 / olfactory_var
            
            # Combined precision and mean
            combined_precision = neural_precision + olfactory_precision
            combined_mean = (neural_precision * neural_mean + olfactory_precision * olfactory_mean) / combined_precision
            combined_var = 1.0 / combined_precision
            
            # Weights based on precision
            neural_weight = neural_precision / combined_precision
            olfactory_weight = olfactory_precision / combined_precision
        else:
            combined_mean = (neural_mean + olfactory_mean) / 2
            combined_var = (neural_var + olfactory_var) / 2
            neural_weight = olfactory_weight = 0.5
        
        # Create fused feature vector
        fused_features = np.array([
            combined_mean,
            combined_var,
            neural_mean,
            olfactory_mean,
            neural_var,
            olfactory_var
        ])
        
        # Add original features with Bayesian weights
        max_len = max(len(neural_vector), len(olfactory_vector))
        neural_padded = np.pad(neural_vector, (0, max_len - len(neural_vector)), mode='constant')
        olfactory_padded = np.pad(olfactory_vector, (0, max_len - len(olfactory_vector)), mode='constant')
        
        weighted_features = neural_weight * neural_padded + olfactory_weight * olfactory_padded
        fused_features = np.concatenate([fused_features, weighted_features])
        
        fusion_weights = np.array([neural_weight, olfactory_weight])
        
        return {
            'fused_features': fused_features,
            'fusion_weights': fusion_weights,
            'fusion_method': 'bayesian_inference',
            'posterior_mean': combined_mean,
            'posterior_variance': combined_var
        }
    
    def _analyze_cross_modal_relationships(
        self,
        neural_features: dict[str, Any],
        olfactory_features: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze relationships between neural and olfactory modalities."""
        
        # Extract key metrics for correlation analysis
        neural_metrics = []
        olfactory_metrics = []
        
        # Neural metrics
        features = neural_features.get('features')
        if features:
            neural_metrics.extend([
                getattr(features, 'mean_amplitude', 0.0),
                getattr(features, 'peak_amplitude', 0.0),
                getattr(features, 'dominant_frequency', 0.0)
            ])
        
        # Olfactory metrics
        molecular_features = olfactory_features.get('molecular_features')
        receptor_activation = olfactory_features.get('receptor_activation', {})
        psychophysical = olfactory_features.get('psychophysical_properties', {})
        
        if molecular_features:
            olfactory_metrics.extend([
                getattr(molecular_features, 'perceived_intensity', 0.0),
                receptor_activation.get('activation_sum', 0.0),
                psychophysical.get('perceived_intensity', 0.0)
            ])
        
        # Compute cross-modal correlation
        correlation = 0.0
        if len(neural_metrics) > 0 and len(olfactory_metrics) > 0:
            # Pad to same length
            max_len = max(len(neural_metrics), len(olfactory_metrics))
            neural_padded = neural_metrics + [0.0] * (max_len - len(neural_metrics))
            olfactory_padded = olfactory_metrics + [0.0] * (max_len - len(olfactory_metrics))
            
            if len(neural_padded) > 1:
                correlation = np.corrcoef(neural_padded, olfactory_padded)[0, 1]
                correlation = correlation if not np.isnan(correlation) else 0.0
        
        # Consistency measure
        consistency = abs(correlation)  # Simple consistency based on correlation strength
        
        # Temporal coupling analysis
        temporal_coupling = self._analyze_temporal_coupling(neural_features, olfactory_features)
        
        return {
            'correlation': correlation,
            'consistency': consistency,
            'temporal_coupling': temporal_coupling,
            'neural_dominance': self._compute_modal_dominance(neural_metrics),
            'olfactory_dominance': self._compute_modal_dominance(olfactory_metrics),
            'cross_modal_coherence': abs(correlation) * consistency
        }
    
    def _analyze_temporal_coupling(
        self,
        neural_features: dict[str, Any],
        olfactory_features: dict[str, Any]
    ) -> float:
        """Analyze temporal coupling between modalities."""
        
        # Get temporal information
        neural_timestamps = neural_features.get('timestamps', np.array([]))
        olfactory_temporal = olfactory_features.get('temporal_profile', {})
        olfactory_timestamps = olfactory_temporal.get('time_points', np.array([]))
        
        if len(neural_timestamps) == 0 or len(olfactory_timestamps) == 0:
            return 0.0
        
        # Compute temporal overlap
        neural_start, neural_end = neural_timestamps[0], neural_timestamps[-1]
        olfactory_start, olfactory_end = olfactory_timestamps[0], olfactory_timestamps[-1]
        
        overlap_start = max(neural_start, olfactory_start)
        overlap_end = min(neural_end, olfactory_end)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_duration = overlap_end - overlap_start
        max_duration = max(neural_end - neural_start, olfactory_end - olfactory_start)
        
        temporal_coupling = overlap_duration / max_duration if max_duration > 0 else 0.0
        
        return temporal_coupling
    
    def _compute_modal_dominance(self, metrics: list[float]) -> float:
        """Compute dominance score for a modality."""
        if not metrics:
            return 0.0
        
        # Normalize metrics and compute dominance
        max_metric = max(metrics) if metrics else 1.0
        normalized_metrics = [m / max_metric for m in metrics] if max_metric > 0 else metrics
        
        return np.mean(normalized_metrics)
    
    def _generate_predictions(
        self,
        fused_features: np.ndarray,
        neural_features: dict[str, Any],
        olfactory_features: dict[str, Any]
    ) -> dict[str, dict[str, float]]:
        """Generate predictions based on fused features."""
        
        # Extract key values from fused features for predictions
        if len(fused_features) > 0:
            fusion_strength = np.mean(fused_features)
            fusion_variability = np.std(fused_features) if len(fused_features) > 1 else 0.0
        else:
            fusion_strength = 0.0
            fusion_variability = 0.0
        
        # Olfactory response prediction
        olfactory_prediction = {
            'predicted_intensity': min(10.0, fusion_strength * 5.0),
            'predicted_pleasantness': max(0.0, min(10.0, 5.0 + fusion_strength)),
            'predicted_familiarity': max(0.0, min(10.0, fusion_strength * 8.0)),
            'predicted_detectability': 1.0 if fusion_strength > 0.1 else 0.0
        }
        
        # Neural activity prediction
        neural_prediction = {
            'predicted_amplitude': fusion_strength * 100.0,  # microvolts
            'predicted_frequency': max(1.0, fusion_strength * 40.0),  # Hz
            'predicted_coherence': min(1.0, fusion_strength),
            'predicted_response_latency': max(50.0, 200.0 - fusion_strength * 100.0)  # ms
        }
        
        # Behavioral prediction
        behavioral_prediction = {
            'recognition_probability': min(1.0, fusion_strength),
            'reaction_time': max(200.0, 1000.0 - fusion_strength * 500.0),  # ms
            'confidence_rating': min(10.0, fusion_strength * 10.0),
            'memory_encoding_strength': fusion_strength * (1.0 - fusion_variability)
        }
        
        return {
            'olfactory_response': olfactory_prediction,
            'neural_activity': neural_prediction,
            'behavioral': behavioral_prediction
        }
    
    def _assess_fusion_quality(
        self,
        neural_vector: np.ndarray,
        olfactory_vector: np.ndarray,
        fusion_results: dict[str, Any]
    ) -> dict[str, float]:
        """Assess quality of the fusion process."""
        
        # Confidence based on input signal quality
        neural_confidence = 1.0 - (np.std(neural_vector) / (np.mean(np.abs(neural_vector)) + 1e-10))
        neural_confidence = max(0.0, min(1.0, neural_confidence))
        
        olfactory_confidence = 1.0 - (np.std(olfactory_vector) / (np.mean(np.abs(olfactory_vector)) + 1e-10))
        olfactory_confidence = max(0.0, min(1.0, olfactory_confidence))
        
        overall_confidence = (neural_confidence + olfactory_confidence) / 2
        
        # Uncertainty based on fusion weights
        fusion_weights = fusion_results['fusion_weights']
        weight_entropy = -np.sum(fusion_weights * np.log2(fusion_weights + 1e-10))
        max_entropy = np.log2(len(fusion_weights))
        uncertainty = weight_entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Stability (inverse of variability in fused features)
        fused_features = fusion_results['fused_features']
        stability = 1.0 / (1.0 + np.std(fused_features)) if len(fused_features) > 1 else 1.0
        
        return {
            'confidence': overall_confidence,
            'uncertainty': uncertainty,
            'stability': stability,
            'neural_confidence': neural_confidence,
            'olfactory_confidence': olfactory_confidence
        }
    
    def _summarize_neural_features(self, neural_features: dict[str, Any]) -> dict[str, Any]:
        """Create summary of neural features."""
        features = neural_features.get('features')
        signal_quality = neural_features.get('signal_quality', {})
        
        return {
            'signal_type': neural_features.get('metadata', {}).get('signal_type', 'unknown'),
            'mean_amplitude': getattr(features, 'mean_amplitude', 0.0) if features else 0.0,
            'dominant_frequency': getattr(features, 'dominant_frequency', 0.0) if features else 0.0,
            'signal_quality': signal_quality.get('overall_quality', 0.0),
            'processing_success': features is not None
        }
    
    def _summarize_olfactory_features(self, olfactory_features: dict[str, Any]) -> dict[str, Any]:
        """Create summary of olfactory features."""
        molecular_features = olfactory_features.get('molecular_features')
        receptor_activation = olfactory_features.get('receptor_activation', {})
        
        return {
            'molecular_weight': getattr(molecular_features, 'molecular_weight', 0.0) if molecular_features else 0.0,
            'perceived_intensity': getattr(molecular_features, 'perceived_intensity', 0.0) if molecular_features else 0.0,
            'receptor_activation_sum': receptor_activation.get('activation_sum', 0.0),
            'odor_character': getattr(molecular_features, 'odor_character', []) if molecular_features else [],
            'analysis_success': molecular_features is not None
        }
    
    def _analyze_temporal_dynamics(
        self,
        neural_features: dict[str, Any],
        olfactory_features: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze temporal dynamics of the fusion process."""
        
        temporal_profile = olfactory_features.get('temporal_profile', {})
        neural_timestamps = neural_features.get('timestamps', np.array([]))
        
        return {
            'olfactory_onset_time': temporal_profile.get('onset_time', 0.0),
            'olfactory_peak_time': temporal_profile.get('peak_time', 0.0),
            'neural_duration': neural_timestamps[-1] - neural_timestamps[0] if len(neural_timestamps) > 1 else 0.0,
            'temporal_resolution': np.mean(np.diff(neural_timestamps)) if len(neural_timestamps) > 1 else 0.0,
            'synchronization_quality': self._analyze_temporal_coupling(neural_features, olfactory_features)
        }

    def analyze_fusion_performance(self) -> dict[str, Any]:
        """Analyze performance across fusion history."""
        
        if not self.fusion_history:
            return {'message': 'No fusion history available'}
        
        # Extract quality metrics from history
        confidence_scores = [result['fusion_results'].fusion_confidence for result in self.fusion_history]
        correlation_scores = [result['fusion_results'].neural_olfactory_correlation for result in self.fusion_history]
        consistency_scores = [result['fusion_results'].cross_modal_consistency for result in self.fusion_history]
        
        return {
            'n_fusions': len(self.fusion_history),
            'average_confidence': np.mean(confidence_scores),
            'confidence_stability': np.std(confidence_scores),
            'average_correlation': np.mean(correlation_scores),
            'average_consistency': np.mean(consistency_scores),
            'performance_trend': 'improving' if len(confidence_scores) > 1 and confidence_scores[-1] > confidence_scores[0] else 'stable',
            'fusion_strategy': self.fusion_strategy.value
        }