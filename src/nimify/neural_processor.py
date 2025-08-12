"""Neural signal processing module for bioneuro-olfactory fusion."""

import numpy as np
from scipy import signal, fft
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from .core import NeuralConfig, NeuralSignalType

logger = logging.getLogger(__name__)


@dataclass
class NeuralFeatures:
    """Container for extracted neural features."""
    
    # Time domain features
    mean_amplitude: float
    std_amplitude: float
    peak_amplitude: float
    
    # Frequency domain features
    dominant_frequency: float
    spectral_power: Dict[str, float]  # alpha, beta, gamma, etc.
    spectral_entropy: float
    
    # Time-frequency features
    phase_locking_value: float
    coherence_map: np.ndarray
    
    # Spatial features (for multi-channel data)
    spatial_patterns: Optional[np.ndarray] = None
    channel_connectivity: Optional[np.ndarray] = None


class NeuralSignalProcessor:
    """Processes neural signals for olfactory response analysis."""
    
    def __init__(self, config: NeuralConfig):
        self.config = config
        self.filters = self._setup_filters()
    
    def _setup_filters(self) -> Dict[str, Any]:
        """Setup preprocessing filters based on signal type."""
        filters = {}
        
        # Standard EEG frequency bands
        if self.config.signal_type == NeuralSignalType.EEG:
            filters.update({
                'delta': (0.5, 4),
                'theta': (4, 8), 
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
            })
            filters['notch_freq'] = 50  # Line noise (50Hz EU, 60Hz US)
            filters['bandpass'] = (0.1, 100)
            
        elif self.config.signal_type == NeuralSignalType.fMRI:
            filters.update({
                'low_freq': (0.01, 0.08),
                'bandpass': (0.01, 0.1)  # BOLD signal range
            })
            
        elif self.config.signal_type == NeuralSignalType.EPHYS:
            filters.update({
                'spike_band': (300, 6000),
                'lfp_band': (1, 300),
                'bandpass': (1, 6000)
            })
        
        return filters
    
    def process(
        self, 
        neural_data: np.ndarray, 
        timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Main processing pipeline for neural signals.
        
        Args:
            neural_data: Neural signal data [channels x time] or [time] for single channel
            timestamps: Time stamps corresponding to data points
            
        Returns:
            Dictionary containing processed features and metadata
        """
        logger.info(f"Processing {self.config.signal_type.value} data: shape {neural_data.shape}")
        
        # Ensure 2D array [channels x time]
        if neural_data.ndim == 1:
            neural_data = neural_data.reshape(1, -1)
        
        # Generate timestamps if not provided
        if timestamps is None:
            timestamps = np.arange(neural_data.shape[1]) / self.config.sampling_rate
        
        # Preprocessing pipeline
        preprocessed_data = self._preprocess(neural_data)
        
        # Feature extraction
        features = self._extract_features(preprocessed_data, timestamps)
        
        # Artifact detection and removal
        if self.config.artifact_removal:
            clean_data, artifact_mask = self._remove_artifacts(preprocessed_data)
        else:
            clean_data, artifact_mask = preprocessed_data, None
        
        return {
            'features': features,
            'preprocessed_data': clean_data,
            'timestamps': timestamps,
            'artifact_mask': artifact_mask,
            'signal_quality': self._assess_signal_quality(clean_data),
            'metadata': {
                'signal_type': self.config.signal_type.value,
                'sampling_rate': self.config.sampling_rate,
                'channels': self.config.channels,
                'processing_timestamp': np.datetime64('now')
            }
        }
    
    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Apply preprocessing filters to neural data."""
        processed_data = data.copy()
        
        for filter_name in self.config.preprocessing_filters:
            if filter_name == "bandpass" and "bandpass" in self.filters:
                processed_data = self._apply_bandpass_filter(
                    processed_data, 
                    self.filters["bandpass"]
                )
            elif filter_name == "notch" and "notch_freq" in self.filters:
                processed_data = self._apply_notch_filter(
                    processed_data,
                    self.filters["notch_freq"]
                )
            elif filter_name == "baseline":
                processed_data = self._baseline_correction(processed_data)
        
        return processed_data
    
    def _apply_bandpass_filter(self, data: np.ndarray, freq_range: Tuple[float, float]) -> np.ndarray:
        """Apply bandpass filter to data."""
        nyquist = self.config.sampling_rate / 2
        low, high = freq_range[0] / nyquist, freq_range[1] / nyquist
        
        # Ensure frequencies are within valid range
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.001, min(high, 0.99))
        
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, data, axis=-1)
    
    def _apply_notch_filter(self, data: np.ndarray, notch_freq: float) -> np.ndarray:
        """Apply notch filter to remove line noise."""
        nyquist = self.config.sampling_rate / 2
        freq_norm = notch_freq / nyquist
        
        if freq_norm >= 1.0:
            logger.warning(f"Notch frequency {notch_freq} exceeds Nyquist frequency")
            return data
        
        b, a = signal.iirnotch(freq_norm, Q=30)
        return signal.filtfilt(b, a, data, axis=-1)
    
    def _baseline_correction(self, data: np.ndarray) -> np.ndarray:
        """Remove baseline drift by detrending."""
        return signal.detrend(data, axis=-1)
    
    def _extract_features(self, data: np.ndarray, timestamps: np.ndarray) -> NeuralFeatures:
        """Extract comprehensive neural features."""
        # Time domain features
        mean_amp = np.mean(np.abs(data), axis=-1).mean()
        std_amp = np.std(data, axis=-1).mean()
        peak_amp = np.max(np.abs(data), axis=-1).mean()
        
        # Frequency domain analysis
        freqs, psd = signal.welch(
            data, 
            fs=self.config.sampling_rate, 
            nperseg=min(data.shape[-1]//4, 1024)
        )
        
        # Extract power in different frequency bands
        spectral_power = {}
        for band_name, (low_freq, high_freq) in self.filters.items():
            if isinstance(low_freq, (int, float)) and isinstance(high_freq, (int, float)):
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                if np.any(band_mask):
                    spectral_power[band_name] = np.mean(psd[:, band_mask])
        
        # Dominant frequency
        dominant_freq = freqs[np.argmax(np.mean(psd, axis=0))]
        
        # Spectral entropy
        psd_norm = psd / np.sum(psd, axis=-1, keepdims=True)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=-1).mean()
        
        # Phase-locking value (simplified)
        phase_locking_value = self._compute_phase_locking_value(data)
        
        # Coherence map (for multi-channel data)
        coherence_map = self._compute_coherence_map(data)
        
        # Spatial patterns (if multi-channel)
        spatial_patterns = None
        channel_connectivity = None
        if data.shape[0] > 1:
            spatial_patterns = self._extract_spatial_patterns(data)
            channel_connectivity = self._compute_channel_connectivity(data)
        
        return NeuralFeatures(
            mean_amplitude=mean_amp,
            std_amplitude=std_amp,
            peak_amplitude=peak_amp,
            dominant_frequency=dominant_freq,
            spectral_power=spectral_power,
            spectral_entropy=spectral_entropy,
            phase_locking_value=phase_locking_value,
            coherence_map=coherence_map,
            spatial_patterns=spatial_patterns,
            channel_connectivity=channel_connectivity
        )
    
    def _compute_phase_locking_value(self, data: np.ndarray) -> float:
        """Compute phase locking value across channels."""
        if data.shape[0] < 2:
            return 0.0
        
        # Apply Hilbert transform to get instantaneous phase
        analytic_signal = signal.hilbert(data, axis=-1)
        phases = np.angle(analytic_signal)
        
        # Compute phase differences between channel pairs
        phase_diffs = []
        for i in range(data.shape[0]):
            for j in range(i+1, data.shape[0]):
                phase_diff = phases[i] - phases[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                phase_diffs.append(plv)
        
        return np.mean(phase_diffs) if phase_diffs else 0.0
    
    def _compute_coherence_map(self, data: np.ndarray) -> np.ndarray:
        """Compute coherence between all channel pairs."""
        n_channels = data.shape[0]
        coherence_map = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    freqs, coherence = signal.coherence(
                        data[i], data[j], 
                        fs=self.config.sampling_rate,
                        nperseg=min(data.shape[-1]//4, 512)
                    )
                    # Average coherence across frequencies
                    coherence_map[i, j] = np.mean(coherence)
                else:
                    coherence_map[i, j] = 1.0
        
        return coherence_map
    
    def _extract_spatial_patterns(self, data: np.ndarray) -> np.ndarray:
        """Extract spatial patterns using PCA or ICA-like approach."""
        # Simple PCA-based spatial pattern extraction
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=min(5, data.shape[0]))
        spatial_patterns = pca.fit_transform(data.T).T
        
        return spatial_patterns
    
    def _compute_channel_connectivity(self, data: np.ndarray) -> np.ndarray:
        """Compute functional connectivity between channels."""
        # Simplified connectivity using correlation
        correlation_matrix = np.corrcoef(data)
        return correlation_matrix
    
    def _remove_artifacts(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect and remove artifacts from neural data."""
        # Simple artifact detection based on amplitude thresholds
        amplitude_threshold = 5 * np.std(data)  # 5 sigma threshold
        artifact_mask = np.abs(data) > amplitude_threshold
        
        # Replace artifacts with interpolated values
        clean_data = data.copy()
        for ch_idx in range(data.shape[0]):
            if np.any(artifact_mask[ch_idx]):
                # Simple linear interpolation
                artifact_indices = np.where(artifact_mask[ch_idx])[0]
                for idx in artifact_indices:
                    # Find nearest non-artifact values for interpolation
                    left_idx = max(0, idx - 10)
                    right_idx = min(data.shape[1], idx + 10)
                    
                    # Simple average interpolation
                    clean_data[ch_idx, idx] = np.mean([
                        clean_data[ch_idx, left_idx:idx].mean(),
                        clean_data[ch_idx, idx+1:right_idx].mean()
                    ])
        
        return clean_data, artifact_mask
    
    def _assess_signal_quality(self, data: np.ndarray) -> Dict[str, float]:
        """Assess overall signal quality metrics."""
        signal_to_noise_ratio = np.mean(data**2) / np.var(data)
        amplitude_variability = np.std(data) / np.mean(np.abs(data))
        
        # Frequency domain quality
        freqs, psd = signal.welch(data, fs=self.config.sampling_rate)
        spectral_flatness = signal.spectral.spectral_flatness(psd)
        
        return {
            'signal_to_noise_ratio': signal_to_noise_ratio,
            'amplitude_variability': amplitude_variability,
            'spectral_flatness': np.mean(spectral_flatness),
            'overall_quality': min(signal_to_noise_ratio / 10, 1.0)  # Normalized quality score
        }

    def analyze_olfactory_response_patterns(
        self,
        neural_data: np.ndarray,
        stimulus_onset_times: np.ndarray,
        pre_stimulus_window: float = 1.0,
        post_stimulus_window: float = 5.0
    ) -> Dict[str, Any]:
        """
        Analyze neural response patterns to olfactory stimuli.
        
        Args:
            neural_data: Preprocessed neural signal data
            stimulus_onset_times: Array of stimulus onset times (in seconds)
            pre_stimulus_window: Time before stimulus for baseline (seconds)
            post_stimulus_window: Time after stimulus to analyze (seconds)
            
        Returns:
            Dictionary containing response analysis results
        """
        response_patterns = []
        
        for onset_time in stimulus_onset_times:
            # Convert time to sample indices
            onset_sample = int(onset_time * self.config.sampling_rate)
            pre_samples = int(pre_stimulus_window * self.config.sampling_rate)
            post_samples = int(post_stimulus_window * self.config.sampling_rate)
            
            # Extract epoch around stimulus
            start_idx = max(0, onset_sample - pre_samples)
            end_idx = min(neural_data.shape[-1], onset_sample + post_samples)
            
            if end_idx - start_idx < post_samples:
                continue  # Skip incomplete epochs
            
            epoch_data = neural_data[:, start_idx:end_idx]
            
            # Baseline correction
            baseline = np.mean(epoch_data[:, :pre_samples], axis=-1, keepdims=True)
            epoch_data = epoch_data - baseline
            
            # Extract response features
            response_features = self._extract_response_features(
                epoch_data, 
                pre_samples,
                post_samples
            )
            response_patterns.append(response_features)
        
        # Aggregate response patterns
        if response_patterns:
            aggregated_response = self._aggregate_response_patterns(response_patterns)
        else:
            aggregated_response = {}
        
        return {
            'individual_responses': response_patterns,
            'aggregated_response': aggregated_response,
            'n_trials': len(response_patterns),
            'response_consistency': self._compute_response_consistency(response_patterns)
        }
    
    def _extract_response_features(
        self, 
        epoch_data: np.ndarray, 
        pre_samples: int, 
        post_samples: int
    ) -> Dict[str, float]:
        """Extract features from single stimulus response epoch."""
        
        # Time-to-peak response
        response_data = epoch_data[:, pre_samples:]  # Post-stimulus data only
        peak_time = np.argmax(np.mean(np.abs(response_data), axis=0)) / self.config.sampling_rate
        
        # Peak amplitude
        peak_amplitude = np.max(np.abs(response_data), axis=-1).mean()
        
        # Response duration (time above threshold)
        threshold = 0.1 * peak_amplitude
        above_threshold = np.mean(np.abs(response_data), axis=0) > threshold
        response_duration = np.sum(above_threshold) / self.config.sampling_rate
        
        # Area under curve
        response_auc = np.trapz(np.mean(np.abs(response_data), axis=0)) / self.config.sampling_rate
        
        # Onset latency
        onset_threshold = 0.05 * peak_amplitude
        onset_samples = np.where(np.mean(np.abs(response_data), axis=0) > onset_threshold)[0]
        onset_latency = onset_samples[0] / self.config.sampling_rate if len(onset_samples) > 0 else np.nan
        
        return {
            'peak_time': peak_time,
            'peak_amplitude': peak_amplitude,
            'response_duration': response_duration,
            'response_auc': response_auc,
            'onset_latency': onset_latency
        }
    
    def _aggregate_response_patterns(self, response_patterns: List[Dict]) -> Dict[str, Any]:
        """Aggregate individual response patterns across trials."""
        aggregated = {}
        
        for key in response_patterns[0].keys():
            values = [r[key] for r in response_patterns if not np.isnan(r[key])]
            if values:
                aggregated[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return aggregated
    
    def _compute_response_consistency(self, response_patterns: List[Dict]) -> float:
        """Compute consistency of responses across trials."""
        if len(response_patterns) < 2:
            return 1.0
        
        # Use coefficient of variation across peak amplitudes as consistency measure
        peak_amplitudes = [r['peak_amplitude'] for r in response_patterns]
        mean_peak = np.mean(peak_amplitudes)
        cv = np.std(peak_amplitudes) / mean_peak if mean_peak > 0 else 0
        
        # Return consistency as 1 - CV (higher is more consistent)
        return max(0, 1 - cv)