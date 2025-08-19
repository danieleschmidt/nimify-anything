"""Research-aware monitoring system with quantum-inspired analytics and adaptive thresholds."""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ResearchMetric:
    """Research-specific metric with statistical properties."""
    name: str
    value: float
    timestamp: float
    statistical_significance: Optional[float]
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    research_context: Dict[str, Any]


@dataclass
class QuantumInsight:
    """Quantum-inspired system insight."""
    coherence_measure: float  # System stability (0-1)
    entanglement_score: float  # Cross-component correlation (0-1)
    tunneling_probability: float  # Likelihood of state transition (0-1)
    uncertainty_principle: float  # Measurement precision trade-off
    decoherence_rate: float  # System degradation rate


@dataclass
class AdaptiveThreshold:
    """Self-adjusting threshold based on system behavior."""
    base_value: float
    current_value: float
    adaptation_rate: float
    learning_window: int
    confidence_level: float
    last_updated: float


class QuantumInspiredAnalyzer:
    """Quantum-inspired system analysis for monitoring."""
    
    def __init__(self, observation_window: int = 1000):
        self.observation_window = observation_window
        self.measurement_history = deque(maxlen=observation_window)
        self.quantum_state_cache = {}
        
    def analyze_system_quantum_state(
        self, 
        metrics: Dict[str, List[float]]
    ) -> QuantumInsight:
        """Analyze system using quantum-inspired principles."""
        
        if not metrics:
            return QuantumInsight(
                coherence_measure=0.5,
                entanglement_score=0.0,
                tunneling_probability=0.1,
                uncertainty_principle=1.0,
                decoherence_rate=0.1
            )
        
        # Coherence: Stability of key metrics
        coherence_scores = []
        for metric_name, values in metrics.items():
            if len(values) > 5:
                # High coherence = low variance relative to mean
                mean_val = np.mean(values)
                if mean_val != 0:
                    stability = 1.0 / (1.0 + np.var(values) / abs(mean_val))
                    coherence_scores.append(min(1.0, stability))
        
        coherence_measure = np.mean(coherence_scores) if coherence_scores else 0.5
        
        # Entanglement: Cross-correlation between metrics
        entanglement_score = self._compute_cross_correlation(metrics)
        
        # Tunneling: Probability of sudden state changes
        tunneling_probability = self._compute_tunneling_probability(metrics)
        
        # Uncertainty: Trade-off in measurement precision
        uncertainty_principle = self._compute_measurement_uncertainty(metrics)
        
        # Decoherence: Rate of system degradation
        decoherence_rate = self._compute_decoherence_rate(metrics)
        
        return QuantumInsight(
            coherence_measure=coherence_measure,
            entanglement_score=entanglement_score,
            tunneling_probability=tunneling_probability,
            uncertainty_principle=uncertainty_principle,
            decoherence_rate=decoherence_rate
        )
    
    def _compute_cross_correlation(self, metrics: Dict[str, List[float]]) -> float:
        """Compute cross-correlation between metrics (entanglement)."""
        metric_pairs = list(metrics.items())
        if len(metric_pairs) < 2:
            return 0.0
        
        correlations = []
        for i in range(len(metric_pairs)):
            for j in range(i + 1, len(metric_pairs)):
                name1, values1 = metric_pairs[i]
                name2, values2 = metric_pairs[j]
                
                # Ensure equal lengths
                min_len = min(len(values1), len(values2))
                if min_len > 5:
                    val1 = values1[-min_len:]
                    val2 = values2[-min_len:]
                    
                    # Compute correlation
                    corr = abs(np.corrcoef(val1, val2)[0, 1])
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _compute_tunneling_probability(self, metrics: Dict[str, List[float]]) -> float:
        """Compute probability of sudden state transitions."""
        jump_probabilities = []
        
        for values in metrics.values():
            if len(values) > 10:
                # Compute rate of large changes
                diffs = np.diff(values)
                std_diff = np.std(diffs)
                if std_diff > 0:
                    # Count "quantum jumps" (changes > 2 std deviations)
                    jumps = np.sum(np.abs(diffs) > 2 * std_diff)
                    jump_prob = jumps / len(diffs)
                    jump_probabilities.append(min(1.0, jump_prob))
        
        return np.mean(jump_probabilities) if jump_probabilities else 0.1
    
    def _compute_measurement_uncertainty(self, metrics: Dict[str, List[float]]) -> float:
        """Compute measurement uncertainty principle."""
        uncertainties = []
        
        for values in metrics.values():
            if len(values) > 5:
                # Uncertainty as relative standard deviation
                mean_val = np.mean(values)
                if mean_val != 0:
                    uncertainty = np.std(values) / abs(mean_val)
                    uncertainties.append(min(2.0, uncertainty))  # Cap at 2.0
        
        return np.mean(uncertainties) if uncertainties else 1.0
    
    def _compute_decoherence_rate(self, metrics: Dict[str, List[float]]) -> float:
        """Compute system decoherence rate."""
        decoherence_rates = []
        
        for values in metrics.values():
            if len(values) > 20:
                # Split into windows and measure stability degradation
                window_size = len(values) // 4
                windows = [
                    values[i:i+window_size] 
                    for i in range(0, len(values), window_size)
                    if len(values[i:i+window_size]) == window_size
                ]
                
                if len(windows) >= 3:
                    # Measure increasing variance over time
                    variances = [np.var(window) for window in windows]
                    if len(variances) > 1:
                        # Linear fit to variance over time
                        time_points = range(len(variances))
                        slope, _ = np.polyfit(time_points, variances, 1)
                        decoherence_rate = max(0.0, slope)  # Only positive slopes
                        decoherence_rates.append(min(1.0, decoherence_rate))
        
        return np.mean(decoherence_rates) if decoherence_rates else 0.1


class AdaptiveThresholdManager:
    """Manages self-adapting thresholds based on system behavior."""
    
    def __init__(self):
        self.thresholds: Dict[str, AdaptiveThreshold] = {}
        self.learning_history: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
        self._lock = threading.RLock()
    
    def get_threshold(self, metric_name: str, base_value: float) -> float:
        """Get adaptive threshold for a metric."""
        with self._lock:
            if metric_name not in self.thresholds:
                self.thresholds[metric_name] = AdaptiveThreshold(
                    base_value=base_value,
                    current_value=base_value,
                    adaptation_rate=0.1,
                    learning_window=100,
                    confidence_level=0.95,
                    last_updated=time.time()
                )
            
            return self.thresholds[metric_name].current_value
    
    def update_threshold(
        self, 
        metric_name: str, 
        metric_value: float, 
        alert_triggered: bool,
        system_quantum_state: QuantumInsight
    ):
        """Update threshold based on recent system behavior."""
        with self._lock:
            if metric_name not in self.thresholds:
                return
            
            threshold = self.thresholds[metric_name]
            
            # Record learning data
            self.learning_history[metric_name].append((metric_value, alert_triggered))
            
            # Keep only recent history
            if len(self.learning_history[metric_name]) > threshold.learning_window:
                self.learning_history[metric_name] = (
                    self.learning_history[metric_name][-threshold.learning_window:]
                )
            
            # Adapt threshold using quantum-inspired algorithm
            if len(self.learning_history[metric_name]) >= 10:
                self._adapt_threshold_quantum(metric_name, system_quantum_state)
    
    def _adapt_threshold_quantum(
        self, 
        metric_name: str, 
        quantum_state: QuantumInsight
    ):
        """Adapt threshold using quantum-inspired principles."""
        threshold = self.thresholds[metric_name]
        history = self.learning_history[metric_name]
        
        # Extract values and alerts
        values = [entry[0] for entry in history]
        alerts = [entry[1] for entry in history]
        
        # Compute alert statistics
        alert_rate = sum(alerts) / len(alerts)
        target_alert_rate = 0.05  # Target 5% alert rate
        
        # Quantum-inspired adaptation
        coherence_factor = quantum_state.coherence_measure
        tunneling_factor = quantum_state.tunneling_probability
        decoherence_factor = quantum_state.decoherence_rate
        
        # Base adaptation based on alert rate difference
        rate_error = alert_rate - target_alert_rate
        base_adjustment = -rate_error * threshold.adaptation_rate
        
        # Quantum corrections
        # High coherence allows more aggressive adaptation
        coherence_boost = coherence_factor * 1.5
        
        # High tunneling probability suggests volatile system, be conservative
        tunneling_damping = (1.0 - tunneling_factor * 0.5)
        
        # Decoherence suggests degrading system, lower thresholds
        decoherence_adjustment = -decoherence_factor * 0.3
        
        # Combined quantum-inspired adjustment
        quantum_adjustment = (
            base_adjustment * coherence_boost * tunneling_damping + 
            decoherence_adjustment * threshold.base_value
        )
        
        # Apply adaptation with bounds
        new_value = threshold.current_value + quantum_adjustment
        
        # Constrain to reasonable bounds (50% to 200% of base value)
        min_threshold = threshold.base_value * 0.5
        max_threshold = threshold.base_value * 2.0
        
        threshold.current_value = np.clip(new_value, min_threshold, max_threshold)
        threshold.last_updated = time.time()
        
        logger.debug(
            f"Adapted threshold for {metric_name}: "
            f"{threshold.current_value:.3f} (was {new_value-quantum_adjustment:.3f}), "
            f"alert_rate={alert_rate:.3f}, coherence={coherence_factor:.3f}"
        )


class ResearchAwareMonitor:
    """Enhanced monitoring system with research insights and quantum analytics."""
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        
        # Core monitoring
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        # Research enhancements
        self.research_metrics: Dict[str, List[ResearchMetric]] = defaultdict(list)
        self.quantum_analyzer = QuantumInspiredAnalyzer()
        self.threshold_manager = AdaptiveThresholdManager()
        
        # State tracking
        self.current_quantum_state: Optional[QuantumInsight] = None
        self.performance_baseline: Dict[str, float] = {}
        self.research_alerts: List[Dict[str, Any]] = []
        
        self._lock = threading.RLock()
        self._analysis_thread = None
        self._stop_analysis = False
        
        # Start background analysis
        self._start_background_analysis()
    
    def record_research_metric(
        self,
        name: str,
        value: float,
        research_context: Dict[str, Any] = None,
        statistical_test: str = None
    ):
        """Record a research-specific metric with statistical properties."""
        with self._lock:
            # Compute statistical significance if context provided
            significance = None
            effect_size = None
            confidence_interval = None
            
            if research_context and statistical_test:
                significance, effect_size, confidence_interval = (
                    self._compute_research_statistics(value, research_context, statistical_test)
                )
            
            research_metric = ResearchMetric(
                name=name,
                value=value,
                timestamp=time.time(),
                statistical_significance=significance,
                effect_size=effect_size,
                confidence_interval=confidence_interval,
                research_context=research_context or {}
            )
            
            self.research_metrics[name].append(research_metric)
            
            # Also record as regular metric
            self.record_gauge(name, value)
    
    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a gauge metric."""
        with self._lock:
            self.gauges[name] = value
            self.metrics[name].append({
                'value': value,
                'timestamp': time.time(),
                'tags': tags or {}
            })
    
    def record_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Record a counter metric."""
        with self._lock:
            self.counters[name] += value
            self.metrics[name].append({
                'value': self.counters[name],
                'timestamp': time.time(),
                'tags': tags or {}
            })
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram metric."""
        with self._lock:
            self.histograms[name].append(value)
            # Keep reasonable history
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
            
            self.metrics[name].append({
                'value': value,
                'timestamp': time.time(),
                'tags': tags or {}
            })
    
    def get_quantum_insights(self) -> QuantumInsight:
        """Get current quantum-inspired system insights."""
        with self._lock:
            if self.current_quantum_state is None:
                # Compute initial state
                metric_values = {}
                for name, points in self.metrics.items():
                    if points:
                        recent_values = [p['value'] for p in list(points)[-100:]]
                        metric_values[name] = recent_values
                
                self.current_quantum_state = self.quantum_analyzer.analyze_system_quantum_state(
                    metric_values
                )
            
            return self.current_quantum_state
    
    def get_adaptive_threshold(self, metric_name: str, base_threshold: float) -> float:
        """Get adaptive threshold for a metric."""
        return self.threshold_manager.get_threshold(metric_name, base_threshold)
    
    def check_research_anomaly(self, metric_name: str, value: float) -> Dict[str, Any]:
        """Check for research-relevant anomalies."""
        with self._lock:
            quantum_state = self.get_quantum_insights()
            
            # Get adaptive threshold
            base_threshold = self.performance_baseline.get(metric_name, value * 1.5)
            adaptive_threshold = self.get_adaptive_threshold(metric_name, base_threshold)
            
            # Check if value exceeds adaptive threshold
            anomaly_detected = value > adaptive_threshold
            
            # Update threshold learning
            self.threshold_manager.update_threshold(
                metric_name, value, anomaly_detected, quantum_state
            )
            
            # Quantum-informed anomaly scoring
            anomaly_score = self._compute_quantum_anomaly_score(
                metric_name, value, quantum_state
            )
            
            result = {
                'anomaly_detected': anomaly_detected,
                'anomaly_score': anomaly_score,
                'adaptive_threshold': adaptive_threshold,
                'quantum_coherence': quantum_state.coherence_measure,
                'system_stability': 1.0 - quantum_state.decoherence_rate,
                'confidence': 1.0 - quantum_state.uncertainty_principle / 2.0
            }
            
            # Log research alert if significant
            if anomaly_detected and anomaly_score > 0.7:
                self.research_alerts.append({
                    'timestamp': time.time(),
                    'metric_name': metric_name,
                    'value': value,
                    'threshold': adaptive_threshold,
                    'anomaly_score': anomaly_score,
                    'quantum_state': quantum_state
                })
            
            return result
    
    def _compute_quantum_anomaly_score(
        self, 
        metric_name: str, 
        value: float, 
        quantum_state: QuantumInsight
    ) -> float:
        """Compute quantum-informed anomaly score."""
        
        if metric_name not in self.metrics:
            return 0.5
        
        # Get recent history
        recent_points = list(self.metrics[metric_name])[-100:]
        if len(recent_points) < 10:
            return 0.5
        
        recent_values = [p['value'] for p in recent_points]
        
        # Statistical anomaly score
        mean_val = np.mean(recent_values)
        std_val = np.std(recent_values)
        
        if std_val > 0:
            z_score = abs(value - mean_val) / std_val
            statistical_score = min(1.0, z_score / 3.0)  # Normalize by 3-sigma
        else:
            statistical_score = 0.0
        
        # Quantum adjustments
        coherence_factor = quantum_state.coherence_measure
        tunneling_factor = quantum_state.tunneling_probability
        decoherence_factor = quantum_state.decoherence_rate
        
        # Low coherence amplifies anomaly concern
        coherence_adjustment = (2.0 - coherence_factor) / 2.0
        
        # High tunneling suggests system volatility (reduce false positives)
        tunneling_adjustment = 1.0 - tunneling_factor * 0.3
        
        # High decoherence suggests degrading system (increase sensitivity)
        decoherence_adjustment = 1.0 + decoherence_factor * 0.5
        
        # Combined quantum-informed score
        quantum_score = (
            statistical_score * 
            coherence_adjustment * 
            tunneling_adjustment * 
            decoherence_adjustment
        )
        
        return min(1.0, quantum_score)
    
    def _compute_research_statistics(
        self, 
        value: float, 
        context: Dict[str, Any], 
        test_type: str
    ) -> Tuple[Optional[float], Optional[float], Optional[Tuple[float, float]]]:
        """Compute statistical significance, effect size, and confidence intervals."""
        
        try:
            if test_type == "t_test" and "baseline_values" in context:
                baseline = context["baseline_values"]
                if len(baseline) > 1:
                    # One-sample t-test
                    t_stat, p_value = stats.ttest_1samp(baseline, value)
                    
                    # Effect size (Cohen's d)
                    baseline_mean = np.mean(baseline)
                    baseline_std = np.std(baseline, ddof=1)
                    if baseline_std > 0:
                        effect_size = (value - baseline_mean) / baseline_std
                    else:
                        effect_size = 0.0
                    
                    # Confidence interval (approximate)
                    se = baseline_std / np.sqrt(len(baseline))
                    margin = 1.96 * se  # 95% CI
                    ci = (value - margin, value + margin)
                    
                    return abs(p_value), abs(effect_size), ci
            
            # Add more statistical tests as needed
            return None, None, None
            
        except Exception as e:
            logger.warning(f"Failed to compute research statistics: {e}")
            return None, None, None
    
    def _start_background_analysis(self):
        """Start background analysis thread."""
        def analysis_loop():
            while not self._stop_analysis:
                try:
                    with self._lock:
                        # Update quantum state
                        metric_values = {}
                        for name, points in self.metrics.items():
                            if points:
                                recent_values = [p['value'] for p in list(points)[-100:]]
                                metric_values[name] = recent_values
                        
                        if metric_values:
                            self.current_quantum_state = (
                                self.quantum_analyzer.analyze_system_quantum_state(metric_values)
                            )
                    
                    # Sleep for analysis interval
                    time.sleep(30)  # Analyze every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Background analysis error: {e}")
                    time.sleep(5)
        
        self._analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
        self._analysis_thread.start()
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get comprehensive research monitoring summary."""
        with self._lock:
            quantum_state = self.get_quantum_insights()
            
            # Compile research metrics statistics
            research_stats = {}
            for metric_name, metrics_list in self.research_metrics.items():
                if metrics_list:
                    values = [m.value for m in metrics_list]
                    significances = [m.statistical_significance for m in metrics_list if m.statistical_significance]
                    effect_sizes = [m.effect_size for m in metrics_list if m.effect_size]
                    
                    research_stats[metric_name] = {
                        'count': len(metrics_list),
                        'mean_value': np.mean(values),
                        'std_value': np.std(values),
                        'mean_significance': np.mean(significances) if significances else None,
                        'mean_effect_size': np.mean(effect_sizes) if effect_sizes else None,
                        'latest_value': values[-1],
                        'latest_timestamp': metrics_list[-1].timestamp
                    }
            
            return {
                'quantum_insights': {
                    'coherence': quantum_state.coherence_measure,
                    'entanglement': quantum_state.entanglement_score,
                    'tunneling_probability': quantum_state.tunneling_probability,
                    'uncertainty': quantum_state.uncertainty_principle,
                    'decoherence_rate': quantum_state.decoherence_rate
                },
                'research_metrics': research_stats,
                'adaptive_thresholds': {
                    name: threshold.current_value 
                    for name, threshold in self.threshold_manager.thresholds.items()
                },
                'recent_alerts': self.research_alerts[-10:],  # Last 10 alerts
                'system_health_score': (
                    quantum_state.coherence_measure * 0.4 +
                    (1.0 - quantum_state.decoherence_rate) * 0.3 +
                    (1.0 - quantum_state.uncertainty_principle / 2.0) * 0.3
                )
            }
    
    def shutdown(self):
        """Shutdown the monitor."""
        self._stop_analysis = True
        if self._analysis_thread:
            self._analysis_thread.join(timeout=5)


# Global research-aware monitor instance
global_research_monitor = ResearchAwareMonitor()


def get_research_monitor() -> ResearchAwareMonitor:
    """Get the global research-aware monitor instance."""
    return global_research_monitor


if __name__ == "__main__":
    # Demonstration
    print("🔬 Research-Aware Monitoring System")
    print("=" * 50)
    
    monitor = ResearchAwareMonitor()
    
    # Simulate some metrics
    import random
    
    for i in range(100):
        # Simulate various metrics with some research context
        accuracy = 0.85 + random.gauss(0, 0.05)
        latency = 50 + random.gauss(0, 10)
        
        monitor.record_research_metric(
            "model_accuracy", 
            accuracy,
            research_context={"baseline_values": [0.75, 0.76, 0.74]},
            statistical_test="t_test"
        )
        
        monitor.record_gauge("response_latency_ms", latency)
        
        # Check for anomalies
        if i % 20 == 0:
            anomaly_result = monitor.check_research_anomaly("response_latency_ms", latency)
            print(f"Step {i}: Latency={latency:.1f}ms, Anomaly Score={anomaly_result['anomaly_score']:.3f}")
    
    # Get research summary
    summary = monitor.get_research_summary()
    
    print("\n🧠 Quantum Insights:")
    qi = summary['quantum_insights']
    print(f"   Coherence: {qi['coherence']:.3f}")
    print(f"   Entanglement: {qi['entanglement']:.3f}")
    print(f"   System Health: {summary['system_health_score']:.3f}")
    
    print(f"\n📊 Research Metrics: {len(summary['research_metrics'])}")
    print(f"🚨 Recent Alerts: {len(summary['recent_alerts'])}")
    
    monitor.shutdown()
    print("✅ Research-aware monitoring demonstration complete!")