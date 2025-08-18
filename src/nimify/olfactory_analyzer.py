"""Olfactory stimulus analysis module for bioneuro-olfactory fusion."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import OlfactoryConfig, OlfactoryMoleculeType

logger = logging.getLogger(__name__)


@dataclass
class MolecularFeatures:
    """Container for extracted molecular features."""
    
    # Basic molecular properties
    molecular_weight: float
    vapor_pressure: float  # mmHg at 20°C
    boiling_point: float   # °C
    melting_point: float   # °C
    
    # Chemical descriptors
    functional_groups: list[str]
    carbon_chain_length: int
    double_bonds: int
    ring_structures: int
    
    # Physical properties
    polarity_index: float  # 0-1 scale
    hydrophobicity: float  # logP
    surface_area: float    # Å²
    volume: float          # Å³
    
    # Olfactory-specific properties
    odor_character: list[str]
    perceived_intensity: float  # 0-10 scale
    threshold_concentration: float  # ppm
    
    # Computational descriptors
    dragon_descriptors: dict[str, float] | None = None
    mordred_descriptors: dict[str, float] | None = None
    rdkit_descriptors: dict[str, float] | None = None


class OlfactoryPerceptionModel:
    """Models human olfactory perception based on molecular structure."""
    
    # Simplified olfactory receptor activation patterns
    OR_ACTIVATION_PATTERNS = {
        OlfactoryMoleculeType.ALDEHYDE: [0.8, 0.3, 0.1, 0.2, 0.9, 0.4],
        OlfactoryMoleculeType.ESTER: [0.2, 0.7, 0.8, 0.1, 0.3, 0.6],
        OlfactoryMoleculeType.KETONE: [0.4, 0.1, 0.6, 0.8, 0.2, 0.5],
        OlfactoryMoleculeType.ALCOHOL: [0.1, 0.5, 0.3, 0.4, 0.7, 0.8],
        OlfactoryMoleculeType.TERPENE: [0.6, 0.8, 0.2, 0.7, 0.1, 0.3],
        OlfactoryMoleculeType.AROMATIC: [0.9, 0.2, 0.4, 0.3, 0.8, 0.1],
    }
    
    @staticmethod
    def predict_or_activation(
        molecule_type: OlfactoryMoleculeType, 
        concentration: float,
        molecular_features: MolecularFeatures
    ) -> np.ndarray:
        """Predict olfactory receptor activation pattern."""
        base_pattern = np.array(
            OlfactoryPerceptionModel.OR_ACTIVATION_PATTERNS[molecule_type]
        )
        
        # Concentration-dependent scaling (Hill equation-like)
        concentration_factor = concentration / (concentration + molecular_features.threshold_concentration)
        
        # Molecular weight influence
        mw_factor = 1.0 / (1.0 + np.exp((molecular_features.molecular_weight - 150) / 50))
        
        # Final activation pattern
        activation_pattern = base_pattern * concentration_factor * mw_factor
        
        return np.clip(activation_pattern, 0, 1)


class OlfactoryAnalyzer:
    """Analyzes olfactory stimuli and predicts neural responses."""
    
    def __init__(self, config: OlfactoryConfig):
        self.config = config
        self.perception_model = OlfactoryPerceptionModel()
        self.molecular_database = self._load_molecular_database()
    
    def _load_molecular_database(self) -> dict[str, Any]:
        """Load molecular database with known odorants."""
        # Simplified molecular database
        database = {
            'vanillin': {
                'type': OlfactoryMoleculeType.ALDEHYDE,
                'molecular_weight': 152.15,
                'vapor_pressure': 0.0133,  # mmHg
                'boiling_point': 285.0,
                'functional_groups': ['aldehyde', 'phenol', 'methoxy'],
                'carbon_chain_length': 8,
                'odor_character': ['vanilla', 'sweet', 'creamy'],
                'threshold_concentration': 0.02  # ppm
            },
            'limonene': {
                'type': OlfactoryMoleculeType.TERPENE,
                'molecular_weight': 136.23,
                'vapor_pressure': 1.98,
                'boiling_point': 176.0,
                'functional_groups': ['alkene', 'cyclohexene'],
                'carbon_chain_length': 10,
                'odor_character': ['citrus', 'fresh', 'orange'],
                'threshold_concentration': 0.01
            },
            'ethyl_acetate': {
                'type': OlfactoryMoleculeType.ESTER,
                'molecular_weight': 88.11,
                'vapor_pressure': 95.1,
                'boiling_point': 77.1,
                'functional_groups': ['ester'],
                'carbon_chain_length': 4,
                'odor_character': ['fruity', 'sweet', 'solvent'],
                'threshold_concentration': 0.87
            },
            'benzaldehyde': {
                'type': OlfactoryMoleculeType.AROMATIC,
                'molecular_weight': 106.12,
                'vapor_pressure': 1.27,
                'boiling_point': 179.0,
                'functional_groups': ['aldehyde', 'aromatic'],
                'carbon_chain_length': 7,
                'odor_character': ['almond', 'sweet', 'bitter'],
                'threshold_concentration': 0.04
            },
            'geraniol': {
                'type': OlfactoryMoleculeType.ALCOHOL,
                'molecular_weight': 154.25,
                'vapor_pressure': 0.04,
                'boiling_point': 230.0,
                'functional_groups': ['alcohol', 'alkene'],
                'carbon_chain_length': 10,
                'odor_character': ['floral', 'rose', 'sweet'],
                'threshold_concentration': 0.01
            }
        }
        
        return database
    
    def analyze(
        self, 
        molecule_data: dict[str, Any], 
        concentration: float
    ) -> dict[str, Any]:
        """
        Analyze olfactory stimulus properties and predict neural responses.
        
        Args:
            molecule_data: Dictionary containing molecule information
            concentration: Stimulus concentration in ppm
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing olfactory stimulus at {concentration} ppm")
        
        # Extract molecular features
        molecular_features = self._extract_molecular_features(molecule_data)
        
        # Predict olfactory receptor activation
        or_activation = self._predict_receptor_activation(
            molecular_features, 
            concentration
        )
        
        # Compute psychophysical properties
        psychophysical_props = self._compute_psychophysical_properties(
            molecular_features,
            concentration
        )
        
        # Predict neural response patterns
        neural_predictions = self._predict_neural_responses(
            molecular_features,
            or_activation,
            concentration
        )
        
        # Temporal dynamics
        temporal_profile = self._generate_temporal_profile(
            molecular_features,
            concentration
        )
        
        return {
            'molecular_features': molecular_features,
            'receptor_activation': or_activation,
            'psychophysical_properties': psychophysical_props,
            'neural_predictions': neural_predictions,
            'temporal_profile': temporal_profile,
            'analysis_metadata': {
                'concentration': concentration,
                'analysis_timestamp': np.datetime64('now'),
                'stimulus_duration': self.config.stimulus_duration,
                'inter_stimulus_interval': self.config.inter_stimulus_interval
            }
        }
    
    def _extract_molecular_features(self, molecule_data: dict[str, Any]) -> MolecularFeatures:
        """Extract comprehensive molecular features."""
        
        # Check if molecule is in our database
        molecule_name = molecule_data.get('name', '').lower()
        if molecule_name in self.molecular_database:
            db_entry = self.molecular_database[molecule_name]
            
            return MolecularFeatures(
                molecular_weight=db_entry['molecular_weight'],
                vapor_pressure=db_entry['vapor_pressure'],
                boiling_point=db_entry['boiling_point'],
                melting_point=molecule_data.get('melting_point', 0.0),
                functional_groups=db_entry['functional_groups'],
                carbon_chain_length=db_entry['carbon_chain_length'],
                double_bonds=molecule_data.get('double_bonds', 0),
                ring_structures=molecule_data.get('ring_structures', 0),
                polarity_index=self._calculate_polarity_index(db_entry['functional_groups']),
                hydrophobicity=self._estimate_logp(db_entry['molecular_weight']),
                surface_area=self._estimate_surface_area(db_entry['molecular_weight']),
                volume=self._estimate_volume(db_entry['molecular_weight']),
                odor_character=db_entry['odor_character'],
                perceived_intensity=molecule_data.get('perceived_intensity', 5.0),
                threshold_concentration=db_entry['threshold_concentration']
            )
        else:
            # Generate features from provided data
            return self._generate_features_from_raw_data(molecule_data)
    
    def _calculate_polarity_index(self, functional_groups: list[str]) -> float:
        """Calculate polarity index based on functional groups."""
        polarity_values = {
            'alcohol': 0.8,
            'aldehyde': 0.6,
            'ketone': 0.5,
            'ester': 0.4,
            'aromatic': 0.2,
            'alkane': 0.1,
            'phenol': 0.9,
            'ether': 0.3
        }
        
        total_polarity = sum(polarity_values.get(group, 0.3) for group in functional_groups)
        return min(1.0, total_polarity / len(functional_groups) if functional_groups else 0.3)
    
    def _estimate_logp(self, molecular_weight: float) -> float:
        """Estimate logP (hydrophobicity) from molecular weight."""
        # Simplified estimation based on MW
        return (molecular_weight - 100) / 100
    
    def _estimate_surface_area(self, molecular_weight: float) -> float:
        """Estimate molecular surface area from molecular weight."""
        # Empirical relationship for organic molecules
        return 4.5 * (molecular_weight ** 0.67)
    
    def _estimate_volume(self, molecular_weight: float) -> float:
        """Estimate molecular volume from molecular weight."""
        # Van der Waals volume estimation
        return molecular_weight / 0.6  # Approximate density
    
    def _generate_features_from_raw_data(self, molecule_data: dict[str, Any]) -> MolecularFeatures:
        """Generate molecular features from raw input data."""
        mw = molecule_data.get('molecular_weight', 150.0)
        
        return MolecularFeatures(
            molecular_weight=mw,
            vapor_pressure=molecule_data.get('vapor_pressure', 1.0),
            boiling_point=molecule_data.get('boiling_point', 200.0),
            melting_point=molecule_data.get('melting_point', 0.0),
            functional_groups=molecule_data.get('functional_groups', ['unknown']),
            carbon_chain_length=molecule_data.get('carbon_chain_length', 6),
            double_bonds=molecule_data.get('double_bonds', 0),
            ring_structures=molecule_data.get('ring_structures', 0),
            polarity_index=molecule_data.get('polarity_index', 0.5),
            hydrophobicity=self._estimate_logp(mw),
            surface_area=self._estimate_surface_area(mw),
            volume=self._estimate_volume(mw),
            odor_character=molecule_data.get('odor_character', ['unknown']),
            perceived_intensity=molecule_data.get('perceived_intensity', 5.0),
            threshold_concentration=molecule_data.get('threshold_concentration', 1.0)
        )
    
    def _predict_receptor_activation(
        self, 
        features: MolecularFeatures, 
        concentration: float
    ) -> dict[str, Any]:
        """Predict olfactory receptor activation patterns."""
        
        # Determine primary molecule type
        primary_type = self._classify_molecule_type(features)
        
        # Get base activation pattern
        activation_pattern = self.perception_model.predict_or_activation(
            primary_type, concentration, features
        )
        
        # Add noise and variability
        noise_level = 0.1
        noisy_activation = activation_pattern + np.random.normal(0, noise_level, activation_pattern.shape)
        noisy_activation = np.clip(noisy_activation, 0, 1)
        
        return {
            'receptor_types': [f'OR{i+1}' for i in range(len(activation_pattern))],
            'activation_strengths': noisy_activation,
            'primary_receptors': np.argsort(activation_pattern)[-3:],  # Top 3 receptors
            'activation_sum': np.sum(activation_pattern),
            'activation_pattern_type': primary_type.value
        }
    
    def _classify_molecule_type(self, features: MolecularFeatures) -> OlfactoryMoleculeType:
        """Classify molecule type based on functional groups."""
        functional_groups = [fg.lower() for fg in features.functional_groups]
        
        if 'aldehyde' in functional_groups:
            return OlfactoryMoleculeType.ALDEHYDE
        elif 'ester' in functional_groups:
            return OlfactoryMoleculeType.ESTER
        elif 'ketone' in functional_groups:
            return OlfactoryMoleculeType.KETONE
        elif 'alcohol' in functional_groups:
            return OlfactoryMoleculeType.ALCOHOL
        elif any(terpene in functional_groups for terpene in ['terpene', 'monoterpene', 'sesquiterpene']):
            return OlfactoryMoleculeType.TERPENE
        elif 'aromatic' in functional_groups or 'benzene' in functional_groups:
            return OlfactoryMoleculeType.AROMATIC
        else:
            # Default classification based on other properties
            return OlfactoryMoleculeType.AROMATIC
    
    def _compute_psychophysical_properties(
        self,
        features: MolecularFeatures,
        concentration: float
    ) -> dict[str, Any]:
        """Compute psychophysical properties (perceived intensity, pleasantness, etc.)."""
        
        # Perceived intensity using Stevens' power law
        if concentration > features.threshold_concentration:
            relative_concentration = concentration / features.threshold_concentration
            perceived_intensity = features.perceived_intensity * (relative_concentration ** 0.3)
        else:
            perceived_intensity = 0.0
        
        # Pleasantness based on odor character and concentration
        pleasantness_map = {
            'vanilla': 7.5, 'sweet': 6.0, 'fruity': 7.0, 'floral': 8.0,
            'citrus': 7.5, 'fresh': 6.5, 'rose': 8.0, 'almond': 6.0,
            'solvent': 2.0, 'bitter': 3.0, 'unknown': 5.0
        }
        
        pleasantness_scores = [
            pleasantness_map.get(char, 5.0) for char in features.odor_character
        ]
        base_pleasantness = np.mean(pleasantness_scores)
        
        # Concentration affects pleasantness (inverted-U shape)
        optimal_concentration = features.threshold_concentration * 10
        concentration_factor = np.exp(-0.5 * ((concentration - optimal_concentration) / optimal_concentration) ** 2)
        pleasantness = base_pleasantness * concentration_factor
        
        # Familiarity (simplified)
        familiarity = 7.0 if any(char in ['vanilla', 'citrus', 'rose', 'fruity'] for char in features.odor_character) else 4.0
        
        return {
            'perceived_intensity': min(10.0, perceived_intensity),
            'pleasantness': max(0.0, min(10.0, pleasantness)),
            'familiarity': familiarity,
            'detectability': 1.0 if concentration > features.threshold_concentration else 0.0,
            'discrimination_threshold': features.threshold_concentration * 0.1,
            'adaptation_rate': self._estimate_adaptation_rate(features)
        }
    
    def _estimate_adaptation_rate(self, features: MolecularFeatures) -> float:
        """Estimate adaptation rate based on molecular properties."""
        # Higher vapor pressure -> faster adaptation
        # Higher molecular weight -> slower adaptation
        adaptation_rate = (features.vapor_pressure / 10) / (features.molecular_weight / 100)
        return min(1.0, max(0.1, adaptation_rate))
    
    def _predict_neural_responses(
        self,
        features: MolecularFeatures,
        receptor_activation: dict[str, Any],
        concentration: float
    ) -> dict[str, Any]:
        """Predict neural response patterns in olfactory processing areas."""
        
        # Olfactory bulb response
        ob_response = self._predict_bulb_response(receptor_activation, concentration)
        
        # Piriform cortex response
        pc_response = self._predict_cortical_response(ob_response, features)
        
        # Orbitofrontal cortex response
        ofc_response = self._predict_ofc_response(pc_response, features, concentration)
        
        # Limbic system response
        limbic_response = self._predict_limbic_response(features, concentration)
        
        return {
            'olfactory_bulb': ob_response,
            'piriform_cortex': pc_response,
            'orbitofrontal_cortex': ofc_response,
            'limbic_system': limbic_response,
            'integrated_response': self._integrate_neural_responses(
                ob_response, pc_response, ofc_response, limbic_response
            )
        }
    
    def _predict_bulb_response(
        self, 
        receptor_activation: dict[str, Any], 
        concentration: float
    ) -> dict[str, float]:
        """Predict olfactory bulb response patterns."""
        activation_sum = receptor_activation['activation_sum']
        
        return {
            'mitral_cell_activity': activation_sum * 0.8,
            'granule_cell_inhibition': activation_sum * 0.3,
            'response_latency': max(50, 200 - concentration * 10),  # ms
            'response_duration': min(2000, 500 + concentration * 50),  # ms
            'gamma_oscillation_power': activation_sum * 0.6,
            'beta_oscillation_power': activation_sum * 0.4
        }
    
    def _predict_cortical_response(
        self,
        bulb_response: dict[str, float],
        features: MolecularFeatures
    ) -> dict[str, float]:
        """Predict piriform cortex response patterns."""
        mitral_input = bulb_response['mitral_cell_activity']
        
        return {
            'pyramidal_cell_activity': mitral_input * 0.7,
            'interneuron_activity': mitral_input * 0.5,
            'pattern_completion_strength': features.perceived_intensity * 0.6,
            'odor_memory_activation': 5.0 if 'familiar' in str(features.odor_character) else 2.0,
            'lateral_inhibition': mitral_input * 0.4,
            'theta_phase_coupling': mitral_input * 0.3
        }
    
    def _predict_ofc_response(
        self,
        cortical_response: dict[str, float],
        features: MolecularFeatures,
        concentration: float
    ) -> dict[str, float]:
        """Predict orbitofrontal cortex response patterns."""
        cortical_input = cortical_response['pyramidal_cell_activity']
        
        # OFC processes valence and reward value
        pleasant_odors = ['vanilla', 'rose', 'fruity', 'citrus', 'sweet']
        valence = 1.0 if any(char in pleasant_odors for char in features.odor_character) else -0.5
        
        return {
            'valence_encoding': valence * cortical_input,
            'reward_prediction': max(0, valence * concentration * 0.1),
            'expectation_mismatch': abs(features.perceived_intensity - 5.0) * 0.2,
            'attention_modulation': cortical_input * 0.6,
            'decision_value': valence * cortical_input * 0.8
        }
    
    def _predict_limbic_response(
        self,
        features: MolecularFeatures,
        concentration: float
    ) -> dict[str, float]:
        """Predict limbic system response (amygdala, hippocampus)."""
        
        # Emotional salience
        emotional_odors = ['vanilla', 'rose', 'citrus']  # Pleasant, emotionally salient
        emotional_salience = 8.0 if any(char in emotional_odors for char in features.odor_character) else 3.0
        
        return {
            'amygdala_activation': emotional_salience * concentration * 0.1,
            'hippocampus_encoding': features.perceived_intensity * 0.5,
            'memory_consolidation': emotional_salience * 0.3,
            'emotional_arousal': emotional_salience * concentration * 0.05,
            'contextual_association': 5.0  # Simplified context
        }
    
    def _integrate_neural_responses(
        self,
        ob_response: dict[str, float],
        pc_response: dict[str, float],
        ofc_response: dict[str, float],
        limbic_response: dict[str, float]
    ) -> dict[str, float]:
        """Integrate responses across brain regions."""
        
        return {
            'overall_activation': (
                ob_response['mitral_cell_activity'] + 
                pc_response['pyramidal_cell_activity'] + 
                ofc_response['valence_encoding'] +
                limbic_response['amygdala_activation']
            ) / 4,
            'perceptual_strength': (ob_response['mitral_cell_activity'] + pc_response['pyramidal_cell_activity']) / 2,
            'emotional_impact': (ofc_response['valence_encoding'] + limbic_response['amygdala_activation']) / 2,
            'memory_formation': limbic_response['memory_consolidation'],
            'attention_capture': max(ofc_response['attention_modulation'], limbic_response['emotional_arousal'])
        }
    
    def _generate_temporal_profile(
        self,
        features: MolecularFeatures,
        concentration: float
    ) -> dict[str, Any]:
        """Generate temporal profile of olfactory response."""
        
        # Time parameters
        duration = self.config.stimulus_duration
        dt = 0.01  # 10ms resolution
        time_points = np.arange(0, duration + dt, dt)
        
        # Response phases
        onset_time = 0.05 + features.molecular_weight / 10000  # Heavier molecules slower
        peak_time = onset_time + 0.5
        adaptation_rate = self._estimate_adaptation_rate(features)
        
        # Generate temporal response profile
        response_profile = np.zeros_like(time_points)
        
        for i, t in enumerate(time_points):
            if t < onset_time:
                response_profile[i] = 0
            elif t < peak_time:
                # Rising phase
                response_profile[i] = concentration * (t - onset_time) / (peak_time - onset_time)
            else:
                # Adaptation phase
                peak_response = concentration
                adaptation_decay = np.exp(-adaptation_rate * (t - peak_time))
                response_profile[i] = peak_response * adaptation_decay
        
        return {
            'time_points': time_points,
            'response_profile': response_profile,
            'onset_time': onset_time,
            'peak_time': peak_time,
            'peak_response': np.max(response_profile),
            'adaptation_time_constant': 1.0 / adaptation_rate if adaptation_rate > 0 else np.inf,
            'area_under_curve': np.trapz(response_profile, time_points)
        }

    def predict_mixture_response(
        self,
        mixture_components: list[tuple[dict[str, Any], float]],
        total_concentration: float
    ) -> dict[str, Any]:
        """
        Predict response to olfactory mixtures.
        
        Args:
            mixture_components: List of (molecule_data, relative_fraction) tuples
            total_concentration: Total mixture concentration in ppm
            
        Returns:
            Dictionary containing mixture analysis results
        """
        
        individual_responses = []
        individual_features = []
        
        # Analyze each component
        for molecule_data, fraction in mixture_components:
            component_concentration = total_concentration * fraction
            
            # Skip components below threshold
            if component_concentration < 0.001:
                continue
            
            component_analysis = self.analyze(molecule_data, component_concentration)
            individual_responses.append(component_analysis)
            individual_features.append(component_analysis['molecular_features'])
        
        # Compute mixture interactions
        mixture_interactions = self._compute_mixture_interactions(
            individual_features,
            [total_concentration * frac for _, frac in mixture_components]
        )
        
        # Aggregate neural responses
        aggregated_neural = self._aggregate_neural_responses(individual_responses)
        
        # Predict emergent mixture properties
        emergent_properties = self._predict_emergent_properties(
            individual_features,
            mixture_interactions
        )
        
        return {
            'individual_components': individual_responses,
            'mixture_interactions': mixture_interactions,
            'aggregated_neural_response': aggregated_neural,
            'emergent_properties': emergent_properties,
            'mixture_metadata': {
                'n_components': len(mixture_components),
                'total_concentration': total_concentration,
                'dominant_component': self._identify_dominant_component(individual_responses)
            }
        }
    
    def _compute_mixture_interactions(
        self,
        component_features: list[MolecularFeatures],
        concentrations: list[float]
    ) -> dict[str, Any]:
        """Compute interactions between mixture components."""
        
        # Competitive inhibition
        competition_matrix = np.zeros((len(component_features), len(component_features)))
        for i, feat_i in enumerate(component_features):
            for j, feat_j in enumerate(component_features):
                if i != j:
                    # Similarity-based competition
                    similarity = self._compute_molecular_similarity(feat_i, feat_j)
                    competition_matrix[i, j] = similarity * concentrations[j]
        
        # Synergistic enhancement
        synergy_scores = []
        for i in range(len(component_features)):
            for j in range(i+1, len(component_features)):
                synergy = self._compute_synergy(component_features[i], component_features[j])
                synergy_scores.append(synergy)
        
        return {
            'competition_matrix': competition_matrix,
            'synergy_scores': synergy_scores,
            'overall_interaction_strength': np.mean(competition_matrix) + np.mean(synergy_scores),
            'interaction_type': 'competitive' if np.mean(competition_matrix) > np.mean(synergy_scores) else 'synergistic'
        }
    
    def _compute_molecular_similarity(
        self,
        feat1: MolecularFeatures,
        feat2: MolecularFeatures
    ) -> float:
        """Compute similarity between two molecules."""
        
        # Functional group overlap
        fg_overlap = len(set(feat1.functional_groups) & set(feat2.functional_groups)) / \
                    max(len(set(feat1.functional_groups) | set(feat2.functional_groups)), 1)
        
        # Molecular weight similarity
        mw_similarity = 1.0 / (1.0 + abs(feat1.molecular_weight - feat2.molecular_weight) / 100)
        
        # Polarity similarity
        polarity_similarity = 1.0 / (1.0 + abs(feat1.polarity_index - feat2.polarity_index))
        
        return (fg_overlap + mw_similarity + polarity_similarity) / 3
    
    def _compute_synergy(self, feat1: MolecularFeatures, feat2: MolecularFeatures) -> float:
        """Compute synergistic interaction potential."""
        
        # Complementary functional groups enhance each other
        complementary_pairs = [
            (['aldehyde'], ['alcohol']),
            (['ester'], ['terpene']),
            (['aromatic'], ['aliphatic'])
        ]
        
        synergy = 0.0
        for groups1, groups2 in complementary_pairs:
            if (any(g in feat1.functional_groups for g in groups1) and 
                any(g in feat2.functional_groups for g in groups2)) or \
               (any(g in feat2.functional_groups for g in groups1) and 
                any(g in feat1.functional_groups for g in groups2)):
                synergy += 0.3
        
        return min(1.0, synergy)
    
    def _aggregate_neural_responses(
        self,
        individual_responses: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Aggregate neural responses across mixture components."""
        
        aggregated = {
            'olfactory_bulb': {},
            'piriform_cortex': {},
            'orbitofrontal_cortex': {},
            'limbic_system': {},
            'integrated_response': {}
        }
        
        for region in aggregated:
            region_responses = [resp['neural_predictions'][region] for resp in individual_responses]
            
            for metric in region_responses[0]:
                values = [resp[metric] for resp in region_responses]
                
                # Different aggregation strategies for different metrics
                if 'activity' in metric or 'activation' in metric:
                    # Sum activities (with saturation)
                    aggregated[region][metric] = min(10.0, sum(values))
                elif 'latency' in metric:
                    # Minimum latency (fastest response)
                    aggregated[region][metric] = min(values)
                elif 'duration' in metric:
                    # Maximum duration
                    aggregated[region][metric] = max(values)
                else:
                    # Average for other metrics
                    aggregated[region][metric] = np.mean(values)
        
        return aggregated
    
    def _predict_emergent_properties(
        self,
        component_features: list[MolecularFeatures],
        interactions: dict[str, Any]
    ) -> dict[str, Any]:
        """Predict emergent properties of the mixture."""
        
        # Emergent odor character
        all_characters = []
        for feat in component_features:
            all_characters.extend(feat.odor_character)
        
        character_counts = {}
        for char in all_characters:
            character_counts[char] = character_counts.get(char, 0) + 1
        
        emergent_character = max(character_counts, key=character_counts.get) if character_counts else 'complex'
        
        # Emergent intensity (non-linear)
        individual_intensities = [feat.perceived_intensity for feat in component_features]
        if interactions['interaction_type'] == 'synergistic':
            emergent_intensity = sum(individual_intensities) * 1.2  # Enhancement
        else:
            emergent_intensity = sum(individual_intensities) * 0.8  # Suppression
        
        return {
            'emergent_odor_character': emergent_character,
            'emergent_intensity': min(10.0, emergent_intensity),
            'complexity_index': len(set(all_characters)) / len(all_characters) if all_characters else 0,
            'mixture_harmony': 1.0 - interactions['overall_interaction_strength'],
            'perceptual_novelty': interactions['overall_interaction_strength']
        }
    
    def _identify_dominant_component(
        self,
        individual_responses: list[dict[str, Any]]
    ) -> int:
        """Identify the dominant component in the mixture."""
        
        dominance_scores = []
        for resp in individual_responses:
            # Score based on receptor activation and perceived intensity
            receptor_score = resp['receptor_activation']['activation_sum']
            psychophysical_score = resp['psychophysical_properties']['perceived_intensity']
            dominance_scores.append(receptor_score + psychophysical_score)
        
        return np.argmax(dominance_scores) if dominance_scores else 0