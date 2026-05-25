"""
Interaction Episode Analysis Module

This module provides functionality to detect, characterize, and analyze
interaction episodes from hand tracking data. Episodes are temporally bounded
sequences representing coherent units of user intent.

Key Features:
- Episode segmentation (velocity-based, spatial, multimodal)
- Kinematic feature extraction (velocity, acceleration, jerk, smoothness)
- Intent characterization (confidence, hesitation, effort)
- Speech-motion alignment
- Episode embedding and similarity
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from scipy import signal
from scipy.interpolate import interp1d


@dataclass
class Episode:
    """Represents a single interaction episode"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    hand: str  # 'left' or 'right'
    
    # Raw data for this episode
    landmarks: np.ndarray = field(default=None)  # (n_frames, 21, 3)
    timestamps: np.ndarray = field(default=None)  # (n_frames,)
    
    # Features
    kinematic_features: Dict[str, float] = field(default_factory=dict)
    intent_features: Dict[str, float] = field(default_factory=dict)
    
    # Multimodal alignment
    aligned_speech: List[Tuple[float, float, str]] = field(default_factory=list)  # [(start, end, text)]
    speech_embedding: Optional[np.ndarray] = None
    
    # Episode classification
    episode_type: str = "unknown"  # 'reach', 'manipulate', 'gesture', 'rest'
    confidence: float = 0.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


class EpisodeSegmenter:
    """Detects episode boundaries from hand tracking data"""
    
    def __init__(self, 
                 velocity_threshold: float = 0.05,
                 min_episode_duration: float = 0.3,
                 merge_gap: float = 0.2):
        """
        Args:
            velocity_threshold: Minimum velocity to consider as intentional motion
            min_episode_duration: Minimum episode duration in seconds
            merge_gap: Merge episodes separated by less than this (seconds)
        """
        self.velocity_threshold = velocity_threshold
        self.min_episode_duration = min_episode_duration
        self.merge_gap = merge_gap
    
    def segment_velocity_based(self, 
                               trace: List[Optional[Tuple[float, float, float]]], 
                               timestamps: np.ndarray,
                               hand: str = 'unknown') -> List[Episode]:
        """
        Segment episodes based on velocity thresholds
        
        Args:
            trace: List of (x, y, z) positions or None
            timestamps: Array of timestamps for each frame
            hand: 'left' or 'right'
            
        Returns:
            List of Episode objects
        """
        if not trace or len(trace) < 2:
            return []
        
        # Extract valid points and compute velocities
        velocities = []
        valid_indices = []
        
        for i in range(len(trace) - 1):
            if trace[i] is not None and trace[i+1] is not None:
                dt = timestamps[i+1] - timestamps[i]
                if dt > 0:
                    dx = np.array(trace[i+1]) - np.array(trace[i])
                    vel = np.linalg.norm(dx) / dt
                    velocities.append(vel)
                    valid_indices.append(i)
                else:
                    velocities.append(0)
                    valid_indices.append(i)
            else:
                velocities.append(0)
                valid_indices.append(i)
        
        if not velocities:
            return []
        
        # Smooth velocities to reduce noise
        if len(velocities) > 5:
            window_size = min(5, len(velocities))
            velocities = signal.savgol_filter(velocities, window_size, 2)
        
        # Detect active regions (velocity > threshold)
        active = np.array(velocities) > self.velocity_threshold
        
        # Find episode boundaries
        episodes = []
        in_episode = False
        start_idx = 0
        
        for i in range(len(active)):
            if active[i] and not in_episode:
                # Start new episode
                start_idx = valid_indices[i] if i < len(valid_indices) else i
                in_episode = True
            elif not active[i] and in_episode:
                # End episode
                end_idx = valid_indices[i-1] if i-1 < len(valid_indices) else i-1
                episodes.append((start_idx, end_idx))
                in_episode = False
        
        # Close final episode if still active
        if in_episode:
            end_idx = valid_indices[-1] if valid_indices else len(trace) - 1
            episodes.append((start_idx, end_idx))
        
        # Filter by minimum duration and merge nearby episodes
        filtered_episodes = []
        for start_frame, end_frame in episodes:
            start_time = timestamps[start_frame]
            end_time = timestamps[end_frame]
            duration = end_time - start_time
            
            if duration >= self.min_episode_duration:
                filtered_episodes.append(Episode(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=start_time,
                    end_time=end_time,
                    hand=hand
                ))
        
        # Merge nearby episodes
        merged = self._merge_nearby_episodes(filtered_episodes)
        
        return merged
    
    def _merge_nearby_episodes(self, episodes: List[Episode]) -> List[Episode]:
        """Merge episodes separated by small gaps"""
        if len(episodes) <= 1:
            return episodes
        
        merged = [episodes[0]]
        
        for ep in episodes[1:]:
            prev = merged[-1]
            gap = ep.start_time - prev.end_time
            
            if gap <= self.merge_gap and ep.hand == prev.hand:
                # Merge with previous
                merged[-1] = Episode(
                    start_frame=prev.start_frame,
                    end_frame=ep.end_frame,
                    start_time=prev.start_time,
                    end_time=ep.end_time,
                    hand=prev.hand
                )
            else:
                merged.append(ep)
        
        return merged


class KinematicFeatureExtractor:
    """Extracts kinematic features from episode data"""
    
    @staticmethod
    def extract_features(trace: List[Optional[Tuple[float, float, float]]], 
                        timestamps: np.ndarray,
                        start_frame: int,
                        end_frame: int) -> Dict[str, float]:
        """
        Extract kinematic features for an episode
        
        Returns dict with keys:
            - mean_velocity, max_velocity, velocity_variance
            - mean_acceleration, acceleration_variance
            - jerk_magnitude
            - smoothness_sparc
            - path_length, path_efficiency
            - spatial_spread_xy, spatial_spread_z
        """
        features = {}
        
        # Extract episode segment
        episode_trace = trace[start_frame:end_frame+1]
        episode_times = timestamps[start_frame:end_frame+1]
        
        # Filter out None values
        valid_points = [(i, p) for i, p in enumerate(episode_trace) if p is not None]
        if len(valid_points) < 2:
            return {k: 0.0 for k in ['mean_velocity', 'max_velocity', 'velocity_variance',
                                      'mean_acceleration', 'acceleration_variance', 'jerk_magnitude',
                                      'smoothness_sparc', 'path_length', 'path_efficiency',
                                      'spatial_spread_xy', 'spatial_spread_z']}
        
        indices, points = zip(*valid_points)
        points = np.array(points)
        times = episode_times[list(indices)]
        
        # Velocity
        velocities = []
        for i in range(len(points) - 1):
            dt = times[i+1] - times[i]
            if dt > 0:
                vel = np.linalg.norm(points[i+1] - points[i]) / dt
                velocities.append(vel)
        
        if velocities:
            features['mean_velocity'] = float(np.mean(velocities))
            features['max_velocity'] = float(np.max(velocities))
            features['velocity_variance'] = float(np.var(velocities))
        else:
            features['mean_velocity'] = 0.0
            features['max_velocity'] = 0.0
            features['velocity_variance'] = 0.0
        
        # Acceleration
        if len(velocities) > 1:
            accelerations = np.diff(velocities) / np.diff(times[:-1])
            features['mean_acceleration'] = float(np.mean(np.abs(accelerations)))
            features['acceleration_variance'] = float(np.var(accelerations))
        else:
            features['mean_acceleration'] = 0.0
            features['acceleration_variance'] = 0.0
        
        # Jerk (rate of change of acceleration)
        if len(velocities) > 2:
            dt_array = np.diff(times[:-1])
            accelerations = np.diff(velocities) / dt_array
            dt_array2 = np.diff(times[:-2])
            jerks = np.diff(accelerations) / dt_array2
            features['jerk_magnitude'] = float(np.mean(np.abs(jerks)))
        else:
            features['jerk_magnitude'] = 0.0
        
        # Path metrics
        path_segments = [np.linalg.norm(points[i+1] - points[i]) for i in range(len(points)-1)]
        features['path_length'] = float(np.sum(path_segments))
        
        direct_distance = np.linalg.norm(points[-1] - points[0])
        if direct_distance > 0:
            features['path_efficiency'] = direct_distance / features['path_length']
        else:
            features['path_efficiency'] = 1.0
        
        # Spatial spread
        features['spatial_spread_xy'] = float(np.std(points[:, :2]))
        features['spatial_spread_z'] = float(np.std(points[:, 2]))
        
        # Smoothness (simplified SPARC - spectral arc length)
        features['smoothness_sparc'] = KinematicFeatureExtractor._compute_sparc(velocities, times[:-1])
        
        return features
    
    @staticmethod
    def _compute_sparc(velocities: List[float], times: np.ndarray) -> float:
        """
        Compute Spectral Arc Length (SPARC) smoothness metric
        Lower values indicate smoother motion
        """
        if len(velocities) < 4:
            return 0.0
        
        try:
            # Compute power spectral density
            vel_array = np.array(velocities)
            freqs, psd = signal.welch(vel_array, fs=1.0/np.mean(np.diff(times)), nperseg=min(len(vel_array), 256))
            
            # Normalize
            psd_norm = psd / np.sum(psd)
            
            # Compute arc length
            arc_length = 0.0
            for i in range(len(psd_norm) - 1):
                df = freqs[i+1] - freqs[i]
                dp = psd_norm[i+1] - psd_norm[i]
                arc_length += np.sqrt(df**2 + dp**2)
            
            return float(arc_length)
        except:
            return 0.0


class IntentFeatureExtractor:
    """Extracts intent-related features (confidence, hesitation, effort)"""
    
    @staticmethod
    def extract_features(trace: List[Optional[Tuple[float, float, float]]],
                        timestamps: np.ndarray,
                        start_frame: int,
                        end_frame: int,
                        kinematic_features: Dict[str, float]) -> Dict[str, float]:
        """
        Extract intent characterization features
        
        Returns dict with keys:
            - initiation_velocity: Peak velocity in first 30% of episode
            - retraction_velocity: Average velocity in last 20% of episode
            - reversal_count: Number of direction changes
            - dwell_ratio: Proportion of time with low velocity
            - hesitation_index: Combined metric of uncertainty
            - effort_proxy: Integrated energy (velocity squared)
        """
        features = {}
        
        # Extract episode segment
        episode_trace = trace[start_frame:end_frame+1]
        episode_times = timestamps[start_frame:end_frame+1]
        
        # Filter valid points
        valid_points = [(i, p) for i, p in enumerate(episode_trace) if p is not None]
        if len(valid_points) < 2:
            return {k: 0.0 for k in ['initiation_velocity', 'retraction_velocity', 
                                      'reversal_count', 'dwell_ratio', 'hesitation_index', 'effort_proxy']}
        
        indices, points = zip(*valid_points)
        points = np.array(points)
        times = episode_times[list(indices)]
        
        # Compute velocities
        velocities = []
        for i in range(len(points) - 1):
            dt = times[i+1] - times[i]
            if dt > 0:
                vel = np.linalg.norm(points[i+1] - points[i]) / dt
                velocities.append(vel)
            else:
                velocities.append(0)
        
        if not velocities:
            return {k: 0.0 for k in ['initiation_velocity', 'retraction_velocity', 
                                      'reversal_count', 'dwell_ratio', 'hesitation_index', 'effort_proxy']}
        
        velocities = np.array(velocities)
        
        # Initiation velocity (first 30%)
        init_cutoff = max(1, int(len(velocities) * 0.3))
        features['initiation_velocity'] = float(np.max(velocities[:init_cutoff]))
        
        # Retraction velocity (last 20%)
        ret_start = max(0, int(len(velocities) * 0.8))
        features['retraction_velocity'] = float(np.mean(velocities[ret_start:]))
        
        # Reversal count (direction changes)
        if len(points) > 2:
            directions = np.diff(points, axis=0)
            # Count sign changes in each dimension
            reversals = 0
            for dim in range(3):
                dir_1d = directions[:, dim]
                sign_changes = np.diff(np.sign(dir_1d))
                reversals += np.sum(np.abs(sign_changes) > 0)
            features['reversal_count'] = float(reversals)
        else:
            features['reversal_count'] = 0.0
        
        # Dwell ratio (time spent nearly stationary)
        dwell_threshold = 0.02  # Adjust based on data scale
        dwell_frames = np.sum(velocities < dwell_threshold)
        features['dwell_ratio'] = dwell_frames / len(velocities)
        
        # Hesitation index (combination of reversals and dwells)
        features['hesitation_index'] = features['reversal_count'] * 0.1 + features['dwell_ratio']
        
        # Effort proxy (integrated kinetic energy)
        if len(times) > 1:
            dt_array = np.diff(times)
            energy = np.sum(velocities**2 * dt_array)
            features['effort_proxy'] = float(energy)
        else:
            features['effort_proxy'] = 0.0
        
        return features


class MultimodalAligner:
    """Aligns speech transcripts with motion episodes"""
    
    @staticmethod
    def align_speech_to_episodes(episodes: List[Episode], 
                                 transcript: List[Tuple[float, float, str]]) -> List[Episode]:
        """
        Align speech segments to episodes based on temporal overlap
        
        Args:
            episodes: List of Episode objects
            transcript: List of (start_time, end_time, text) tuples
            
        Returns:
            Updated episodes with aligned_speech populated
        """
        if not transcript:
            return episodes
        
        for episode in episodes:
            # Find overlapping speech segments
            for start, end, text in transcript:
                # Check for temporal overlap
                if not (end < episode.start_time or start > episode.end_time):
                    # Calculate overlap ratio
                    overlap_start = max(start, episode.start_time)
                    overlap_end = min(end, episode.end_time)
                    overlap_duration = overlap_end - overlap_start
                    
                    # Only include if significant overlap (>20% of speech segment)
                    speech_duration = end - start
                    if speech_duration > 0 and overlap_duration / speech_duration > 0.2:
                        episode.aligned_speech.append((start, end, text))
        
        return episodes
    
    @staticmethod
    def extract_deictic_references(speech_text: str) -> List[str]:
        """Extract deictic terms from speech (this, that, here, there)"""
        deictic_terms = ['this', 'that', 'here', 'there', 'these', 'those']
        words = speech_text.lower().split()
        return [w for w in words if w in deictic_terms]


class EpisodeAnalyzer:
    """Main interface for episode analysis"""
    
    def __init__(self, 
                 velocity_threshold: float = 0.05,
                 min_episode_duration: float = 0.3):
        self.segmenter = EpisodeSegmenter(
            velocity_threshold=velocity_threshold,
            min_episode_duration=min_episode_duration
        )
        self.kinematic_extractor = KinematicFeatureExtractor()
        self.intent_extractor = IntentFeatureExtractor()
        self.multimodal_aligner = MultimodalAligner()
    
    def analyze_trace(self, 
                     trace: List[Optional[Tuple[float, float, float]]],
                     timestamps: np.ndarray,
                     hand: str = 'unknown',
                     transcript: Optional[List[Tuple[float, float, str]]] = None) -> List[Episode]:
        """
        Complete episode analysis pipeline
        
        Args:
            trace: Hand position trace
            timestamps: Time array
            hand: 'left' or 'right'
            transcript: Optional speech transcript
            
        Returns:
            List of fully analyzed Episode objects
        """
        # Segment into episodes
        episodes = self.segmenter.segment_velocity_based(trace, timestamps, hand)
        
        # Extract features for each episode
        for episode in episodes:
            # Kinematic features
            episode.kinematic_features = self.kinematic_extractor.extract_features(
                trace, timestamps, episode.start_frame, episode.end_frame
            )
            
            # Intent features
            episode.intent_features = self.intent_extractor.extract_features(
                trace, timestamps, episode.start_frame, episode.end_frame,
                episode.kinematic_features
            )
            
            # Classify episode type based on features
            episode.episode_type = self._classify_episode(episode)
            episode.confidence = self._compute_confidence(episode)
        
        # Align speech if provided
        if transcript:
            episodes = self.multimodal_aligner.align_speech_to_episodes(episodes, transcript)
        
        return episodes
    
    def _classify_episode(self, episode: Episode) -> str:
        """Classify episode type based on features"""
        kf = episode.kinematic_features
        intf = episode.intent_features
        
        # Simple heuristic classification
        if kf.get('mean_velocity', 0) < 0.03:
            return 'rest'
        elif intf.get('initiation_velocity', 0) > 0.15:
            return 'reach'
        elif kf.get('path_efficiency', 1) > 0.7:
            return 'direct_action'
        elif intf.get('reversal_count', 0) > 5:
            return 'exploratory'
        else:
            return 'manipulation'
    
    def _compute_confidence(self, episode: Episode) -> float:
        """Compute confidence score based on hesitation indicators"""
        intf = episode.intent_features
        
        # Lower hesitation = higher confidence
        hesitation = intf.get('hesitation_index', 0)
        confidence = 1.0 / (1.0 + hesitation)
        
        return float(np.clip(confidence, 0, 1))
    
    def compute_episode_embedding(self, episode: Episode, embed_dim: int = 32) -> np.ndarray:
        """
        Create feature vector for episode (for clustering/similarity)
        
        Args:
            episode: Episode object
            embed_dim: Target embedding dimension
            
        Returns:
            Feature vector (embed_dim,)
        """
        # Combine all features into vector
        features = []
        
        # Kinematic features
        for key in ['mean_velocity', 'max_velocity', 'smoothness_sparc', 
                   'path_length', 'path_efficiency', 'jerk_magnitude']:
            features.append(episode.kinematic_features.get(key, 0))
        
        # Intent features
        for key in ['initiation_velocity', 'retraction_velocity', 
                   'reversal_count', 'hesitation_index', 'effort_proxy']:
            features.append(episode.intent_features.get(key, 0))
        
        # Temporal
        features.append(episode.duration)
        features.append(episode.num_frames)
        
        # Pad or truncate to embed_dim
        features = np.array(features, dtype=np.float32)
        if len(features) < embed_dim:
            features = np.pad(features, (0, embed_dim - len(features)))
        else:
            features = features[:embed_dim]
        
        # Normalize
        if np.linalg.norm(features) > 0:
            features = features / np.linalg.norm(features)
        
        return features
