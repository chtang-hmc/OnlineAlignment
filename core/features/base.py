"""Base class for feature extractors."""

# standard imports
from abc import ABC, abstractmethod

# library imports
import numpy as np


class FeatureExtractor(ABC):
    """Base class for feature extraction from audio signals.

    Feature extractors convert raw audio signals or feature representations
    into feature vectors suitable for alignment algorithms.
    """

    def __init__(self, n_features: int | None = None):
        """Initialize the feature extractor.

        Args:
            n_features: Number of features to extract per frame.
                If None, determined automatically by the extractor.
        """
        self.n_features = n_features

    @abstractmethod
    def extract(self, signal: np.ndarray) -> np.ndarray:
        """Extract features from a signal.

        Args:
            signal: Input signal. Shape depends on extractor type.
                For audio: (n_samples,) or (n_channels, n_samples)
                For features: (n_features, n_frames)

        Returns:
            Feature matrix. Shape (n_features, n_frames)
        """
        pass

    @abstractmethod
    def extract_frame(self, signal_frame: np.ndarray) -> np.ndarray:
        """Extract features from a single frame.

        Args:
            signal_frame: Single frame of input signal.
                Shape depends on extractor type.

        Returns:
            Feature vector. Shape (n_features, 1)
        """
        pass
