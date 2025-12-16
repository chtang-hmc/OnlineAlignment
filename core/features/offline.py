"""Offline/batch feature extraction."""

# standard imports
from abc import abstractmethod

# library imports
import numpy as np

from .base import FeatureExtractor


class OfflineFeatureExtractor(FeatureExtractor):
    """Base class for offline feature extraction.

    Offline extractors process complete signals at once, allowing for
    more efficient batch processing and global optimizations.
    """

    @abstractmethod
    def extract(
        self,
        signal: np.ndarray,
        frame_size: int | None = None,
        hop_size: int | None = None,
    ) -> np.ndarray:
        """Extract features from a complete signal.

        Args:
            signal: Complete input signal.
                Shape (n_samples,) or (n_channels, n_samples)
            frame_size: Size of each frame in samples.
                If None, uses default for extractor.
            hop_size: Number of samples between consecutive frames.
                If None, uses default for extractor.

        Returns:
            Feature matrix. Shape (n_features, n_frames)
        """
        pass
