"""Online feature extraction for streaming signals."""

# standard imports
from abc import abstractmethod
from typing import Iterator

# library imports
import numpy as np

from .base import FeatureExtractor


class OnlineFeatureExtractor(FeatureExtractor):
    """Base class for online feature extraction.

    Online extractors process signals frame-by-frame as they arrive,
    maintaining internal state for efficient streaming processing.
    """

    def __init__(self, frame_size: int, hop_size: int, n_features: int | None = None):
        """Initialize online feature extractor.

        Args:
            frame_size: Size of each frame in samples.
            hop_size: Number of samples between consecutive frames.
            n_features: Number of features per frame.
        """
        super().__init__(n_features=n_features)
        self.frame_size = frame_size
        self.hop_size = hop_size
        self._buffer = np.zeros(frame_size, dtype=np.float32)
        self._buffer_idx = 0

    def reset(self):
        """Reset internal state for new signal."""
        self._buffer.fill(0)
        self._buffer_idx = 0

    @abstractmethod
    def feed(self, samples: np.ndarray) -> Iterator[np.ndarray]:
        """Feed new samples and yield features for complete frames.

        This method processes incoming samples and yields feature vectors
        as complete frames become available. This allows for efficient
        streaming processing without accumulating all features in memory.

        Args:
            samples: New audio samples. Shape (n_samples,)

        Yields:
            Feature vectors, one per complete frame.
            Each vector has shape (n_features, 1)

        Example:
            >>> extractor = OnlineMFCCExtractor(frame_size=2048, hop_size=512)
            >>> for audio_chunk in audio_stream:
            ...     for feature_frame in extractor.feed(audio_chunk):
            ...         process_feature(feature_frame)
        """
        pass

    @abstractmethod
    def flush(self) -> Iterator[np.ndarray]:
        """Flush remaining buffered samples and yield final features.

        Call this method after feeding all samples to process any
        remaining buffered samples.

        Yields:
            Feature vectors from remaining buffered samples.
            Each vector has shape (n_features, 1)

        Example:
            >>> for feature_frame in extractor.flush():
            ...     process_feature(feature_frame)
        """
        pass
