"""Offline time scale modification for complete signals."""

# library imports
import numpy as np

from .base import TimeScaleModifier


class OfflineTimeScaleModifier(TimeScaleModifier):
    """Base class for offline time scale modification.

    Offline TSM processes complete signals with full alignment paths,
    allowing for more sophisticated algorithms and optimizations.
    """

    def modify(
        self,
        signal: np.ndarray,
        alignment_path: np.ndarray,
    ) -> np.ndarray:
        """Modify time scale of complete signal based on alignment path.

        Args:
            signal: Complete input signal to modify.
                Shape (n_samples,) or (n_channels, n_samples)
            alignment_path: Complete alignment path from alignment algorithm.
                Shape (2, n_points) where [0, :] are reference indices
                and [1, :] are query indices.

        Returns:
            Modified signal with adjusted time scale.
            Shape matches input signal.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
