"""Online time scale modification for streaming signals."""

# library imports
import numpy as np

from .base import TimeScaleModifier


class OnlineTimeScaleModifier(TimeScaleModifier):
    """Base class for online time scale modification.

    Online TSM processes signals incrementally as alignment information
    becomes available, suitable for real-time applications.
    """

    def __init__(self):
        """Initialize online time scale modifier."""
        super().__init__()
        self._buffer = []

    def reset(self):
        """Reset internal state for new signal."""
        self._buffer.clear()

    def feed(
        self,
        signal_frame: np.ndarray,
        alignment_info: dict,
    ) -> np.ndarray | None:
        """Feed a signal frame and alignment info, return modified frame if ready.

        Args:
            signal_frame: Frame of input signal.
                Shape (n_samples,) or (n_channels, n_samples)
            alignment_info: Dictionary containing alignment information
                for this frame (e.g., stretch factor, target position).

        Returns:
            Modified signal frame if ready, None otherwise.
            Shape matches input signal_frame.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def flush(self) -> np.ndarray:
        """Flush remaining buffered samples.

        Returns:
            Remaining modified signal samples.
        """
        raise NotImplementedError
