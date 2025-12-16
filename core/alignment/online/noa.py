"""Naive Online Alignment algorithm."""

from ..base import OnlineAlignment


class NOA(OnlineAlignment):
    """Naive Online Alignment Algorithm."""

    def __init__(self, *args, **kwargs):
        """Initialize NOA algorithm.

        Args:
            *args: Arguments passed to OnlineAlignment base class.
            **kwargs: Keyword arguments passed to OnlineAlignment base class.
        """
        super().__init__(*args, **kwargs)
        raise NotImplementedError
