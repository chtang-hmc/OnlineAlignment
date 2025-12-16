"""Time scale modification for audio signals."""

from .base import TimeScaleModifier
from .online import OnlineTimeScaleModifier
from .offline import OfflineTimeScaleModifier

__all__ = [
    "TimeScaleModifier",
    "OnlineTimeScaleModifier",
    "OfflineTimeScaleModifier",
]
