"""Online audio alignment package.

This package provides scalable and extensible code for online audio alignment
with implementations of NOA (Naive Online Alignment) and OLTW (Online Time Warping).

The package is organized into core modules:
    - core.alignment: Alignment algorithms (online and offline)
    - core.cost: Cost metrics for computing distances
    - core.features: Feature extraction (online and offline)
    - core.timescale: Time scale modification (online and offline)
"""

# Core modules
from . import core

# Alignment algorithms
from .core.alignment import AlignmentBase, OnlineAlignment
from .core.alignment.online import NOA, OLTW

# Cost metrics
from .core.cost import (
    CostMetric,
    CosineDistance,
    EuclideanDistance,
    get_cost_metric,
)

# Feature extraction
from .core.features import (
    FeatureExtractor,
    OnlineFeatureExtractor,
    OfflineFeatureExtractor,
)

__version__ = "0.1.0"

__all__ = [
    # Core modules
    "core",
    # Alignment
    "AlignmentBase",
    "OnlineAlignment",
    "NOA",
    "OLTW",
    # Cost metrics
    "CostMetric",
    "CosineDistance",
    "EuclideanDistance",
    "get_cost_metric",
    # Features
    "FeatureExtractor",
    "OnlineFeatureExtractor",
    "OfflineFeatureExtractor",
]
