"""Cost metric registry and factory functions."""

# standard imports
from typing import Callable

# custom imports
from .cost_metric import CostMetric
from .cosine import CosineDistance
from .euclidean import EuclideanDistance

# Registry of supported cost metrics
_COST_REGISTRY: dict[str, CostMetric] = {
    "cosine": CosineDistance(),
    "euclidean": EuclideanDistance(),
}


def get_cost_metric(
    cost_metric: str | Callable | CostMetric,
) -> CostMetric:
    """Get a CostMetric instance from various input types.

    This function provides a convenient way to create CostMetric instances
    from string names, callable functions, or existing CostMetric instances.

    Args:
        cost_metric: Can be:
            - A string name ("cosine", "euclidean")
            - A callable function that computes vector-to-vector distance
            - An existing CostMetric instance

    Returns:
        CostMetric instance ready to use.

    Raises:
        ValueError: If string name is not in the registry.
        TypeError: If input type is not supported.

    Examples:
        >>> metric = get_cost_metric("cosine")
        >>> metric = get_cost_metric(CosineDistance())
        >>> metric = get_cost_metric(my_custom_distance_function)
    """
    if isinstance(cost_metric, CostMetric):
        return cost_metric
    if isinstance(cost_metric, str):
        return _get_cost_from_str(cost_metric)
    if isinstance(cost_metric, Callable):
        return _get_cost_from_callable(cost_metric)
    raise TypeError(
        f'Variable "cost_metric" must be str, Callable, or CostMetric, '
        f"got {type(cost_metric).__name__}"
    )


def _get_cost_from_str(cost_metric: str) -> CostMetric:
    """Get cost metric from string name.

    Args:
        cost_metric: String name of the cost metric.

    Returns:
        CostMetric: Desired cost metric instance.

    Raises:
        ValueError: If cost metric name is not supported.
    """
    if cost_metric not in _COST_REGISTRY:
        raise ValueError(
            f"Input cost metric '{cost_metric}' not yet supported. "
            f"Supported metrics: {list[str](_COST_REGISTRY.keys())}"
        )
    return _COST_REGISTRY[cost_metric]


def _get_cost_from_callable(cost_metric: Callable) -> CostMetric:
    """Create cost metric from a callable distance function.

    Args:
        cost_metric: Vector-to-vector distance function.

    Returns:
        CostMetric: Custom-built cost metric based on the provided function.
    """
    # TODO: Add validation that cost_metric is a valid v2v cost function
    return CostMetric(v2v_cost=cost_metric, name=cost_metric.__name__)
