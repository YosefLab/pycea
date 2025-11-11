from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

_AggregatorFn = Callable[[np.ndarray], np.ndarray | float]
_Aggregator = Literal["mean", "median", "sum", "var"]


def _reduce(fn, X: NDArray[np.generic]) -> NDArray[np.generic] | float:
    axis = 0 if X.ndim == 2 else None
    return fn(X, axis=axis) if axis is not None else fn(X)


def _var(X: NDArray[np.generic]) -> NDArray[np.generic] | float:
    axis = 0 if X.ndim == 2 else None
    return np.var(X, axis=axis, ddof=1)


_REGISTRY: dict[str, _AggregatorFn] = {
    "mean": lambda X: _reduce(np.mean, X),
    "median": lambda X: _reduce(np.median, X),
    "sum": lambda X: _reduce(np.sum, X),
    "var": _var,
}


def _get_aggregator(name_or_fn: _Aggregator | _AggregatorFn) -> _AggregatorFn:
    """Return a predefined aggregator or accept a custom callable."""
    if callable(name_or_fn) and not isinstance(name_or_fn, str):
        return name_or_fn

    key = str(name_or_fn).lower().strip()
    try:
        return _REGISTRY[key]
    except KeyError:
        raise ValueError(f"Unknown aggregator '{name_or_fn}'. Available: {', '.join(_REGISTRY)}") from None
