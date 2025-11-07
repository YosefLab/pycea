from __future__ import annotations
from typing import Protocol, Callable, Dict, Any, Union
import numpy as np
from typing import Literal

_AggregatorFn = Callable[[np.ndarray], np.ndarray | float]
_Aggregator = Literal["mean", "median", "sum", "var"]


class _AggregatorFn(Protocol):
    """Callable that maps (n, k) -> (k,) and (n,) -> scalar."""
    def __call__(self, X: np.ndarray) -> np.ndarray | float: ...


class _Mean:
    def __call__(self, X: np.ndarray) -> np.ndarray | float:
        return np.mean(X, axis=0) if X.ndim == 2 else np.mean(X)


class _Median:
    def __call__(self, X: np.ndarray) -> np.ndarray | float:
        return np.median(X, axis=0) if X.ndim == 2 else np.median(X)


class _Sum:
    def __call__(self, X: np.ndarray) -> np.ndarray | float:
        return np.sum(X, axis=0) if X.ndim == 2 else np.sum(X)


class _Var:
    def __call__(self, X: np.ndarray) -> np.ndarray | float:
        # match R's var() => sample variance (ddof=1)
        return np.var(X, axis=0, ddof=1) if X.ndim == 2 else np.var(X, ddof=1)


class Aggregator:
    """Factory for simple aggregation functions (mean, median, sum, var)."""

    _REGISTRY: Dict[str, Callable[[], _AggregatorFn]] = {
        "mean": _Mean,
        "median": _Median,
        "sum": _Sum,
        "var": _Var,
    }

    @classmethod
    def get_aggregator(cls, name_or_fn: _Aggregator | _AggregatorFn) -> _AggregatorFn:
        """Return an aggregator function, either predefined or custom."""
        # If user directly passed a callable, return it as-is
        if callable(name_or_fn) and not isinstance(name_or_fn, str):
            return name_or_fn

        # Otherwise, look up the predefined aggregator
        key = str(name_or_fn).lower().strip()
        if key not in cls._REGISTRY:
            raise ValueError(
                f"Unknown aggregator '{name_or_fn}'. "
                f"Available: {', '.join(cls._REGISTRY)}"
            )
        return cls._REGISTRY[key]()
